import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

class FrappeStudentModel(nn.Module):
    """
    轻量级Frappe CTR预测学生模型
    
    使用知识蒸馏和迁移学习从教师模型获取知识
    """
    def __init__(
        self,
        sparse_features: List[str],
        embedding_dim: int = 10,
        hidden_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化学生模型
        
        Args:
            sparse_features: 稀疏特征列表
            embedding_dim: 嵌入维度(需与教师模型一致以便迁移学习)
            hidden_units: 隐藏层单元数
            dropout_rate: Dropout率
            l2_reg: L2正则化系数
            device: 计算设备
        """
        super(FrappeStudentModel, self).__init__()
        
        self.sparse_features = sparse_features
        self.embedding_dim = embedding_dim
        self.feature_num = len(sparse_features)
        self.device = device
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        # 存储特征映射
        self.feature_info = {}
        self.encoders = {}
        
        # 创建嵌入层
        self.embedding_dict = nn.ModuleDict()
        self.embedding_initialized = False
        
        # 交互层 - 简化的特征交互
        interaction_input_dim = self.feature_num * embedding_dim
        
        # 简化的DNN层 - 只使用1-2个隐藏层
        self.dnn = nn.Sequential()
        input_dim = interaction_input_dim
        
        # 添加隐藏层 (最多2层)
        for i, units in enumerate(hidden_units):
            self.dnn.add_module(f'dense_{i}', nn.Linear(input_dim, units))
            self.dnn.add_module(f'bn_{i}', nn.BatchNorm1d(units))
            self.dnn.add_module(f'relu_{i}', nn.ReLU())
            self.dnn.add_module(f'dropout_{i}', nn.Dropout(dropout_rate))
            input_dim = units
        
        # 输出层
        self.dnn_linear = nn.Linear(input_dim, 1)
        
        # 蒸馏温度
        self.temperature = 1.0
        
        # 迁移学习配置
        self.is_transfer_learning = False
        
        # 将模型移至指定设备
        self.to(device)
    
    def initialize_embeddings(self, vocab_sizes):
        """
        初始化嵌入层并确保词汇表足够大
        
        Args:
            vocab_sizes: 每个特征的词汇表大小字典
        """
        for feat in self.sparse_features:
            # 确保词汇表至少有1000个条目(或者使用from teacher获取的)
            size = max(vocab_sizes.get(feat, 1000), 1000)
            self.embedding_dict[feat] = nn.Embedding(
                size, 
                self.embedding_dim,
                sparse=False
            ).to(self.device)
        
        self.embedding_initialized = True
        
    def transfer_embeddings(self, teacher_model):
        """
        从教师模型迁移嵌入层参数
        
        Args:
            teacher_model: 教师模型实例
        """
        print("从教师模型迁移嵌入参数...")
        
        # 获取教师模型的嵌入参数
        teacher_embeddings = None
        
        # 方法1: 尝试直接从模型的embedding_dict获取
        try:
            if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'embedding_dict'):
                teacher_embeddings = teacher_model.model.embedding_dict
            elif hasattr(teacher_model, 'embedding_dict'):
                teacher_embeddings = teacher_model.embedding_dict
        except Exception as e:
            print(f"直接获取教师嵌入失败: {e}")
        
        # 方法2: 通过get_hidden_outputs获取
        if teacher_embeddings is None and hasattr(teacher_model, 'get_hidden_outputs'):
            try:
                # 构造一个小的样本数据集
                sample_input = {feat: np.array([0]) for feat in self.sparse_features}
                hidden_outputs = teacher_model.get_hidden_outputs(sample_input)
                if 'model_params' in hidden_outputs:
                    teacher_params = hidden_outputs['model_params']
                    # 提取嵌入层参数
                    for feat in self.sparse_features:
                        embed_key = f'embedding_dict.{feat}.weight'
                        if embed_key in teacher_params:
                            if feat in self.embedding_dict:
                                # 迁移参数
                                with torch.no_grad():
                                    self.embedding_dict[feat].weight.data[:len(teacher_params[embed_key])] = teacher_params[embed_key]
                                print(f"已迁移特征 '{feat}' 的嵌入")
                            else:
                                print(f"学生模型中缺少特征 '{feat}' 的嵌入层")
                    self.is_transfer_learning = True
                    return True
            except Exception as e:
                print(f"通过hidden_outputs获取嵌入失败: {e}")
        
        # 方法3: 直接从存储的模型参数中获取
        if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'state_dict'):
            try:
                teacher_state_dict = teacher_model.model.state_dict()
                transfer_count = 0
            
                for feat in self.sparse_features:
                    teacher_key = f'embedding_dict.{feat}.weight'
                    if teacher_key in teacher_state_dict and feat in self.embedding_dict:
                        # 获取教师权重和学生权重的形状
                        teacher_weight = teacher_state_dict[teacher_key].to(self.device)  # 移至正确设备
                    
                        # 确定可以迁移的行数
                        rows = min(teacher_weight.shape[0], self.embedding_dict[feat].weight.shape[0])
                        cols = min(teacher_weight.shape[1], self.embedding_dict[feat].weight.shape[1])

                        # 迁移权重
                        with torch.no_grad():
                            self.embedding_dict[feat].weight.data[:rows, :cols] = teacher_weight[:rows, :cols]
                    
                        transfer_count += 1

                if transfer_count > 0:
                    print(f"成功从状态字典迁移了 {transfer_count} 个特征的嵌入")
                    self.is_transfer_learning = True
                    return True
            except Exception as e:
                print(f"从状态字典迁移嵌入失败: {e}")
    
        print("嵌入迁移失败，使用随机初始化")
        return False
    
    def forward(self, x):
        """
        前向传播，增加索引安全检查
        
        Args:
            x: 输入特征字典 {feature_name: tensor}
                
        Returns:
            预测logits
        """
        # 特征嵌入
        sparse_embedding_list = []
        
        for feat in self.sparse_features:
            if feat in self.embedding_dict and feat in x:
                # 获取输入特征张量并确保移至正确的设备
                feat_tensor = x[feat].to(self.device)
                
                # 添加安全检查：确保索引不超出嵌入表大小
                vocab_size = self.embedding_dict[feat].weight.shape[0]
                # 将超出范围的索引截断到有效范围内 (使用clamp)
                feat_tensor = torch.clamp(feat_tensor, 0, vocab_size-1)
                
                # 获取嵌入
                embed = self.embedding_dict[feat](feat_tensor)
                sparse_embedding_list.append(embed)
                    
        # 检查是否有嵌入
        if not sparse_embedding_list:
            raise ValueError("没有可用的特征嵌入")
                
        # 连接所有嵌入
        sparse_embedding = torch.cat(sparse_embedding_list, dim=-1)
        
        # 压平成二维张量以便输入DNN
        batch_size = sparse_embedding.shape[0]
        flat_sparse_embedding = sparse_embedding.reshape(batch_size, -1)
        
        # 通过DNN层
        dnn_output = self.dnn(flat_sparse_embedding)
        
        # 最终输出层
        y_pred = torch.sigmoid(self.dnn_linear(dnn_output))
        
        return y_pred
    
    def _compute_loss(self, y_pred, y_true, teacher_pred=None, alpha=0.5):
        """
        计算损失函数，支持知识蒸馏
        
        Args:
            y_pred: 学生模型预测
            y_true: 真实标签
            teacher_pred: 教师模型预测（用于知识蒸馏）
            alpha: 蒸馏损失权重
            
        Returns:
            总损失
        """
        # 硬目标损失 - 二元交叉熵
        hard_loss = F.binary_cross_entropy(y_pred.view(-1), y_true.float())
        
        # 如果有教师预测，计算蒸馏损失
        if teacher_pred is not None:
            # 软目标损失 - KL散度
            student_logits = torch.log(y_pred / (1 - y_pred + 1e-7) + 1e-7) / self.temperature
            teacher_probs = teacher_pred.view(-1)
            teacher_logits = torch.log(teacher_probs / (1 - teacher_probs + 1e-7) + 1e-7) / self.temperature
            
            soft_loss = F.kl_div(
                F.log_softmax(student_logits.view(-1, 1), dim=1),
                F.softmax(teacher_logits.view(-1, 1), dim=1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            # 总损失 = alpha * 软目标损失 + (1 - alpha) * 硬目标损失
            return alpha * soft_loss + (1 - alpha) * hard_loss
        
        # 如果没有教师预测，只使用硬目标损失
        return hard_loss


class StudentFrappeTrainer:
    """学生模型训练器，支持知识蒸馏和迁移学习"""
    
    def __init__(
        self,
        model,
        teacher_model=None,
        learning_rate=0.001,
        weight_decay=1e-5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化训练器
        
        Args:
            model: 学生模型实例
            teacher_model: 教师模型实例（可选）
            learning_rate: 学习率
            weight_decay: 权重衰减系数
            device: 计算设备
        """
        self.model = model
        self.teacher_model = teacher_model
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 是否使用知识蒸馏
        self.use_distillation = teacher_model is not None
        
        # 蒸馏超参数
        self.distill_alpha = 0.5  # 蒸馏损失权重
        self.temperature = 3.0    # 蒸馏温度
        
        if self.use_distillation:
            self.model.temperature = self.temperature
    
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs=10,
        distill_alpha=0.5,
        temperature=3.0,
        patience=3,
        verbose=True
    ):
        """
        训练模型
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器（可选）
            epochs: 训练轮数
            distill_alpha: 蒸馏损失权重
            temperature: 蒸馏温度
            patience: 早停耐心值
            verbose: 是否打印详细信息
            
        Returns:
            训练历史记录
        """
        # 设置蒸馏参数
        self.distill_alpha = distill_alpha
        self.temperature = temperature
        self.model.temperature = temperature
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': []
        }
        
        # 早停设置
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # 开始训练
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            epoch_loss = 0
            batch_count = 0
            
            # 遍历训练批次
            for batch in train_dataloader:
                batch_count += 1
                x, y_true = batch
                
                # 将数据移至指定设备
                x = {k: v.to(self.device) for k, v in x.items()}
                y_true = y_true.float().to(self.device)
                
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 学生模型前向传播
                y_pred = self.model(x)
                
                # 如果使用知识蒸馏，获取教师预测
                teacher_pred = None
                # 在训练器的 train 方法中，修改获取教师预测的部分
                if self.use_distillation:
                    with torch.no_grad():
                        if hasattr(self.teacher_model, 'predict'):
                            try:
                                # 将学生模型的输入张量字典转换为适合教师模型的 DataFrame 格式
                                import pandas as pd
                                x_for_teacher = {feat: x[feat].cpu().numpy().flatten()
                                                 for feat in self.teacher_model.sparse_features if feat in x}
                                x_for_teacher_df = pd.DataFrame(x_for_teacher)
                                
                                # 调用教师模型的预测方法
                                teacher_pred = torch.tensor(
                                    self.teacher_model.predict(x_for_teacher_df),
                                    device=self.device
                                )
                            except Exception as e:
                                print(f"获取教师预测出错: {e}")
                                # 出错时不使用教师预测
                                teacher_pred = None
            
                
                # 计算损失
                loss = self.model._compute_loss(y_pred, y_true, teacher_pred, self.distill_alpha)
                
                # 反向传播与优化
                loss.backward()
                self.optimizer.step()
                
                # 累加损失
                epoch_loss += loss.item()
            
            # 计算平均训练损失
            avg_train_loss = epoch_loss / batch_count
            history['train_loss'].append(avg_train_loss)
            
            # 验证
            val_metrics = None
            if val_dataloader:
                val_metrics = self.evaluate(val_dataloader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_auc'].append(val_metrics['auc'])
                
                # 早停检查
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # 打印训练信息
            if verbose:
                log_message = f"Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.4f}"
                if val_metrics:
                    log_message += f", 验证损失: {val_metrics['loss']:.4f}, 验证AUC: {val_metrics['auc']:.4f}"
                print(log_message)
            
            # 检查是否早停
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                # 恢复最佳模型
                if best_model_state:
                    self.model.load_state_dict(best_model_state)
                break
        
        return history
    
    def evaluate(self, dataloader):
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            包含评估指标的字典
        """
        self.model.eval()
        total_loss = 0
        y_true_list = []
        y_pred_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                x, y_true = batch
                x = {k: v.to(self.device) for k, v in x.items()}
                y_true = y_true.float().to(self.device)
                
                # 学生模型预测
                y_pred = self.model(x)
                
                # 计算损失
                loss = self.model._compute_loss(y_pred, y_true)
                total_loss += loss.item()
                
                # 保存预测和真实值
                y_true_list.extend(y_true.cpu().numpy())
                y_pred_list.extend(y_pred.view(-1).cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        
        # 计算AUC
        auc = roc_auc_score(y_true_list, y_pred_list)
        
        # 计算准确率
        y_pred_binary = (np.array(y_pred_list) > 0.5).astype(int)
        accuracy = accuracy_score(y_true_list, y_pred_binary)
        
        return {
            'loss': avg_loss,
            'auc': auc,
            'accuracy': accuracy
        }


class FrappeDataset(Dataset):
    """Frappe数据集包装器"""
    
    def __init__(self, data, feature_cols, label_col='label', encoders=None):
        """
        初始化数据集
        
        Args:
            data: pandas DataFrame
            feature_cols: 特征列名列表
            label_col: 标签列名
            encoders: 特征编码器字典
        """
        self.data = data
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.encoders = encoders
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本，不移动到设备上（由DataLoader负责）"""
        # 获取当前样本
        row = self.data.iloc[idx]
        
        # 构建特征字典，创建但不移动到设备
        x = {col: torch.tensor([row[col]], dtype=torch.long) for col in self.feature_cols}
        
        # 获取标签
        y = torch.tensor(row[self.label_col], dtype=torch.float)
        
        return x, y


def create_student_model(teacher_model, sparse_features=None, hidden_units=None):
    """
    根据教师模型创建学生模型
    
    Args:
        teacher_model: 教师模型实例
        sparse_features: 特征列表，如为None则从teacher获取
        hidden_units: 隐藏层单元数列表
        
    Returns:
        学生模型实例
    """
    # 从教师模型获取特征信息
    if sparse_features is None:
        if hasattr(teacher_model, 'sparse_features'):
            sparse_features = teacher_model.sparse_features
        else:
            raise ValueError("未指定特征列表且无法从教师模型获取")
    
    # 设置默认隐藏层单元数
    if hidden_units is None:
        hidden_units = [64, 32]
    
    # 获取嵌入维度
    embedding_dim = 10
    if hasattr(teacher_model, 'embedding_dim'):
        embedding_dim = teacher_model.embedding_dim
    
    # 创建学生模型
    student_model = FrappeStudentModel(
        sparse_features=sparse_features,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
        dropout_rate=0.1,  # 轻量级模型使用较小的dropout
        l2_reg=1e-6  # 较小的正则化
    )
    
    return student_model


def distill_knowledge(
    teacher_model,
    train_data,
    val_data=None,
    batch_size=1024,
    epochs=10,
    hidden_units=None,
    learning_rate=0.001,
    distill_alpha=0.7,
    temperature=3.0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    通过知识蒸馏训练学生模型
    
    Args:
        teacher_model: 教师模型实例
        train_data: 训练数据
        val_data: 验证数据
        batch_size: 批次大小
        epochs: 训练轮数
        hidden_units: 隐藏层单元数列表
        learning_rate: 学习率
        distill_alpha: 蒸馏损失权重
        temperature: 蒸馏温度
        
    Returns:
        训练好的学生模型和训练历史
    """
    print("开始知识蒸馏过程...")
    
    # 获取特征列和词汇表大小
    sparse_features = teacher_model.sparse_features
    
    # 创建学生模型
    print(f"创建学生模型, 隐藏层: {hidden_units if hidden_units else [64, 32]}")
    student_model = create_student_model(teacher_model, sparse_features, hidden_units)
    student_model = student_model.to(device)

    # 获取词汇表大小
    vocab_sizes = {}
    if hasattr(teacher_model, 'feature_info') and 'vocab_sizes' in teacher_model.feature_info:
        vocab_sizes = teacher_model.feature_info['vocab_sizes']
    elif hasattr(teacher_model, 'encoders'):
        # 从编码器推断词汇表大小
        for feat, encoder in teacher_model.encoders.items():
            if hasattr(encoder, 'classes_'):
                vocab_sizes[feat] = len(encoder.classes_) + 1  # +1 for unknown values
    
    # 初始化学生模型的嵌入层
    student_model.initialize_embeddings(vocab_sizes)
    
    # 迁移嵌入
    student_model.transfer_embeddings(teacher_model)
    
    # 创建数据集和数据加载器
    train_dataset = FrappeDataset(train_data, sparse_features)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if val_data is not None:
        val_dataset = FrappeDataset(val_data, sparse_features)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建训练器
    trainer = StudentFrappeTrainer(
        model=student_model,
        teacher_model=teacher_model,
        learning_rate=learning_rate
    )
    
    # 训练模型
    print(f"开始训练学生模型, epochs={epochs}, batch_size={batch_size}, 蒸馏权重={distill_alpha}, 温度={temperature}")
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        distill_alpha=distill_alpha,
        temperature=temperature
    )
    
    return student_model, history


def save_student_model(model, path):
    """保存学生模型"""
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # 构建模型状态
    model_state = {
        'model_state_dict': model.state_dict(),
        'sparse_features': model.sparse_features,
        'embedding_dim': model.embedding_dim,
        'is_transfer_learning': model.is_transfer_learning
    }
    
    # 保存模型
    torch.save(model_state, path)
    print(f"学生模型已保存至 {path}")


def load_student_model(path, hidden_units=None):
    """加载学生模型"""
    checkpoint = torch.load(path)
    
    # 读取模型配置
    sparse_features = checkpoint['sparse_features']
    embedding_dim = checkpoint['embedding_dim']
    
    # 创建模型
    if hidden_units is None:
        hidden_units = [64, 32]
        
    model = FrappeStudentModel(
        sparse_features=sparse_features,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置迁移学习标志
    if 'is_transfer_learning' in checkpoint:
        model.is_transfer_learning = checkpoint['is_transfer_learning']
    
    return model


if __name__ == "__main__":
    # 加载教师模型(假设已经训练好)
    from teacher_frappe import CTRFrappeTeacherModel, load_frappe_dataset
    import os
    
    print("测试学生模型框架...")
    
    # 获取当前脚本所在目录的上级目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    teacher_model_path = os.path.join(base_dir, "models", "teacher_frappe_model.pth")
    
    # 加载数据和教师模型
    try:
        print("加载Frappe数据集...")
        train_data, test_data = load_frappe_dataset(use_stratify=True)
        
        print("加载教师模型...")
        teacher_model = CTRFrappeTeacherModel()
        teacher_model.load_model(teacher_model_path)
        
        # 创建并训练学生模型
        print("创建学生模型...")
        # 使用较小的隐藏层
        hidden_units = [64, 32]
        
        # 知识蒸馏
        print("开始知识蒸馏...")
        student_model, history = distill_knowledge(
            teacher_model,
            train_data.sample(5000),  # 使用小数据集进行演示
            test_data.sample(1000),   # 使用小的验证集
            batch_size=512,
            epochs=10,                 # 演示时使用较少的轮数
            hidden_units=hidden_units,
            distill_alpha=0.7,        # 较高的蒸馏权重
            temperature=3.0           # 较高温度使软目标更平滑
        )
        
        # 保存学生模型
        student_model_path = os.path.join(base_dir, "models", "student_frappe_0_model.pth")
        save_student_model(student_model, student_model_path)
        print(f"学生模型已保存至 {student_model_path}")
        
    except Exception as e:
        print(f"运行时出错: {e}")
        import traceback
        traceback.print_exc()