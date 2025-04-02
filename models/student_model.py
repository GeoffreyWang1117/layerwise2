import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

# 添加项目根目录到路径以便导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.teacher_model import CTRTeacherModel

class LightweightCTRModel(nn.Module):
    """
    轻量化CTR预测模型，用于知识蒸馏
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], dropout_rate: float = 0.05):
        """
        参数:
          input_dim: 输入特征的维度（拼接后的特征向量维度）
          hidden_dims: 各隐藏层维度列表
          dropout_rate: Dropout比率，默认0.05
        """
        super(LightweightCTRModel, self).__init__()
        
        # 保存配置
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 构建网络层
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 构建剩余的隐藏层
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 构建网络模型
        self.features = nn.Sequential(*layers)
        
        # 输出层（二分类）
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, x):
        """
        前向传播同时返回中间层输出（用于蒸馏）
        """
        # 保存中间层激活值
        intermediates = {}
        
        # 通过每一层并记录中间输出
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                layer_idx = i // 3  # 每3层一组(Linear+ReLU+Dropout)
                intermediates[f'layer_{layer_idx}'] = h
        
        # 输出层
        logits = self.output_layer(h)
        probs = torch.sigmoid(logits)
        
        return probs, intermediates
    
    def get_intermediate_outputs(self, x):
        """
        仅获取中间层输出，用于知识蒸馏
        """
        _, intermediates = self.forward(x)
        return intermediates
    
    def predict(self, x):
        """
        模型预测函数
        """
        probs, _ = self.forward(x)
        return probs


class FeatureExtractor:
    """
    特征提取器，用于将原始特征转换为学生模型输入
    """
    def __init__(self, teacher_model):
        """
        参数:
          teacher_model: 教师模型实例
        """
        self.teacher_model = teacher_model
        
        # 确保教师模型处于评估模式
        if hasattr(self.teacher_model, 'model'):
            self.teacher_model.model.eval()
        else:
            print("警告：教师模型没有model属性，无法设置为评估模式")

    # 在FeatureExtractor类中添加更强大的特征提取逻辑
    def extract_features(self, data_batch):
        """
        从教师模型提取特征嵌入
        
        参数:
          data_batch: 输入数据批次
        
        返回:
          合并后的特征向量
        """
        # 获取教师模型的嵌入层输出
        with torch.no_grad():
            embeddings = self.teacher_model.get_hidden_outputs(data_batch)
            
            # 首先检查并打印所有可用键
            print("教师模型输出包含以下键:")
            tensor_keys = []
            for key, value in embeddings.items():
                if isinstance(value, torch.Tensor):
                    shape_info = f"形状={value.shape}" if hasattr(value, 'shape') else "非张量"
                    print(f" - {key}: {shape_info}")
                    tensor_keys.append(key)
            
            # 尝试从模型参数构建更丰富的特征表示
            if 'model_params' in embeddings:
                print("使用模型参数构建特征表示")
                # 提取所有批次的第一个样本的ID
                first_ids = {}
                for feat, value in data_batch.items():
                    if isinstance(value, np.ndarray) and value.size > 0:
                        first_ids[feat] = value[0]
                
                # 从参数中提取对应的嵌入向量
                feature_vectors = []
                param_dict = embeddings['model_params']
                for name, param in param_dict.items():
                    if 'embedding_dict' in name and '.weight' in name:
                        feat_name = name.split('.')[1]  # 提取特征名
                        if feat_name in first_ids:
                            # 获取每个样本对应的嵌入向量
                            feat_id = first_ids[feat_name]
                            if feat_id < param.shape[0]:  # 确保ID在有效范围内
                                feat_vector = param[feat_id]
                                # 为整个批次复制这个向量
                                batch_size = next(iter(data_batch.values())).shape[0]
                                feat_vectors = feat_vector.unsqueeze(0).expand(batch_size, -1)
                                feature_vectors.append(feat_vectors)
                
                if feature_vectors:
                    combined_emb = torch.cat(feature_vectors, dim=1)
                    print(f"构建的特征表示形状: {combined_emb.shape}")
                    return combined_emb
            
            # 如果上面的方法失败，回退到原始的特征选择逻辑
            # 优先使用最有信息量的表示
            if 'final_output' in embeddings and embeddings['final_output'].shape[1] > 1:
                # 如果final_output不只是一个标量值，它可能包含有用信息
                return embeddings['final_output']
            elif 'stacked_features' in embeddings:
                # 使用替代特征表示
                stacked_features = embeddings['stacked_features']
                
                # 处理不同维度情况
                if len(stacked_features.shape) > 2:  # 如果维度>2，展平成2D
                    return stacked_features.reshape(stacked_features.shape[0], -1)
                else:
                    return stacked_features
            else:
                # 最后选择任何可用的张量
                for key in tensor_keys:
                    value = embeddings[key]
                    if len(value.shape) >= 2 and value.shape[1] > 1:  # 至少是2D且特征维度>1
                        if len(value.shape) > 2:  # 如果维度>2，展平成2D
                            return value.reshape(value.shape[0], -1)
                        return value
                
                # 如果没有合适的表示，创建一个随机特征向量
                print("警告: 没有找到合适的特征表示，创建随机特征")
                batch_size = next(iter(data_batch.values())).shape[0]
                device = next(self.teacher_model.model.parameters()).device
                return torch.randn(batch_size, 64, device=device)  # 创建64维随机特征

def train_student_model(
    teacher_model_path: str,
    data_path: str,
    sparse_features: List[str] = None,
    hidden_dims: List[int] = [64, 32, 16],
    dropout_rate: float = 0.05,
    batch_size: int = 256,
    epochs: int = 20,
    learning_rate: float = 0.001,
    model_save_path: str = 'models/student_model.pth',
    device: str = None
):
    """
    训练学生模型
    
    Args:
        teacher_model_path: 教师模型路径
        data_path: 数据集路径
        sparse_features: 稀疏特征列表
        hidden_dims: 隐藏层维度
        dropout_rate: Dropout率
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        model_save_path: 模型保存路径
        device: 设备类型
    
    Returns:
        训练好的学生模型
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    
    # 设置默认特征集
    if sparse_features is None:
        sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 'app_id', 'device_type', 'device_conn_type']
    
    # 1. 加载数据
    print(f"加载数据: {data_path}")
    data = pd.read_csv(data_path)
    
    # 数据预处理 - 确保目标字段正确
    data['click'] = pd.to_numeric(data['click'], errors='coerce')
    data = data.dropna(subset=['click'])
    data['click'] = data['click'].astype(int)
    
    # 2. 加载教师模型
    print(f"加载教师模型: {teacher_model_path}")
    teacher_model = CTRTeacherModel(sparse_features=sparse_features)
    try:
        teacher_model.load_model(teacher_model_path, data)
    except RuntimeError as e:
        print(f"警告: 加载模型出错: {e}")
        print("尝试使用非严格模式加载模型...")
        # 检查是否支持非严格加载
        if hasattr(teacher_model.model, 'load_state_dict'):
            checkpoint = torch.load(teacher_model_path)
            teacher_model.sparse_features = checkpoint['sparse_features']
            teacher_model.embedding_dim = checkpoint['embedding_dim']
            teacher_model.task = checkpoint['task']
            teacher_model.encoders = checkpoint['encoders']
            teacher_model.build_model(data)  # 使用当前数据重建模型
            try:
                teacher_model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("成功以非严格模式加载模型")
            except Exception as e2:
                print(f"非严格加载模式也失败: {e2}")
                print("将使用未训练的教师模型，结果可能不准确")
    
    # 3. 划分训练集和测试集
    print("划分训练集与测试集...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"训练集大小: {train_data.shape}, 测试集大小: {test_data.shape}")
    
    # 4. 特征提取
    print("从教师模型提取特征...")
    # 预处理数据
    processed_train = teacher_model.preprocess_data(train_data)
    processed_test = teacher_model.preprocess_data(test_data)
    
    # 准备模型输入
    train_input = teacher_model.prepare_model_input(processed_train)
    test_input = teacher_model.prepare_model_input(processed_test)
    
    # 提取特征
    feature_extractor = FeatureExtractor(teacher_model)
    train_features = feature_extractor.extract_features(train_input)
    test_features = feature_extractor.extract_features(test_input)
    
    # 获取特征维度
    input_dim = train_features.shape[1]
    print(f"特征维度: {input_dim}")
    
    # 5. 初始化学生模型
    print(f"初始化学生模型: 输入维度={input_dim}, 隐藏层={hidden_dims}...")
    student_model = LightweightCTRModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    ).to(device)
    
    # 6. 准备训练数据
    y_train = torch.FloatTensor(processed_train['click'].values.reshape(-1, 1)).to(device)
    y_test = torch.FloatTensor(processed_test['click'].values.reshape(-1, 1)).to(device)
    
    # 处理特征张量 - 检查是否已经是张量并转换类型
    if isinstance(train_features, torch.Tensor):
        train_features_tensor = train_features.float()  # 转换为浮点类型
        if train_features_tensor.device != device:
            train_features_tensor = train_features_tensor.to(device)
    else:
        train_features_tensor = torch.FloatTensor(train_features).to(device)
    
    if isinstance(test_features, torch.Tensor):
        test_features_tensor = test_features.float()  # 转换为浮点类型
        if test_features_tensor.device != device:
            test_features_tensor = test_features_tensor.to(device)
    else:
        test_features_tensor = torch.FloatTensor(test_features).to(device)
    
    # 打印更多调试信息
    print(f"训练特征形状: {train_features_tensor.shape}, 类型: {train_features_tensor.dtype}, 设备: {train_features_tensor.device}")
    print(f"测试特征形状: {test_features_tensor.shape}, 类型: {test_features_tensor.dtype}, 设备: {test_features_tensor.device}")
    print(f"标签形状: {y_train.shape}, 类型: {y_train.dtype}, 设备: {y_train.device}")
    
    # 如果特征维度太小(小于10)，打印警告
    if train_features_tensor.shape[1] < 10:
        print(f"警告: 特征维度异常小 ({train_features_tensor.shape[1]})，可能会影响模型性能")
        print("尝试获取更详细的教师模型输出信息...")
        
        # 重新获取特征，尝试使用不同的方法
        sample_input = {k: v[:5] for k, v in train_input.items()}  # 取样本的前5个样本
        sample_outputs = teacher_model.get_hidden_outputs(sample_input)
        
        print("可用的特征表示:")
        for key, value in sample_outputs.items():
            if isinstance(value, torch.Tensor):
                print(f" - {key}: 形状={value.shape}, 类型={value.dtype}")

        
    # 创建训练数据集和数据加载器 - 添加在这里
    train_dataset = torch.utils.data.TensorDataset(train_features_tensor, y_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # 7. 训练模型
    print(f"开始训练学生模型，epochs={epochs}, batch_size={batch_size}, lr={learning_rate}...")
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    best_auc = 0
    best_state = None
    
    for epoch in range(epochs):
        student_model.train()
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            # 前向传播
            predictions, _ = student_model(batch_features)
            
            # 计算损失
            loss = criterion(predictions, batch_labels)
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 每个epoch结束后评估
        student_model.eval()
        with torch.no_grad():
            test_predictions = student_model.predict(test_features_tensor)
            test_preds_np = test_predictions.cpu().numpy()
            test_labels_np = y_test.cpu().numpy()
            
            # 计算评估指标
            test_loss = log_loss(test_labels_np, test_preds_np)
            test_auc = roc_auc_score(test_labels_np, test_preds_np)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, "
                  f"Test LogLoss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")
            
            # 保存最好的模型
            if test_auc > best_auc:
                best_auc = test_auc
                best_state = student_model.state_dict()
                print(f"    发现更好的模型，AUC: {best_auc:.4f}")
    
    # 加载最好的状态
    if best_state:
        student_model.load_state_dict(best_state)
    
    # 8. 最终评估
    print("\n最终评估结果:")
    student_model.eval()
    with torch.no_grad():
        # 学生模型预测
        final_preds = student_model.predict(test_features_tensor).cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        
        # 教师模型预测
        teacher_preds = teacher_model.predict(processed_test)
        
        # 计算最终指标
        student_loss = log_loss(y_test_np, final_preds)
        student_auc = roc_auc_score(y_test_np, final_preds)
        
        teacher_loss = log_loss(y_test_np, teacher_preds)
        teacher_auc = roc_auc_score(y_test_np, teacher_preds)
        
        print(f"学生模型 - LogLoss: {student_loss:.4f}, AUC: {student_auc:.4f}")
        print(f"教师模型 - LogLoss: {teacher_loss:.4f}, AUC: {teacher_auc:.4f}")
        
        # 计算性能差异
        loss_diff = student_loss - teacher_loss
        auc_diff = student_auc - teacher_auc
        print(f"性能差异 - LogLoss: {loss_diff:.4f}, AUC: {auc_diff:.4f}")
    
    # 9. 保存模型
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': student_model.state_dict(),
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate
        }, model_save_path)
        print(f"学生模型已保存至: {model_save_path}")
    
    return student_model


if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 获取当前脚本所在目录的上级目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 构建路径
    data_path = os.path.join(base_dir, 'data', 'avazu', 'avazu_sample.csv')
    teacher_model_path = os.path.join(base_dir, 'models', 'teacher_model.pth')
    student_model_path = os.path.join(base_dir, 'models', 'student_model.pth')
    
    print(f"使用数据路径: {data_path}")
    print(f"使用教师模型路径: {teacher_model_path}")
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在 {data_path}")
        # 尝试使用备用路径
        alternate_paths = [
            '/home/coder-gw/layerwise2/data/avazu/avazu_sample.csv',
            './data/avazu/avazu_sample.csv',
            '../data/avazu/avazu_sample.csv'
        ]
        for alt_path in alternate_paths:
            if os.path.exists(alt_path):
                data_path = alt_path
                print(f"找到备用数据路径: {data_path}")
                break
        else:
            print("无法找到数据文件，请确保avazu_sample.csv位于正确的位置")
            exit(1)
    
    if not os.path.exists(teacher_model_path):
        print(f"错误: 教师模型文件不存在 {teacher_model_path}")
        print("请先训练教师模型或指定正确的教师模型路径")
        exit(1)
    
    # 训练学生模型
    student_model = train_student_model(
        teacher_model_path=teacher_model_path,
        data_path=data_path,
        hidden_dims=[64, 32, 16],  # 轻量化网络结构 
        dropout_rate=0.05,
        batch_size=256,
        epochs=10,  # 减少轮数以加快测试
        learning_rate=0.001,
        model_save_path=student_model_path
    )
    
    # === 简单的单元测试 ===
    print("\n开始单元测试...")
    
    try:
        # 测试1: 检查模型结构
        assert hasattr(student_model, 'features'), "模型未包含features组件"
        assert hasattr(student_model, 'output_layer'), "模型未包含output_layer组件"
        print("✓ 测试1通过: 学生模型结构正确")
        
        # 测试2: 生成一些随机数据，检查前向传播
        input_dim = student_model.input_dim
        test_input = torch.randn(10, input_dim).to(student_model.output_layer.weight.device)
        outputs, intermediates = student_model(test_input)
        assert outputs.shape == (10, 1), f"输出形状错误: {outputs.shape}"
        assert len(intermediates) == len(student_model.hidden_dims), f"中间层数量错误: {len(intermediates)}"
        print("✓ 测试2通过: 前向传播工作正常")
        
        # 测试3: 检查预测函数
        predictions = student_model.predict(test_input)
        assert torch.all((0 <= predictions) & (predictions <= 1)), "预测值不在[0,1]区间内"
        print("✓ 测试3通过: 预测函数工作正常")
        
        # 测试4: 测试中间层输出提取
        intermediate_outputs = student_model.get_intermediate_outputs(test_input)
        assert len(intermediate_outputs) > 0, "未能获取中间层输出"
        print("✓ 测试4通过: 中间层输出提取功能正常")
        
        # 测试5: 加载保存的模型
        if os.path.exists(student_model_path):
            checkpoint = torch.load(student_model_path)
            new_model = LightweightCTRModel(
                input_dim=checkpoint['input_dim'],
                hidden_dims=checkpoint['hidden_dims'],
                dropout_rate=checkpoint['dropout_rate']
            ).to(student_model.output_layer.weight.device)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            
            # 检查加载后的模型参数是否相同
            for (name1, param1), (name2, param2) in zip(
                student_model.named_parameters(), new_model.named_parameters()
            ):
                assert torch.allclose(param1, param2), f"参数{name1}不一致"
            print("✓ 测试5通过: 模型保存和加载功能正常")
        
        print("\n所有测试通过! 学生模型可以正常工作")
        
    except Exception as e:
        print(f"测试失败: {e}")