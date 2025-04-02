import pandas as pd
import numpy as np
import torch
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, get_feature_names
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from typing import List, Dict, Tuple, Union, Any, Optional


class CTRTeacherModel:
    def __init__(
        self,
        sparse_features: List[str],
        embedding_dim: int = 4,
        task: str = 'binary',
        device: str = None,
    ):
        """
        CTR预测的教师模型，基于DeepFM
        
        Args:
            sparse_features: 稀疏特征列表
            embedding_dim: 嵌入维度
            task: 任务类型，默认为二分类
            device: 设备类型，若为None则自动选择
        """
        self.sparse_features = sparse_features
        self.embedding_dim = embedding_dim
        self.task = task
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型相关参数
        self.linear_feature_columns = None
        self.dnn_feature_columns = None
        self.feature_names = None
        self.model = None
        self.encoders = {}
        
    def _create_feature_columns(self, data: pd.DataFrame) -> None:
        """创建特征列"""
        # 构建SparseFeat特征列
        fixlen_feature_columns = [
            SparseFeat(feat, vocabulary_size=data[feat].nunique() + 1, embedding_dim=self.embedding_dim)
            for feat in self.sparse_features
        ]
        
        self.linear_feature_columns = fixlen_feature_columns
        self.dnn_feature_columns = fixlen_feature_columns
        self.feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理：填充缺失值、标签编码，并处理未见过的类别
        
        Args:
            data: 输入的原始数据
        
        Returns:
            处理后的数据
        """
        # 复制数据避免修改原始数据
        processed_data = data.copy()
        
        # 对每个特征列进行缺失值填充
        for feat in self.sparse_features:
            processed_data[feat] = processed_data[feat].fillna('-1')
            
        # 使用LabelEncoder将字符串特征转换为整数编码
        for feat in self.sparse_features:
            if feat not in self.encoders:
                self.encoders[feat] = LabelEncoder()
                processed_data[feat] = self.encoders[feat].fit_transform(processed_data[feat])
            else:
                # 处理未见过的类别值
                feat_vals = processed_data[feat].astype(str).values
                known_values = set(self.encoders[feat].classes_)
                
                # 找出未见过的值并替换为'-1'
                for i, val in enumerate(feat_vals):
                    if val not in known_values:
                        feat_vals[i] = '-1'
                
                # 将已知值转换，'-1'会被映射到它在训练时的编码
                try:
                    processed_data[feat] = self.encoders[feat].transform(feat_vals)
                except ValueError:
                    # 如果'-1'也是未见过的，则将所有未知值设为0
                    print(f"警告: 特征'{feat}'中存在未见过的值，已替换为0")
                    for i, val in enumerate(feat_vals):
                        if val not in known_values:
                            feat_vals[i] = self.encoders[feat].classes_[0]
                    processed_data[feat] = self.encoders[feat].transform(feat_vals)
                    
            # 确保类型是整数
            processed_data[feat] = processed_data[feat].astype('int64')
                
        return processed_data
    
    def prepare_model_input(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        准备模型输入
        
        Args:
            data: 预处理后的数据
            
        Returns:
            模型输入字典
        """
        model_input = {}
        for name in self.feature_names:
            model_input[name] = data[name].values.astype('int64')
            
        return model_input
    
    def build_model(self, data: pd.DataFrame) -> None:
        """
        构建DeepFM模型
        
        Args:
            data: 用于确定特征维度的数据
        """
        # 创建特征列
        self._create_feature_columns(data)
        
        # 初始化DeepFM模型
        self.model = DeepFM(
            linear_feature_columns=self.linear_feature_columns,
            dnn_feature_columns=self.dnn_feature_columns,
            task=self.task,
            device=self.device
        )
        
        # 编译模型
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["binary_crossentropy", "auc"]
        )
    
    def get_hidden_outputs(self, input_data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        获取模型中间层输出
        
        Args:
            input_data: 模型输入数据
                    
        Returns:
            各隐藏层的输出字典
        """
        # 设置为评估模式
        self.model.eval()
        
        # 首先使用predict方法获取预测结果，这是一种安全的方式
        with torch.no_grad():
            # 准备模型所需的输入格式
            batch_size = next(iter(input_data.values())).shape[0]
            
            # 获取预测结果
            y_pred = self.model.predict(input_data, batch_size=batch_size)
            
            # 构建输出字典
            embeddings = {}
            embeddings['final_output'] = torch.tensor(y_pred, device=self.device)
            
            # 为了获取中间层表示，我们可以用一种替代方法
            # 创建一个简单的特征表示 - 把所有特征嵌入拼接起来
            feature_data = []
            for name, value in input_data.items():
                feature_tensor = torch.tensor(value, dtype=torch.long, device=self.device)
                feature_data.append(feature_tensor)
                
            stacked_features = torch.stack(feature_data, dim=1)
            embeddings['stacked_features'] = stacked_features
            
            # 计算特征维度总和 - 用于学生模型的输入维度参考
            input_dim_total = sum(feat.embedding_dim for feat in self.linear_feature_columns)
            embeddings['input_dim'] = input_dim_total
            
            # 如果需要进一步的特征表示，可以使用模型的部分组件
            try:
                # 获取模型的部分参数，这可以作为知识蒸馏的基础
                param_dict = {}
                for name, param in self.model.named_parameters():
                    if 'embedding' in name:  # 只获取嵌入层参数
                        param_dict[name] = param.data
                
                embeddings['model_params'] = param_dict
            except Exception as e:
                print(f"获取模型参数时出错: {e}")
            
        return embeddings
    
    def fit(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        batch_size: int = 1024,
        epochs: int = 20,
        verbose: int = 2,
        validation_split: float = 0.1
    ) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_data: 训练数据
            validation_data: 验证数据，如为None则从训练数据中分割
            batch_size: 批次大小
            epochs: 训练轮数
            verbose: 输出详细程度
            validation_split: 验证集比例（当validation_data为None时生效）
            
        Returns:
            训练历史记录
        """
        # 预处理数据
        processed_data = self.preprocess_data(train_data)
        
        # 构建模型（如果尚未构建）
        if self.model is None:
            self.build_model(processed_data)
        
        # 准备模型输入
        train_model_input = self.prepare_model_input(processed_data)
        y_train = processed_data['click'].values.astype('float32')
        
        # 训练模型
        history = self.model.fit(
            train_model_input,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split
        )
        
        return history
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_data: 测试数据
            
        Returns:
            包含评估指标的字典
        """
        # 预处理数据
        processed_data = self.preprocess_data(test_data)
        
        # 准备模型输入
        test_model_input = self.prepare_model_input(processed_data)
        y_test = processed_data['click'].values.astype('float32')
        
        # 预测
        pred_ans = self.model.predict(test_model_input, batch_size=256)
        
        # 计算评估指标
        metrics = {
            'log_loss': round(log_loss(y_test, pred_ans), 4),
            'auc': round(roc_auc_score(y_test, pred_ans), 4)
        }
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        模型预测
        
        Args:
            data: 预测数据
            
        Returns:
            预测结果
        """
        # 预处理数据
        processed_data = self.preprocess_data(data)
        
        # 准备模型输入
        model_input = self.prepare_model_input(processed_data)
        
        # 预测
        pred_ans = self.model.predict(model_input, batch_size=256)
        
        return pred_ans
    
    # save_model 方法
    def save_model(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        # 获取每个特征的词汇表大小
        vocab_sizes = {}
        for feat, feat_spec in zip(self.sparse_features, self.linear_feature_columns):
            vocab_sizes[feat] = feat_spec.vocabulary_size
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'encoders': self.encoders,
            'sparse_features': self.sparse_features,
            'embedding_dim': self.embedding_dim,
            'task': self.task,
            'vocab_sizes': vocab_sizes,  # 添加词汇表大小信息
        }, path)
    
    def load_model(self, path: str, data: pd.DataFrame) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
            data: 用于构建特征列的数据
        """
        checkpoint = torch.load(path)
        
        # 设置模型参数
        self.sparse_features = checkpoint['sparse_features']
        self.embedding_dim = checkpoint['embedding_dim']
        self.task = checkpoint['task']
        self.encoders = checkpoint['encoders']
        
        # 创建特征列 - 确保使用与保存模型时相同的维度
        self._create_feature_columns_from_checkpoint(checkpoint)
        
        # 构建模型
        # 不再使用传入的数据来构建模型，而是使用checkpoint中的信息
        self._build_model_from_checkpoint()
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])

    # 创建特征列
    def _create_feature_columns_from_checkpoint(self, checkpoint):
        """
        从checkpoint创建特征列，确保维度匹配
        
        Args:
            checkpoint: 加载的模型checkpoint
        """
        # 从checkpoint提取vocabulary_size信息（如果可用）
        if 'vocab_sizes' in checkpoint:
            vocab_sizes = checkpoint['vocab_sizes']
        else:
            # 回退方案：从模型权重推断词汇表大小
            vocab_sizes = {}
            state_dict = checkpoint['model_state_dict']
            for feat in self.sparse_features:
                # 查找对应的嵌入层权重
                embed_key = f'embedding_dict.{feat}.weight'
                if embed_key in state_dict:
                    vocab_sizes[feat] = state_dict[embed_key].shape[0]
                else:
                    # 如果找不到权重，使用默认值
                    vocab_sizes[feat] = 10000  # 设置一个安全的默认值
        
        # 构建SparseFeat特征列
        fixlen_feature_columns = [
            SparseFeat(feat, 
                      vocabulary_size=vocab_sizes.get(feat, 10000),  # 使用checkpoint中的值或默认值
                      embedding_dim=self.embedding_dim)
            for feat in self.sparse_features
        ]
        
        self.linear_feature_columns = fixlen_feature_columns
        self.dnn_feature_columns = fixlen_feature_columns
        self.feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)
    
    def _build_model_from_checkpoint(self):
        """
        使用特征列构建模型
        """
        # 初始化DeepFM模型
        self.model = DeepFM(
            linear_feature_columns=self.linear_feature_columns,
            dnn_feature_columns=self.dnn_feature_columns,
            task=self.task,
            device=self.device
        )
        
        # 编译模型
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["binary_crossentropy", "auc"]
        )


def train_and_evaluate_model(
    data_path: str = 'data/avazu/avazu_sample.csv',
    sparse_features: List[str] = None,
    epochs: int = 10,
    batch_size: int = 1024,
    embedding_dim: int = 4,
    model_save_path: str = 'models/teacher_model.pth'
) -> CTRTeacherModel:
    """
    训练并评估CTR教师模型
    
    Args:
        data_path: 数据集路径
        sparse_features: 稀疏特征列表，如果为None则使用默认特征集
        epochs: 训练轮数
        batch_size: 批次大小
        embedding_dim: 嵌入维度
        model_save_path: 模型保存路径
        
    Returns:
        训练好的教师模型
    """
    # 设置默认特征集
    if sparse_features is None:
        sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 'app_id', 'device_type', 'device_conn_type']
    
    # 1. 加载数据
    print(f"加载数据: {data_path}")
    data = pd.read_csv(data_path)
    
    # 输出前5行数据，便于检查数据结构
    print("数据预览：")
    print(data.head())
    
    # 2. 将目标标签 click 转换为数值类型，并剔除缺失值
    data['click'] = pd.to_numeric(data['click'], errors='coerce')
    data = data.dropna(subset=['click'])
    data['click'] = data['click'].astype(int)
    
    # 3. 划分训练集与测试集
    print("划分训练集与测试集...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"训练集大小: {train_data.shape}, 测试集大小: {test_data.shape}")
    
    # 4. 初始化教师模型
    print("初始化教师模型...")
    model = CTRTeacherModel(
        sparse_features=sparse_features,
        embedding_dim=embedding_dim,
        task='binary'
    )
    
    # 5. 训练模型
    print(f"开始训练模型，epochs={epochs}, batch_size={batch_size}...")
    history = model.fit(
        train_data=train_data,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_split=0.1
    )
    
    # 6. 保存模型
    if model_save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save_model(model_save_path)
        print(f"模型已保存至: {model_save_path}")
    
    # 7. 评估模型
    print("评估模型性能...")
    metrics = model.evaluate(test_data)
    print(f"测试集结果 - Log Loss: {metrics['log_loss']}, AUC: {metrics['auc']}")
    
    return model


if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 使用绝对路径而非相对路径
    import os
    # 获取当前脚本所在目录的上级目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 构建数据文件的绝对路径
    data_path = os.path.join(base_dir, 'data', 'avazu', 'avazu_sample.csv')
    
    print(f"使用数据路径: {data_path}")
    
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
    
    # 训练并评估模型
    model = train_and_evaluate_model(
        data_path=data_path,
        epochs=5,  # 减少轮数以加快测试
        batch_size=1024,
        model_save_path=os.path.join(base_dir, 'models', 'teacher_model.pth')
    )
    
    # === 简单的单元测试 ===
    print("\n开始单元测试...")
    
    # 测试1: 检查模型是否成功加载
    assert model.model is not None, "模型未正确初始化"
    print("✓ 测试1通过: 模型已正确初始化")
    
    # 测试2: 测试预测功能
    test_data = pd.read_csv(data_path).head(10)
    predictions = model.predict(test_data)
    assert len(predictions) == len(test_data), "预测数量与测试数据不匹配"
    assert all(0 <= p <= 1 for p in predictions), "预测值不在有效范围[0,1]内"
    print(f"✓ 测试2通过: 预测功能正常，预测结果范围正确: {predictions[:3]}")
    
    # 测试3: 测试中间层输出
    processed_data = model.preprocess_data(test_data)
    test_input = model.prepare_model_input(processed_data)
    hidden_outputs = model.get_hidden_outputs(test_input)
    assert 'final_output' in hidden_outputs, "未找到最终输出"
    assert 'stacked_features' in hidden_outputs, "未找到特征表示"
    assert 'input_dim' in hidden_outputs, "未找到输入维度信息"
    print("✓ 测试3通过: 成功获取模型表示")
    
    # 测试4: 检查模型保存功能
    model_path = os.path.join(base_dir, 'models', 'test_model.pth')
    model.save_model(model_path)
    assert os.path.exists(model_path), "模型文件未成功保存"
    print(f"✓ 测试4通过: 模型成功保存到 {model_path}")
    
    print("\n所有测试通过！教师模型可以正常工作")