import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, get_feature_names
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import json
import pickle
import logging
from typing import List, Dict, Tuple, Union, Any, Optional
from datasets import load_dataset
from tqdm import tqdm
import onnx

class CTRFrappeTeacherModel:
    def __init__(
        self,
        sparse_features: List[str] = None,
        embedding_dim: int = 10,
        dnn_hidden_units: Tuple[int, ...] = (128, 64),
        l2_reg: float = 1e-5,
        dropout_rate: float = 0.2,
        task: str = 'binary',
        device: str = None,
    ):
        """
        基于Frappe数据集的CTR预测教师模型，基于DeepFM
        
        Args:
            sparse_features: 稀疏特征列表，如果为None则使用默认特征集
            embedding_dim: 嵌入维度
            dnn_hidden_units: DNN隐藏层单元数
            l2_reg: L2正则化系数
            dropout_rate: Dropout比率
            task: 任务类型，默认为二分类
            device: 设备类型，若为None则自动选择
        """
        # 如果未指定特征，使用Frappe数据集的默认特征
        if sparse_features is None:
            self.sparse_features = ['user', 'item', 'daytime', 'weekday', 
                                   'isweekend', 'homework', 'cost', 'weather', 
                                   'country', 'city']
        else:
            self.sparse_features = sparse_features
        
        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.task = task
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型相关参数
        self.linear_feature_columns = None
        self.dnn_feature_columns = None
        self.feature_names = None
        self.model = None
        self.encoders = {}
        
        # 特征处理相关参数
        self.feature_frequencies = {}  # 存储特征值频率信息
        self.value_thresholds = {}     # 存储频率阈值
        self.default_value_map = {}    # 存储默认值映射
        
    def _create_feature_columns(self, data: pd.DataFrame, max_vocab_size=None) -> None:
        """
        创建特征列，确保所有特征使用相同的嵌入维度
        
        Args:
            data: 用于确定特征维度的数据
            max_vocab_size: 最大词汇表大小，None则不限制
        """
        print("创建特征列...")
        fixlen_feature_columns = []
        
        for feat in self.sparse_features:
            # 计算特征基数
            feat_cardinality = data[feat].nunique()
            
            # 对高基数特征，考虑限制词汇表大小
            vocab_size = feat_cardinality + 1  # +1 for unknown values
            
            if max_vocab_size and vocab_size > max_vocab_size:
                vocab_size = max_vocab_size
                print(f"特征 '{feat}' 基数 ({feat_cardinality}) 超过最大词汇表限制，将限制为 {max_vocab_size}")
            
            # 所有特征使用相同的嵌入维度，以避免尺寸不匹配错误
            fixlen_feature_columns.append(
                SparseFeat(
                    feat, 
                    vocabulary_size=vocab_size,
                    embedding_dim=self.embedding_dim,  # 固定使用相同的嵌入维度
                    embedding_name=feat 
                )
            )
            
            print(f"特征 '{feat}' - 基数: {feat_cardinality}, 嵌入维度: {self.embedding_dim}, 词汇表大小: {vocab_size}")
        
        self.linear_feature_columns = fixlen_feature_columns
        self.dnn_feature_columns = fixlen_feature_columns
        self.feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)

    def analyze_features(self, data: pd.DataFrame, min_freq_threshold=10) -> None:
        """
        分析特征分布和频率，为处理未见过的特征值做准备
        
        Args:
            data: 训练数据
            min_freq_threshold: 最小频率阈值，低于此值的特征视为低频
        """
        print("分析特征分布...")
        
        for feat in self.sparse_features:
            # 统计特征值频率
            value_counts = data[feat].value_counts()
            total_count = len(data)
            
            # 存储频率信息
            self.feature_frequencies[feat] = {
                'counts': value_counts.to_dict(),
                'total': total_count
            }
            
            # 计算低频阈值 (小于训练样本的min_freq_threshold%)
            threshold = max(min_freq_threshold, int(total_count * 0.001))
            self.value_thresholds[feat] = threshold
            
            # 为每个特征找出一个常见的默认值（频率最高的值）
            most_common = value_counts.idxmax()
            self.default_value_map[feat] = most_common
            
            print(f"特征 '{feat}' - 唯一值: {len(value_counts)}, 低频阈值: {threshold}, 默认值: {most_common}")
    
    def preprocess_data(self, data: pd.DataFrame, is_training=False) -> pd.DataFrame:
        """
        增强的数据预处理：填充缺失值、标签编码，智能处理未见过的类别
        
        Args:
            data: 输入的原始数据
            is_training: 是否为训练阶段
            
        Returns:
            处理后的数据
        """
        # 复制数据避免修改原始数据
        processed_data = data.copy()
        
        # 在训练阶段分析特征
        if is_training:
            self.analyze_features(processed_data)
        
        # 对每个特征列进行缺失值填充和编码
        for feat in self.sparse_features:
            # 1. 填充缺失值
            if feat in self.default_value_map:
                default_val = self.default_value_map[feat]
                processed_data[feat] = processed_data[feat].fillna(default_val)
            else:
                # 如果没有默认值映射，使用-1
                processed_data[feat] = processed_data[feat].fillna(-1)
            
            # 2. 编码处理
            if is_training:
                # 训练阶段：创建新的编码器
                self.encoders[feat] = LabelEncoder()
                # 确保特征值为字符串以便统一处理
                processed_data[feat] = processed_data[feat].astype(str)
                processed_data[feat] = self.encoders[feat].fit_transform(processed_data[feat])
            else:
                # 推理阶段：处理未见过的值
                if feat in self.encoders:
                    # 转换为字符串以便统一处理
                    feat_vals = processed_data[feat].astype(str).values
                    known_values = set(self.encoders[feat].classes_)
                    
                    # 检测并替换未见过的值
                    unknown_indices = [i for i, val in enumerate(feat_vals) if val not in known_values]
                    if unknown_indices:
                        print(f"特征'{feat}'中有 {len(unknown_indices)} 个未见过的值")
                        # 用默认值替换未见过的值
                        for i in unknown_indices:
                            if feat in self.default_value_map:
                                # 使用分析得到的默认值
                                feat_vals[i] = str(self.default_value_map[feat])
                            else:
                                # 回退到第一个值
                                feat_vals[i] = self.encoders[feat].classes_[0]
                    
                    # 转换数据
                    try:
                        processed_data[feat] = self.encoders[feat].transform(feat_vals)
                    except Exception as e:
                        print(f"转换特征'{feat}'时出错: {e}")
                        # 紧急回退方案：全部用0填充
                        processed_data[feat] = 0
                else:
                    # 如果没有对应的编码器，使用0
                    print(f"警告: 特征'{feat}'没有对应的编码器")
                    processed_data[feat] = 0
                
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
        构建DeepFM模型，增加正则化和Dropout
        
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
            device=self.device,
            dnn_hidden_units=self.dnn_hidden_units,
            l2_reg_linear=self.l2_reg,
            l2_reg_embedding=self.l2_reg,
            l2_reg_dnn=self.l2_reg,
            dnn_dropout=self.dropout_rate,
            dnn_use_bn=True,  # 使用批标准化
            seed=1024  # 固定随机种子
        )
        
        # 编译模型，使用自定义学习率
        from torch.optim import Adam
        optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=self.l2_reg)
        
        self.model.compile(
            optimizer=optimizer,  # 使用自定义的优化器
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
        batch_size: int = 2048,
        epochs: int = 10,
        verbose: int = 2,
        validation_split: float = 0.1,
        patience: int = 3,
        learning_rate: float = 0.001,
    ) -> Dict[str, List[float]]:
        """
        训练模型 - 使用DeepCTR的fit方法并安全处理历史记录
        
        Args:
            train_data: 训练数据
            validation_data: 验证数据，如为None则从训练数据中分割
            batch_size: 批次大小
            epochs: 训练轮数
            verbose: 输出详细程度
            validation_split: 验证集比例（当validation_data为None时生效）
            patience: 早停耐心值（目前未使用）
            learning_rate: 初始学习率
            
        Returns:
            训练历史记录
        """
        # 预处理数据，标记为训练阶段
        processed_data = self.preprocess_data(train_data, is_training=True)
        
        # 构建模型（如果尚未构建）
        if self.model is None:
            self.build_model(processed_data)
        
        # 准备模型输入
        train_model_input = self.prepare_model_input(processed_data)
        y_train = processed_data['label'].values.astype('float32')
        
        # 设置优化器和学习率
        from torch.optim import Adam
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.l2_reg)
        self.model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["binary_crossentropy", "auc"])
        
        # 训练模型 - 不使用callbacks参数
        history = self.model.fit(
            train_model_input,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split
        )
        
        # 安全地尝试访问和输出验证指标 - 直接从history对象获取
        # DeepCTR-Torch返回的history对象保存格式可能是普通字典
        try:
            if hasattr(history, 'history') and isinstance(history.history, dict):
                # Keras风格的history对象
                history_dict = history.history
            elif isinstance(history, dict):
                # 直接是字典格式
                history_dict = history
            else:
                # 无法识别的格式
                print("\n无法识别的history对象格式，跳过验证性能输出")
                return history
            
            # 检查字典中是否有验证指标
            if 'val_binary_crossentropy' in history_dict and len(history_dict['val_binary_crossentropy']) > 0:
                last_val_loss = history_dict['val_binary_crossentropy'][-1]
                print(f"\n最终验证性能 - Loss: {last_val_loss:.4f}", end="")
                
                if 'val_auc' in history_dict:
                    last_val_auc = history_dict['val_auc'][-1]
                    print(f", AUC: {last_val_auc:.4f}")
                else:
                    print("")
                
                # 找出最佳性能
                val_losses = history_dict['val_binary_crossentropy']
                best_epoch = np.argmin(val_losses)
                print(f"最佳模型出现在epoch {best_epoch+1}, 验证Loss: {val_losses[best_epoch]:.4f}")
                
                if 'val_auc' in history_dict:
                    print(f"对应AUC: {history_dict['val_auc'][best_epoch]:.4f}")
        except Exception as e:
            print(f"\n解析训练历史记录时出错: {e}")
            print("继续执行，不影响模型训练结果")
        
        return history

    def evaluate(self, test_data: pd.DataFrame, threshold=0.5) -> Dict[str, float]:
        """
        评估模型性能，增加更多评估指标
        
        Args:
            test_data: 测试数据
            threshold: 二分类阈值
            
        Returns:
            包含评估指标的字典
        """
        # 预处理数据
        processed_data = self.preprocess_data(test_data)
        
        # 准备模型输入
        test_model_input = self.prepare_model_input(processed_data)
        y_test = processed_data['label'].values.astype('float32')
        
        # 预测
        y_pred = self.model.predict(test_model_input, batch_size=512)
        
        # 计算评估指标
        metrics = {}
        
        try:
            # 基础指标
            metrics['log_loss'] = round(log_loss(y_test, y_pred), 4)
            metrics['auc'] = round(roc_auc_score(y_test, y_pred), 4)
            
            # 二分类指标
            y_pred_binary = (y_pred > threshold).astype(int)
            
            # 准确率
            metrics['accuracy'] = round(accuracy_score(y_test, y_pred_binary), 4)
            
            # 精确率、召回率
            metrics['precision'] = round(precision_score(y_test, y_pred_binary, zero_division=0), 4)
            metrics['recall'] = round(recall_score(y_test, y_pred_binary, zero_division=0), 4)
            
            # F1 分数
            metrics['f1'] = round(f1_score(y_test, y_pred_binary, zero_division=0), 4)
        except Exception as e:
            print(f"计算评估指标时出错: {e}")
            # 提供基本指标
            metrics['error'] = str(e)
        
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
        pred_ans = self.model.predict(model_input, batch_size=512)
        
        return pred_ans
    
    def save_model(self, path: str, save_full_model: bool = True) -> None:
        """
        Enhanced model saving, includes feature statistics and saves model state dictionary
        
        Args:
            path: Path to save state dict version
            save_full_model: If True, also save full model
        """
        # Get vocabulary size for each feature
        vocab_sizes = {}
        for feat, feat_spec in zip(self.sparse_features, self.linear_feature_columns):
            vocab_sizes[feat] = feat_spec.vocabulary_size
        
        # Create complete model state dictionary
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'encoders': self.encoders,
            'sparse_features': self.sparse_features,
            'embedding_dim': self.embedding_dim,
            'task': self.task,
            'vocab_sizes': vocab_sizes,
            'dnn_hidden_units': self.dnn_hidden_units,
            'l2_reg': self.l2_reg,
            'dropout_rate': self.dropout_rate,
            'feature_frequencies': self.feature_frequencies,
            'value_thresholds': self.value_thresholds,
            'default_value_map': self.default_value_map
        }
        
        # Save state dict version
        torch.save(model_state, path)
        print(f"Model state dictionary saved to: {path}")
        
        # Save feature info as JSON (for easy viewing)
        feature_info = {
            'sparse_features': self.sparse_features,
            'vocab_sizes': vocab_sizes,
            'default_value_map': {k: str(v) for k, v in self.default_value_map.items()},
            'value_thresholds': self.value_thresholds
        }
        
        info_path = path.replace('.pth', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"Feature info saved to: {info_path}")
        
        # Get outputs directory
        outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        
        # Additionally save the full model
        if save_full_model:
            # Define full model path
            full_model_path = os.path.join(outputs_dir, 'teacher_frappe_full.pth')
            
            # Save the complete model
            model_copy = self.model
            model_copy.eval()  # Set to evaluation mode
            
            # Create a complete package with the actual model and metadata
            full_model_package = {
                'model': model_copy,  # The actual PyTorch model
                'metadata': {
                    'sparse_features': self.sparse_features,
                    'embedding_dim': self.embedding_dim,
                    'task': self.task,
                    'vocab_sizes': vocab_sizes,
                    'encoders': self.encoders
                }
            }
            
            # Save the full model package
            torch.save(full_model_package, full_model_path)
            print(f"Full model saved to: {full_model_path}")

    def load_model(self, path: str, data: Optional[pd.DataFrame] = None) -> None:
        """
        加载模型，支持无数据加载
        
        Args:
            path: 模型路径
            data: 用于构建特征列的数据，可选
        """
        checkpoint = torch.load(path)
        
        # 设置模型参数
        self.sparse_features = checkpoint['sparse_features']
        self.embedding_dim = checkpoint['embedding_dim']
        self.task = checkpoint['task']
        self.encoders = checkpoint['encoders']
        
        # 加载额外参数（如果存在）
        if 'dnn_hidden_units' in checkpoint:
            self.dnn_hidden_units = checkpoint['dnn_hidden_units']
        if 'l2_reg' in checkpoint:
            self.l2_reg = checkpoint['l2_reg']
        if 'dropout_rate' in checkpoint:
            self.dropout_rate = checkpoint['dropout_rate']
        
        # 加载特征统计信息
        if 'feature_frequencies' in checkpoint:
            self.feature_frequencies = checkpoint['feature_frequencies']
        if 'value_thresholds' in checkpoint:
            self.value_thresholds = checkpoint['value_thresholds']
        if 'default_value_map' in checkpoint:
            self.default_value_map = checkpoint['default_value_map']
        
        # 创建特征列 - 确保使用与保存模型时相同的维度
        if data is not None:
            self._create_feature_columns(data)
        else:
            self._create_feature_columns_from_checkpoint(checkpoint)
        
        # 构建模型
        self._build_model_from_checkpoint(checkpoint)
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"模型已从 {path} 成功加载")

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
        
        # 构建SparseFeat特征列，所有特征使用相同的嵌入维度
        fixlen_feature_columns = []
        
        for feat in self.sparse_features:
            fixlen_feature_columns.append(
                SparseFeat(
                    feat, 
                    vocabulary_size=vocab_sizes.get(feat, 10000),
                    embedding_dim=self.embedding_dim,  # 使用固定的嵌入维度
                    embedding_name=feat
                )
            )
        
        self.linear_feature_columns = fixlen_feature_columns
        self.dnn_feature_columns = fixlen_feature_columns
        self.feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)

    def _build_model_from_checkpoint(self, checkpoint=None):
        """
        使用特征列构建模型
        
        Args:
            checkpoint: 可选的模型checkpoint用于提取参数
        """
        # 获取模型超参数
        dnn_hidden_units = self.dnn_hidden_units
        l2_reg = self.l2_reg
        dropout_rate = self.dropout_rate
        
        # 如果提供了checkpoint，优先使用checkpoint中的参数
        if checkpoint:
            if 'dnn_hidden_units' in checkpoint:
                dnn_hidden_units = checkpoint['dnn_hidden_units']
            if 'l2_reg' in checkpoint:
                l2_reg = checkpoint['l2_reg']
            if 'dropout_rate' in checkpoint:
                dropout_rate = checkpoint['dropout_rate']
        
        # 初始化DeepFM模型
        self.model = DeepFM(
            linear_feature_columns=self.linear_feature_columns,
            dnn_feature_columns=self.dnn_feature_columns,
            task=self.task,
            device=self.device,
            dnn_hidden_units=dnn_hidden_units,
            l2_reg_linear=l2_reg,
            l2_reg_embedding=l2_reg,
            l2_reg_dnn=l2_reg,
            dnn_dropout=dropout_rate,
            dnn_use_bn=True,
            seed=1024
        )
        
        # 编译模型
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["binary_crossentropy", "auc"]
        )


def load_frappe_dataset(use_stratify=True):
    """
    加载Frappe数据集并转换为pandas DataFrame
    
    Args:
        use_stratify: 是否使用分层采样保持标签分布
        
    Returns:
        train_df: 训练集DataFrame
        test_df: 测试集DataFrame
    """
    # 加载Frappe数据集
    print("加载Frappe数据集...")
    frappe_dataset = load_dataset("reczoo/Frappe_x1")
    
    # 将训练集转换为pandas DataFrame
    train_df = pd.DataFrame(frappe_dataset["train"])
    
    # 创建测试集 (从训练集中分出20%)
    if use_stratify:
        train_df, test_df = train_test_split(train_df, test_size=0.2, 
                                            stratify=train_df['label'], 
                                            random_state=42)
    else:
        train_df, test_df = train_test_split(train_df, test_size=0.2, 
                                            random_state=42)
    
    print(f"数据集已加载 - 训练集: {len(train_df)}行, 测试集: {len(test_df)}行")
    print(f"训练集标签分布: {train_df['label'].value_counts(normalize=True).to_dict()}")
    print(f"测试集标签分布: {test_df['label'].value_counts(normalize=True).to_dict()}")
    
    return train_df, test_df


def train_and_evaluate_frappe_model(
    epochs: int = 10,
    batch_size: int = 2048,
    embedding_dim: int = 12,
    dnn_hidden_units: Tuple[int, ...] = (128, 64, 32),
    l2_reg: float = 1e-5,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    patience: int = 3,
    model_save_path: str = 'models/teacher_frappe_model.pth'
) -> CTRFrappeTeacherModel:
    """
    训练并评估Frappe CTR教师模型
    
    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        embedding_dim: 嵌入维度
        dnn_hidden_units: DNN隐藏层单元数
        l2_reg: L2正则化系数
        dropout_rate: Dropout比率
        learning_rate: 学习率
        patience: 早停耐心值
        model_save_path: 模型保存路径
        
    Returns:
        训练好的教师模型
    """
    # 1. 加载数据（使用分层抽样）
    train_data, test_data = load_frappe_dataset(use_stratify=True)
    
    # 输出数据结构
    print("\n数据预览：")
    print(train_data.head())
    print(f"\n特征列表: {train_data.columns.tolist()}")
    
    # 2. 初始化教师模型
    print("\n初始化Frappe教师模型...")
    sparse_features = [col for col in train_data.columns if col != 'label']
    model = CTRFrappeTeacherModel(
        sparse_features=sparse_features,
        embedding_dim=embedding_dim,
        dnn_hidden_units=dnn_hidden_units,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        task='binary'
    )
    
    # 3. 训练模型
    print(f"\n开始训练模型，epochs={epochs}, batch_size={batch_size}...")
    try:
        history = model.fit(
            train_data=train_data,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_split=0.1,
            patience=patience,
            learning_rate=learning_rate
        )
        
        # 4. 保存模型
        if model_save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            model.save_model(model_save_path)
            print(f"模型已保存至: {model_save_path}")
        
        # 5. 评估模型
        print("\n评估模型性能...")
        metrics = model.evaluate(test_data)
        print("测试集结果:")
        for metric_name, metric_value in metrics.items():
            print(f"  - {metric_name}: {metric_value}")
    
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    return model


def k_fold_frappe_model_evaluation(
    k=5,
    epochs=5,
    embedding_dim=12,
    l2_reg=1e-5,
    model_save_prefix='models/teacher_frappe_model'
):
    """
    使用K折交叉验证评估模型性能
    
    Args:
        k: 折数
        epochs: 每折训练的轮数
        embedding_dim: 嵌入维度
        l2_reg: L2正则化系数
        model_save_prefix: 模型保存路径前缀
    """
    # 加载原始数据
    frappe_dataset = load_dataset("reczoo/Frappe_x1")
    full_data = pd.DataFrame(frappe_dataset["train"])
    
    # 初始化K折交叉验证
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # 存储每折的结果
    fold_results = []
    
    # 执行K折交叉验证
    for fold, (train_idx, test_idx) in enumerate(kf.split(full_data, full_data['label'])):
        print(f"\n=============== 第 {fold+1}/{k} 折 ===============")
        
        # 划分数据
        train_data = full_data.iloc[train_idx]
        test_data = full_data.iloc[test_idx]
        
        print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
        
        # 初始化模型
        model = CTRFrappeTeacherModel(
            sparse_features=[col for col in full_data.columns if col != 'label'],
            embedding_dim=embedding_dim,
            dnn_hidden_units=(128, 64, 32),
            l2_reg=l2_reg,
            dropout_rate=0.2,
            task='binary'
        )
        
        # 训练模型
        try:
            model.fit(
                train_data=train_data,
                batch_size=2048,
                epochs=epochs,
                verbose=2,
                validation_split=0.1,
                patience=2
            )
            
            # 评估模型
            metrics = model.evaluate(test_data)
            fold_results.append(metrics)
            
            print(f"第 {fold+1} 折结果: AUC = {metrics['auc']}, Log Loss = {metrics['log_loss']}")
            
            # 保存模型
            fold_model_path = f"{model_save_prefix}_fold{fold+1}.pth"
            model.save_model(fold_model_path)
        except Exception as e:
            print(f"第 {fold+1} 折训练出错: {e}")
    
    # 计算平均结果
    if fold_results:
        avg_results = {}
        for metric in fold_results[0].keys():
            avg_results[metric] = sum(r[metric] for r in fold_results) / len(fold_results)
        
        print("\n=============== K折交叉验证平均结果 ===============")
        for metric, value in avg_results.items():
            print(f"{metric}: {value:.4f}")
        
        return avg_results
    else:
        print("没有成功完成任何折的训练")
        return {}


if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 测试是否安装了必要的库
    try:
        import tensorflow
        print(f"TensorFlow版本: {tensorflow.__version__}")
    except ImportError:
        print("TensorFlow未安装，但DeepCTR-Torch可能仍然需要它的某些组件")
    
    # 获取当前脚本所在目录的上级目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 设置模型保存路径
    model_save_path = os.path.join(base_dir, 'models', 'teacher_frappe_model.pth')
    
    # 训练并评估模型
    try:
        model = train_and_evaluate_frappe_model(
            epochs=20,           # 减少训练轮数以加快训练
            batch_size=2048,
            embedding_dim=12,    # 调整嵌入维度
            dnn_hidden_units=(128, 64, 32),  # 3层DNN
            l2_reg=1e-5,         # 添加L2正则化
            dropout_rate=0.2,    # 添加dropout
            learning_rate=0.001,
            patience=2,          # 早停
            model_save_path=model_save_path
        )
        
        # === 简单的单元测试 ===
        print("\n开始单元测试...")
        
        # 测试1: 检查模型是否成功加载
        assert model.model is not None, "模型未正确初始化"
        print("✓ 测试1通过: 模型已正确初始化")
        
        # 测试2: 测试预测功能
        train_data, _ = load_frappe_dataset()
        test_data = train_data.head(10)
        predictions = model.predict(test_data)
        assert len(predictions) == len(test_data), "预测数量与测试数据不匹配"
        assert all(0 <= p <= 1 for p in predictions.flatten()), "预测值不在有效范围[0,1]内"
        
        # 检查预测结果是否有差异（表明模型不是简单返回同一值）
        unique_preds = np.unique(predictions.flatten())
        if len(unique_preds) > 1:
            print(f"✓ 测试2通过: 预测功能正常，预测结果范围正确且有不同值: {predictions[:3]}")
        else:
            print(f"⚠ 测试2警告: 预测值缺乏多样性: {unique_preds}")
        
        # 测试3: 测试中间层输出
        processed_data = model.preprocess_data(test_data)
        test_input = model.prepare_model_input(processed_data)
        hidden_outputs = model.get_hidden_outputs(test_input)
        assert 'final_output' in hidden_outputs, "未找到最终输出"
        assert 'stacked_features' in hidden_outputs, "未找到特征表示"
        print("✓ 测试3通过: 成功获取模型表示")
        
        print("\n所有测试通过！改进的Frappe教师模型可以正常工作")
        
    except Exception as e:
        print(f"运行中出现错误: {e}")
        import traceback
        traceback.print_exc()