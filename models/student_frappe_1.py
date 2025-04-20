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
from collections import defaultdict
import matplotlib.pyplot as plt

class FrappeStudentModel(nn.Module):
    """
    轻量级Frappe CTR预测学生模型 - 增强版
    
    - 使用知识蒸馏和迁移学习从教师模型获取知识
    - 使用Fisher信息矩阵进行结构化剪枝
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
        
        # Fisher信息矩阵
        self.fisher = {}
        self.fisher_initialized = False
        self.ewc_lambda = 15.0  # EWC正则化强度
        self.parameter_importance = {}
        self.mask = {}  # 用于存储参数掩码 (1=保留, 0=剪枝)
        
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
                        teacher_weight = teacher_state_dict[teacher_key].to(self.device)
                    
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
            task_loss = alpha * soft_loss + (1 - alpha) * hard_loss
        else:
            # 如果没有教师预测，只使用硬目标损失
            task_loss = hard_loss
            
        # 如果Fisher信息已初始化，添加EWC正则化损失
        ewc_loss = 0
        if self.fisher_initialized:
            for name, param in self.named_parameters():
                if name in self.fisher and name in self.parameter_importance:
                    # 根据参数的重要程度应用EWC损失
                    importance = self.parameter_importance[name]
                    
                    # 修复：使用 gt(0).any() 来检查tensor中是否有大于0的元素
                    if torch.gt(importance, 0).any():
                        fisher = self.fisher[name]
                        ewc_loss += torch.sum(fisher * (param - param.data) ** 2)
            
            # 添加EWC损失
            task_loss += self.ewc_lambda * ewc_loss
            
        return task_loss
    
    def compute_fisher_information(self, data_loader, sample_size=1000, task_id=None):
        """
        计算Fisher信息矩阵，支持任务ID以追踪随时间变化的重要性
        
        Args:
            data_loader: 数据加载器
            sample_size: 用于计算Fisher的样本数量
            task_id: 可选的任务ID，用于标记不同训练阶段的Fisher信息
        """
        print("计算Fisher信息矩阵...")
        self.eval()  # 设置为评估模式
        
        # 保存当前参数作为该任务的最优参数
        task_id = task_id or f"task_{len(getattr(self, 'fisher_history', {})) + 1}"
        self.theta_star = {}
        for name, param in self.named_parameters():
            self.theta_star[name] = param.data.clone()
        
        # 初始化Fisher矩阵存储
        fisher = {}
        for name, param in self.named_parameters():
            fisher[name] = torch.zeros_like(param, device=self.device)
        
        # 处理的样本数量
        samples_processed = 0
        
        for batch in data_loader:
            if samples_processed >= sample_size:
                break
                
            x, y_true = batch
            x = {k: v.to(self.device) for k, v in x.items()}
            y_true = y_true.float().to(self.device)
            
            batch_size = y_true.size(0)
            
            # 确保不超过样本大小
            if samples_processed + batch_size > sample_size:
                # 调整批次大小以匹配所需的样本数
                remainder = sample_size - samples_processed
                for k in x:
                    x[k] = x[k][:remainder]
                y_true = y_true[:remainder]
                batch_size = remainder
            
            # 前向传播
            y_pred = self(x)
            
            # 计算对数似然
            log_likelihood = torch.mean(
                y_true * torch.log(y_pred + 1e-7) + 
                (1 - y_true) * torch.log(1 - y_pred + 1e-7)
            )
            
            # 反向传播
            self.zero_grad()
            log_likelihood.backward()
            
            # 累积Fisher信息
            for name, param in self.named_parameters():
                if param.grad is not None:
                    # Fisher = E[(∂log p(y|x)/∂θ)^2]
                    fisher[name] += batch_size * (param.grad ** 2) / sample_size
            
            samples_processed += batch_size
        
        # 存储Fisher信息矩阵
        self.fisher = fisher
        
        # 存储历史Fisher信息
        if not hasattr(self, 'fisher_history'):
            self.fisher_history = {}
        
        # 仅存储每个参数的平均Fisher值（减少存储）
        compressed_fisher = {}
        for name, f_matrix in fisher.items():
            compressed_fisher[name] = float(f_matrix.abs().mean().item())
        
        self.fisher_history[task_id] = {
            'compressed_fisher': compressed_fisher,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 计算参数重要性分数
        self.compute_parameter_importance()
        
        self.fisher_initialized = True
        print(f"Fisher信息矩阵计算完成，使用了{samples_processed}个样本，任务ID: {task_id}")
    
    def compute_parameter_importance(self):
        """计算每个参数的重要性分数"""
        print("计算参数重要性...")
        
        # 参数重要性字典
        parameter_importance = {}
        
        # 对每个参数计算重要性
        for name, param in self.named_parameters():
            if name in self.fisher:
                # 使用Fisher作为重要性度量
                raw_importance = self.fisher[name].abs()
                
                # 对于大型参数（如嵌入），计算每行的均值重要性
                if len(param.shape) > 1 and 'embedding_dict' in name:
                    # 对每个嵌入向量计算重要性
                    row_importance = raw_importance.mean(dim=1)
                    parameter_importance[name] = row_importance
                else:
                    parameter_importance[name] = raw_importance
        
        self.parameter_importance = parameter_importance
        print("参数重要性计算完成")
    
    def prune_parameters(self, pruning_rate=0.3):
        """
        基于Fisher信息矩阵剪枝参数
        
        Args:
            pruning_rate: 要剪枝的参数比例 (0.0-1.0)
        
        Returns:
            剪枝的参数数量
        """
        if not self.fisher_initialized:
            print("错误：需要先计算Fisher信息矩阵才能进行剪枝")
            return 0
        
        print(f"使用Fisher信息进行参数剪枝，剪枝率={pruning_rate}...")
        
        # 初始化掩码字典
        self.mask = {}
        for name, param in self.named_parameters():
            self.mask[name] = torch.ones_like(param, device=self.device)
        
        # 分别处理嵌入层和其他层
        pruned_count = 0
        total_params = 0
        
        # 剪枝嵌入层
        for name, param in self.named_parameters():
            # 只处理具有重要性分数的参数
            if name not in self.parameter_importance:
                continue
                
            total_params += param.numel()
            importance = self.parameter_importance[name]
            
            # 特殊处理嵌入层
            if 'embedding_dict' in name:
                # 获取嵌入表中每一行的重要性（每个特征值的嵌入）
                if len(importance.shape) == 1:  # 行重要性
                    # 提取特征名
                    parts = name.split('.')
                    if len(parts) >= 3:
                        feature_name = parts[1]
                        
                        # 确保不剪枝常用特征值
                        vocab_size = param.shape[0]
                        protect_size = min(100, vocab_size // 10)  # 保护前N个嵌入
                        
                        # 获取重要性排序后的索引
                        sorted_idx = importance.argsort()
                        
                        # 只剪枝重要性较低的嵌入行，但不包括保护的部分
                        if vocab_size > protect_size:
                            # 计算要剪枝的行数
                            prune_count = int((vocab_size - protect_size) * pruning_rate)
                            
                            # 实际剪枝
                            for idx in sorted_idx[:prune_count]:
                                # 跳过受保护的嵌入
                                if idx >= protect_size:
                                    self.mask[name][idx] = 0.0
                                    pruned_count += self.embedding_dim
            else:
                # 处理DNN层
                flat_importance = importance.flatten()
                total_weights = flat_importance.numel()
                
                # 计算阈值
                k = int(total_weights * pruning_rate)
                if k > 0:
                    threshold = torch.kthvalue(flat_importance, k).values
                    
                    # 应用掩码
                    self.mask[name] = (importance > threshold).float()
                    pruned_count += total_weights - torch.sum(self.mask[name]).item()
        
        # 应用掩码
        self.apply_mask()
        
        pruning_percentage = 100.0 * pruned_count / total_params if total_params > 0 else 0
        print(f"参数剪枝完成: 共剪枝{pruned_count}个参数 (总计{total_params}个, {pruning_percentage:.2f}%)")
        
        return pruned_count
    
    def apply_mask(self):
        """应用参数掩码，将被剪枝的参数设为零"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self.mask:
                    param.mul_(self.mask[name])
    
    def forward_with_mask(self, x):
        """
        应用掩码的前向传播，用于评估剪枝效果
        
        Args:
            x: 输入特征
            
        Returns:
            模型预测
        """
        # 保存原始参数
        original_params = {}
        for name, param in self.named_parameters():
            original_params[name] = param.data.clone()
            
        # 应用掩码
        self.apply_mask()
        
        # 前向传播
        output = self.forward(x)
        
        # 恢复原始参数
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])
                    
        return output

    def summarize_layerwise_contribution(self):
        """
        分析并汇总各层对模型预测的相对重要性贡献
        
        Returns:
            字典，包含各层的重要性分数和百分比
        """
        if not self.fisher_initialized or not self.parameter_importance:
            print("错误: 需要先计算Fisher信息矩阵才能分析层级贡献")
            return None
            
        print("分析各层对模型的相对贡献...")
        summary = {}
        total_importance = 0
        
        # 按层级对参数重要性进行分组
        for name, importance in self.parameter_importance.items():
            # 提取层名称 (e.g., "embedding_dict.user" -> "embedding_dict")
            parts = name.split('.')
            if len(parts) >= 2:
                if 'embedding_dict' in name:
                    # 对于嵌入层，进一步按特征名分组
                    layer = f"embedding_dict.{parts[1]}"  # e.g., embedding_dict.user
                elif 'dnn' in name:
                    # 对于DNN层，区分不同的隐藏层
                    if 'dense' in name:
                        layer_num = name.split('_')[1].split('.')[0]  # e.g., dense_0 -> 0
                        layer = f"dnn_layer_{layer_num}"
                    else:
                        layer = "dnn_other"
                else:
                    layer = parts[0]  # 一般层名
            else:
                layer = name
                
            # 计算该参数的总体重要性
            importance_score = float(importance.abs().sum().item())
            total_importance += importance_score
            
            # 累加到对应层
            if layer not in summary:
                summary[layer] = 0
            summary[layer] += importance_score
        
        # 计算百分比，并按重要性降序排序
        result = {
            "raw_scores": {},
            "percentages": {},
            "total_importance": total_importance
        }
        
        for layer, score in summary.items():
            result["raw_scores"][layer] = score
            result["percentages"][layer] = (score / total_importance) * 100 if total_importance > 0 else 0
            
        # 排序
        result["raw_scores"] = dict(sorted(result["raw_scores"].items(), 
                                           key=lambda x: x[1], reverse=True))
        result["percentages"] = dict(sorted(result["percentages"].items(), 
                                           key=lambda x: x[1], reverse=True))
        
        print(f"层级贡献分析完成. 总重要性分数: {total_importance:.4f}")
        return result
    
    def plot_layerwise_contribution(self, save_path=None):
        """
        绘制各层对模型预测的相对重要性贡献饼图
        
        Args:
            save_path: 图表保存路径，如不提供则显示图表
        """
        contribution = self.summarize_layerwise_contribution()
        if not contribution:
            return
            
        percentages = contribution["percentages"]
        
        # 绘制饼图
        plt.figure(figsize=(10, 7))
        labels = list(percentages.keys())
        sizes = list(percentages.values())
        
        # 限制显示的层数，将小于1%的层归为"其他"
        if len(labels) > 8:  # 如果超过8个层，合并较小的层
            threshold = 1.0  # 显示贡献大于1%的层
            main_layers = {k: v for k, v in percentages.items() if v >= threshold}
            other_layers = {k: v for k, v in percentages.items() if v < threshold}
            
            if other_layers:
                main_layers["其他层"] = sum(other_layers.values())
                
            labels = list(main_layers.keys())
            sizes = list(main_layers.values())
        
        # 创建饼图
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
        plt.axis('equal')  # 保持圆形
        plt.title('各层对模型预测的重要性贡献')
        
        if save_path:
            plt.savefig(save_path)
            print(f"层级贡献饼图已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # 也创建一个条形图
        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, sizes)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.title('各层对模型预测的重要性贡献')
        plt.ylabel('重要性百分比 (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            bar_path = save_path.replace('.png', '_bar.png')
            plt.savefig(bar_path)
            print(f"层级贡献条形图已保存至: {bar_path}")
        else:
            plt.show()
            
        plt.close()
        
        return contribution

    def analyze_feature_importance(self):
        """
        分析嵌入层中的特征重要性，用于特征选择
        
        Returns:
            特征重要性排名和统计信息
        """
        if not self.fisher_initialized:
            print("错误：需要先计算Fisher信息矩阵才能分析特征重要性")
            return None
        
        feature_importance = {}
        
        # 收集所有嵌入层特征的重要性
        for name, importance in self.parameter_importance.items():
            if 'embedding_dict' in name and len(importance.shape) == 1:  # 嵌入行的重要性
                parts = name.split('.')
                if len(parts) >= 3:
                    feature_name = parts[1]  # 特征名，如'user'
                    
                    # 计算该特征的平均重要性和中位数重要性
                    mean_imp = float(importance.mean().item())
                    median_imp = float(importance.median().item())
                    max_imp = float(importance.max().item())
                    min_imp = float(importance.min().item())
                    
                    # 获取嵌入表中最不重要的行索引（可能对应于无用的特征值）
                    k = min(10, len(importance))  # 最多10个最不重要的索引
                    least_important_indices = importance.argsort()[:k].cpu().numpy().tolist()
                    
                    # 获取嵌入表中最重要的行索引
                    most_important_indices = importance.argsort(descending=True)[:k].cpu().numpy().tolist()
                    
                    feature_importance[feature_name] = {
                        'mean_importance': mean_imp,
                        'median_importance': median_imp,
                        'max_importance': max_imp,
                        'min_importance': min_imp,
                        'least_important_indices': least_important_indices,
                        'most_important_indices': most_important_indices
                    }
        
        # 按平均重要性对特征排序
        sorted_features = sorted(feature_importance.items(), 
                                key=lambda x: x[1]['mean_importance'], 
                                reverse=True)
        
        result = {
            'feature_ranking': {name: rank+1 for rank, (name, _) in enumerate(sorted_features)},
            'feature_details': feature_importance
        }
        
        # 输出特征重要性摘要
        print("\n特征重要性分析:")
        print("-" * 50)
        print(f"{'特征名':15} {'平均重要性':15} {'排名':5} {'可剪枝索引'}")
        print("-" * 50)
        
        for rank, (name, details) in enumerate(sorted_features):
            print(f"{name:15} {details['mean_importance']:.6f}      {rank+1:<5} {details['least_important_indices'][:3]}")
        
        return result

    def plot_feature_importance(self, save_path=None):
        """
        绘制特征重要性条形图
        
        Args:
            save_path: 图表保存路径，如不提供则显示图表
        """
        feature_analysis = self.analyze_feature_importance()
        if not feature_analysis:
            return
            
        # 提取数据
        features = list(feature_analysis['feature_details'].keys())
        importances = [feature_analysis['feature_details'][f]['mean_importance'] 
                      for f in features]
        
        # 按重要性排序
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        # 绘制条形图
        plt.figure(figsize=(12, 6))
        bars = plt.bar(sorted_features, sorted_importances)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.6f}', ha='center', va='bottom', rotation=45, fontsize=8)
        
        plt.title('特征重要性分析')
        plt.ylabel('平均重要性')
        plt.xlabel('特征名')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"特征重要性图已保存至: {save_path}")
        else:
            plt.show()
            
        plt.close()
        
        return feature_analysis

    def generate_model_summary(self):
        """
        生成模型参数重要性和剪枝情况的结构化摘要
        
        Returns:
            JSON格式的模型摘要
        """
        if not self.fisher_initialized:
            print("警告：Fisher信息未初始化，摘要可能不完整")
            
        summary = {
            "model_type": "FrappeStudentModel",
            "embedding_dim": self.embedding_dim,
            "feature_count": self.feature_num,
            "sparse_features": self.sparse_features,
            "hidden_units": [],
            "is_transfer_learning": self.is_transfer_learning,
            "parameters": {}
        }
        
        # 提取隐藏层单元数
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and 'dnn' in name and 'linear' not in name:
                summary["hidden_units"].append(module.out_features)
        
        # 计算总参数量和已剪枝参数量
        total_params = 0
        pruned_params = 0
        
        # 层级参数统计
        layer_stats = {}
        
        for name, param in self.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            # 提取层名称
            parts = name.split('.')
            layer = parts[0]
            if len(parts) > 1 and parts[0] == 'embedding_dict':
                layer = f"{parts[0]}.{parts[1]}"  # 例如 embedding_dict.user
                
            # 统计剪枝情况
            pruned_count = 0
            pruning_rate = 0
            
            if hasattr(self, 'mask') and name in self.mask:
                mask = self.mask[name]
                pruned_count = param_count - int(mask.sum().item())
                pruning_rate = pruned_count / param_count if param_count > 0 else 0
            
            # 统计重要性
            importance_score = 0
            if hasattr(self, 'parameter_importance') and name in self.parameter_importance:
                importance = self.parameter_importance[name]
                importance_score = float(importance.abs().mean().item())
            
            # 更新层统计
            if layer not in layer_stats:
                layer_stats[layer] = {
                    "param_count": 0,
                    "pruned_count": 0,
                    "importance_score": 0
                }
            
            layer_stats[layer]["param_count"] += param_count
            layer_stats[layer]["pruned_count"] += pruned_count
            layer_stats[layer]["importance_score"] += importance_score
            
            # 添加参数详情
            summary["parameters"][name] = {
                "shape": list(param.shape),
                "param_count": param_count,
                "pruned_count": pruned_count,
                "pruning_rate": float(pruning_rate),
                "importance_score": float(importance_score)
            }
        
        # 计算总体剪枝情况
        pruned_params = sum(info["pruned_count"] for info in layer_stats.values())
        pruning_rate = pruned_params / total_params if total_params > 0 else 0
        
        # 添加摘要统计
        summary["stats"] = {
            "total_params": total_params,
            "pruned_params": pruned_params,
            "pruning_rate": float(pruning_rate),
            "layers": layer_stats
        }
        
        # 添加参数重要性汇总
        if hasattr(self, 'fisher_initialized') and self.fisher_initialized:
            layer_contribution = self.summarize_layerwise_contribution()
            if layer_contribution:
                summary["layer_contribution"] = {
                    "percentages": layer_contribution["percentages"]
                }
        
        return summary
    
    def save_model_summary(self, path):
        """
        将模型摘要保存为JSON和Markdown文件
        
        Args:
            path: 保存路径（不含扩展名）
        """
        summary = self.generate_model_summary()
        
        # 保存JSON
        json_path = f"{path}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        md_path = f"{path}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Frappe学生模型分析报告\n\n")
            
            # 模型基本信息
            f.write("## 模型基本信息\n\n")
            f.write(f"- 模型类型: {summary['model_type']}\n")
            f.write(f"- 特征数量: {summary['feature_count']}\n")
            f.write(f"- 嵌入维度: {summary['embedding_dim']}\n")
            f.write(f"- 隐藏层单元: {summary['hidden_units']}\n")
            f.write(f"- 使用迁移学习: {'是' if summary['is_transfer_learning'] else '否'}\n\n")
            
            # 整体参数统计
            stats = summary['stats']
            f.write("## 参数统计\n\n")
            f.write(f"- 总参数量: {stats['total_params']:,}\n")
            f.write(f"- 已剪枝参数: {stats['pruned_params']:,}\n")
            f.write(f"- 剪枝率: {stats['pruning_rate']*100:.2f}%\n\n")
            
            # 层级参数统计
            f.write("## 层级参数统计\n\n")
            f.write("| 层名称 | 参数量 | 已剪枝 | 剪枝率 | 重要性分数 |\n")
            f.write("|--------|--------|--------|--------|----------|\n")
            
            for layer, layer_stats in stats['layers'].items():
                param_count = layer_stats["param_count"]
                pruned_count = layer_stats["pruned_count"]
                pruning_rate = pruned_count / param_count if param_count > 0 else 0
                importance = layer_stats["importance_score"]
                
                f.write(f"| {layer} | {param_count:,} | {pruned_count:,} | {pruning_rate*100:.2f}% | {importance:.6f} |\n")
            
            f.write("\n")
            
            # 层级贡献
            if "layer_contribution" in summary:
                f.write("## 层级贡献分析\n\n")
                f.write("| 层名称 | 贡献百分比 |\n")
                f.write("|--------|------------|\n")
                
                for layer, percentage in summary["layer_contribution"]["percentages"].items():
                    f.write(f"| {layer} | {percentage:.2f}% |\n")
                
                f.write("\n")
            
            # 特征重要性
            f.write("## 特征重要性分析\n\n")
            
            # 将嵌入层参数按特征名分组
            feature_importance = {}
            for name, param_info in summary["parameters"].items():
                if 'embedding_dict' in name:
                    parts = name.split('.')
                    if len(parts) >= 3:
                        feature_name = parts[1]
                        if feature_name not in feature_importance:
                            feature_importance[feature_name] = {
                                "importance": param_info["importance_score"],
                                "pruning_rate": param_info["pruning_rate"]
                            }
            
            # 按重要性排序
            sorted_features = sorted(feature_importance.items(), 
                                    key=lambda x: x[1]["importance"], 
                                    reverse=True)
            
            if sorted_features:
                f.write("| 特征名 | 重要性分数 | 剪枝率 |\n")
                f.write("|--------|------------|--------|\n")
                
                for feature, info in sorted_features:
                    f.write(f"| {feature} | {info['importance']:.6f} | {info['pruning_rate']*100:.2f}% |\n")
        
        print(f"模型摘要已保存至: {json_path} 和 {md_path}")


class StudentFrappeTrainer:
    """学生模型训练器，支持知识蒸馏和迁移学习，并集成Fisher信息剪枝"""
    
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
            
        # Fisher信息和剪枝相关参数
        self.pruning_frequency = 5  # 每隔多少个epoch执行一次剪枝
        self.pruning_rate = 0.1     # 每次剪枝的比例
        self.cumulative_pruning_rate = 0.5  # 累积剪枝率上限
        self.current_pruning = 0.0  # 当前累积剪枝率
    
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs=10,
        distill_alpha=0.5,
        temperature=3.0,
        patience=3,
        verbose=True,
        compute_fisher=True,
        pruning_frequency=5,
        initial_pruning_rate=0.1,
        max_pruning_rate=0.5,
        output_dir=None
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
            compute_fisher: 是否计算Fisher信息并剪枝
            pruning_frequency: 剪枝频率（每隔多少个epoch）
            initial_pruning_rate: 初始剪枝率
            max_pruning_rate: 最大累积剪枝率
            output_dir: 分析结果保存目录
            
        Returns:
            训练历史记录
        """
        # 设置蒸馏参数
        self.distill_alpha = distill_alpha
        self.temperature = temperature
        self.model.temperature = temperature
        
        # 设置剪枝参数
        self.pruning_frequency = pruning_frequency
        self.pruning_rate = initial_pruning_rate
        self.cumulative_pruning_rate = max_pruning_rate
        
        # 训练历史 - 增强版本，记录更多指标
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_accuracy': [],
            'pruned_params': [],       # 每次剪枝的参数数量
            'pruned_ratio': [],        # 累计剪枝比例
            'pruning_epochs': [],      # 执行剪枝的epoch
            'pruning_metrics': {       # 记录每次剪枝前后的指标变化
                'before': {
                    'loss': [],
                    'auc': [],
                    'accuracy': []
                },
                'after': {
                    'loss': [],
                    'auc': [],
                    'accuracy': []
                }
            }
        }
        
        # 早停设置
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # 计算Fisher信息（如果启用）
        if compute_fisher and hasattr(self.model, 'compute_fisher_information'):
            task_id = f"initial_training"
            self.model.compute_fisher_information(train_dataloader, task_id=task_id)
        
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
                
                # 应用参数掩码（如果已计算Fisher信息）
                if hasattr(self.model, 'mask') and self.model.mask:
                    self.model.apply_mask()
                
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
                history['val_accuracy'].append(val_metrics['accuracy'])
                
                # 早停检查
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # 执行参数剪枝（如果启用Fisher信息）
            pruned_params = 0
            if compute_fisher and hasattr(self.model, 'prune_parameters'):
                # 检查是否需要在当前epoch执行剪枝
                if (epoch + 1) % self.pruning_frequency == 0:
                    # 计算当前剪枝率，确保不超过累积上限
                    current_rate = min(self.pruning_rate, self.cumulative_pruning_rate - self.current_pruning)
                    
                    if current_rate > 0:
                        # 记录剪枝前的性能
                        if val_dataloader:
                            pre_pruning_metrics = self.evaluate(val_dataloader)
                            history['pruning_metrics']['before']['loss'].append(pre_pruning_metrics['loss'])
                            history['pruning_metrics']['before']['auc'].append(pre_pruning_metrics['auc'])
                            history['pruning_metrics']['before']['accuracy'].append(pre_pruning_metrics['accuracy'])
                        
                        # 执行剪枝
                        pruned_params = self.model.prune_parameters(pruning_rate=current_rate)
                        self.current_pruning += current_rate
                        
                        # 记录剪枝统计
                        history['pruned_params'].append(pruned_params)
                        history['pruned_ratio'].append(self.current_pruning)
                        history['pruning_epochs'].append(epoch + 1)
                        
                        # 重新评估剪枝后的性能
                        if val_dataloader:
                            print("评估剪枝后的模型性能...")
                            post_pruning_metrics = self.evaluate(val_dataloader)
                            history['pruning_metrics']['after']['loss'].append(post_pruning_metrics['loss'])
                            history['pruning_metrics']['after']['auc'].append(post_pruning_metrics['auc'])
                            history['pruning_metrics']['after']['accuracy'].append(post_pruning_metrics['accuracy'])
                            
                            # 计算性能变化百分比
                            auc_change = ((post_pruning_metrics['auc'] - pre_pruning_metrics['auc']) / 
                                         pre_pruning_metrics['auc'] * 100)
                            
                            print(f"剪枝后性能 - 损失: {post_pruning_metrics['loss']:.4f}, " 
                                 f"AUC: {post_pruning_metrics['auc']:.4f} ({auc_change:+.2f}%), "
                                 f"准确率: {post_pruning_metrics['accuracy']:.4f}")
                        
                        # 在剪枝后更新Fisher信息（可选，体现参数重要性的变化）
                        if hasattr(self.model, 'compute_fisher_information') and epoch < epochs - 1:
                            task_id = f"after_pruning_epoch_{epoch+1}"
                            self.model.compute_fisher_information(train_dataloader, 
                                                                 sample_size=min(1000, len(train_dataloader.dataset)),
                                                                 task_id=task_id)
            
            # 打印训练信息
            if verbose:
                log_message = f"Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.4f}"
                if val_metrics:
                    log_message += f", 验证损失: {val_metrics['loss']:.4f}, 验证AUC: {val_metrics['auc']:.4f}"
                if pruned_params > 0:
                    log_message += f", 本次剪枝参数: {pruned_params} (累计剪枝率: {self.current_pruning*100:.1f}%)"
                print(log_message)
            
            # 检查是否早停
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                # 恢复最佳模型
                if best_model_state:
                    self.model.load_state_dict(best_model_state)
                break
        
        # 训练结束后，分析层级贡献
        if compute_fisher and hasattr(self.model, 'summarize_layerwise_contribution'):
            print("\n分析训练后的层级贡献...")
            layerwise_contribution = self.model.summarize_layerwise_contribution()
            history['layerwise_contribution'] = layerwise_contribution
            
            # 分析特征重要性
            if hasattr(self.model, 'analyze_feature_importance'):
                print("\n分析特征重要性...")
                feature_importance = self.model.analyze_feature_importance()
                history['feature_importance'] = feature_importance
        
        # 保存分析结果
        if output_dir:
            print(f"保存分析结果至 {output_dir}...")
            self.model.save_model_summary(os.path.join(output_dir, "model_summary"))
            self.plot_pruning_performance(history, os.path.join(output_dir, "pruning_performance.png"))
            self.model.plot_layerwise_contribution(os.path.join(output_dir, "layerwise_contribution.png"))
            self.model.plot_feature_importance(os.path.join(output_dir, "feature_importance.png"))
        
        return history
    
    def plot_pruning_performance(self, history, save_path=None):
        """
        绘制剪枝率与性能的关系图
        
        Args:
            history: 训练历史记录
            save_path: 图表保存路径，如不提供则显示图表
        """
        if 'pruned_ratio' not in history or not history['pruned_ratio']:
            print("错误：没有剪枝历史记录可供绘制")
            return
            
        pruned_ratios = [0] + history['pruned_ratio']  # 加入初始未剪枝状态
        
        # 提取剪枝前后的AUC
        before_metrics = history['pruning_metrics']['before']
        after_metrics = history['pruning_metrics']['after']
        
        # 合并初始AUC和剪枝后的AUC
        aucs = [history['val_auc'][0]] + after_metrics['auc']
        accuracies = [history['val_accuracy'][0]] + after_metrics['accuracy']
        
        # 将百分比转换为更易读的格式
        x_labels = [f"{ratio*100:.1f}%" for ratio in pruned_ratios]
        
        plt.figure(figsize=(12, 6))
        
        # 绘制AUC
        plt.subplot(1, 2, 1)
        plt.plot(pruned_ratios, aucs, 'bo-', label='AUC')
        plt.xlabel('累计剪枝率')
        plt.ylabel('AUC')
        plt.title('剪枝率与AUC的关系')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴标签
        plt.xticks(pruned_ratios, x_labels, rotation=45)
        
        # 绘制准确率
        plt.subplot(1, 2, 2)
        plt.plot(pruned_ratios, accuracies, 'ro-', label='准确率')
        plt.xlabel('累计剪枝率')
        plt.ylabel('准确率')
        plt.title('剪枝率与准确率的关系')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴标签
        plt.xticks(pruned_ratios, x_labels, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"剪枝性能图已保存至: {save_path}")
        else:
            plt.show()
            
        plt.close()

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
    compute_fisher=True,
    pruning_frequency=5,
    initial_pruning_rate=0.1,
    max_pruning_rate=0.5,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_dir=None
):
    """
    通过知识蒸馏训练学生模型，并使用Fisher信息矩阵进行剪枝
    
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
        compute_fisher: 是否计算Fisher信息并剪枝
        pruning_frequency: 剪枝频率（每隔多少个epoch）
        initial_pruning_rate: 初始剪枝率
        max_pruning_rate: 最大累积剪枝率
        output_dir: 分析结果保存目录
        
    Returns:
        训练好的学生模型和训练历史
    """
    print("开始知识蒸馏过程，集成Fisher信息剪枝...")
    
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
    print(f"Fisher信息剪枝: {'启用' if compute_fisher else '禁用'}, 剪枝频率={pruning_frequency}, 初始剪枝率={initial_pruning_rate}")
    
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        distill_alpha=distill_alpha,
        temperature=temperature,
        compute_fisher=compute_fisher,
        pruning_frequency=pruning_frequency,
        initial_pruning_rate=initial_pruning_rate,
        max_pruning_rate=max_pruning_rate,
        output_dir=output_dir
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
        'is_transfer_learning': model.is_transfer_learning,
        'fisher': model.fisher if model.fisher_initialized else None,
        'parameter_importance': model.parameter_importance if hasattr(model, 'parameter_importance') else None,
        'mask': model.mask if hasattr(model, 'mask') else None
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
    
    # 加载Fisher信息和相关数据
    if 'fisher' in checkpoint and checkpoint['fisher'] is not None:
        model.fisher = checkpoint['fisher']
        model.fisher_initialized = True
        
    if 'parameter_importance' in checkpoint and checkpoint['parameter_importance'] is not None:
        model.parameter_importance = checkpoint['parameter_importance']
        
    if 'mask' in checkpoint and checkpoint['mask'] is not None:
        model.mask = checkpoint['mask']
    
    return model


if __name__ == "__main__":
    # 加载教师模型(假设已经训练好)
    from teacher_frappe import CTRFrappeTeacherModel, load_frappe_dataset
    import os
    import matplotlib.pyplot as plt
    
    print("测试基于Fisher信息剪枝的学生模型框架...")
    
    # 获取当前脚本所在目录的上级目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    teacher_model_path = os.path.join(base_dir, "models", "teacher_frappe_model.pth")
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "analysis", "student_frappe_1")
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        # 知识蒸馏与Fisher信息剪枝
        print("开始知识蒸馏与Fisher信息剪枝...")
        student_model, history = distill_knowledge(
            teacher_model,
            train_data.sample(5000),  # 使用小数据集进行演示
            test_data.sample(1000),   # 使用小的验证集
            batch_size=512,
            epochs=10,                 # 演示时使用较少的轮数
            hidden_units=hidden_units,
            distill_alpha=0.7,        # 较高的蒸馏权重
            temperature=3.0,          # 较高温度使软目标更平滑
            compute_fisher=True,      # 启用Fisher信息剪枝
            pruning_frequency=2,      # 每2个epoch剪枝一次
            initial_pruning_rate=0.1, # 初始剪枝率10%
            max_pruning_rate=0.5,     # 最大累积剪枝率50%
            output_dir=output_dir     # 保存分析结果
        )
        
        # 保存学生模型
        student_model_path = os.path.join(base_dir, "models", "student_frappe_1_model.pth")
        save_student_model(student_model, student_model_path)
        print(f"基于Fisher信息剪枝的学生模型已保存至 {student_model_path}")
        
        # 创建训练历史图表
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制AUC曲线
        plt.subplot(1, 2, 2)
        if 'val_auc' in history and history['val_auc']:
            plt.plot(history['val_auc'], 'g-', label='验证AUC')
            # 标记剪枝点
            if 'pruning_epochs' in history and history['pruning_epochs']:
                pruning_epochs = [e-1 for e in history['pruning_epochs']]  # 转为索引
                pruning_aucs = [history['val_auc'][i] for i in pruning_epochs if i < len(history['val_auc'])]
                plt.scatter(pruning_epochs, pruning_aucs, c='r', marker='o', 
                            label='剪枝点', zorder=5)
                
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('验证AUC')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图表
        history_plot_path = os.path.join(output_dir, "training_history.png")
        plt.savefig(history_plot_path)
        print(f"训练历史图表已保存至 {history_plot_path}")
        
        print("Fisher信息剪枝学生模型训练和分析完成!")
        
    except Exception as e:
        print(f"运行时出错: {e}")
        import traceback
        traceback.print_exc()