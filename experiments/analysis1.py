#!/usr/bin/env python
"""
analysis1.py

专注于教师模型的逐层梯度/权重贡献分析（任务 1）:
- 记录训练中每层梯度范数变化
- 跟踪权重更新幅度
- 可视化不同层的梯度和更新模式

基于 DeepCTR-Torch 的 DeepFM 教师模型，使用 Avazu 小样本数据。
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径以便导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.teacher_model import CTRTeacherModel

# ---------------------------
# 1. 模型加载与数据预处理
# ---------------------------
def load_teacher_model_and_data(model_path, data_path, device='cuda'):
    """
    加载教师模型和数据
    
    优先从models目录加载预训练模型，如果失败则考虑其他方式
    """
    print(f"加载数据: {data_path}")
    
    # 加载数据
    data = pd.read_csv(data_path)
    
    # 选择特征
    sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 
                      'app_id', 'device_type', 'device_conn_type']
    
    # 数据预处理 - 确保目标字段正确
    data['click'] = pd.to_numeric(data['click'], errors='coerce')
    data = data.dropna(subset=['click'])
    data['click'] = data['click'].astype(int)
    
    # 检查models目录下是否存在预训练模型
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model_path = os.path.join(base_dir, 'models', 'teacher_model.pth')
    
    # 优先使用models目录下的teacher_model.pth
    if os.path.exists(default_model_path):
        print(f"找到预训练模型: {default_model_path}")
        model_path = default_model_path
    elif os.path.exists(model_path):
        print(f"使用指定模型: {model_path}")
    else:
        print(f"警告: 未找到预训练模型文件 {default_model_path} 或 {model_path}")
        print("将尝试从源代码初始化模型")
    
    # 创建教师模型实例
    teacher_model = CTRTeacherModel(sparse_features=sparse_features)
    
    if os.path.exists(model_path):
        # 尝试加载模型
        print(f"加载教师模型: {model_path}")
        try:
            teacher_model.load_model(model_path, data)
            print("成功加载模型")
        except Exception as e:
            print(f"警告: 加载模型出错: {e}")
            print("尝试使用非严格模式加载模型...")
            try:
                checkpoint = torch.load(model_path)
                teacher_model.sparse_features = checkpoint['sparse_features']
                teacher_model.embedding_dim = checkpoint['embedding_dim']
                teacher_model.task = checkpoint['task']
                teacher_model.encoders = checkpoint['encoders']
                teacher_model.build_model(data)
                teacher_model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("非严格模式加载成功")
            except Exception as e2:
                print(f"非严格模式加载也失败: {e2}")
                print("将使用源代码初始化模型")
                teacher_model.build_model(data)
    else:
        # 如果没有模型文件，初始化一个新模型
        print("初始化新模型")
        teacher_model.build_model(data)
    
    # 将模型移至指定设备
    teacher_model.model.to(device)
    teacher_model.model.eval()
    
    # 划分训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
    return teacher_model, train_data, test_data, sparse_features

# ---------------------------
# 2. 逐层梯度/权重贡献分析
# ---------------------------
def track_gradient_during_training(model, train_data, test_data, 
                                  sparse_features, epochs=1, 
                                  batch_size=8,  # Very small batch size
                                  learning_rate=0.001,
                                  device='cuda'):
    """
    分析模型层的重要性，通过梯度和参数大小
    """
    print("\n=== 逐层梯度/权重重要性分析 ===")
    
    # 获取模型
    model_to_analyze = model.model
    model_to_analyze.eval()  # 分析模式而非训练模式
    
    # 保存参数信息
    param_info = {}
    
    # 1. 首先收集模型参数的基本信息
    print("收集模型参数信息...")
    for name, param in model_to_analyze.named_parameters():
        if param.requires_grad:
            param_norm = torch.norm(param.data).item()
            param_shape = tuple(param.shape)
            param_size = np.prod(param_shape)
            
            param_info[name] = {
                'norm': param_norm,
                'shape': param_shape,
                'size': param_size
            }
    
    # 2. 记录少量样本的梯度信息（无需训练，只需前向传播+反向传播）
    print("\n计算样本梯度信息...")
    
    # 从测试集中抽取少量样本
    sample_size = min(100, len(test_data))
    sample_data = test_data.sample(n=sample_size, random_state=42)
    
    # 预处理样本数据
    processed_sample = model.preprocess_data(sample_data)
    
    # 为每个参数收集梯度信息
    grad_info = {}
    
    try:
        # 批量处理，每次处理一个样本
        for idx, row in tqdm(processed_sample.iterrows(), desc="处理样本", total=len(processed_sample)):
            # 将单个样本转换为批次格式
            single_sample = processed_sample.iloc[[processed_sample.index.get_loc(idx)]]
            x = model.prepare_model_input(single_sample)
            y = torch.FloatTensor([single_sample['click'].values[0]]).to(device)
            
            # 确保梯度跟踪
            model_to_analyze.zero_grad()
            for param in model_to_analyze.parameters():
                if param.requires_grad:
                    param.requires_grad_(True)
            
            # 前向传播
            pred = model_to_analyze(x).squeeze()
            
            # 使用均方误差计算梯度（简单指标）
            loss = torch.nn.functional.mse_loss(pred, y)
            
            # 反向传播
            loss.backward()
            
            # 记录梯度
            for name, param in model_to_analyze.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if name not in grad_info:
                        grad_info[name] = []
                    
                    grad_norm = torch.norm(param.grad).item() if param.grad is not None else 0
                    grad_info[name].append(grad_norm)
            
    except Exception as e:
        print(f"梯度计算时发生错误: {e}")
    
    # 3. 计算每个参数的平均梯度
    avg_grad_norms = {}
    for name in param_info.keys():
        if name in grad_info and len(grad_info[name]) > 0:
            avg_grad_norms[name] = np.mean(grad_info[name])
        else:
            avg_grad_norms[name] = 0.0
    
    # 4. 计算参数重要性得分：梯度范数 * 参数范数
    importance_scores = {}
    for name in param_info.keys():
        importance_scores[name] = avg_grad_norms[name] * param_info[name]['norm']
    
    # 5. 按重要性排序并显示顶部参数
    sorted_params = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n参数重要性排名 (前15):")
    for i, (name, score) in enumerate(sorted_params[:15]):
        grad = avg_grad_norms[name]
        weight = param_info[name]['norm']
        shape = param_info[name]['shape']
        print(f"{i+1}. {name}: 重要性={score:.6f}, 梯度={grad:.6f}, 权重={weight:.6f}, 形状={shape}")
    
    # 6. 分析不同类型的层
    layer_types = {
        'embedding': {'importance': 0, 'count': 0},
        'dnn': {'importance': 0, 'count': 0},
        'fm': {'importance': 0, 'count': 0},
        'linear': {'importance': 0, 'count': 0},
        'other': {'importance': 0, 'count': 0}
    }
    
    for name, score in importance_scores.items():
        if 'embed' in name.lower():
            layer_types['embedding']['importance'] += score
            layer_types['embedding']['count'] += 1
        elif 'dnn' in name.lower():
            layer_types['dnn']['importance'] += score
            layer_types['dnn']['count'] += 1
        elif 'fm' in name.lower():
            layer_types['fm']['importance'] += score
            layer_types['fm']['count'] += 1
        elif 'linear' in name.lower():
            layer_types['linear']['importance'] += score
            layer_types['linear']['count'] += 1
        else:
            layer_types['other']['importance'] += score
            layer_types['other']['count'] += 1
    
    # 计算每种层类型的平均重要性
    print("\n各类型层的平均重要性:")
    for layer_type, data in layer_types.items():
        if data['count'] > 0:
            avg_importance = data['importance'] / data['count']
            print(f"{layer_type}: {avg_importance:.6f} (参数数量: {data['count']})")
    
    # 整理结果
    results = {
        "param_info": param_info,
        "grad_info": grad_info,
        "avg_grad_norms": avg_grad_norms,
        "importance_scores": importance_scores,
        "layer_types": layer_types
    }
    
    return results

# ---------------------------
# 3. 可视化梯度分析结果
# ---------------------------
def visualize_gradient_results(gradient_results):
    """
    可视化梯度变化趋势和权重更新
    """
    print("\n=== 梯度分析结果可视化 ===")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. 可视化DNN层梯度范数变化趋势
    ax1 = axes[0]
    for layer_name, grad_history in gradient_results["dnn_grad_history"].items():
        ax1.plot(range(1, len(grad_history)+1), grad_history, marker='o', label=layer_name)
    
    ax1.set_title('DNN层梯度范数变化趋势')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('平均梯度范数')
    ax1.grid(True)
    ax1.legend()
    
    # 2. 可视化嵌入层梯度范数变化趋势
    ax2 = axes[1]
    for layer_name, grad_history in gradient_results["embedding_grad_history"].items():
        if len(grad_history) > 0:  # 确保有数据
            ax2.plot(range(1, len(grad_history)+1), grad_history, marker='s', label=layer_name)
    
    ax2.set_title('嵌入层梯度范数变化趋势')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('平均梯度范数')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('gradient_analysis_results.png', dpi=300)
    print("分析结果已保存为: gradient_analysis_results.png")

# ---------------------------
# 4. 生成梯度分析报告
# ---------------------------
def generate_gradient_report(gradient_results):
    """
    生成梯度分析报告
    """
    print("\n=== 梯度分析报告 ===")
    
    # 分析DNN层梯度
    dnn_grads = gradient_results["dnn_grad_history"]
    if dnn_grads:
        max_grad_layer = max(dnn_grads.items(), key=lambda x: np.mean(x[1]))
        min_grad_layer = min(dnn_grads.items(), key=lambda x: np.mean(x[1]))
        
        print("\nDNN层梯度分析结果:")
        print(f"  - 梯度范数最大的层: {max_grad_layer[0]}, 平均值: {np.mean(max_grad_layer[1]):.6f}")
        print(f"  - 梯度范数最小的层: {min_grad_layer[0]}, 平均值: {np.mean(min_grad_layer[1]):.6f}")
    
    # 分析嵌入层梯度
    embedding_grads = gradient_results["embedding_grad_history"]
    if embedding_grads:
        if len(embedding_grads) > 0:
            max_embed_layer = max(embedding_grads.items(), key=lambda x: np.mean(x[1]) if len(x[1]) > 0 else 0)
            
            print("\n嵌入层梯度分析结果:")
            print(f"  - 梯度范数最大的嵌入层: {max_embed_layer[0]}, 平均值: {np.mean(max_embed_layer[1]):.6f}")
    
    # 保存报告到文件
    with open('gradient_analysis_report.txt', 'w') as f:
        f.write("DeepFM教师模型梯度分析报告\n")
        f.write("==========================\n\n")
        
        f.write("1. DNN层梯度分析结果:\n")
        if dnn_grads:
            f.write(f"  - 梯度范数最大的层: {max_grad_layer[0]}, 平均值: {np.mean(max_grad_layer[1]):.6f}\n")
            f.write(f"  - 梯度范数最小的层: {min_grad_layer[0]}, 平均值: {np.mean(min_grad_layer[1]):.6f}\n\n")
        
        f.write("2. 嵌入层梯度分析结果:\n")
        if embedding_grads and len(embedding_grads) > 0:
            f.write(f"  - 梯度范数最大的嵌入层: {max_embed_layer[0]}, 平均值: {np.mean(max_embed_layer[1]):.6f}\n\n")
        
        f.write("3. 模型优化建议:\n")
        if dnn_grads:
            f.write(f"  - 关注梯度最活跃的层 ({max_grad_layer[0]}), 可能是模型中最重要的特征提取部分\n")
            if np.mean(max_grad_layer[1]) / np.mean(min_grad_layer[1]) > 10:
                f.write("  - 梯度分布不均匀，考虑使用自适应学习率优化器如Adam或调整学习率\n")
            
    print("\n梯度分析报告已保存至: gradient_analysis_report.txt")
    return

# ---------------------------
# 5. 主函数
# ---------------------------
def main():
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 设置路径
    model_path = os.path.join(base_dir, 'models', 'teacher_model.pth')
    data_path = os.path.join(base_dir, 'data', 'avazu', 'avazu_sample.csv')
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 1. 加载模型和数据
    teacher_model, train_data, test_data, sparse_features = load_teacher_model_and_data(
        model_path, data_path, device)
    
    # 2. 执行梯度/权重贡献分析
    gradient_results = track_gradient_during_training(
        teacher_model, train_data, test_data, sparse_features, 
        epochs=3, batch_size=256, device=device)
    
    # 3. 可视化结果
    visualize_gradient_results(gradient_results)
    
    # 4. 生成分析报告
    generate_gradient_report(gradient_results)

if __name__ == "__main__":
    main()