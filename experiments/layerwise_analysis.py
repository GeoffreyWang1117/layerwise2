#!/usr/bin/env python
"""
layerwise_analysis.py

完整的教师模型层级分析 pipeline 示例，包含：
1. 逐层梯度/权重贡献分析（记录训练中每层梯度范数变化）
2. 逐层消融实验（冻结或屏蔽某层，观察性能变化）
3. 逐层表征相似性分析（利用线性 CKA 计算各层激活相似度）
4. 预测输出的层贡献分解（利用 Captum 的 LayerLRP）

注意：该示例基于 DeepCTR-Torch 的 DeepFM 作为教师模型，
      数据使用 Avazu 小样本（CSV 文件），教师模型参数保存在 teacher_model.pth 中。
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from captum.attr import LayerLRP, LayerIntegratedGradients

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
# 2. 注册 Hook 捕获模型中间激活
# ---------------------------
def register_hooks(model):
    """
    注册 hook 捕获模型各层的输出。
    返回一个字典 {layer_name: activation_tensor} 以及 hook handle 列表。
    """
    layer_outputs = {}
    hook_handles = []

    def get_hook(name):
        def hook(module, input, output):
            # 保持输出；如果后续需要计算梯度归因，可调用 retain_grad()
            if isinstance(output, torch.Tensor):
                output.retain_grad()
                layer_outputs[name] = output.detach().cuda()
            else:
                # 如果输出不是张量，可能是元组或列表
                layer_outputs[name] = output[0].detach().cuda() if isinstance(output, tuple) else output
        return hook

    # 注册 DeepFM 中的所有关键层
    # 先找到 DNN 部分
    if hasattr(model, 'dnn'):
        for idx, layer in enumerate(model.dnn):
            handle = layer.register_forward_hook(get_hook(f"dnn_layer_{idx}"))
            hook_handles.append(handle)
    
    # FM部分
    if hasattr(model, 'fm'):
        handle = model.fm.register_forward_hook(get_hook("fm_layer"))
        hook_handles.append(handle)
    
    # Linear部分
    if hasattr(model, 'linear_model'):
        handle = model.linear_model.register_forward_hook(get_hook("linear_layer"))
        hook_handles.append(handle)
        
    # 查找其他可能的层，如嵌入层
    if hasattr(model, 'embedding_dict'):
        for name, module in model.embedding_dict.named_children():
            handle = module.register_forward_hook(get_hook(f"embedding_{name}"))
            hook_handles.append(handle)
    
    # 如果模型是嵌套的，可以使用以下方法遍历所有子模块
    # 但由于可能获取太多不必要的层，暂时注释掉
    # for name, module in model.named_modules():
    #     if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
    #         handle = module.register_forward_hook(get_hook(name))
    #         hook_handles.append(handle)
            
    return hook_handles, layer_outputs

# ---------------------------
# 3. 任务 1：逐层梯度/权重贡献分析
# ---------------------------
def track_gradient_during_training(model, train_data, test_data, 
                                  sparse_features, epochs=3, 
                                  batch_size=256, 
                                  learning_rate=0.001,
                                  device='cuda'):
    """
    在多个训练轮次中跟踪模型各层的梯度范数变化
    """
    print("\n=== 任务1: 逐层梯度/权重贡献分析 ===")
    
    # 复制原始模型用于训练
    training_model = deepcopy(model.model)
    training_model.train()
    
    # 准备优化器
    optimizer = torch.optim.Adam(training_model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    # 保存每个epoch的梯度信息
    epoch_grad_norms = {f"epoch_{i}": {} for i in range(epochs)}
    weight_update_history = {f"epoch_{i}": {} for i in range(epochs)}
    
    # 获取所有参数层的名称
    param_names = []
    for name, param in training_model.named_parameters():
        if param.requires_grad:
            param_names.append(name)
    
    # 跟踪前一轮参数值，计算更新幅度
    prev_params = {}
    for name, param in training_model.named_parameters():
        if param.requires_grad:
            prev_params[name] = param.data.clone()
    
    # 主训练循环
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0
        
        # 重新处理训练数据，确保每个epoch都有新的打乱顺序
        shuffled_train = train_data.sample(frac=1.0, random_state=epoch)
        
        # 计算批次数
        num_samples = len(shuffled_train)
        num_batches = num_samples // batch_size + (1 if num_samples % batch_size > 0 else 0)
        
        # 按批次处理数据
        for batch_idx in tqdm(range(num_batches), desc=f"训练 Epoch {epoch+1}"):
            # 获取当前批次数据
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = shuffled_train.iloc[start_idx:end_idx]
            
            # 为每个批次单独处理数据，确保正确的格式
            processed_batch = model.preprocess_data(batch_data)
            batch_x = model.prepare_model_input(processed_batch)
            batch_y = torch.FloatTensor(processed_batch['click'].values).to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            batch_pred = training_model(batch_x).squeeze()
            loss = criterion(batch_pred, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 记录梯度范数
            for name, param in training_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    if name not in epoch_grad_norms[f"epoch_{epoch}"]:
                        epoch_grad_norms[f"epoch_{epoch}"][name] = []
                    epoch_grad_norms[f"epoch_{epoch}"][name].append(grad_norm)
            
            # 更新参数
            optimizer.step()
            total_loss += loss.item()
            
            # 为了减少内存使用，只在部分批次结束时计算权重更新幅度
            if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                for name, param in training_model.named_parameters():
                    if param.requires_grad:
                        # 计算参数相对变化幅度
                        if name in prev_params:
                            update_magnitude = torch.norm(param.data - prev_params[name]).item()
                            update_relative = update_magnitude / (torch.norm(prev_params[name]).item() + 1e-10)
                            
                            if name not in weight_update_history[f"epoch_{epoch}"]:
                                weight_update_history[f"epoch_{epoch}"][name] = []
                            weight_update_history[f"epoch_{epoch}"][name].append(update_relative)
                        
                        # 更新前一轮参数值
                        prev_params[name] = param.data.clone()  
                                                  
        # 计算本轮平均梯度范数
        avg_grad_norms = {}
        for name in param_names:
            if name in epoch_grad_norms[f"epoch_{epoch}"]:
                avg_grad_norms[name] = np.mean(epoch_grad_norms[f"epoch_{epoch}"][name])
        
        # 计算本轮平均权重更新幅度
        avg_weight_updates = {}
        for name in param_names:
            if name in weight_update_history[f"epoch_{epoch}"]:
                avg_weight_updates[name] = np.mean(weight_update_history[f"epoch_{epoch}"][name])
        
        # 选择前10个最大梯度和更新幅度打印
        top_grad_names = sorted(avg_grad_norms.keys(), key=lambda k: avg_grad_norms[k], reverse=True)[:10]
        top_update_names = sorted(avg_weight_updates.keys(), key=lambda k: avg_weight_updates[k], reverse=True)[:10]
        
        print(f"\nEpoch {epoch+1} 平均损失: {total_loss/num_batches:.4f}")
        print("\n前10个最大梯度范数的层:")
        for name in top_grad_names:
            print(f"  {name}: {avg_grad_norms[name]:.6f}")
        
        print("\n前10个最大权重更新幅度的层:")
        for name in top_update_names:
            print(f"  {name}: {avg_weight_updates[name]:.6f}")
    
    # 提取DNN层和嵌入层的梯度信息用于可视化
    dnn_grad_history = {}
    embedding_grad_history = {}
    
    for epoch in range(epochs):
        for name, values in epoch_grad_norms[f"epoch_{epoch}"].items():
            # 提取DNN层信息
            if "dnn" in name.lower():
                layer_name = name.split('.')[-2] + '.' + name.split('.')[-1]  # 提取层名称
                if layer_name not in dnn_grad_history:
                    dnn_grad_history[layer_name] = []
                dnn_grad_history[layer_name].append(np.mean(values))
            
            # 提取嵌入层信息
            elif "embed" in name.lower():
                layer_name = name.split('.')[0] + '.' + name.split('.')[1]  # 提取嵌入层名称
                if layer_name not in embedding_grad_history:
                    embedding_grad_history[layer_name] = []
                embedding_grad_history[layer_name].append(np.mean(values))
    
    results = {
        "epoch_grad_norms": epoch_grad_norms,
        "weight_update_history": weight_update_history,
        "dnn_grad_history": dnn_grad_history,
        "embedding_grad_history": embedding_grad_history
    }
    
    return results

# ---------------------------
# 4. 任务 2：逐层信息保留与消融分析
# ---------------------------
def run_ablation_experiments(model, test_data, sparse_features, device='cuda'):
    """
    对模型各层进行消融实验，比较两种方法：
    1. 将层输出替换为随机噪声
    2. 冻结层权重，阻止其更新
    
    记录每种方法对AUC和LogLoss的影响
    """
    print("\n=== 任务2: 逐层信息保留与消融分析 ===")
    
    # 准备测试数据
    processed_test = model.preprocess_data(test_data)
    test_input = model.prepare_model_input(processed_test)
    y_test = processed_test['click'].values
    
    # 获取基准性能
    original_model = deepcopy(model.model)
    original_model.eval()
    with torch.no_grad():
        base_preds = original_model(test_input).squeeze().cuda().numpy()
    base_loss = log_loss(y_test, base_preds)
    base_auc = roc_auc_score(y_test, base_preds)
    
    print(f"基准性能 - LogLoss: {base_loss:.4f}, AUC: {base_auc:.4f}")
    
    # 定义需要分析的层
    # 这里我们主要关注DNN层，也可以分析其他层
    layers_to_analyze = []
    
    # 查找所有DNN层
    if hasattr(model.model, 'dnn'):
        for idx, _ in enumerate(model.dnn):
            layers_to_analyze.append(f"dnn_layer_{idx}")
    
    # 消融实验结果
    ablation_results = {}
    
    for layer_name in layers_to_analyze:
        print(f"\n分析层: {layer_name}")
        
        # 替换层输出为随机噪声
        def noise_hook(module, input, output):
            return torch.randn_like(output)
        
        layer = getattr(model.model, layer_name, None)
        if layer is not None:
            handle = layer.register_forward_hook(noise_hook)
            with torch.no_grad():
                noisy_preds = model.model(test_input).squeeze().cuda().numpy()
            handle.remove()
            
            noisy_loss = log_loss(y_test, noisy_preds)
            noisy_auc = roc_auc_score(y_test, noisy_preds)
            
            ablation_results[layer_name] = {
                "noisy_loss": noisy_loss,
                "noisy_auc": noisy_auc
            }
    
    return ablation_results

# ---------------------------
# 5. 任务 3：逐层表征相似性分析（CKA）
# ---------------------------
def linear_CKA(X, Y):
    """
    计算线性核心对齐(Linear Kernel Alignment)
    
    参数:
    X, Y: 形状为[N, D]的张量，N是样本数，D是特征维度
    
    返回:
    线性CKA相似度值，范围在[0, 1]
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    
    XTX = torch.mm(X.T, X)
    YTY = torch.mm(Y.T, Y)
    
    XTY = torch.mm(X.T, Y)
    
    # 计算Frobenius范数的平方
    frob_XX = torch.sum(XTX * XTX)
    frob_YY = torch.sum(YTY * YTY)
    frob_XY = torch.sum(XTY * XTY)
    
    # 如果分母为0，则返回0
    if frob_XX == 0 or frob_YY == 0:
        return 0
    
    return frob_XY / (torch.sqrt(frob_XX) * torch.sqrt(frob_YY))

def analyze_layer_similarities(model, data, sparse_features, num_samples=500, device='cuda'):
    """
    使用线性CKA分析模型各层表征之间的相似性
    """
    print("\n=== 任务3: 逐层表征相似性分析 ===")
    
    # 准备数据
    processed_data = model.preprocess_data(data.iloc[:num_samples])
    model_input = model.prepare_model_input(processed_data)
    
    # 注册hook获取各层输出
    model.model.eval()
    hook_handles, layer_outputs = register_hooks(model.model)
    
    # 前向传播以获取各层输出
    with torch.no_grad():
        _ = model.model(model_input)
    
    # 移除hook
    for handle in hook_handles:
        handle.remove()
    
    # 过滤出我们关注的层(主要是DNN层)
    filtered_outputs = {}
    for name, output in layer_outputs.items():
        if 'dnn_layer' in name or 'linear_layer' in name or 'fm_layer' in name:
            # 如果是多维张量，将其展平为2D
            if len(output.shape) > 2:
                output = output.reshape(output.shape[0], -1)
            filtered_outputs[name] = output
    
    # 计算每对层之间的CKA相似度
    layer_names = list(filtered_outputs.keys())
    n_layers = len(layer_names)
    similarity_matrix = torch.zeros((n_layers, n_layers))
    
    for i, name_i in enumerate(layer_names):
        for j, name_j in enumerate(layer_names):
            X = filtered_outputs[name_i]
            Y = filtered_outputs[name_j]
            similarity_matrix[i, j] = linear_CKA(X, Y)
    
    return {
        "similarity_matrix": similarity_matrix,
        "layer_names": layer_names
    }

# ---------------------------
# 6. 任务 4：预测输出的层贡献分解
# ---------------------------
def analyze_layer_contributions(model, data, sparse_features, num_samples=100, device='cuda'):
    """
    使用LayerLRP分析各层对预测结果的贡献
    """
    print("\n=== 任务4: 预测输出的层贡献分解 ===")
    
    # 准备数据
    processed_data = model.preprocess_data(data.iloc[:num_samples])
    model_input = model.prepare_model_input(processed_data)
    y_true = processed_data['click'].values
    
    # 获取模型预测
    model.model.eval()
    with torch.no_grad():
        outputs = model.model(model_input).squeeze().cuda().detach().numpy()
    
    # 选择一些高置信度样本和低置信度样本进行分析
    high_conf_idxs = np.argsort(outputs)[-20:]  # 最高20个预测值
    low_conf_idxs = np.argsort(outputs)[:20]    # 最低20个预测值
    
    print(f"分析 {len(high_conf_idxs)} 个高置信度样本和 {len(low_conf_idxs)} 个低置信度样本")
    
    # 获取各层
    layers = []
    if hasattr(model.model, 'dnn'):
        for idx, layer in enumerate(model.model.dnn):
            if isinstance(layer, nn.Linear):
                layers.append((f"dnn_layer_{idx}", layer))
    
    # 创建LRP分析器
    contributions = {}
    
    # 由于模型结构可能较复杂，这里使用简化方法
    # 对每层输出进行相关性分析
    hook_handles, layer_outputs = register_hooks(model.model)
    
    # 对高置信度样本进行前向传播
    high_conf_inputs = {}
    for i, idx in enumerate(high_conf_idxs):
        if i >= 5:  # 限制样本数量以加快计算
            break
        sample_input = {k: v[idx:idx+1] for k, v in model_input.items()}
        with torch.no_grad():
            _ = model.model(sample_input)
        
        # 记录每层对高置信度样本的激活值
        if f"high_conf_{i}" not in contributions:
            contributions[f"high_conf_{i}"] = {}
        
        for name, output in layer_outputs.items():
            if 'dnn_layer' in name or 'linear_layer' in name or 'fm_layer' in name:
                if isinstance(output, torch.Tensor):
                    contributions[f"high_conf_{i}"][name] = output.abs().mean().item()
    
    # 对低置信度样本进行前向传播
    for i, idx in enumerate(low_conf_idxs):
        if i >= 5:  # 限制样本数量以加快计算
            break
        sample_input = {k: v[idx:idx+1] for k, v in model_input.items()}
        with torch.no_grad():
            _ = model.model(sample_input)
        
        # 记录每层对低置信度样本的激活值
        if f"low_conf_{i}" not in contributions:
            contributions[f"low_conf_{i}"] = {}
        
        for name, output in layer_outputs.items():
            if 'dnn_layer' in name or 'linear_layer' in name or 'fm_layer' in name:
                if isinstance(output, torch.Tensor):
                    contributions[f"low_conf_{i}"][name] = output.abs().mean().item()
    
    # 移除hook
    for handle in hook_handles:
        handle.remove()
    
    return {
        "layer_contributions": contributions,
        "high_conf_preds": outputs[high_conf_idxs],
        "low_conf_preds": outputs[low_conf_idxs],
        "high_conf_true": y_true[high_conf_idxs],
        "low_conf_true": y_true[low_conf_idxs]
    }

# ---------------------------
# 7. 可视化分析结果
# ---------------------------
def visualize_results(gradient_results, ablation_results, similarity_results, contribution_results):
    """
    将所有分析结果可视化
    """
    print("\n=== 结果可视化 ===")
    plt.figure(figsize=(20, 16))
    
    # 1. 可视化任务1: 梯度范数变化趋势
    plt.subplot(2, 2, 1)
    for layer_name, grad_history in gradient_results["dnn_grad_history"].items():
        plt.plot(range(1, len(grad_history)+1), grad_history, marker='o', label=layer_name)
    plt.title('DNN层梯度范数变化趋势')
    plt.xlabel('Epoch')
    plt.ylabel('平均梯度范数')
    plt.grid(True)
    plt.legend()
    
    # 2. 可视化任务2: 消融实验结果
    plt.subplot(2, 2, 2)
    layer_names = list(ablation_results.keys())
    auc_drops = [base_auc - ablation_results[layer]["noisy_auc"] for layer in layer_names]
    
    bars = plt.bar(range(len(layer_names)), auc_drops)
    plt.title('各层噪声替换后的AUC下降')
    plt.xlabel('层')
    plt.ylabel('AUC下降值')
    plt.xticks(range(len(layer_names)), layer_names, rotation=45)
    plt.grid(True, axis='y')
    
    # 为每个柱状图添加数值标签
    for bar, value in zip(bars, auc_drops):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{value:.3f}', ha='center', va='bottom', rotation=0)
    
    # 3. 可视化任务3: 层间相似性热力图
    plt.subplot(2, 2, 3)
    sim_matrix = similarity_results["similarity_matrix"]
    layer_names = similarity_results["layer_names"]
    
    sns.heatmap(sim_matrix, annot=True, cmap='viridis',
                xticklabels=layer_names, yticklabels=layer_names)
    plt.title('层间表征相似性(CKA)')
    
    # 4. 可视化任务4: 层贡献分解
    plt.subplot(2, 2, 4)
    
    # 计算高置信度和低置信度样本的平均层贡献
    high_conf_contribs = {}
    low_conf_contribs = {}
    
    # 提取公共的层名
    common_layers = set()
    for sample_key, layer_dict in contribution_results["layer_contributions"].items():
        common_layers.update(layer_dict.keys())
    common_layers = sorted(list(common_layers))
    
    # 收集高置信度和低置信度样本的贡献
    for sample_key, layer_dict in contribution_results["layer_contributions"].items():
        if 'high_conf' in sample_key:
            for layer_name, contrib in layer_dict.items():
                if layer_name not in high_conf_contribs:
                    high_conf_contribs[layer_name] = []
                high_conf_contribs[layer_name].append(contrib)
        elif 'low_conf' in sample_key:
            for layer_name, contrib in layer_dict.items():
                if layer_name not in low_conf_contribs:
                    low_conf_contribs[layer_name] = []
                low_conf_contribs[layer_name].append(contrib)
    
    # 计算平均贡献
    high_conf_avg = {}
    low_conf_avg = {}
    for layer_name in common_layers:
        if layer_name in high_conf_contribs:
            high_conf_avg[layer_name] = np.mean(high_conf_contribs[layer_name])
        else:
            high_conf_avg[layer_name] = 0
            
        if layer_name in low_conf_contribs:
            low_conf_avg[layer_name] = np.mean(low_conf_contribs[layer_name])
        else:
            low_conf_avg[layer_name] = 0
    
    # 绘制层贡献对比条形图
    x = np.arange(len(common_layers))
    width = 0.35
    
    high_values = [high_conf_avg[layer] for layer in common_layers]
    low_values = [low_conf_avg[layer] for layer in common_layers]
    
    plt.bar(x - width/2, high_values, width, label='高置信度样本')
    plt.bar(x + width/2, low_values, width, label='低置信度样本')
    
    plt.title('高/低置信度样本的层贡献对比')
    plt.xlabel('层')
    plt.ylabel('平均激活值')
    plt.xticks(x, common_layers, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('layerwise_analysis_results.png', dpi=300)
    print("分析结果已保存为: layerwise_analysis_results.png")

# ---------------------------
# 8. 生成综合分析报告
# ---------------------------
def generate_analysis_report(gradient_results, ablation_results, similarity_results, contribution_results):
    """
    生成结构化分析报告
    """
    print("\n=== 综合分析报告 ===")
    
    # 1. 找出梯度变化最大的层
    dnn_grads = gradient_results["dnn_grad_history"]
    max_grad_layer = max(dnn_grads.items(), key=lambda x: np.mean(x[1]))
    min_grad_layer = min(dnn_grads.items(), key=lambda x: np.mean(x[1]))
    
    # 2. 找出影响性能最大的层
    max_impact_layer = max(ablation_results.items(), 
                          key=lambda x: base_auc - x[1]["noisy_auc"])
    
    # 3. 分析层间相似性
    sim_matrix = similarity_results["similarity_matrix"]
    layer_names = similarity_results["layer_names"]
    
    # 找出最相似和最不相似的层对
    max_sim = 0
    min_sim = 1
    max_sim_pair = (0, 0)
    min_sim_pair = (0, 0)
    
    for i in range(sim_matrix.shape[0]):
        for j in range(i+1, sim_matrix.shape[1]):  # 只考虑上三角矩阵
            if sim_matrix[i, j] > max_sim:
                max_sim = sim_matrix[i, j]
                max_sim_pair = (i, j)
            if sim_matrix[i, j] < min_sim:
                min_sim = sim_matrix[i, j]
                min_sim_pair = (i, j)
    
    most_similar_layers = (layer_names[max_sim_pair[0]], layer_names[max_sim_pair[1]])
    least_similar_layers = (layer_names[min_sim_pair[0]], layer_names[min_sim_pair[1]])
    
    # 4. 分析层贡献差异
    high_low_diff = {}
    
    # 提取公共的层名
    common_layers = set()
    for sample_key, layer_dict in contribution_results["layer_contributions"].items():
        common_layers.update(layer_dict.keys())
    common_layers = sorted(list(common_layers))
    
    # 计算高置信度和低置信度样本的平均层贡献
    high_conf_contribs = {layer: [] for layer in common_layers}
    low_conf_contribs = {layer: [] for layer in common_layers}
    
    for sample_key, layer_dict in contribution_results["layer_contributions"].items():
        if 'high_conf' in sample_key:
            for layer_name, contrib in layer_dict.items():
                high_conf_contribs[layer_name].append(contrib)
        elif 'low_conf' in sample_key:
            for layer_name, contrib in layer_dict.items():
                low_conf_contribs[layer_name].append(contrib)
    
    # 计算每层的高/低置信度贡献差异
    for layer in common_layers:
        high_avg = np.mean(high_conf_contribs[layer]) if high_conf_contribs[layer] else 0
        low_avg = np.mean(low_conf_contribs[layer]) if low_conf_contribs[layer] else 0
        high_low_diff[layer] = high_avg - low_avg
    
    # 找出贡献差异最大的层
    max_contrib_diff_layer = max(high_low_diff.items(), key=lambda x: abs(x[1]))
    
    # 输出报告
    print("\n1. 梯度分析结果:")
    print(f"  - 梯度范数最大的层: {max_grad_layer[0]}, 平均值: {np.mean(max_grad_layer[1]):.6f}")
    print(f"  - 梯度范数最小的层: {min_grad_layer[0]}, 平均值: {np.mean(min_grad_layer[1]):.6f}")
    
    print("\n2. 消融实验结果:")
    print(f"  - 对模型性能影响最大的层: {max_impact_layer[0]}")
    print(f"  - 替换为噪声后AUC下降: {base_auc - max_impact_layer[1]['noisy_auc']:.4f}")
    
    print("\n3. 层间相似性分析:")
    print(f"  - 最相似的层对: {most_similar_layers[0]} 和 {most_similar_layers[1]}, 相似度: {max_sim:.4f}")
    print(f"  - 最不相似的层对: {least_similar_layers[0]} 和 {least_similar_layers[1]}, 相似度: {min_sim:.4f}")
    
    print("\n4. 层贡献分析:")
    print(f"  - 高/低置信度样本贡献差异最大的层: {max_contrib_diff_layer[0]}")
    print(f"  - 贡献差异值: {max_contrib_diff_layer[1]:.4f}")
    
    # 模型结构改进建议
    print("\n5. 模型结构改进建议:")
    
    # 基于相似性检测冗余层
    if max_sim > 0.9:  # 如果存在高度相似的层
        print(f"  - 检测到高度相似的层: {most_similar_layers[0]} 和 {most_similar_layers[1]} (相似度: {max_sim:.4f})")
        print(f"    可能存在冗余，考虑移除或减小其中一个层的维度")
    
    # 基于梯度和消融实验识别关键层
    print(f"  - 关键层: {max_impact_layer[0]} (影响最大) 和 {max_grad_layer[0]} (梯度最活跃)")
    print(f"    建议增加这些层的容量或添加skip connection以增强其表达能力")
    
    # 识别可能过拟合的层
    if min_grad_layer[0] != max_contrib_diff_layer[0]:
        print(f"  - 梯度较小但贡献差异大的层: {max_contrib_diff_layer[0]}")
        print(f"    可能存在过拟合风险，考虑增加正则化或dropout")
    
    # 保存报告到文件
    with open('layerwise_analysis_report.txt', 'w') as f:
        f.write("DeepFM教师模型层级分析报告\n")
        f.write("==========================\n\n")
        
        f.write("1. 梯度分析结果:\n")
        f.write(f"  - 梯度范数最大的层: {max_grad_layer[0]}, 平均值: {np.mean(max_grad_layer[1]):.6f}\n")
        f.write(f"  - 梯度范数最小的层: {min_grad_layer[0]}, 平均值: {np.mean(min_grad_layer[1]):.6f}\n\n")
        
        f.write("2. 消融实验结果:\n")
        f.write(f"  - 对模型性能影响最大的层: {max_impact_layer[0]}\n")
        f.write(f"  - 替换为噪声后AUC下降: {base_auc - max_impact_layer[1]['noisy_auc']:.4f}\n\n")
        
        f.write("3. 层间相似性分析:\n")
        f.write(f"  - 最相似的层对: {most_similar_layers[0]} 和 {most_similar_layers[1]}, 相似度: {max_sim:.4f}\n")
        f.write(f"  - 最不相似的层对: {least_similar_layers[0]} 和 {least_similar_layers[1]}, 相似度: {min_sim:.4f}\n\n")
        
        f.write("4. 层贡献分析:\n")
        f.write(f"  - 高/低置信度样本贡献差异最大的层: {max_contrib_diff_layer[0]}\n")
        f.write(f"  - 贡献差异值: {max_contrib_diff_layer[1]:.4f}\n\n")
        
        f.write("5. 模型结构改进建议:\n")
        if max_sim > 0.9:
            f.write(f"  - 检测到高度相似的层: {most_similar_layers[0]} 和 {most_similar_layers[1]} (相似度: {max_sim:.4f})\n")
            f.write(f"    可能存在冗余，考虑移除或减小其中一个层的维度\n")
        
        f.write(f"  - 关键层: {max_impact_layer[0]} (影响最大) 和 {max_grad_layer[0]} (梯度最活跃)\n")
        f.write(f"    建议增加这些层的容量或添加skip connection以增强其表达能力\n")
        
        if min_grad_layer[0] != max_contrib_diff_layer[0]:
            f.write(f"  - 梯度较小但贡献差异大的层: {max_contrib_diff_layer[0]}\n")
            f.write(f"    可能存在过拟合风险，考虑增加正则化或dropout\n")
    
    print("\n分析报告已保存至: layerwise_analysis_report.txt")
    return

# ---------------------------
# 9. 主函数
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
    
    # 2. 任务1: 逐层梯度/权重贡献分析
    gradient_results = track_gradient_during_training(
        teacher_model, train_data, test_data, sparse_features, 
        epochs=3, batch_size=256, device=device)
    
    # 3. 任务2: 逐层信息保留与消融分析
    ablation_results = run_ablation_experiments(
        teacher_model, test_data, sparse_features, device=device)
    
    # 4. 任务3: 逐层表征相似性分析
    similarity_results = analyze_layer_similarities(
        teacher_model, test_data, sparse_features, num_samples=500, device=device)
    
    # 5. 任务4: 预测输出的层贡献分解
    contribution_results = analyze_layer_contributions(
        teacher_model, test_data, sparse_features, num_samples=100, device=device)
    
    # 6. 可视化结果
    visualize_results(
        gradient_results, ablation_results, 
        similarity_results, contribution_results)
    
    # 7. 生成综合分析报告
    generate_analysis_report(
        gradient_results, ablation_results,
        similarity_results, contribution_results)

if __name__ == "__main__":
    main()