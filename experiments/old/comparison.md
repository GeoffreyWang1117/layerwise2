# 模型比较器使用文档

## 1. 概述

模型比较器(`ModelComparator`)是一个用于比较教师模型和学生模型性能差异的工具类，它通过多个维度的定量分析，帮助评估知识蒸馏的效果。主要功能包括模型大小对比、推理速度测试、预测性能评估和特征表示分析。

## 2. 主要功能

### 2.1 初始化和数据准备

```python
comparator = ModelComparator(
    teacher_model_path="models/teacher_model.pth",
    student_model_path="models/student_model.pth",
    data_path="data/avazu/avazu_sample.csv"
)
```

- 加载教师模型和学生模型
- 准备测试数据和特征提取器
- 自动处理设备选择（CPU/GPU）

### 2.2 模型大小比较

```python
size_results = comparator.compare_model_size()
```

比较两个模型在以下方面的差异：
- 参数数量对比
- 模型文件大小对比
- 计算参数量和模型大小的减少比例

### 2.3 推理速度比较

```python
speed_results = comparator.compare_inference_speed(num_runs=10, batch_size=1000)
```

通过多次运行测量并比较：
- 教师模型和学生模型的平均推理时间
- 学生模型相对教师模型的加速比
- 支持批量大小和运行次数配置

### 2.4 预测性能比较

```python
perf_results = comparator.compare_prediction_performance()
```

使用多种评估指标比较：
- Log Loss（对数损失）对比
- AUC（曲线下面积）对比 
- 计算性能差距，评估知识迁移效果

### 2.5 特征表示比较

```python
feat_results = comparator.compare_feature_representations(num_samples=1000)
```

分析两个模型的内部表示：
- 提取并比较中间层输出
- 计算各层特征的统计信息（均值、标准差、最大/最小值）
- 支持自定义样本数量

## 3. 可视化功能

### 3.1 预测结果可视化

```python
comparator.plot_prediction_comparison()
```

生成包含以下内容的图表：
- ROC曲线对比
- 预测概率分布对比

### 3.2 综合性能可视化

```python
comparator.visualize_model_comparison()
```

生成包含以下内容的图表：
- 参数数量对比（对数尺度）
- 推理时间对比
- AUC指标对比
- Log Loss对比

## 4. 一键比较

```python
results = comparator.run_comprehensive_comparison()
```

一次性运行所有比较并生成：
- 详细的终端输出报告
- 可视化图表
- 包含所有比较结果的字典

## 5. 使用示例

```python
# 导入必要的库
import os
from experiments.comparison import ModelComparator

# 设置路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'data', 'avazu', 'avazu_sample.csv')
teacher_model_path = os.path.join(base_dir, 'models', 'teacher_model.pth')
student_model_path = os.path.join(base_dir, 'models', 'student_model.pth')

# 创建比较器并运行比较
comparator = ModelComparator(
    teacher_model_path=teacher_model_path,
    student_model_path=student_model_path,
    data_path=data_path
)

# 运行全面比较
results = comparator.run_comprehensive_comparison()
```

## 6. 输出结果

比较运行后，将生成以下输出：
- 终端中的详细比较结果
- `prediction_comparison.png`：ROC曲线和预测分布对比图
- `model_comparison.png`：模型大小、推理速度和性能指标的对比图

## 7. 结果解读

- **参数量减少**：学生模型相比教师模型参数量减少的百分比，越高说明压缩效果越好
- **推理加速比**：学生模型推理速度相比教师模型提升的倍数，越高说明性能优化越好
- **AUC差距**：学生模型与教师模型在AUC指标上的差异，接近0说明知识保留得好
- **Log Loss差距**：学生模型与教师模型在对数损失上的差异，越小越好