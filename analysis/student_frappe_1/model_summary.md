# Frappe学生模型分析报告

## 模型基本信息

- 模型类型: FrappeStudentModel
- 特征数量: 10
- 嵌入维度: 12
- 隐藏层单元: [64, 32]
- 使用迁移学习: 是

## 参数统计

- 总参数量: 167,045
- 已剪枝参数: 13,303
- 剪枝率: 7.96%

## 层级参数统计

| 层名称 | 参数量 | 已剪枝 | 剪枝率 | 重要性分数 |
|--------|--------|--------|--------|----------|
| embedding_dict.user | 12,000 | 864 | 7.20% | 0.000000 |
| embedding_dict.item | 48,996 | 3,576 | 7.30% | 0.000000 |
| embedding_dict.daytime | 12,000 | 0 | 0.00% | 0.000000 |
| embedding_dict.weekday | 12,000 | 0 | 0.00% | 0.000000 |
| embedding_dict.isweekend | 12,000 | 0 | 0.00% | 0.000000 |
| embedding_dict.homework | 12,000 | 0 | 0.00% | 0.000000 |
| embedding_dict.cost | 12,000 | 0 | 0.00% | 0.000000 |
| embedding_dict.weather | 12,000 | 0 | 0.00% | 0.000000 |
| embedding_dict.country | 12,000 | 0 | 0.00% | 0.000000 |
| embedding_dict.city | 12,000 | 0 | 0.00% | 0.000000 |
| dnn | 10,016 | 8,837 | 88.23% | 0.021718 |
| dnn_linear | 33 | 26 | 78.79% | 0.009718 |

## 层级贡献分析

| 层名称 | 贡献百分比 |
|--------|------------|
| dnn_layer_0 | 99.93% |
| dnn_layer_1 | 0.05% |
| dnn_other | 0.02% |
| embedding_dict.item | 0.00% |
| embedding_dict.isweekend | 0.00% |
| embedding_dict.weather | 0.00% |
| embedding_dict.city | 0.00% |
| embedding_dict.daytime | 0.00% |
| embedding_dict.country | 0.00% |
| embedding_dict.weekday | 0.00% |
| embedding_dict.homework | 0.00% |
| embedding_dict.cost | 0.00% |
| embedding_dict.user | 0.00% |

## 特征重要性分析

| 特征名 | 重要性分数 | 剪枝率 |
|--------|------------|--------|
| isweekend | 0.000000 | 0.00% |
| weather | 0.000000 | 0.00% |
| city | 0.000000 | 0.00% |
| daytime | 0.000000 | 0.00% |
| country | 0.000000 | 0.00% |
| weekday | 0.000000 | 0.00% |
| item | 0.000000 | 7.30% |
| homework | 0.000000 | 0.00% |
| cost | 0.000000 | 0.00% |
| user | 0.000000 | 7.20% |
