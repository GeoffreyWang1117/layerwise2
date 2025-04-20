# DeepFM教师模型分析报告

## 模型基本信息

- 模型类型: DeepFM
- 特征数量: 10
- 嵌入维度: 12
- 隐藏层单元: (128, 64, 32)

## 参数统计

- 总参数量: 96,401

## 层级参数统计

| 层名称 | 参数量 | 重要性分数 |
|--------|--------|----------|
| dnn_linear | 25,856 | 6.512866 |
| embedding.cost | 36 | 0.527110 |
| embedding.homework | 48 | 0.039590 |
| embedding.daytime | 96 | 0.038968 |
| embedding.weekday | 96 | 0.030077 |
| embedding.weather | 120 | 0.028812 |
| embedding.isweekend | 36 | 0.020305 |
| embedding.country | 972 | 0.007514 |
| embedding.user | 11,496 | 0.004661 |
| embedding.embedding_dict | 5,392 | 0.003198 |
| embedding.city | 2,808 | 0.003027 |
| embedding.item | 48,996 | 0.002589 |
| other | 1 | 0.002423 |
| dnn_layer_0 | 448 | 0.002050 |

## 层级贡献分析

| 层名称 | 贡献百分比 |
|--------|------------|
| dnn_linear | 97.96% |
| embedding.item | 1.12% |
| embedding.user | 0.48% |
| embedding.cost | 0.17% |
| embedding.city | 0.08% |
| embedding.country | 0.06% |
| embedding.daytime | 0.03% |
| embedding.weather | 0.03% |
| embedding.weekday | 0.03% |
| dnn_layer_0 | 0.02% |
| embedding.homework | 0.02% |
| embedding.isweekend | 0.01% |
| embedding.embedding_dict | 0.00% |
| other | 0.00% |

## 特征重要性分析

| 特征名 | 重要性分数 | 排名 |
|--------|------------|------|
| isweekend | 0.000832 | 1 |
| cost | 0.000817 | 2 |
| homework | 0.000644 | 3 |
| weekday | 0.000323 | 4 |
| daytime | 0.000307 | 5 |
| weather | 0.000232 | 6 |
| country | 0.000029 | 7 |
| city | 0.000010 | 8 |
| user | 0.000003 | 9 |
| item | 0.000001 | 10 |
