# DeepFM 教师模型详解

## 1. 模型概述

### 1.1 CTR预测任务介绍

点击率(Click-Through Rate, CTR)预测是在线广告和推荐系统中的核心技术，目标是预测用户对特定物品(广告、商品等)点击的概率。CTR预测模型输出的是一个介于0到1之间的概率值，代表用户点击的可能性。

### 1.2 DeepFM模型架构

DeepFM模型是一种兼顾了低阶和高阶特征交互的CTR预测模型，它结合了因子分解机(FM)和深度神经网络(DNN)的优点，能够同时学习低阶和高阶特征交互。模型架构如下：

1. **线性部分(Linear)**: 捕获一阶特征重要性
2. **FM部分**: 捕获二阶特征交互
3. **Deep部分**: 捕获高阶特征交互

最终的预测公式为：
$$\hat{y} = sigmoid(y_{linear} + y_{FM} + y_{Deep})$$

其中：
- $y_{linear}$ 是线性部分的输出
- $y_{FM}$ 是FM部分的输出
- $y_{Deep}$ 是Deep部分的输出
- $sigmoid$ 是激活函数，将输出压缩到[0,1]区间

## 2. 模型设计原理

### 2.1 特征表示与嵌入

在CTR预测任务中，输入特征通常是高维稀疏的分类特征。我们使用嵌入层将这些高维稀疏特征转换为低维稠密向量。

对于每个分类特征 $i$，其嵌入表示为：
$$\mathbf{v_i} = Embedding(feature_i) \in \mathbb{R}^k$$

其中 $k$ 是嵌入维度(本代码中默认为4)。

### 2.2 模型各部分数学原理

#### 2.2.1 线性部分(Linear)

线性部分捕获一阶特征对结果的直接影响：

$$y_{linear} = \mathbf{w_0} + \sum_{i=1}^n \mathbf{w_i} \cdot \mathbf{x_i}$$

其中：
- $\mathbf{w_0}$ 是偏置项
- $\mathbf{w_i}$ 是特征 $i$ 的权重
- $\mathbf{x_i}$ 是特征 $i$ 的输入值

#### 2.2.2 FM部分(Factorization Machine)

FM部分捕获二阶特征交互：

$$y_{FM} = \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v_i}, \mathbf{v_j} \rangle \cdot \mathbf{x_i} \cdot \mathbf{x_j}$$

其中：
- $\langle \mathbf{v_i}, \mathbf{v_j} \rangle$ 表示特征 $i$ 和特征 $j$ 的嵌入向量的内积
- $\mathbf{x_i}$ 和 $\mathbf{x_j}$ 是输入值

这可以优化为：

$$y_{FM} = \frac{1}{2} \sum_{f=1}^k \left( \left( \sum_{i=1}^n \mathbf{v_{i,f}} \mathbf{x_i} \right)^2 - \sum_{i=1}^n \mathbf{v_{i,f}}^2 \mathbf{x_i}^2 \right)$$

这一优化将计算复杂度从 $O(kn^2)$ 降低到 $O(kn)$。

#### 2.2.3 Deep部分(DNN)

Deep部分使用多层神经网络捕获高阶特征交互：

$$\mathbf{a^{(0)}} = concat(\mathbf{v_1}, \mathbf{v_2}, ..., \mathbf{v_n})$$

$$\mathbf{a^{(l+1)}} = \sigma(\mathbf{W^{(l)}} \mathbf{a^{(l)}} + \mathbf{b^{(l)}})$$

$$y_{Deep} = \mathbf{W^{(L)}} \mathbf{a^{(L)}} + \mathbf{b^{(L)}}$$

其中：
- $\mathbf{a^{(0)}}$ 是所有特征嵌入的拼接
- $\mathbf{W^{(l)}}$, $\mathbf{b^{(l)}}$ 是第 $l$ 层的权重和偏置
- $\sigma$ 是激活函数(通常是ReLU)
- $L$ 是网络深度

## 3. 代码实现详解

### 3.1 类初始化与配置

```python
def __init__(
    self,
    sparse_features: List[str],
    embedding_dim: int = 4,
    task: str = 'binary',
    device: str = None,
):
```

此构造函数初始化模型的基本配置：
- `sparse_features`: 稀疏特征列表
- `embedding_dim`: 嵌入向量维度，默认为4
- `task`: 任务类型，默认为二分类
- `device`: 计算设备(CPU或GPU)

### 3.2 特征处理

```python
def _create_feature_columns(self, data: pd.DataFrame) -> None:
    # 构建SparseFeat特征列
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=data[feat].nunique() + 1, embedding_dim=self.embedding_dim)
        for feat in self.sparse_features
    ]
    
    self.linear_feature_columns = fixlen_feature_columns
    self.dnn_feature_columns = fixlen_feature_columns
    self.feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)
```

此方法创建特征列，关键点：
- 为每个稀疏特征创建`SparseFeat`对象
- 设置词汇表大小为特征唯一值数量+1(为缺失值预留)
- 为每个特征设置相同的嵌入维度
- 线性部分和DNN部分使用相同的特征列

### 3.3 数据预处理

```python
def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
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
```

数据预处理流程：
1. 复制原始数据避免修改
2. 填充缺失值为'-1'
3. 对每个特征进行标签编码(LabelEncoder)
4. 处理未见过的特征值，替换为已知值
5. 确保所有特征都是整数类型

### 3.4 模型构建与编译

```python
def build_model(self, data: pd.DataFrame) -> None:
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
```

模型构建步骤：
1. 创建特征列
2. 初始化DeepFM模型，设置线性和DNN部分的特征列
3. 编译模型，使用Adam优化器和二元交叉熵损失函数

损失函数(二元交叉熵)的数学表达式为：
$$L = -\frac{1}{N}\sum_{i=1}^N [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

### 3.5 模型训练

```python
def fit(
    self,
    train_data: pd.DataFrame,
    validation_data: Optional[pd.DataFrame] = None,
    batch_size: int = 1024,
    epochs: int = 20,
    verbose: int = 2,
    validation_split: float = 0.1
) -> Dict[str, List[float]]:
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
```

训练流程：
1. 预处理训练数据
2. 如需要，构建模型
3. 准备模型输入(特征)和输出(点击标签)
4. 使用小批量梯度下降法训练模型
5. 返回训练历史记录

### 3.6 中间层输出提取

```python
def get_hidden_outputs(self, input_data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    # 设置为评估模式
    self.model.eval()
    
    # 转换为张量
    tensors = {}
    for feat, value in input_data.items():
        tensors[feat] = torch.tensor(value, dtype=torch.long, device=self.device)
    
    # 获取嵌入层输出
    embeddings = {}
    with torch.no_grad():
        # DeepFM模型内部结构分析
        sparse_embedding_list = self.model.embedding_dict(tensors)
        linear_logit = self.model.linear_model(tensors)
        
        # 获取FM部分的输出
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        fm_logit = self.model.fm(fm_input)
        
        # 获取Deep部分中各层的输出
        dnn_input = torch.flatten(fm_input, start_dim=1)
        dnn_output = self.model.dnn(dnn_input)
        
        embeddings['sparse_embeddings'] = sparse_embedding_list
        embeddings['linear_logit'] = linear_logit
        embeddings['fm_logit'] = fm_logit
        embeddings['dnn_output'] = dnn_output
    
    return embeddings
```

此方法是知识蒸馏的核心，它提取模型各层的中间输出：
1. 将模型设为评估模式
2. 将输入数据转换为张量
3. 提取嵌入层输出
4. 提取线性部分输出
5. 提取FM部分输出
6. 提取DNN部分输出
7. 返回所有层的输出

## 4. 应用场景与最佳实践

### 4.1 模型应用场景

1. **在线广告CTR预测**：预测用户点击广告的概率
2. **推荐系统**：预测用户对物品的兴趣程度
3. **知识蒸馏**：作为教师模型，将知识传递给轻量级学生模型

### 4.2 最佳实践

1. **特征工程**：
   - 选择有代表性的特征
   - 处理缺失值和异常值
   - 标准化特征编码过程

2. **模型调优**：
   - 调整嵌入维度(embedding_dim)
   - 调整DNN层数和神经元数量
   - 使用适当的批量大小和学习率

3. **知识蒸馏**：
   - 使用`get_hidden_outputs`方法提取中间层表示
   - 通过蒸馏损失函数训练学生模型
   - 平衡原始任务损失和蒸馏损失

## 5. 总结

本教师模型基于DeepFM架构，结合了线性模型、因子分解机和深度神经网络的优点，能够有效捕获不同阶的特征交互。模型实现了完整的数据预处理、模型构建、训练评估和中间层提取功能，可作为知识蒸馏的基础教师模型，传递知识给轻量级的学生模型。