import pandas as pd
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, get_feature_names
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ===============================
# 1. 加载数据
# ===============================
data = pd.read_csv('data/avazu/avazu_sample.csv')

# 输出前5行数据，便于检查数据结构
print("数据预览：")
print(data.head())

# ===============================
# 2. 特征选择与数据预处理
# ===============================
# 目标标签（点击率）
target = ['click']

# 选择用于训练的特征
sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 'app_id', 'device_type', 'device_conn_type']

# 将目标标签 click 转换为数值类型，并剔除缺失值
data['click'] = pd.to_numeric(data['click'], errors='coerce')
data = data.dropna(subset=['click'])
data['click'] = data['click'].astype(int)

# 对每个特征列进行缺失值填充
for feat in sparse_features:
    data[feat] = data[feat].fillna('-1')
    
# ===============================
# 3. 对特征进行标签编码
# ===============================
# 使用LabelEncoder将字符串特征转换为整数编码
encoders = {}
for feat in sparse_features:
    encoders[feat] = LabelEncoder()
    data[feat] = encoders[feat].fit_transform(data[feat])
    # 确保类型是整数
    data[feat] = data[feat].astype('int64')

# ===============================
# 4. 构造特征列
# ===============================
# 使用 deepctr_torch 中的 SparseFeat 构造固定长度特征列
fixlen_feature_columns = [
    SparseFeat(feat, vocabulary_size=data[feat].nunique() + 1, embedding_dim=4)
    for feat in sparse_features
]
# 线性部分与 DNN 部分均使用相同的特征列
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns

# 生成特征名称列表，用于构造模型输入字典
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# ===============================
# 5. 划分训练集与测试集
# ===============================
train, test = train_test_split(data, test_size=0.2, random_state=42)

# 构造模型输入数据（确保所有值都是整数类型）
train_model_input = {}
test_model_input = {}

for name in feature_names:
    train_model_input[name] = train[name].values.astype('int64')
    test_model_input[name] = test[name].values.astype('int64')

# ===============================
# 6. 初始化 DeepFM 模型
# ===============================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])

print("目标标签数据类型：", data['click'].dtype)
print("特征数据类型示例：")
for name in list(train_model_input.keys())[:2]:
    print(f"{name}: {train_model_input[name].dtype}")

# ===============================
# 7. 模型训练
# ===============================
y_train = train['click'].values.astype('float32')
model.fit(train_model_input, y_train,
          batch_size=1024, epochs=20, verbose=2, validation_split=0.1)

# ===============================
# 8. 模型评估
# ===============================
pred_ans = model.predict(test_model_input, 256)
y_test = test['click'].values.astype('float32')

print("Test LogLoss:", round(log_loss(y_test, pred_ans), 4))
print("Test AUC:", round(roc_auc_score(y_test, pred_ans), 4))