from datasets import load_dataset

# 加载 Frappe 数据集
ds = load_dataset("reczoo/Frappe_x1")

# 假设数据集的 split 名为 "train"，你可以根据数据集实际的 split 名称进行调整
train_data = ds["train"]

# 打印前 30 条数据
for i in range(30):
    print(f"样本 {i}:")
    print(train_data[i])
    print("------")