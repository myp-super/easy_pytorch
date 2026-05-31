# 第十章：Dataset 与 DataLoader — 管理数据

## 10.0 本章导引

前九章你的"数据"都是手动创建的几个 Tensor。但真实的深度学习项目有成千上万条数据——图片、文本、音频……你需要一套高效的机制来**加载、分批、打乱、预处理**它们。

PyTorch 提供了三个工具：

```
Dataset     → "数据在哪？怎么取？"
DataLoader  → "怎么分批？怎么打乱？多线程？"
transforms  → "怎么预处理？（归一化、裁剪、增强……）"
```

---

## 10.1 没有 DataLoader 时的痛苦

```python
# %%
import torch

X = torch.randn(1000, 10)    # 1000 条数据
y = torch.randint(0, 3, (1000,))

batch_size = 32

# 手动分批——代码冗长且容易出错
for i in range(0, len(X), batch_size):
    x_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    # ... 训练 ...

# 痛点：
# - 没 shuffle（数据顺序固定）
# - 最后一个 batch 大小可能不同（需要额外处理）
# - 不能多线程加载（慢）
# - 每次都要写重复代码
```

---

## 10.2 TensorDataset —— 最简单的 Dataset

```python
# %%
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(1000, 10)
y = torch.randint(0, 3, (1000,))

# 把 X 和 y "打包"成一个 Dataset
dataset = TensorDataset(X, y)

# 像访问列表一样访问
x0, y0 = dataset[0]
print(f"第 0 个样本: x shape={x0.shape}, y={y0}")

print(f"数据集大小: {len(dataset)}")   # 1000
```

**TensorDataset 做了什么？**

```
dataset[i] = (X[i], y[i])

就是把第一个 Tensor 的第 i 个元素、第二个 Tensor 的第 i 个元素……组成一个元组。
```

---

## 10.3 DataLoader —— 自动分批

### 10.3.1 核心参数

```python
# %%
dataloader = DataLoader(
    dataset,          # 数据集
    batch_size=32,    # 每批多少条
    shuffle=True      # 每个 epoch 开始时是否打乱顺序
)

# 遍历
for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
    print(f"Batch {batch_idx}: x.shape = {list(x_batch.shape)}, "
          f"y.shape = {list(y_batch.shape)}")
    if batch_idx >= 2:
        break
```

**输出**：
```
Batch 0: x.shape = [32, 10], y.shape = [32]
Batch 1: x.shape = [32, 10], y.shape = [32]
Batch 2: x.shape = [32, 10], y.shape = [32]
```

### 10.3.2 在训练中使用 DataLoader

```python
# %%
# 标准训练循环（终于用上了 DataLoader！）
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        # x_batch: [batch_size, features]
        # y_batch: [batch_size]
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
```

### 10.3.3 shuffle=True 的效果

```
epoch 1 shuffle=True:  数据顺序：[3, 7, 1, 9, 2, ...]  (打乱的)
epoch 2 shuffle=True:  数据顺序：[8, 2, 5, 1, 4, ...]  (又打乱了)
epoch 3 shuffle=True:  数据顺序：[6, 1, 9, 3, 7, ...]  (再次不同)

shuffle=False: 每个 epoch 的顺序完全一样 → 模型可能学到"顺序规律"
```

---

## 10.4 自定义 Dataset

真实数据很少以 Tensor 形式存在。通常是图片文件、CSV、数据库。你需要写自己的 Dataset。

### 10.4.1 三个必须实现的方法

```
自定义 Dataset = 继承 Dataset + 三个方法

class MyDataset(Dataset):
    def __init__(self, ...):    # 初始化：加载数据
    def __len__(self):          # 返回总样本数
    def __getitem__(self, idx): # 返回第 idx 个样本
```

**为什么是这三个？** DataLoader 内部会：
- 调用 `len(dataset)` 知道有多少样本
- 反复调用 `dataset[idx]`（即 `__getitem__`）获取每个样本
- 把获取的样本堆叠成 batch

### 10.4.2 模板

```python
# %%
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X, y):
        """初始化：接收并存储数据"""
        self.X = X
        self.y = y
    
    def __len__(self):
        """返回总样本数"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """返回第 idx 个样本（一个元组）"""
        return self.X[idx], self.y[idx]

# 使用——和 TensorDataset 一样
dataset = MyDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 10.4.3 从 CSV 文件读取——一个完整示例

```python
# %%
import csv
import torch
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    def __init__(self, csv_path):
        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)   # 跳过表头
            for row in reader:
                # 最后列是标签，其余是特征
                features = [float(v) for v in row[:-1]]
                label = float(row[-1])
                self.samples.append((features, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, label = self.samples[idx]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

# 使用
# dataset = CSVDataset('data.csv')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 10.5 transforms —— 数据预处理

```python
# %%
from torchvision import transforms

# Compose：把多个变换串在一起
transform = transforms.Compose([
    transforms.ToTensor(),              # PIL Image → Tensor + [0,255] → [0,1]
    transforms.Normalize(               # (x - mean) / std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ToTensor 做了什么：
# 1. H×W×C (PIL) → C×H×W (Tensor)
# 2. 像素值 [0, 255] → [0, 1]

# Normalize 做了什么：
# 让数据均值变 0、标准差变 1 → 训练更稳定
```

---

## 10.6 torchvision.datasets —— 内置数据集

```python
# %%
from torchvision import datasets

# MNIST：手写数字
mnist_train = datasets.MNIST(
    root='./data',          # 下载到哪里
    train=True,             # True = 训练集
    download=True,          # 没下载就自动下载
    transform=transforms.ToTensor()
)
mnist_test = datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor()
)

print(f"训练集大小: {len(mnist_train)}")   # 60,000
print(f"测试集大小: {len(mnist_test)}")    # 10,000

# 取第一个样本
img, label = mnist_train[0]
print(f"图片 shape: {img.shape}")   # [1, 28, 28]
print(f"标签: {label}")             # 5
```

---

## 10.7 本章总结

```
    数据流全景：

    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Dataset  │ → │DataLoader│ → │  Model   │
    │ 原始数据  │    │分批+打乱  │    │ 前向传播  │
    └──────────┘    └──────────┘    └──────────┘
         │               │
    TensorDataset    batch_size
    自定义Dataset    shuffle=True
    torchvision     num_workers
```

---

## 10.8 本章练习

### 练习 10-1：TensorDataset + DataLoader

```python
# X=randn(200,5), y=randn(200,1)
# TensorDataset → DataLoader(batch_size=20, shuffle=True)
# 遍历并打印每个 batch 的 shape
```

### 练习 10-2：观察 shuffle

```python
# X=arange(20).view(20,1)
# 分别用 shuffle=True 和 False 各遍历一个 epoch
# 打印数据顺序
```

### 练习 10-3：自定义 Dataset

```python
# 写 ListDataset(Dataset):
# __init__(self, data_list)
# __len__, __getitem__
# DataLoader(batch_size=5) 遍历
```

### 练习 10-4：生成数据 Dataset

```python
# 写 MathDataset(n=500):
# __init__: 生成 n 个 (x, y=x²+noise)
# DataLoader 加载
```

### 练习 10-5：加载 MNIST

```python
# 加载 MNIST 训练集和测试集
# 打印各集大小
# 取第一个样本，打印图片和标签的 shape
```

### 练习 10-6：不看答案——独立完成

> 关闭所有文档，独立写出：
> 自定义 Dataset → DataLoader(batch_size=32, shuffle=True) → 遍历 2 个 epoch

---

> **下一步**：所有零件准备就绪。进入[第十一章：完整 MNIST 训练](./11_mnist_training.md)——第一次真正的深度学习训练。
