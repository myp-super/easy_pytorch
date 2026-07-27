# 第十一章：完整训练 — 第一次真正训练 MNIST

## 11.0 本章导引

**这是整套教程最重要的章节。**

前十章你学了所有零件。现在你把它们组合成一台完整的机器，训练一个能识别手写数字的神经网络。

本章结束时，你应该能**不看任何资料，独立写出约 50 行的 MNIST 完整训练代码。**

如果做不到——回到对应章节，把那章的"不看答案练习"重做一遍。

---

## 11.1 MNIST 是什么

- 70,000 张手写数字图片（60,000 训练 + 10,000 测试）
- 每张是 28×28 的灰度图（只有一个颜色通道）
- 标签是 0-9 的整数
- 这是深度学习界的 "Hello World"

**输入和输出：**

```
输入：28×28 = 784 个像素值（展平成一个向量）
输出：10 个类别的分数（0, 1, 2, ..., 9 各一个）
```

**网络结构设计：**

```
    输入 [batch, 784]         ← 展平后的像素
        │
    Linear(784, 128)
        │
    ReLU                       ← 非线性
        │
    Linear(128, 64)
        │
    ReLU
        │
    Linear(64, 10)             ← 输出 10 个类别的 logits
        │
    CrossEntropyLoss            ← 内含 Softmax，比较概率分布
```

---

## 11.2 完整代码——分 9 步构建

### Step 1：导入

```python
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

### Step 2：定义网络

```python
# %%
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)         # [batch, 1, 28, 28] → [batch, 784]
        x = torch.relu(self.fc1(x))       # [batch, 128]
        x = torch.relu(self.fc2(x))       # [batch, 64]
        x = self.fc3(x)                   # [batch, 10]  ← logits
        return x
```

**`x.view(x.size(0), -1)` 详解：**

```
输入 x 的 shape: [batch_size, 1, 28, 28]
    这是 MNIST 图片的标准格式（1个灰度通道）

x.size(0) → batch_size（比如 64）
-1        → 自动推算 = 1×28×28 = 784

.view(64, -1) → .view(64, 784)
结果: [64, 784]
```

### Step 3：准备数据

```python
# %%
transform = transforms.Compose([
    transforms.ToTensor(),                          # [0,255] → [0,1]
    transforms.Normalize((0.1307,), (0.3081,))     # MNIST 的标准均值和标准差
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练样本: {len(train_dataset):,}")
print(f"测试样本: {len(test_dataset):,}")
print(f"每个 epoch 的 batch 数: {len(train_loader)}")
# 60,000 / 64 ≈ 938
```

### Step 4：创建模型、损失函数、优化器

```python
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"使用设备: {device}")
```

### Step 5：写训练函数

```python
# %%
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()                      # ① 清零
        outputs = model(images)                    # ② 前向
        loss = criterion(outputs, labels)          # ③ 损失
        loss.backward()                            # ④ 反向
        optimizer.step()                           # ⑤ 更新
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)       # 取分数最大的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(dataloader), 100.0 * correct / total
```

**`torch.max(outputs, 1)` 详解：**

```
outputs shape: [64, 10]  → 64 个样本，每个有 10 个分数

torch.max(outputs, 1):
    dim=1 → 在"类别"那个维度上取最大值
    返回值：(最大值, 最大值的索引)
    
    最大值    → 不需要（用 _ 忽略）
    最大索引   → predicted（模型认为最可能的类别）
```

### Step 6：写评估函数

```python
# %%
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():                         # 不追踪梯度
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(dataloader), 100.0 * correct / total
```

### Step 7：训练循环

```python
# %%
num_epochs = 5

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.2f}%")
```

### Step 8：期望输出

```
Epoch 1/5:
  Train Loss: 0.3245, Train Acc: 90.53%
  Test  Loss: 0.1582, Test  Acc: 95.12%
Epoch 2/5:
  Train Loss: 0.1312, Train Acc: 96.04%
  Test  Loss: 0.1108, Test  Acc: 96.64%
Epoch 3/5:
  Train Loss: 0.0901, Train Acc: 97.23%
  Test  Loss: 0.0889, Test  Acc: 97.18%
Epoch 4/5:
  Train Loss: 0.0678, Train Acc: 97.83%
  Test  Loss: 0.0823, Test  Acc: 97.45%
Epoch 5/5:
  Train Loss: 0.0523, Train Acc: 98.31%
  Test  Loss: 0.0805, Test  Acc: 97.56%
```

**准确率从随机猜的 10% 提升到 97%+ 。** 这就是深度学习的威力。

### Step 9：保存模型

```python
# %%
torch.save(model.state_dict(), 'mnist_model.pth')
print("模型已保存！")
```

---

## 11.3 完整代码（一次性展示 ~50 行）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 网络
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=64, shuffle=False
)

# 3. 模型、损失、优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练
for epoch in range(5):
    # 训练
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # 评估
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch+1}: Accuracy = {100*correct/total:.2f}%")
```

---

## 11.4 常见错误排查

```
┌──────────────────────────────────────┬────────────────────────────┐
│ 错误                                  │ 原因和解决                   │
├──────────────────────────────────────┼────────────────────────────┤
│ shape mismatch                        │ 忘记展平：x.view(x.size(0), -1)│
│ expected float but found Byte        │ 没加 transforms.ToTensor()  │
│ mat1 and mat2 shapes cannot multiply │ 维度不匹配：检查 Linear 的维度  │
│ loss 不下降                           │ 1) 忘了 zero_grad           │
│                                      │ 2) lr 太小或太大            │
│                                      │ 3) 忘了 Normalize           │
│ 准确率 10%（等于随机）                  │ 网络没在学习：检查整个训练流程  │
│ CUDA out of memory                   │ batch_size 太大：减半        │
└──────────────────────────────────────┴────────────────────────────┘
```

---

## 11.5 本章练习

### 练习 11-1：跟着敲一遍

把完整代码自己敲一遍（不是复制粘贴），运行直到准确率 ≥ 97%。

### 练习 11-2：改网络结构

尝试：A) 加宽 784→256→128→10；B) 加深 784→256→128→64→32→10。比较准确率。

### 练习 11-3：改学习率

lr = 0.01, 0.001, 0.0001 各训练一次，观察收敛速度。

### 练习 11-4：换 FashionMNIST

把 MNIST 换成 FashionMNIST。代码修改不超过 3 行。目标准确率 ≥ 88%。

### 练习 11-5：不看答案——FashionMNIST 独立完成

> 关闭所有文档，在 FashionMNIST 上独立完成完整训练。

### 练习 11-6：终极检验——不看答案 MNIST 零参考

> **这是整个教程最重要的练习。**
>
> 关闭所有文档和浏览器。从空白文件开始，独立写出 MNIST 完整训练代码，运行到 ≥ 95% 准确率。
>
> 如果做不到——记录卡在哪一章，回去重学。

---

> **下一步**：模型训练好了，怎么用它？进入[第十二章：推理](12_inference.md)。
