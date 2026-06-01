# 第十五章：CNN — 卷积神经网络

## 15.0 本章导引

### 回顾：全连接网络的问题

在基础篇中，你只用 `nn.Linear` 搭建网络。它对 MNIST（28×28=784 像素）能跑到 97%+ 准确率。

但 CIFAR-10（32×32×3=3072 像素）呢？全连接网络只能跑到约 50%。为什么？

**因为全连接网络把图片当成"一堆互不相关的像素"。它不理解"相邻像素构成眼睛、边缘、纹理"。**

```
一张 224×224×3 的图片，用全连接网络：
    输入维度：224 × 224 × 3 = 150,528
    第一层如果 1024 个神经元：
    参数数量：150,528 × 1,024 ≈ 1.54 亿个！

    1.5 亿个参数 → 需要巨量显存 → 需要海量数据 → 太慢

CNN 解决这个问题，用了几百万倍的参数缩减。
```

### 本章地图

```
15.1 CNN 的核心直觉            ← 为什么 CNN 理解图片更好
15.2 Conv2d：滑动窗口检测器     ← 卷积层的数学和直觉
15.3 Pooling：降采样           ← 缩小特征图
15.4 完整 CNN 架构              ← 卷积层 + 池化层 + 全连接层
15.5 LeNet-5 实现              ← 经典 CNN 架构
15.6 CIFAR-10 CNN 训练          ← 目标 75%+ 准确率
15.7 练习
```

---

## 15.1 CNN 的三大核心直觉

### 15.1.1 直觉一：局部感受野 — 不用看整张图

人眼看东西时，不是一次看整张脸，而是先看到眼睛、鼻子、嘴巴这些**局部**特征。

```
全连接网络：每个神经元连接到所有像素
    → 每个神经元都能"看到"整张图
    → 需要巨量参数

CNN：每个神经元只连接到一小块区域（比如 3×3）
    → 每个神经元只"看到"局部
    → 参数大幅减少
```

```
    全连接：[输入 784 个像素] → [每个神经元 784 个连接]
    CNN：   [输入 28×28 的图] → [每个卷积核只看 3×3=9 个像素]
```

### 15.1.2 直觉二：权重共享 — 同一个检测器在整张图上滑动

全连接中每个连接都有独立的权重。CNN 中**同一个卷积核（filter）的权重在整张图上共享。**

```
类比：你用一个"竖线检测器"扫过整张图。
    在左上角检测竖线 → 用同一个检测器
    在右下角检测竖线 → 还是用同一个检测器
    
    不管竖线出现在哪，同一个检测器都能发现它。

    这就是权重共享：
    一个 3×3 的卷积核只有 9 个参数，但可以用在图的任何位置。
```

### 15.1.3 直觉三：层次化特征 — 从简单到复杂

CNN 的不同层学习不同级别的特征：

```
第 1 层：检测简单边缘 → 水平线、竖线、斜线、颜色块
第 2 层：组合简单边缘 → 角、弧线、简单纹理
第 3 层：组合纹理    → 眼睛、鼻子、轮子
第 4 层：组合部件    → 脸、汽车
```

**这和人脑视觉皮层的工作方式高度类似。**

---

## 15.2 Conv2d — 滑动窗口检测器

### 15.2.1 什么是卷积（直觉版）

```
卷积 = 一个小的"权重矩阵"（卷积核）在输入图片上滑动，
      每次计算一个加权和，生成一个新的"特征图"。

    输入图片 (5×5)          卷积核 (3×3)          输出特征图 (3×3)
    ┌─────────────────┐    ┌───────────┐    ┌───────────────┐
    │ 1  2  3  4  1   │    │ 1  0  1   │    │               │
    │ 5  6  7  8  1   │    │ 0  1  0   │    │  ?  ?  ?     │
    │ 9  8  7  6  1   │    │ 1  0  1   │    │  ?  ?  ?     │
    │ 5  4  3  2  1   │    └───────────┘    │  ?  ?  ?     │
    │ 1  2  3  4  5   │                     └───────────────┘
    └─────────────────┘

    第一步：卷积核放在左上角 3×3 区域：
    1×1 + 2×0 + 3×1 + 5×0 + 6×1 + 7×0 + 9×1 + 8×0 + 7×1
    = 1+0+3+0+6+0+9+0+7 = 26  → 特征图[0,0] = 26

    第二步：卷积核向右滑动一格：
    2×1 + 3×0 + 4×1 + 6×0 + 7×1 + 8×0 + 8×1 + 7×0 + 6×1
    = 2+0+4+0+7+0+8+0+6 = 27  → 特征图[0,1] = 27

    ...依此类推...
```

### 15.2.2 nn.Conv2d 的参数

```python
# %%
import torch
import torch.nn as nn

# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
conv = nn.Conv2d(
    in_channels=3,      # 输入有 3 个通道（RGB 图片）
    out_channels=16,    # 输出 16 个通道（16 个不同的卷积核）
    kernel_size=3,      # 每个卷积核是 3×3
    stride=1,           # 每次滑动 1 格
    padding=1           # 在输入周围补一圈 0（保持尺寸不变）
)
```

**参数逐个解释：**

```
in_channels = 3    → RGB 图片有 3 个颜色通道
                     每个卷积核也有 3 个"深度"，分别对应 R、G、B

out_channels = 16  → 用 16 个不同的卷积核
                     每个卷积核学一种不同的模式（竖线、横线、颜色...）
                     输出就有 16 个"特征图"

kernel_size = 3    → 每个卷积核的大小是 3×3
                     3×3 是最常用的选择

stride = 1         → 卷积核每次移动 1 个像素
                     stride=2 → 每次移动 2 个像素（输出尺寸减半）

padding = 1        → 在输入外围补一圈 0
                     这样输出尺寸和输入保持一致
```

### 15.2.3 输入输出 shape

```python
# %%
import torch
import torch.nn as nn

# 输入：16 张 RGB 32×32 图片
x = torch.randn(16, 3, 32, 32)    # [batch, channels, height, width]
print(f"输入 shape: {x.shape}")

# 卷积层
conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
y = conv(x)
print(f"输出 shape: {y.shape}")    # [16, 16, 32, 32]
# batch=16, 通道=16（out_channels），高=32，宽=32
```

**输出尺寸公式：**

```
output_height = (H + 2×padding - kernel_size) / stride + 1
output_width  = (W + 2×padding - kernel_size) / stride + 1

例：H=32, kernel_size=3, stride=1, padding=1
    output_height = (32 + 2 - 3) / 1 + 1 = 32  ✓ 保持不变

例：H=32, kernel_size=3, stride=2, padding=1
    output_height = (32 + 2 - 3) / 2 + 1 = 16.5 → 16（向下取整）
```

```python
# %%
# stride=2 → 输出尺寸减半（降采样）
conv_stride2 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
y2 = conv_stride2(x)
print(f"stride=2 输出: {y2.shape}")  # [16, 16, 16, 16]

# padding=0 → 输出尺寸缩小
conv_no_pad = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
y3 = conv_no_pad(x)
print(f"padding=0 输出: {y3.shape}")  # [16, 16, 30, 30]  ← 32-3+1=30
```

### 15.2.4 一张图理解卷积的维度变化

```
    输入: [batch, in_channels, H, W]
               │
          nn.Conv2d(in_channels, out_channels, kernel_size)
               │
               ▼
    输出: [batch, out_channels, H_out, W_out]

    batch     → 不变（自动处理）
    通道      → in_channels  → out_channels
    空间尺寸   → 取决于 kernel_size、stride、padding
```

### 15.2.5 理解 out_channels：多个卷积核

```python
# %%
# out_channels=32 → 32 个不同的卷积核
conv = nn.Conv2d(3, 32, 3, padding=1)

# 查看卷积核的 shape
print(f"卷积核 shape: {conv.weight.shape}")
# torch.Size([32, 3, 3, 3])
#   ↑        ↑   ↑  ↑  ↑
#   out=32  in=3  h=3 w=3

# 32 个卷积核，每个是 [3, 3, 3]（in_channels × kernel_size × kernel_size）
# 每个卷积核学习检测不同的模式
```

---

## 15.3 MaxPool2d — 降采样

### 15.3.1 为什么需要池化

卷积层提取了特征图。但特征图可能很大。池化层做**降采样**——缩小特征图尺寸。

```
    好处：
    1. 减少计算量（后续层的输入更小）
    2. 增加感受野（同样的卷积核能"看到"更大的区域）
    3. 提供平移不变性（小的位移不影响检测结果）
```

### 15.3.2 最大池化

```python
# %%
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 输入：16 张 16 通道 32×32 特征图
x = torch.randn(16, 16, 32, 32)
y = pool(x)
print(f"池化后 shape: {y.shape}")  # [16, 16, 16, 16]
# 空间尺寸减半：32 → 16
```

**最大池化做了什么：**

```
    输入 4×4，kernel_size=2，stride=2：

    ┌─────────────┐
    │ 1  3  2  4  │      取每个 2×2 窗口的最大值
    │ 5  6  7  8  │
    │ 9  2  1  0  │      ┌────┐
    │ 4  3  2  1  │  →   │ 6 8│
    └─────────────┘      │ 9 2│
                          └────┘
                          输出 2×2
```

---

## 15.4 完整 CNN 架构

一个典型的 CNN 由三种层交替组成：

```
    Conv2d → ReLU → MaxPool2d → Conv2d → ReLU → MaxPool2d
    → ... → Flatten → Linear → Linear → output

    卷积部分：提取特征（空间尺寸逐渐减小，通道数逐渐增加）
    全连接部分：分类决策
```

```python
# %%
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取器（卷积部分）
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)      # [3, 32, 32] → [32, 32, 32]
        self.pool1 = nn.MaxPool2d(2, 2)                   # → [32, 16, 16]
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)     # → [64, 16, 16]
        self.pool2 = nn.MaxPool2d(2, 2)                   # → [64, 8, 8]
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)    # → [128, 8, 8]
        self.pool3 = nn.MaxPool2d(2, 2)                   # → [128, 4, 4]
        
        # 分类器（全连接部分）
        self.fc1 = nn.Linear(128 * 4 * 4, 256)            # 展平后 → 256
        self.fc2 = nn.Linear(256, num_classes)             # → 10
    
    def forward(self, x):
        # 输入： [batch, 3, 32, 32]
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        print(f"conv1+pool1: {tuple(x.shape)}")    # [b, 32, 16, 16]
        
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        print(f"conv2+pool2: {tuple(x.shape)}")    # [b, 64, 8, 8]
        
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        print(f"conv3+pool3: {tuple(x.shape)}")    # [b, 128, 4, 4]
        
        # 展平
        x = x.view(x.size(0), -1)                  # [b, 128*4*4] = [b, 2048]
        print(f"展平后:      {tuple(x.shape)}")
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        print(f"输出:        {tuple(x.shape)}")    # [b, 10]
        return x

model = SimpleCNN()
x = torch.randn(4, 3, 32, 32)    # 4 张 CIFAR-10 图片
output = model(x)
```

**关键观察——通道数 vs 空间尺寸的变化趋势：**

```
    卷积网络中：
        空间尺寸逐渐减少（池化降采样）
        通道数逐渐增加（更多卷积核提取不同特征）

    [3, 32, 32]
        ↓ conv1
    [32, 32, 32]
        ↓ pool
    [32, 16, 16]    空间减半
        ↓ conv2
    [64, 16, 16]
        ↓ pool
    [64, 8, 8]      空间再减半
        ↓ conv3
    [128, 8, 8]
        ↓ pool
    [128, 4, 4]     通道从 3→32→64→128，空间从 32→16→8→4
        ↓ flatten
    [2048]           → 进入全连接
```

---

## 15.5 LeNet-5 — 经典 CNN 完整实现

LeNet-5 是 1998 年 Yann LeCun 提出的 CNN 架构，用于手写数字识别。今天它依然是一个很好的教学案例。

```python
# %%
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 卷积部分
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # MNIST是灰度图，in_channels=1
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 全连接部分
        # MNIST 28×28 → conv1(5,p=2) → 28×28 → pool1 → 14×14
        # → conv2(5) → 10×10 → pool2 → 5×5
        # 通道数=16 → 展平后 = 16×5×5 = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 验证 shape
model = LeNet5()
x = torch.randn(1, 1, 28, 28)
print(f"输入: {tuple(x.shape)}")
# 手动追踪 shape
x = model.conv1(x)
print(f"conv1后: {tuple(x.shape)}")     # [1, 6, 28, 28]
x = model.pool1(x)
print(f"pool1后: {tuple(x.shape)}")     # [1, 6, 14, 14]
x = model.conv2(x)
print(f"conv2后: {tuple(x.shape)}")     # [1, 16, 10, 10]
x = model.pool2(x)
print(f"pool2后: {tuple(x.shape)}")     # [1, 16, 5, 5]
```

**shape 追踪的重要性**：在设计 CNN 时，**一定要手动推算每层后的 shape**。最常用的调试方法就是在 `forward` 中加 `print(x.shape)`。

---

## 15.6 CIFAR-10 CNN 完整训练

```python
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# === 网络 ===
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # 32 → pool → 16 → pool → 8 → pool → 4
        # 128 × 4 × 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # [b,3,32,32] → [b,32,16,16]
        x = self.pool(torch.relu(self.conv2(x)))   # [b,32,16,16] → [b,64,8,8]
        x = self.pool(torch.relu(self.conv3(x)))   # [b,64,8,8] → [b,128,4,4]
        x = x.view(x.size(0), -1)                   # [b, 2048]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# === 数据 ===
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),    # 数据增强：随机水平翻转
    transforms.RandomCrop(32, padding=4),       # 随机裁剪
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_loader = DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform_train),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transform_test),
    batch_size=64, shuffle=False
)

# === 训练 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFAR10CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch+1:2d}: "
          f"Loss={train_loss/len(train_loader):.4f}, "
          f"Acc={100*correct/total:.2f}%")
```

**期望输出**：

```
Epoch  1: Loss=1.5213, Acc=45.23%
Epoch  5: Loss=0.8234, Acc=68.91%
Epoch 10: Loss=0.5123, Acc=74.56%
Epoch 15: Loss=0.3891, Acc=77.23%
Epoch 20: Loss=0.3120, Acc=79.15%
```

**CNN 对比全连接（基础篇 CIFAR-10 项目）：50% → 79%+，几乎翻倍。**

---

## 15.7 本章总结

```
    CNN 三大核心思想：
        1. 局部感受野  → 每个神经元只看局部，大幅减少参数
        2. 权重共享    → 同一个检测器在整张图上滑动
        3. 层次化特征   → 低层检测边缘，高层检测语义

    Conv2d: 滑动窗口做加权和
        in_channels → out_channels（最重要的维度变换）

    MaxPool2d: 降采样
        空间尺寸减半，不变通道数

    CNN 架构模式：
        Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → FC
        （空间↓ 通道↑）                      （分类器）

    设计 CNN 时必须手动推算每层的 shape 变化！
```

---

## 15.8 本章练习

### 练习 15-1：理解 Conv2d 参数

```python
# 输入 [8, 3, 64, 64]
# 分别计算以下 Conv2d 的输出 shape（不运行代码）：
# 1. Conv2d(3, 16, 3, padding=1)
# 2. Conv2d(3, 32, 5, stride=2, padding=2)
# 3. Conv2d(3, 8, 1)
# 然后运行验证
```

### 练习 15-2：追踪 CNN shape

```python
# 设计一个 CNN：
# Conv2d(3,16,3,p=1) → ReLU → MaxPool(2,2)
# → Conv2d(16,32,3,p=1) → ReLU → MaxPool(2,2)
# → Conv2d(32,64,3,p=1) → ReLU → MaxPool(2,2)
# 输入 [1, 3, 64, 64]
# 手动推算每层后的 shape，然后运行验证
```

### 练习 15-3：实现 LeNet-5

```python
# 不看上面的代码，独立实现 LeNet-5
# 用 torch.randn(1, 1, 28, 28) 测试
# 打印每层后的 shape
```

### 练习 15-4：训练 CIFAR-10 CNN

```python
# 完成 CIFAR-10 CNN 训练，目标准确率 ≥ 75%
```

### 练习 15-5：修改架构

```python
# 在 15-4 基础上，尝试：
# A. 增加卷积层数
# B. 改变卷积核数（32→64→128→256）
# C. 增加 Dropout
# 比较准确率
```

### 练习 15-6：不看答案——独立实现 CNN + CIFAR-10

> 关闭所有文档，独立写出 CNN 架构 + CIFAR-10 完整训练的代码。

---

> **下一步**：CNN 训练有时不稳定。进入[第十六章：Batch Normalization 与 Dropout](16_batchnorm_dropout.md)，让训练更稳定。
