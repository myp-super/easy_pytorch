# 第十五章补充：ResNet 与跳跃连接 — 为什么深的网络反而难训练

## 15B.0 本章导引

第十五章你学会了 CNN——卷积 + 池化 + 全连接。按照直觉，层数越多，网络越强。

但实验发现了一个反直觉的现象：**20 层的网络反而比 56 层的网络准确率高。**

不是过拟合——56 层网络的**训练**准确率也比 20 层低。

这意味着深层网络连"记住训练数据"都做不到——它根本**训不动**。

ResNet 用一个极其优雅的想法解决了这个问题：**跳跃连接**。这个想法后来被 Transformer 继承，成了现代深度学习的基石。

---

## 15B.1 问题：深层网络的退化

### 15B.1.1 退化问题（Degradation Problem）

```
    20 层网络：训练准确率 95%，测试准确率 92%
    56 层网络：训练准确率 92%，测试准确率 90%  ← 训练都更低！

    这不是过拟合（过拟合训练准确率高、测试低）。
    这是退化——加了 36 层，网络反而连训练数据都拟合不好了。
```

**为什么会退化？**

```
    原因很复杂，但直觉是：
    梯度在经过 56 层反向传播时，信号越来越弱。
    即使有 BN 和 ReLU，深层的梯度仍然可能"迷路"。

    类比：你让一个 20 人的传话队伍传一句话 → 基本正确
          你让一个 56 人的传话队伍传一句话 → 面目全非
```

### 15B.1.2 ResNet 的核心洞察

```
    56 层网络至少应该能做得和 20 层一样好——因为你可以让后面 36 层
    "什么都不做"（学成恒等映射 f(x)=x），不就等于 20 层了吗？

    问题在于：让一堆非线性层去学 f(x)=x 出奇地困难。
    
    ResNet 的解决方案：
        与其让网络学 f(x)，不如让它学 f(x) - x（残差）。
        如果最优映射是恒等映射，网络只需学到 f(x)-x = 0 → 输出全零即可。
        这比学 f(x)=x 容易得多！
```

---

## 15B.2 跳跃连接 — 整个 ResNet 的精髓

### 15B.2.1 什么是跳跃连接

```
    传统网络：
        input → [Conv → BN → ReLU → Conv → BN] → ReLU → output
        ↑                                    ↑
        直接学习 input → output 的映射

    ResNet（带跳跃连接）：
        input → [Conv → BN → ReLU → Conv → BN] → + → ReLU → output
        │                                      ↑
        └──────────────────────────────────────┘
                    跳跃连接（skip connection）

        网络学习的是 output - input（残差），不是 output 本身。
        output = input + F(input)
```

### 15B.2.2 用代码理解

```python
# %%
import torch
import torch.nn as nn

# 传统卷积块
class TraditionalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out)              # ← 直接输出

# ResNet 残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x                          # ← 保存输入
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity                  # ← 关键！加上输入
        return torch.relu(out)

# 区别就在这一行：
#   传统：return relu(conv(x))
#   ResNet：return relu(conv(x) + x)   ← 跳跃连接！
```

**跳跃连接 = 把输入直接加到输出上。** 就这么简单。但这简单的一行代码，让训练 152 层的网络成为可能。

### 15B.2.3 跳跃连接的直觉

```
    类比：你在学一门新课。

    传统学习：
        每节课都基于上一节课推导出全新的内容
        → 如果第 5 节没听懂 → 第 6-20 节全完蛋

    ResNet 式学习：
        每节课告诉你"这一节和上一节有什么区别"
        → 你始终知道上一节的内容（跳跃连接保留输入）
        → 你只需要学"增量"（残差）
        → 如果某节课的增量是 0 → 你至少还有上一节的内容（恒等映射）
```

**梯度也能直接流过跳跃连接：**

```
    反向传播时，梯度有两条路径：
    1. 穿过卷积层（可能衰减）
    2. 直接通过跳跃连接（完全不衰减！）

    → 即使卷积层的梯度消失了，跳跃连接仍然能传梯度
    → 深层也能收到"新鲜"的梯度信号
```

---

## 15B.3 完整的 ResNet 残差块

### 15B.3.1 通道数变化时的处理

当跳跃连接的输入和输出通道数不同时，需要一个 1×1 卷积来"对齐"维度：

```python
# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接：如果输入输出不匹配 → 用 1×1 卷积对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)            # 跳跃连接（可能对齐维度）
        
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity                   # F(x) + x
        return torch.relu(out)
```

### 15B.3.2 构建 ResNet

```python
# %%
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积（替代原 ResNet 的 7×7 大卷积，适配 CIFAR-10 的 32×32 图片）
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 四个阶段，每个阶段的通道数翻倍，空间尺寸减半
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        # 第一个块可能需要降采样
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        # 后续块保持通道数不变
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ResNet-18: 每阶段 2 个残差块
def ResNet18(num_classes=10):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)

# ResNet-34: 每阶段 [3, 4, 6, 3] 个残差块
def ResNet34(num_classes=10):
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes)

# 测试
model = ResNet18()
x = torch.randn(2, 3, 32, 32)
y = model(x)
print(f"ResNet-18 输入:  {tuple(x.shape)}")
print(f"ResNet-18 输出:  {tuple(y.shape)}")     # [2, 10]

# 参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params:,}")               # ~11M
```

---

## 15B.4 手动推算 ResNet-18 每层的 shape

这是理解 ResNet 的关键——脑内模拟数据的 shape 变化：

```
    输入：[3, 32, 32]
    
    conv1 (3→64, s=1, p=1):    [64, 32, 32]     通道↑ 尺寸不变
    
    layer1 (64→64, s=1):       [64, 32, 32]      2个残差块，尺寸不变
    layer2 (64→128, s=2):      [128, 16, 16]     通道↑ 尺寸↓
    layer3 (128→256, s=2):     [256, 8, 8]       通道↑ 尺寸↓
    layer4 (256→512, s=2):     [512, 4, 4]       通道↑ 尺寸↓
    
    avgpool → [512, 1, 1]
    flatten → [512]
    fc → [10]
```

**趋势总结**：通道数 3→64→128→256→512，空间尺寸 32→16→8→4→1。CNN 的经典模式。

---

## 15B.5 ResNet 在 CIFAR-10 上的训练

```python
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_loader = DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform_train),
    batch_size=128, shuffle=True
)
test_loader = DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transform_test),
    batch_size=128, shuffle=False
)

# 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 训练
for epoch in range(200):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    scheduler.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}: Acc={100*correct/total:.2f}%")
```

**期望结果**：ResNet-18 在 CIFAR-10 上可达 **93-95%** 准确率——比第十五章的普通 CNN（~79%）高出近 15 个百分点。

---

## 15B.6 跳跃连接改变了什么——一个实验

```python
# %%
# 比较：同样的架构，有无跳跃连接的区别

class NoSkipNet(nn.Module):
    """不带跳跃连接的深层网络"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(64 if i > 0 else 3, 64, 3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU()
            ) for i in range(10)  # 10 层
        ])
        self.fc = nn.Linear(64 * 32 * 32, 10)
    
    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SkipNet(nn.Module):
    """带跳跃连接的深层网络（ResNet 风格）"""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
        self.blocks = nn.ModuleList([
            ResidualBlock(64, 64) for _ in range(10)
        ])
        self.fc = nn.Linear(64 * 32 * 32, 10)
    
    def forward(self, x):
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 训练 NoSkipNet 和 SkipNet，比较训练和测试准确率。
# SkipNet 明显优于 NoSkipNet，尤其在深层时差距更大。
```

---

## 15B.7 残差连接的变体

```
    原始 ResNet（上图）：Conv → BN → ReLU → Conv → BN → + → ReLU
    预激活 ResNet（改进版）：BN → ReLU → Conv → BN → ReLU → Conv → +
                          ↑ 把激活放在卷积前面，效果稍好

    瓶颈残差块（ResNet-50/101/152 用）：
        1×1 Conv（降维） → 3×3 Conv → 1×1 Conv（升维） → +
        用 1×1 卷积减少参数，让更深网络成为可能
```

---

## 15B.8 本章总结

```
    退化问题：深层网络连训练数据都拟合不好

    ResNet 核心：
        跳跃连接：output = F(x) + x
        网络学习残差 F(x) = output - x，而不是直接学 output
        梯度可通过跳跃连接直接回流 → 深层也能训得动

    残差块的参数：
        in_channels → out_channels，stride 控制下采样
        通道不匹配时用 1×1 卷积对齐

    ResNet-18: [2,2,2,2] 个残差块，~11M 参数，CIFAR-10 ≥ 93%

    跳跃连接是现代深度学习的基石 → Transformer 的残差连接就是从这里来的
```

---

## 15B.9 本章练习

### 练习 15B-1：手写 ResidualBlock

```python
# 不看上面代码，独立实现：
# ResidualBlock(in_channels, out_channels, stride)
# 包含：两个 Conv+BN+ReLU + 跳跃连接（需要时用 1×1 卷积对齐维度）
```

### 练习 15B-2：对比有无跳跃连接

```python
# 创建两个 10 层网络（有/无跳跃连接），在 CIFAR-10 上训练
# 对比训练 loss 下降速度和最终准确率
```

### 练习 15B-3：实现 ResNet-18

```python
# 独立实现完整 ResNet-18，CIFAR-10 ≥ 90%
```

### 练习 15B-4：修改 ResNet 深度

```python
# 尝试 ResNet-10（[1,1,1,1]）、ResNet-18（[2,2,2,2]）
# 比较训练时间和准确率的权衡
```

### 练习 15B-5：不看答案——独立实现完整的 ResNet-18

> 关闭所有文档，独立写出 ResidualBlock + ResNet18 + CIFAR-10 完整训练。

---

> **下一步**：ResNet 让训练更深成为可能，但要充分利用深度，还需要一些稳定训练的技术。进入[第十六章：Batch Normalization 与 Dropout](16_batchnorm_dropout.md)。
