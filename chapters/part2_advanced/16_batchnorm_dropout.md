# 第十六章：Batch Normalization 与 Dropout — 让训练更稳定

## 16.0 本章导引

第十五章你训练了第一个 CNN，准确率 79%。但你可能注意到：
- 训练初期 loss 震荡
- 训练和测试准确率差距较大（过拟合）
- 训练速度不够快

本章学两个"即插即用"的技术来解决这些问题：
- **Batch Normalization**：稳定训练、加速收敛
- **Dropout**：防止过拟合

这两个技术是当代神经网络的标配——几乎所有现代架构都在用。

---

## 16.1 Batch Normalization

### 16.1.1 问题：内部协变量偏移（Internal Covariate Shift）

```
深层网络的困境：
    第 3 层的参数更新了 → 第 3 层的输出分布变了
    → 第 4 层接收到的输入分布突然改变
    → 第 4 层之前学的参数不再适合新的输入分布
    → 第 4 层被迫重新适应

就像一个工厂流水线：
    上一道工序的产出规格突然变了
    → 下一道工序的机器需要重新调试
    → 整个流水线效率下降
```

**BatchNorm 的解决方案**：在每一层后面，把输出"标准化"——强制均值=0、标准差=1。这样无论前面层的参数怎么变，下一层接收到的输入分布总是稳定的。

### 16.1.2 BatchNorm 做了什么

```
对当前 batch 的每个特征通道：
    1. 计算该通道的均值 μ 和标准差 σ
    2. 归一化：x̂ = (x - μ) / σ
    3. 缩放和平移：y = γ × x̂ + β
                     ↑      ↑
                 可学习参数（让网络能"撤销"BN 如果它有害）

γ（gamma）：缩放参数（初始为 1）
β（beta）：偏移参数（初始为 0）
```

```python
# %%
import torch
import torch.nn as nn

# 对一个 batch 做 BatchNorm 的直观演示
x = torch.randn(4, 3)   # 4 个样本，每个 3 个特征
print(f"BN 前:")
print(f"  均值: {x.mean(dim=0)}")
print(f"  标准差: {x.std(dim=0)}")

bn = nn.BatchNorm1d(3)   # 3 个特征
y = bn(x)
print(f"\nBN 后:")
print(f"  均值: {y.mean(dim=0)}")    # 接近 [0, 0, 0]
print(f"  标准差: {y.std(dim=0)}")   # 接近 [1, 1, 1]
```

### 16.1.3 在 CNN 中使用 BatchNorm2d

```python
# %%
class CNNWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)    # ← 注意：参数是通道数
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # Conv → BN → ReLU → Pool
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        return x
```

**标准顺序**：

```
    Conv2d → BatchNorm2d → ReLU → MaxPool2d
       ↑                     ↑
      线性变换         归一化后再激活
```

> 为什么 BN 放在 ReLU 之前？因为 ReLU 会把负值清零，破坏归一化的分布。先归一化再激活，让 ReLU 接收分布良好的输入。

### 16.1.4 BatchNorm 的训练 vs 推理模式——关键区别！

```python
# %%
bn = nn.BatchNorm2d(16)

# 训练模式
bn.train()
x_train = torch.randn(32, 16, 8, 8)
y_train = bn(x_train)
# 使用当前 batch 的均值和标准差

# 推理模式
bn.eval()
x_test = torch.randn(1, 16, 8, 8)    # 单张图片
y_test = bn(x_test)
# 使用训练时累积的全局均值和标准差（running_mean, running_var）
```

**这就是为什么基础篇反复强调 `model.train()` 和 `model.eval()` 的重要性！**

```
训练时：用当前 batch 的统计量归一化
推理时：不能用单张图片的统计量（没有意义），用训练时积累的全局统计量

BatchNorm 是 train()/eval() 行为不同的最典型案例。
如果你训练时忘记 model.train() → BN 用了全局统计量 → 训练效果差
如果你评估时忘记 model.eval() → BN 用了 batch 统计量 → 评估不准确
```

---

## 16.2 Dropout

### 16.2.1 问题：过拟合

```
过拟合 = 模型"背"了训练数据，而不是"理解"了规律

症状：
    训练准确率很高（99%），但测试准确率明显低（85%）
    训练 loss 持续下降，验证 loss 不降反升
```

### 16.2.2 Dropout 做了什么

```
训练时：以概率 p 随机"关闭"一些神经元（输出置零）
    每次前向传播，关闭的神经元不同
    → 网络不能依赖任何一个神经元
    → 迫使网络学出更多"冗余"的表示
    → 相当于隐式地训练了很多个子网络的平均

推理时：所有神经元都参与（不关闭），但输出乘以 (1-p)
```

```python
# %%
dropout = nn.Dropout(p=0.5)   # 50% 概率关闭

x = torch.ones(10)             # 10 个神经元，全是 1

# 训练模式
dropout.train()
print(f"训练时: {dropout(x)}")
# tensor([2., 0., 2., 0., 2., 2., 0., 2., 0., 2.])
# 约一半是 0，保留的除以 (1-p)=0.5 → 变成 2

# 推理模式
dropout.eval()
print(f"推理时: {dropout(x)}")
# tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
# 全部保留，不缩放
```

### 16.2.3 在网络中放置 Dropout

```python
# %%
class CNNWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=0.3)    # 30% 关闭率
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)                 # 全连接层之后加 Dropout
        x = self.fc2(x)
        return x
```

**Dropout 通常放在全连接层的 ReLU 之后**，卷积层较少用（因为卷积已经有空间结构，随机丢像素不太合理）。

### 16.2.4 Dropout 的常用值

```
全连接层后：p=0.3 ~ 0.5
卷积层后：  p=0.1 ~ 0.2（少用）
小网络：    p 小一点（0.1 ~ 0.3）
大网络：    p 大一点（0.3 ~ 0.5）
```

---

## 16.3 BN + Dropout 完整示例

```python
# %%
class ModernCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类器
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Block 1: Conv → BN → ReLU → Pool
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        
        # 分类器
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

**使用这个网络，CIFAR-10 准确率可以从 79% 提升到 85%+。**

---

## 16.4 本章总结

```
BatchNorm：
    作用：稳定训练、加速收敛、允许更大的学习率
    位置：Conv → BN → ReLU → Pool
    关键：训练用 batch 统计量，推理用全局统计量 → 必须切换 train/eval

Dropout：
    作用：防止过拟合
    位置：全连接层的激活函数之后
    关键：训练时随机关闭神经元，推理时全部参与
```

---

## 16.5 本章练习

### 练习 16-1：观察 BN 的效果

```python
# 创建两个网络（有无 BN），训练 5 个 epoch
# 对比 loss 下降速度和最终准确率
```

### 练习 16-2：观察 Dropout 的效果

```python
# 创建两个网络（有无 Dropout），小数据集训练
# 对比训练/测试准确率的差距（过拟合程度）
```

### 练习 16-3：BN + Dropout 训练 CIFAR-10

```python
# 在第十五章 CNN 基础上加 BN + Dropout
# 目标准确率 ≥ 85%
```

### 练习 16-4：验证 train/eval 切换的重要性

```python
# 训练时故意不用 model.train()
# 观察 loss 是否正常下降
```

### 练习 16-5：不看答案——独立实现

> 关闭所有文档，独立写出带 BN + Dropout 的 CNN + CIFAR-10 完整训练代码。

---

> **下一步**：[第十七章：数据增强与学习率调度](17_augmentation_scheduler.md)。
