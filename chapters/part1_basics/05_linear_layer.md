# 第五章：单层网络 — 理解数据如何流动

## 5.0 本章导引

第四章你学会了网络的"骨架"（`class X(nn.Module)`）。现在把血肉填进去。

本章只做一件事：**让你彻底理解 `nn.Linear`——它做了什么，数据怎么流过去的。**

本章不训练、不求梯度、不计算 loss。就只是观察——像一个医生看 X 光片一样，看数据在网络里的 shape 变化。

### 本章地图

```
5.1 nn.Linear 的数学           ← y = Wx + b
5.2 输入输出维度               ← 最容易踩的坑
5.3 单层网络完整示例            ← 从创建到输出
5.4 观察参数                   ← W 和 b 长什么样
5.5 多层堆叠                   ← 维度必须匹配
5.6 用 print 追踪数据流         ← 最重要的调试技巧
5.7 练习
```

---

## 5.1 nn.Linear —— 全连接层

### 5.1.1 公式

```
y = x @ W^T + b

x: 输入，shape [*, in_features]
W: 权重矩阵，shape [out_features, in_features]
b: 偏置向量，shape [out_features]
y: 输出，shape [*, out_features]
```

### 5.1.2 W 是什么——用直觉理解

```
假设你在预测房价。你只知道两个信息：面积和房间数。

输入 x = [面积, 房间数] = [100, 3]    (100平米，3个房间)

权重 W 是一个矩阵：
        面积    房间数
    ┌─────────────────┐
    │ w11    w12      │ → 这行决定了"第一个隐藏特征"
    │ w21    w22      │ → 这行决定了"第二个隐藏特征"
    │ w31    w32      │ → 这行决定了"第三个隐藏特征"
    └─────────────────┘

W 的每一行 = "我有多关注每个输入特征？"
如果 w11 很大 → "面积"对第一个隐藏特征很重要
如果 w12 很小 → "房间数"对第一个隐藏特征不太重要

偏置 b = [b1, b2, b3]：
    如果 b1 = 10 → 即使面积=0, 房间数=0，第一个隐藏特征也有 10 的基础值
```

**现实世界类比**：

```
W = 每个输入信号经过的"音量旋钮"。
    旋钮拧得大 → 这个信号传输得多
    旋钮拧得小 → 这个信号被抑制

b = "基础音量"。
    即使所有输入都是 0，输出也有一个基础值（就像音箱的底噪）
```

### 5.1.3 创建 nn.Linear

```python
# %%
import torch
import torch.nn as nn

# in_features=4：每个输入样本有 4 个特征
# out_features=3：每个输出有 3 个特征
linear = nn.Linear(4, 3)

# 这个层从 4 维空间"映射"到 3 维空间
```

**两个参数的含义**：

```
nn.Linear(in_features, out_features)
              ↑              ↑
         输入有几个数    输出有几个数

    和 batch 无关！batch 维度是自动处理的。
```

---

## 5.2 nn.Linear 的输入输出维度

### 5.2.1 最简单的例子

```python
# %%
import torch
import torch.nn as nn

linear = nn.Linear(4, 3)       # 4 输入 → 3 输出

# 输入：2 个样本，每个样本 4 个特征
x = torch.randn(2, 4)          # shape: [2, 4]
print(f"输入 shape: {x.shape}")

# 前向传播
y = linear(x)
print(f"输出 shape: {y.shape}")  # [2, 3]
```

**维度变化图**：

```
输入 x:          [batch, in_features]    = [2, 4]
                       │
              nn.Linear(4, 3)
                       │
                       ▼
输出 y:          [batch, out_features]   = [2, 3]
```

### 5.2.2 最容易犯的错误——给 nn.Linear 传了 batch 维度

```python
# %%
# ❌ 错误做法：把 batch 也写进 Linear 定义
# linear = nn.Linear(2, 4, 3)  ← nn.Linear 只接受两个参数！

# ✅ 正确：nn.Linear 只关心"特征"维度，batch 维度自动适应
linear = nn.Linear(4, 3)        # 4 个特征 → 3 个特征

# 无论你输入多少样本，Layer 都能处理：
x_small = torch.randn(1, 4)     # batch=1
x_large = torch.randn(128, 4)   # batch=128

print(f"batch=1:   {linear(x_small).shape}")   # [1, 3]
print(f"batch=128: {linear(x_large).shape}")   # [128, 3]
```

### 5.2.3 支持任意多的前导维度

```python
# %%
# nn.Linear 实际上支持任意多前导维度
# 它对最后两维做矩阵乘法，前面的维度都视为"batch 维度"

x_3d = torch.randn(4, 8, 10)     # [4, 8, 10]
linear = nn.Linear(10, 5)        # 10 → 5
y_3d = linear(x_3d)
print(f"3D 输入 → 3D 输出: {y_3d.shape}")  # [4, 8, 5]

# 内部等价于：
# 把 [4, 8, 10] 当成 32 个样本，每个 [10]
# W^T: [10, 5]
# 结果: [32, 5] → reshape → [4, 8, 5]
```

---

## 5.3 完整单层网络示例

```python
# %%
import torch
import torch.nn as nn

# Step 1: 定义网络
class SingleLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)      # 4 → 3
    
    def forward(self, x):
        return self.fc(x)

# Step 2: 创建网络实例
model = SingleLayerNet()

# Step 3: 创建输入数据
x = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0]
])  # shape: [2, 4] — 2 个样本，每个 4 个特征

print(f"输入:")
print(x)
print(f"输入 shape: {x.shape}\n")

# Step 4: 前向传播
y = model(x)
print(f"输出:")
print(y)
print(f"输出 shape: {y.shape}")
```

---

## 5.4 观察参数

### 5.4.1 查看 W 和 b

```python
# %%
linear = nn.Linear(4, 3)

print("权重 W:")
print(f"  shape: {linear.weight.shape}")    # [3, 4]
print(f"  值:\n{linear.weight.data}")

print("\n偏置 b:")
print(f"  shape: {linear.bias.shape}")      # [3]
print(f"  值: {linear.bias.data}")
```

### 5.4.2 为什么 W 是 [out_features, in_features] 而不是反过来？

```
W shape: [3, 4]  =  [out_features, in_features]

因为 PyTorch 内部做的是：
    y = x @ W^T + b

    x:      [2, 4]
    W^T:    [4, 3]
    x @ W^T: [2, 4] @ [4, 3] = [2, 3]

这样矩阵乘法的维度自然就对了。
如果 W 是 [4, 3]，就需要 x @ W（不用转置），但这样不符合数学惯例。
```

### 5.4.3 初始值是怎么来的

```python
# %%
# 每次创建同一个层，参数不同（随机初始化）
net1 = nn.Linear(4, 3)
net2 = nn.Linear(4, 3)

print(f"W 相同？ {torch.equal(net1.weight, net2.weight)}")  # False
print(f"b 相同？ {torch.equal(net1.bias, net2.bias)}")      # False

# PyTorch 默认用 kaiming_uniform 初始化 W
# 偏置 b 默认用 uniform(-1/sqrt(in_features), 1/sqrt(in_features))
```

---

## 5.5 多层 nn.Linear 堆叠

### 5.5.1 关键规则

```
上一层的 out_features = 下一层的 in_features

    [batch, 10]
        │
    nn.Linear(10, 8)    ← out_features=8
        │
    [batch, 8]
        │
    nn.Linear(8, 5)     ← in_features=8（必须等于上一层的 out_features）
        │
    [batch, 5]
        │
    nn.Linear(5, 2)     ← in_features=5（必须等于上一层的 out_features）
        │
    [batch, 2]
```

### 5.5.2 完整示例

```python
# %%
class ThreeLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)   # 784 → 256
        self.fc2 = nn.Linear(256, 128)   # 256 → 128
        self.fc3 = nn.Linear(128, 10)    # 128 → 10
    
    def forward(self, x):
        print(f"输入:     {tuple(x.shape)}")
        x = self.fc1(x)
        print(f"fc1 后:   {tuple(x.shape)}")
        x = self.fc2(x)
        print(f"fc2 后:   {tuple(x.shape)}")
        x = self.fc3(x)
        print(f"fc3 后:   {tuple(x.shape)}")
        return x

model = ThreeLayerNet()
x = torch.randn(32, 784)   # 32 张"伪 MNIST 图片"
output = model(x)
```

**输出**：
```
输入:     (32, 784)
fc1 后:   (32, 256)
fc2 后:   (32, 128)
fc3 后:   (32, 10)
```

### 5.5.3 维度不匹配的经典 bug

```python
# %%
# ❌ 这个网络能工作吗？
class BadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 8)   # 输出 8
        self.fc2 = nn.Linear(5, 3)    # 期望输入 5 ← 不匹配！
    
    def forward(self, x):
        x = self.fc1(x)    # [batch, 10] → [batch, 8]
        x = self.fc2(x)    # ❌ RuntimeError! fc2 期望 [batch, 5] 但收到 [batch, 8]
        return x

# 修复：把 fc2 改成 nn.Linear(8, 3)
```

---

## 5.6 用 print 追踪数据流动——最重要的调试技巧

在每个深度学习工程师的日常工作中，**在 forward 中加 `print(x.shape)` 是最常用的调试手段。**

当你写一个新的网络结构时，第一次运行一定要打印 shape，确保维度变化和你预期的一致。

```python
# %%
class DebugNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        # 动态创建多层 Linear
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.output = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f"Layer {i} ({layer.in_features}→{layer.out_features}): {tuple(x.shape)}")
        x = self.output(x)
        print(f"Output ({self.output.in_features}→{self.output.out_features}): {tuple(x.shape)}")
        return x

# 测试：784 → 256 → 128 → 64 → 10
model = DebugNet(784, [256, 128, 64], 10)
x = torch.randn(32, 784)
output = model(x)
```

---

## 5.7 本章总结

```
关键公式: y = x @ W^T + b
              [*, in] @ [in, out]^T + [out] = [*, out]

维度铁律: 
    nn.Linear(in, out).weight.shape = [out, in]
    nn.Linear(in, out).bias.shape   = [out]
    
    输入: [*, in_features]
    输出: [*, out_features]
    
    上一层的 out_features == 下一层的 in_features

调试口诀:
    写新网络 → 先 print shape → 确认维度 → 再训练
```

---

## 5.8 本章练习

### 练习 5-1：创建单层并验证

```python
# 创建 nn.Linear(5, 2)
# 输入 torch.randn(3, 5)
# 验证输出 shape 是 [3, 2]
# 查看 W 和 b 的 shape
```

### 练习 5-2：手动计算验证

```python
# 创建一个 nn.Linear(2, 1)
# 手动取出 W 和 b
# 输入 x = tensor([[2., 3.]])
# 分别用 linear(x) 和手动 x@W^T+b 计算
# 验证结果相同
```

### 练习 5-3：设计并追踪两层网络

```python
# 10 → 5 → 2
# 在 forward 中打印每层后的 shape
# 输入 torch.randn(4, 10)
```

### 练习 5-4：设计 MNIST 雏形

```python
# 784 → 128 → 64 → 10
# 打印每层后的 shape（输入 batch=32）
```

### 练习 5-5：Debug 练习

```python
# 找出问题（不运行代码）：
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 15)
        self.fc2 = nn.Linear(10, 5)   # 问题在哪？
    
    def forward(self, x):
        return self.fc2(self.fc1(x))
```

### 练习 5-6：不看答案——独立设计并追踪

> 关闭所有文档，独立完成：
> 设计一个 5 → 3 → 1 的网络。
> 打印 W/b 的 shape、每层后的数据 shape。
> 自己验证维度是否匹配。

---

> **下一步**：数据能流动了。但只用 Linear 堆叠 100 层也等于 1 层。需要[第六章：激活函数](./06_activation.md)来赋予网络"非线性"。
