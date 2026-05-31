# 第六章：激活函数 — 为什么需要非线性

## 6.0 本章导引

第五章你学了 `nn.Linear`——它做的是 `y = Wx + b`。这是**线性**变换。

现在的问题是：如果你堆叠 10 层 `nn.Linear`，不加任何其他东西……

**10 层 = 1 层。**

这不是夸张。这是数学定理。本章用代码证明给你看，然后告诉你解决方案。

---

## 6.1 为什么 Linear 堆叠再多层也没用

### 6.1.1 数学证明

```
第 1 层：y₁ = W₁x + b₁
第 2 层：y₂ = W₂y₁ + b₂
         = W₂(W₁x + b₁) + b₂
         = (W₂W₁)x + (W₂b₁ + b₂)
         = W'x + b'

其中 W' = W₂W₁，b' = W₂b₁ + b₂

结果：两层 Linear = 一个等效的单层 Linear
     100 层 Linear = 也是一个等效的单层 Linear
```

**因为：线性函数的复合仍然是线性函数。**

### 6.1.2 用代码验证

```python
# %%
import torch
import torch.nn as nn

# 10 层纯 Linear（不加激活函数）
class TenLinearLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(2, 2) for _ in range(10)])
    
    def forward(self, x):
        return self.layers(x)

# 1 层 Linear（等效网络）
class OneLinearLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)
    
    def forward(self, x):
        return self.fc(x)

# 比较：在不同输入上的输出
x = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, -1.0]
])

# 训练 OneLinearLayer 来"模仿" TenLinearLayers
# 因为 10 层线性 = 1 层线性，所以 1 层完全可以学到相同映射！
# （这里我们不是真的训练，而是说明数学原理）
```

### 6.1.3 现实世界类比

```
只用直尺，你只能画直线。
无论你有 10 把直尺还是 1 把直尺——画出来的都是直线。

如果你想把点和点之间用曲线连起来，你需要"能弯曲的东西"。
激活函数就是那个"能弯曲的东西"。
```

---

## 6.2 ReLU — 最重要的激活函数

### 6.2.1 ReLU 的数学

```
ReLU(x) = max(0, x)

即：
    x > 0  → 保持原值
    x ≤ 0  → 变成 0
```

```python
# %%
import torch

x = torch.tensor([-3.0, -1.5, 0.0, 1.5, 3.0])
print(f"输入:     {x}")
print(f"ReLU 后:  {torch.relu(x)}")
# tensor([0., 0., 0., 1.5, 3.])
```

**ReLU 图像**（在脑海中画）：

```
    y
    │         /
    │        /
    │       /
    │      /
    │     /
    ├────/────────── x
    │
    │ (x≤0 时 y=0)
```

### 6.2.2 为什么 ReLU 如此流行

| 优点 | 说明 |
|------|------|
| **计算极简单** | 就是 `if x > 0: 保留 else: 置零`，硬件上几乎零开销 |
| **梯度不消失** | 正数区域梯度恒为 1——梯度不会越传越小 |
| **稀疏激活** | 约 50% 的神经元输出 0 → 只有部分神经元在"工作" → 类似人脑 |
| **在实践中效果最好** | 几十年经验证明 |

### 6.2.3 在网络中使用 ReLU

```python
# %%
import torch.nn as nn

class NetWithReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()        # 激活函数也是一个"层"
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)      # [batch, 784] → [batch, 128]
        x = self.relu(x)     # [batch, 128] → [batch, 128]  ← shape 不变！
        x = self.fc2(x)      # [batch, 128] → [batch, 10]
        return x
```

**关键理解**：激活函数**不改变 shape**！它只改变值。

### 6.2.4 模块式 vs 函数式

```python
# %%
x = torch.randn(3, 5)

# 模块式：nn.ReLU() —— 可以放在网络中
relu_module = nn.ReLU()
y1 = relu_module(x)

# 函数式：torch.relu() —— 直接在 forward 中调用
y2 = torch.relu(x)

print(torch.equal(y1, y2))  # True

# 两者效果相同。在 forward 中常用函数式（更简洁）：
# x = torch.relu(self.fc1(x))
```

---

## 6.3 Sigmoid — 把输出压缩到 (0, 1)

### 6.3.1 Sigmoid 的数学

```
σ(x) = 1 / (1 + e^(-x))

x → -∞ 时，σ(x) → 0
x → +∞ 时，σ(x) → 1
x = 0   时，σ(x) = 0.5
```

```python
# %%
x = torch.linspace(-5, 5, 11)
s = torch.sigmoid(x)

for xi, si in zip(x, s):
    print(f"sigmoid({xi:5.1f}) = {si:.4f}")
```

**输出**：
```
sigmoid( -5.0) = 0.0067
sigmoid( -4.0) = 0.0180
sigmoid( -3.0) = 0.0474
sigmoid( -2.0) = 0.1192
sigmoid( -1.0) = 0.2689
sigmoid(  0.0) = 0.5000
sigmoid(  1.0) = 0.7311
sigmoid(  2.0) = 0.8808
sigmoid(  3.0) = 0.9526
sigmoid(  4.0) = 0.9820
sigmoid(  5.0) = 0.9933
```

**直观理解**：Sigmoid 把任何实数"挤压"到 (0, 1) 之间。

### 6.3.2 什么时候用 Sigmoid

**二分类问题的输出层。**

如果你的网络要回答"是/否"问题（如"这张图是猫吗？"），最后一层用 1 个输出 + Sigmoid → 输出一个 (0,1) 之间的概率。

### 6.3.3 Sigmoid 的致命缺陷：梯度消失

```python
# %%
x_far = torch.tensor(10.0, requires_grad=True)
y = torch.sigmoid(x_far)
y.backward()
print(f"sigmoid(10) 的梯度: {x_far.grad:.8f}")
# ≈ 0.00004540 —— 几乎是 0！

# 当输入在 ±5 之外时，Sigmoid 的梯度接近于 0
# 梯度 → 0 → 参数不更新 → 前面的层学不到东西 → 网络"死"了
```

**这就是为什么 ReLU 取代了 Sigmoid 成为隐藏层的默认选择。**

---

## 6.4 Tanh — 压缩到 (-1, 1)

```python
# %%
x = torch.linspace(-3, 3, 7)
t = torch.tanh(x)

for xi, ti in zip(x, t):
    print(f"tanh({xi:5.1f}) = {ti:.4f}")
# tanh( -3.0) = -0.9951
# tanh( -2.0) = -0.9640
# tanh( -1.0) = -0.7616
# tanh(  0.0) =  0.0000
# tanh(  1.0) =  0.7616
# tanh(  2.0) =  0.9640
# tanh(  3.0) =  0.9951
```

**Tanh vs Sigmoid**：

```
Sigmoid: (0, 1)  —— 以 0.5 为中心 → 输出总是正的 → 可能减慢收敛
Tanh:    (-1, 1) —— 以 0 为中心   → 正负均衡 → 通常比 Sigmoid 好
```

**但两者都有梯度消失问题。** ReLU 已经取代了它们成为隐藏层的默认选择。

---

## 6.5 激活函数在网络中的位置——标准模式

### 6.5.1 标准结构

```
Linear → ReLU → Linear → ReLU → Linear → [可选：Sigmoid/Softmax]

规则：
    1. 每个 Linear 后面通常跟一个激活函数（除了最后一层）
    2. 隐藏层几乎永远用 ReLU 或其变体
    3. 最后一层的激活取决于任务：
       分类 → 不加（CrossEntropyLoss 内置 Softmax）
       二分类 → Sigmoid（配合 BCEWithLogitsLoss）
       回归 → 不加
```

### 6.5.2 完整网络示例

```python
# %%
class StandardClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)    # 最后一层：无激活
    
    def forward(self, x):
        x = x.view(x.size(0), -1)             # 展平
        x = torch.relu(self.fc1(x))           # Linear → ReLU
        x = torch.relu(self.fc2(x))           # Linear → ReLU
        x = self.fc3(x)                       # Linear（无激活！）
        return x
        # 返回 raw logits → 配合 CrossEntropyLoss（内置 Softmax）
```

### 6.5.3 最后一层不要加 Softmax

```python
# %%
# ❌ 错误
class BadClassifier(nn.Module):
    def forward(self, x):
        return torch.softmax(self.fc3(x), dim=1)  # 加了 Softmax

# ✅ 正确
class GoodClassifier(nn.Module):
    def forward(self, x):
        return self.fc3(x)  # 返回 logits

# 原因：CrossEntropyLoss = LogSoftmax + NLLLoss
# 如果网络做了 Softmax，CrossEntropyLoss 还会再做一次
# 两次 Softmax → 结果不对 + 数值不稳定
```

---

## 6.6 有/无激活函数的对比实验

```python
# %%
import torch
import torch.nn as nn

# 两个网络：一个有 ReLU，一个没有
class LinearOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(*[nn.Linear(2, 2) for _ in range(5)])
    def forward(self, x): return self.net(x)

class WithReLU(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(5):
            layers.append(nn.Linear(2, 2))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# 测试：输入 4 个点
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])

net_linear = LinearOnly()
net_relu = WithReLU()

print("纯线性网络输出：")
print(net_linear(x))
print("\n带 ReLU 网络输出：")
print(net_relu(x))

# 纯线性网络输出的是输入的线性变换 → 所有输出点都在同一个平面上
# 带 ReLU 的输出是非线性的 → 输出点可以被"掰弯"
```

---

## 6.7 本章总结

```
    为什么需要激活函数？
        线性 + 线性 = 线性 → 100 层等于 1 层
        激活函数打破"线性链" → 网络有了真正的表达能力

    激活函数速查表：
        ReLU:    隐藏层默认选择  f(x) = max(0, x)
        Sigmoid: 二分类输出层     f(x) = 1/(1+e^(-x))  → (0, 1)
        Tanh:    替代 Sigmoid     f(x) → (-1, 1)
        Softmax: 多分类输出       → 概率分布（第七章讲）

    标准模式：
        Linear → ReLU → Linear → ReLU → Linear
        最后一层不加激活（除非输出层有特殊要求）
```

---

## 6.8 本章练习

### 练习 6-1：手动实现 ReLU

```python
# 用 torch.max 实现 ReLU
# 验证和 torch.relu 结果一致
```

### 练习 6-2：构建带激活的三层网络

```python
# 784 → ReLU → 256 → ReLU → 128 → ReLU → 10
# 输入 batch=32，打印每层后的 shape
```

### 练习 6-3：纯线性 vs 带激活 对比实验

```python
# 两个 3 层网络（A：不加激活，B：加 ReLU）
# 给相同输入，观察输出分布差异
# 验证：纯线性网络的输出是否是输入的线性组合
```

### 练习 6-4：观察 Sigmoid 和 Tanh

```python
# 对于 [-5, -3, -1, 0, 1, 3, 5]：
# 1. 计算 sigmoid 和 tanh
# 2. 哪个关于原点对称？
# 3. 哪个的范围更大？
```

### 练习 6-5：不看答案——独立完成

> 关闭所有文档，独立写出：
> `784 → ReLU → 256 → ReLU → 128 → ReLU → 64 → ReLU → 10`
> 打印每层后的 shape。确保所有维度匹配。

---

> **下一步**：网络有输出，但输出好不好？进入[第七章：损失函数](./07_loss.md)。
