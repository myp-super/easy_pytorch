# 第二章：自动求导 autograd — 神经网络学习的引擎

## 2.0 本章导引

### 为什么这一章决定了你能走多远

第一章你学会了操作数据（Tensor）。但神经网络不只是"操作数据"——它要**从数据中学习**。

"学习"的数学本质是什么？

```
学习 = 自动调整参数，使得"预测值"不断接近"真实值"
    = 找到让"误差"最小的那组参数
    = 一个优化问题

而解决优化问题的核心工具就是：梯度
```

如果你不理解梯度是怎么算出来的、怎么流动的，那么：
- `loss.backward()` 对你来说就是一句咒语
- `optimizer.step()` 对你来说就是另一个咒语
- 你永远在"模仿代码"，而不是"理解代码"

本章的目标：**让你从"咒语使用者"变成"理解咒语原理的人"。**

### 本章地图

```
2.1 为什么需要梯度          ← 从调空调温度讲起
2.2 计算图                  ← PyTorch 后台的秘密
2.3 requires_grad            ← 告诉 PyTorch"我要追踪"
2.4 .backward()             ← 触发反向传播
2.5 .grad                   ← 查看梯度值
2.6 梯度累积（重要陷阱！）    ← 为什么需要 zero_grad()
2.7 完整示例：用 autograd 求极值
2.8 no_grad() 和 detach()
2.9 练习
```

---

## 2.1 为什么需要梯度

### 2.1.1 从"调空调"开始理解

你走进一个陌生房间，觉得**太热**。你不知道最佳温度是多少，但你知道：
- 如果当前太热 → 把温度调低一点
- 如果当前太冷 → 把温度调高一点
- 每次调多少？→ 感觉越热就调越多

把这个过程写成伪代码：

```
当前温度 = whatever
目标舒适度 = 未知（你不知道具体数值）
当前不舒适 = |当前温度带来的体感 - 舒适体感|

while 不舒适:
    如果 太热:
        调低温度（幅度正比于"有多热"）
    如果 太冷:
        调高温度（幅度正比于"有多冷"）
    重新感受不舒适程度
```

**这个循环就是梯度下降的直觉版本！**

翻译成数学语言：

```
梯度 = "往哪个方向走，不舒适度会增加"
所以我们要往梯度的反方向走
每一步的大小 = 学习率 × 梯度的大小
```

### 2.1.2 用一个简单函数来看"梯度"

考虑 `y = x²`。在 x=2 这个点：

```
y = 2² = 4

在 x=2 附近：
  如果 x 变成 2.1 → y = 4.41  (y 变大了)
  如果 x 变成 1.9 → y = 3.61  (y 变小了)

所以在 x=2 处，梯度 = 2x = 4（正数）
→ 增大 x 会让 y 增大
→ 如果要最小化 y，应该减小 x
```

```python
# %%
import torch
import matplotlib.pyplot as plt

# 可视化 y = x²
# (在你的环境中运行以下代码看图像)
x = torch.linspace(-4, 4, 100)
y = x ** 2

# plt.plot(x.numpy(), y.numpy())
# plt.xlabel('x')
# plt.ylabel('y = x²')
# plt.title('y = x²')
# plt.axvline(x=2, color='r', linestyle='--', label='x=2')
# plt.legend()
# plt.grid(True)
# plt.show()
```

### 2.1.3 梯度 vs 导数

```
导数（Derivative）：单变量函数的"变化率"
    例：f(x) = x² → f'(x) = 2x

梯度（Gradient）：多变量函数的"变化率向量"
    例：f(x, y) = x² + y² → ∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]

梯度 = 导数的多维推广
```

在深度学习中，我们几乎总在处理多变量函数（网络有成千上万个参数），所以一直说"梯度"。

### 2.1.4 为什么神经网络需要梯度

```
神经网络 = 一个巨大的函数 f(输入; 参数)
    参数 = 所有权重 W 和偏置 b（成千上万个）

损失函数 L = 衡量 f(输入; 参数) 和"正确答案"的差距

目标：找到让 L 最小的那组参数

方法：
    1. 计算 ∇L（L 对每个参数的梯度）
    2. 参数 ← 参数 - 学习率 × ∇L
    3. 重复，直到 L 足够小
```

**如果没有自动求导**，你需要手算成千上万个导数。这不仅容易出错，而且当网络结构改变时，你需要重新手算。

**有了自动求导**，PyTorch 帮你完成这一切。你只需要调用 `.backward()`。

---

## 2.2 计算图 —— PyTorch 后台的秘密

### 2.2.1 什么是计算图

当你在 PyTorch 中做运算时，PyTorch 悄悄在后台画了一张图。这张图记录了：**每个结果是由哪些数据、经过哪些运算得出的。**

看一个最简单的例子：

```python
# %%
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
b = torch.tensor(1.0)

y = w * x + b    # y = 2*3 + 1 = 7
```

PyTorch 在后台记录的图：

```
    w(2.0) ──┐
              ├──→ [×] ──→ (6.0) ──┐
    x(3.0) ──┘                      ├──→ [+] ──→ y(7.0)
                                     │
    b(1.0) ─────────────────────────┘


    图例：
    w(2.0)  = 值为 2.0 的叶子节点（requires_grad=True）
    [×]     = 乘法运算节点
    [+]     = 加法运算节点
    y(7.0)  = 输出节点
```

### 2.2.2 前向传播 vs 反向传播

```
前向传播（Forward）：
    输入 → 一层一层计算 → 输出
    w, x, b  →  w×x  →  w×x + b  →  y

反向传播（Backward / Backpropagation）：
    输出 → 一层一层回传梯度 → 输入
    y  →  ∂y/∂(w×x)  →  ∂y/∂w, ∂y/∂x, ∂y/∂b
```

**在图上直观理解：**

```
前向（黑色箭头沿计算方向）：
    w ──→ [×] ──→ [+] ──→ y
    x ──→  ↑              ↑
    b ────────────────────→

反向（红色箭头沿反向传梯度）：
    w ←── [×] ←── [+] ←── y
    x ←──  ↑              ↑
    b ←────────────────────

    梯度从 y 出发，沿着计算图回流到每个叶子节点
```

### 2.2.3 手算一遍：感受"自动求导"替我们做了什么

以 `loss = (w*x + b - target)²` 为例（w=2, x=3, b=1, target=8）：

```
前向传播：
    u = w × x     = 2 × 3 = 6
    v = u + b     = 6 + 1 = 7
    loss = (v - 8)² = (-1)² = 1

反向传播（链式法则）：
    ∂loss/∂v     = 2(v - 8) = 2(-1) = -2
    ∂loss/∂u     = ∂loss/∂v × ∂v/∂u = -2 × 1 = -2
    ∂loss/∂b     = ∂loss/∂v × ∂v/∂b = -2 × 1 = -2
    ∂loss/∂w     = ∂loss/∂u × ∂u/∂w = -2 × 3 = -6
    ∂loss/∂x     = ∂loss/∂u × ∂u/∂x = -2 × 2 = -4

如果只优化 w 和 b（固定 x）：
    w.grad = -6  → 增大 w 会减小 loss（因为梯度是负的）
    b.grad = -2  → 增大 b 会减小 loss
```

**PyTorch 用一行代码就完成了以上所有计算：**

```python
# %%
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
b = torch.tensor(1.0, requires_grad=True)
target = torch.tensor(8.0)

loss = (w * x + b - target) ** 2
loss.backward()

print(f"w.grad = {w.grad}")    # -6.0  ← 和我们手算的一样！
print(f"b.grad = {b.grad}")    # -2.0  ← 和我们手算的一样！
```

### 2.2.4 叶子节点 vs 中间节点

```python
# %%
w = torch.tensor(2.0, requires_grad=True)    # ← 叶子节点
x = torch.tensor(3.0)                         # ← 叶子节点
b = torch.tensor(1.0, requires_grad=True)    # ← 叶子节点

u = w * x        # ← 中间节点（由运算产生）
v = u + b        # ← 中间节点
loss = (v - 8) ** 2  # ← 中间节点

print(f"w.is_leaf:    {w.is_leaf}")      # True
print(f"u.is_leaf:    {u.is_leaf}")      # False
print(f"loss.is_leaf: {loss.is_leaf}")   # False
```

**关键理解**：
- **叶子节点** = 你自己创建的 Tensor。只有它们的 `.grad` 会被保留。
- **中间节点** = 由运算产生的 Tensor。它们的梯度在反向传播后被释放（以节省内存）。

```
为什么中间节点的梯度被释放？

因为优化器只需要更新参数（叶子节点）。
保留中间节点的梯度纯属浪费内存（对一个百万参数的模型来说这是巨大的量）。
```

---

## 2.3 requires_grad —— 告诉 PyTorch 你想追踪谁

### 2.3.1 设置 requires_grad

```python
# %%
# 方式 1：创建时指定
w = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"w.requires_grad: {w.requires_grad}")  # True

# 方式 2：创建后修改（_ 后缀 = 原地操作）
x = torch.tensor([1.0, 2.0])
print(f"之前: {x.requires_grad}")             # False
x.requires_grad_(True)
print(f"之后: {x.requires_grad}")             # True
```

### 2.3.2 什么该设为 True，什么该设为 False

```
┌──────────────────────┬──────────────────┬──────────────────┐
│ 设 True               │ 设 False          │ 原因              │
├──────────────────────┼──────────────────┼──────────────────┤
│ 网络参数（W, b）       │ 输入数据          │ 只需优化参数       │
│ 需要优化的变量         │ 固定不变的量       │ 不需要更新数据     │
│                       │ 标签（label）     │ 标签是"正确答案"   │
│                       │ 推理时的所有 Tensor│ 推理不更新参数     │
└──────────────────────┴──────────────────┴──────────────────┘
```

### 2.3.3 只有 requires_grad=True 的叶子节点才有 .grad

```python
# %%
a = torch.tensor(2.0, requires_grad=True)   # 叶子 + 需要梯度
b = torch.tensor(3.0)                        # 叶子 + 不需要梯度
c = a * b                                     # 中间节点

loss = c ** 2
loss.backward()

print(f"a.grad: {a.grad}")    # 12.0  ✅
print(f"b.grad: {b.grad}")    # None   ❌ requires_grad=False
print(f"c.grad: {c.grad}")    # None   ❌ 不是叶子节点
```

---

## 2.4 .backward() —— 触发反向传播

### 2.4.1 从一个标量开始

```python
# %%
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2       # y = 4

y.backward()     # 计算 dy/dx = 2x |_{x=2} = 4

print(f"x.grad: {x.grad}")  # 4.0
# 验证：导数就是 2x，在 x=2 处就是 4
```

### 2.4.2 计算图只能用一次（重要！）

```python
# %%
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(f"第一次 backward: x.grad = {x.grad}")  # 4.0

# y.backward()  # ❌ RuntimeError!
# "Trying to backward through the graph a second time"
# 翻译：计算图已经释放了，不能第二次反向传播
```

**为什么计算图只能用一次？**

```
原因：节省内存。反向传播结束后，中间结果（计算图）就没用了。
PyTorch 默认立即释放它们。

解决方案：
    如果你确实需要多次 backward（极少见），用 retain_graph=True：
    
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)   # 保留计算图
y.backward()                     # 可以再跑了
print(f"x.grad: {x.grad}")       # 8.0（梯度累加了 ← 2.5节细讲）
```

### 2.4.3 backward() 只能对标量调用

```python
# %%
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2    # [1, 4, 9] — 这是一个向量

# y.backward()  # ❌ RuntimeError: grad can be implicitly created
#                  only for scalar outputs

# 原因：你需要一个标量作为"总损失"，才能计算梯度
# 解决：把向量变成标量
y_scalar = y.sum()    # [1, 4, 9] → 14
y_scalar.backward()
print(f"x.grad: {x.grad}")  # [2, 4, 6] = [2*1, 2*2, 2*3]

# 或者取均值
# y.mean().backward()
```

**为什么必须是标量？**

```
梯度回答的问题是：
"如果我微调 x，这个损失（一个数字）会怎么变化？"

你不可能同时优化三个不同的目标（除非你把它们加权求和）。
```

---

## 2.5 .grad —— 梯度的家

### 2.5.1 梯度存储在哪里

梯度存储在叶子节点的 `.grad` 属性中。在调用 `.backward()` 之前，`.grad` 是 `None`。调用之后，梯度被写入。

```python
# %%
w = torch.tensor(3.0, requires_grad=True)
print(f"backward 前: w.grad = {w.grad}")  # None

loss = (w - 1) ** 2    # (3-1)² = 4
loss.backward()
print(f"backward 后: w.grad = {w.grad}")  # 4.0 (= 2*(w-1)|_{w=3})
```

### 2.5.2 梯度会累积！！！（最重要的一节）

**这是整个教程中最重要的陷阱之一。请仔细读。**

```python
# %%
w = torch.tensor(3.0, requires_grad=True)

for i in range(3):
    loss = (w - 1) ** 2
    loss.backward()
    print(f"第 {i+1} 次 backward 后: w.grad = {w.grad}")
```

**输出：**

```
第 1 次 backward 后: w.grad = 4.0
第 2 次 backward 后: w.grad = 8.0    ← 不是 4.0！
第 3 次 backward 后: w.grad = 12.0   ← 不是 4.0！
```

**发生了什么？**

```
每次 backward() 计算的梯度会加（accumulate）到 .grad 上，而不是覆盖。

第 1 次：.grad 从 None → 4.0
第 2 次：.grad 从 4.0 → 4.0 + 4.0 = 8.0
第 3 次：.grad 从 8.0 → 8.0 + 4.0 = 12.0
```

**用图理解：**

```
backward() 之前:
    .grad = None

第 1 次 backward():
    .grad += 4.0  → .grad = 4.0

第 2 次 backward():
    .grad += 4.0  → .grad = 8.0    ← 叠加了！

第 3 次 backward():
    .grad += 4.0  → .grad = 12.0   ← 继续叠加！
```

### 2.5.3 为什么要这样设计？

PyTorch 故意让梯度累积，是为了支持一个高级技巧：**梯度累积**。

```
当你显存不够，放不下 batch_size=128 时：
    → 分 4 次，每次 batch_size=32
    → 4 次的梯度自动累加
    → 一次性更新参数
    → 等效于 batch_size=128 的训练效果
```

但对于普通训练，梯度累积是 bug 的来源。解决办法：

### 2.5.4 清零梯度

```python
# %%
w = torch.tensor(3.0, requires_grad=True)

for i in range(3):
    loss = (w - 1) ** 2
    loss.backward()
    print(f"backward 后: w.grad = {w.grad}")
    
    # 清零梯度
    w.grad.zero_()   # 或者 w.grad = None
```

**两种清零方式：**

```python
# 方式 1：原地清零
param.grad.zero_()

# 方式 2：设为 None（推荐，PyTorch 内部处理更好）
param.grad = None

# 在训练中，你通常用：
optimizer.zero_grad()   # 它会清零所有参数（第八章详讲）
```

---

## 2.6 完整示例：用 autograd 找到函数的最小值

现在我们把所有知识串起来，做一个完整的梯度下降。

**问题**：找到使 `y = x²` 最小的 x。（答案是 0，但我们假装不知道）

```python
# %%
import torch

# 1. 初始化：随便猜一个 x
x = torch.tensor(4.0, requires_grad=True)
learning_rate = 0.1

print(f"初始: x = {x.item():.4f}")
print("-" * 40)

for step in range(20):
    # 2. 前向：计算当前的 y 值
    y = x ** 2
    
    # 3. 反向：计算 dy/dx
    y.backward()
    
    # 4. 更新：x = x - lr * grad
    #    注意：更新必须用 no_grad()，否则更新操作也会被追踪
    with torch.no_grad():
        x -= learning_rate * x.grad
    
    # 5. 清零梯度（准备下一步）
    x.grad.zero_()
    
    if step % 5 == 0:
        print(f"Step {step:2d}: x = {x.item():.4f}, y = {y.item():.4f}")

print("-" * 40)
print(f"最终: x ≈ {x.item():.6f} (期望 0.0)")
```

**期望输出：**

```
初始: x = 4.0000
----------------------------------------
Step  0: x = 3.2000, y = 16.0000
Step  5: x = 0.8389, y = 1.0486
Step 10: x = 0.2200, y = 0.0671
Step 15: x = 0.0576, y = 0.0046
----------------------------------------
最终: x ≈ 0.015104 (期望 0.0)
```

**逐步解读这个"训练循环"：**

```
Step 0: x=4.0  → y = 16.0 → dy/dx = 8.0  → x -= 0.1×8.0 = 3.2
Step 1: x=3.2  → y = 10.24 → dy/dx = 6.4 → x -= 0.1×6.4 = 2.56
Step 2: x=2.56 → y = 6.55  → dy/dx = 5.12 → x -= 0.1×5.12 = 2.048
...
x 不断向 0（最小值）靠近。
每一步的幅度在减小（因为越靠近最小值，梯度越小）。
```

**这 20 行代码包含了神经网络训练的全部精髓！**

```
    y = x ** 2            ← 相当于 model(x)
    y.backward()          ← 相当于 loss.backward()
    x -= lr * x.grad      ← 相当于 optimizer.step()
    x.grad.zero_()        ← 相当于 optimizer.zero_grad()
```

---

## 2.7 with torch.no_grad() —— 关掉梯度追踪

### 2.7.1 为什么需要它

当你更新参数时，你不想把"更新操作"也记录到计算图中。

```python
# %%
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()

# ❌ 错误：直接更新会将"更新操作"加入计算图
# x = x - 0.1 * x.grad  # 这会创建一个新节点

# ✅ 正确：在 no_grad() 中更新
with torch.no_grad():
    x -= 0.1 * x.grad

print(f"更新后 x: {x.item()}")

# no_grad() 做的事：
# "在这个上下文里，所有操作都不要被追踪。不要建计算图。"
```

### 2.7.2 其他使用场景

```python
# %%
# 场景 1：推理/评估时（第十二章详讲）
model.eval()
with torch.no_grad():
    predictions = model(test_data)  # 不追踪梯度，省内存

# 场景 2：计算评估指标（准确率等，不需要梯度）
with torch.no_grad():
    accuracy = (predictions == labels).float().mean()

# 场景 3：任何你不需要梯度的计算
with torch.no_grad():
    # 做任何操作，都不会构建计算图
    pass
```

### 2.7.3 no_grad vs detach

```python
# %%
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

# detach()：创建一个"值相同但不追踪梯度"的 Tensor 副本
y_detached = y.detach()
print(f"y.requires_grad:          {y.requires_grad}")           # True
print(f"y_detached.requires_grad: {y_detached.requires_grad}")  # False
print(f"值相同: {y == y_detached}")                              # True

# no_grad vs detach：
# - no_grad：一个上下文管理器，"在这个范围内，所有操作都不追踪"
# - detach：对单个 Tensor，"创建这个 Tensor 的无梯度版本"
```

---

## 2.8 本章总结：自动求导心智模型

```
                    ┌──────────────────────┐
                    │  requires_grad=True   │
                    │  "我要追踪这个 Tensor"  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │       计算图          │
                    │  PyTorch 后台自动构建   │
                    │  记录所有运算关系       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │     .backward()       │
                    │  "沿着计算图反向传播"   │
                    │  链式法则自动计算梯度    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │       .grad           │
                    │  叶子节点的梯度存储处   │
                    │   ⚠️ 梯度会累积！      │
                    │   需要 .zero_() 清零   │
                    └──────────────────────┘
```

**五个核心铁律：**

| 序号 | 铁律 | 说明 |
|------|------|------|
| 1 | `requires_grad=True` → 追踪 | 网络参数设 True，数据设 False |
| 2 | `.backward()` → 触发 | 必须在标量上调用 |
| 3 | `.grad` → 查看 | 存在叶子节点上，backward 前为 None |
| 4 | **梯度会累积** | 每次 backward 前必须清零 |
| 5 | `no_grad()` → 关掉 | 推理、评估、参数更新时使用 |

---

## 2.9 本章练习

### 练习 2-1：手动画计算图

在纸上画出 `loss = (w*x + b - target)²` 的计算图，其中 `w=1, x=2, b=0.5, target=3`。

标注每个节点的值和反向传播的梯度。

### 练习 2-2：PyTorch 验证

用代码验证你手算的梯度。

### 练习 2-3：梯度累积实验

```python
# 不运行代码，先预测输出：
x = torch.tensor(1.0, requires_grad=True)
for i in range(5):
    y = x ** 3    # dy/dx = 3x²|_{x=1} = 3
    y.backward()
    print(f"Iter {i}: x.grad = {x.grad}")

# 然后运行验证。你的预测对吗？
```

### 练习 2-4：retain_graph 实验

```python
# 不运行代码，先预测 x.grad 的最终值：
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2       # dy/dx = 4
y.backward(retain_graph=True)
y.backward()
print(x.grad)    # ?
```

### 练习 2-5：用 autograd 求多变量函数的最小值

```python
# 找到使 f(x, y) = x² + y² 最小的 (x, y)
# 从 (x=3, y=4) 开始，lr=0.1，迭代 30 步
# 提示：需要两个 requires_grad=True 的变量
```

### 练习 2-6：no_grad 实验

```python
# 预测以下两段代码的输出有什么区别：
# 
# 代码 A：
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
x = x - 0.1  # 没有 no_grad
# y.backward()  # 取消注释会怎样？

# 代码 B：
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
with torch.no_grad():
    x = x - 0.1  # 有 no_grad
# y.backward()  # 取消注释会怎样？
```

### 练习 2-7：不看答案 —— 完整梯度下降

> **核心检验**。关闭所有文档和代码，独立完成：

```python
# 用 PyTorch autograd 找到 f(x) = (x - 5)⁴ 的最小值
# 从 x = 10 开始
# lr = 0.01
# 迭代 50 步
# 每 10 步打印 x 的值
# 确保最终 x ≈ 5.0
```

---

## 答案与提示

> **请先独立完成练习再看答案。**

<details>
<summary>练习 2-3 答案</summary>

```
每次 dy/dx = 3x²|_{x=1} = 3，梯度会累加：
Iter 0: 3
Iter 1: 6   (3+3)
Iter 2: 9   (6+3)
Iter 3: 12  (9+3)
Iter 4: 15  (12+3)

修复：在每次 backward 前加 x.grad = None
```
</details>

<details>
<summary>练习 2-4 答案</summary>

```
x.grad = 8.0

第 1 次 backward: dy/dx = 4，累加 → grad = 4
第 2 次 backward: dy/dx = 4，累加 → grad = 8

（retain_graph=True 让计算图在第一次 backward 后没有被释放）
```
</details>

<details>
<summary>练习 2-5 提示</summary>

```python
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)
lr = 0.1

for step in range(30):
    f = x**2 + y**2
    f.backward()
    
    with torch.no_grad():
        x -= lr * x.grad
        y -= lr * y.grad
    
    x.grad.zero_()
    y.grad.zero_()

# 最终 x ≈ 0, y ≈ 0（抛物线的最低点）
```
</details>

<details>
<summary>练习 2-7 参考答案</summary>

```python
import torch

x = torch.tensor(10.0, requires_grad=True)
lr = 0.01

for step in range(50):
    f = (x - 5) ** 4   # f(x) = (x-5)^4
    f.backward()
    
    with torch.no_grad():
        x -= lr * x.grad
    
    x.grad.zero_()
    
    if step % 10 == 0:
        print(f"Step {step:2d}: x = {x.item():.4f}")

print(f"最终 x = {x.item():.4f} (期望 5.0)")
```
</details>

---

> **下一步**：现在你理解了梯度，但要写神经网络还需要理解 Python 类。进入[第三章：Python 面向对象基础](./03_oop_basics.md)。
