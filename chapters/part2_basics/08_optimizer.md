# 第八章：优化器 — 让网络"学习"的核心

## 8.0 本章导引

第七章你算出了梯度（`.grad`）。但**梯度本身不改变参数**。

你需要用梯度来**更新参数**——这就是优化器（Optimizer）的工作。

```
梯度     = "往哪个方向走，loss 会增大"（方向 + 大小）
优化器   = "用这个信息，更新参数，让 loss 减小"（行动）

梯度是地图，优化器是腿。
```

---

## 8.1 从手工更新到 Optimizer

### 8.1.1 手工实现——只用 autograd

```python
# %%
import torch

# 问题：找到使 y = x² 最小的 x
x = torch.tensor(4.0, requires_grad=True)
learning_rate = 0.1

for step in range(20):
    # 前向
    y = x ** 2
    # 反向
    y.backward()
    # 更新：x = x - lr * grad
    with torch.no_grad():
        x -= learning_rate * x.grad
    # 清零
    x.grad.zero_()
    
    if step % 5 == 0:
        print(f"Step {step:2d}: x = {x.item():.4f}")

print(f"最终 x ≈ {x.item():.6f}")
# x 应该接近 0（y=x² 的最小值）
```

**这 8 行代码就是整个训练的本质。** 但如果你有 1 万个参数，`x -= lr * x.grad` 要写 1 万遍。Optimizer 替你做了这件事。

### 8.1.2 什么是 Optimizer

```
Optimizer = 参数更新管理器

它回答两个问题：
    1. 更新哪些参数？（model.parameters()）
    2. 怎么更新？（SGD 的公式？Adam 的公式？）

给你两个关键方法：
    zero_grad()  → 把所有参数的 .grad 清零
    step()       → 用梯度更新所有参数
```

---

## 8.2 SGD —— 最经典的优化器

### 8.2.1 SGD 的公式

```
param = param - lr × param.grad

非常简单：沿梯度的反方向走一小步。
步长 = 学习率 × 梯度大小
```

### 8.2.2 用 SGD 改写上面的例子

```python
# %%
import torch
import torch.nn as nn

# 用一个 Linear 层来模拟"求 y=x² 最小值"的问题
model = nn.Linear(1, 1)   # 输入 1 → 输出 1（虽然这不是合理用法，但能演示）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.tensor([[2.0]])   # 输入 = 2
target = torch.tensor([[0.0]])  # 期望输出 = 0

for step in range(20):
    optimizer.zero_grad()        # 清零
    output = model(x)            # 前向
    loss = (output - target) ** 2  # 损失
    loss.backward()              # 反向
    optimizer.step()             # 更新（替代了手动 for param in params: param -= lr*grad）
    
    if step % 5 == 0:
        print(f"Step {step:2d}: loss = {loss.item():.6f}")
```

### 8.2.3 optimizer.zero_grad() —— 为什么必须在每步开始时调用

回顾第二章的铁律：**梯度会累积！**

```python
# %%
# 演示：忘记 zero_grad 的后果
x = torch.tensor(3.0, requires_grad=True)

for i in range(5):
    y = x ** 2
    y.backward()
    # 忘记 zero_grad！
    print(f"Iter {i}: grad = {x.grad}")  # 3, 6, 9, 12, 15  ← 梯度在叠加！
```

```
每次 backward 把新梯度加到旧梯度上 → 参数更新越来越离谱。

optimizer.zero_grad() 做的事情：
    for param in model.parameters():
        param.grad = None   # 清空所有参数的梯度
```

### 8.2.4 SGD 的 momentum —— 带上"惯性"

```python
# %%
# momentum=0.9: 不只考虑当前梯度，还考虑之前的更新方向
# 就像球下山——有惯性，不会在每个小凹坑里停下来
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

```
没有 momentum：每一步只看当前梯度 → 容易卡在"小坑"里
有 momentum：之前的更新方向会"带着"当前步 → 更容易跳过小坑
```

---

## 8.3 Adam —— 最常用的优化器

### 8.3.1 Adam 好在哪

**SGD**：所有参数共享同一个学习率。如果某个参数已经接近最优值，你希望它步子小一点；如果某个参数离最优值还远，你希望它步子大一点。SGD 做不到。

**Adam**：每个参数有自己的**自适应学习率**。它会根据历史梯度自动调整每个参数的步长——梯度大的参数步子会变小，梯度小的参数步子会变大。

```
类比：
    SGD = 所有人穿同一双鞋走山路 → 有人鞋大有人鞋小
    Adam = 每个人穿自己的鞋 → 自动适配
```

### 8.3.2 使用 Adam

```python
# %%
model = nn.Linear(10, 2)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,                    # Adam 的经典默认值
    betas=(0.9, 0.999),          # 几乎永远不用改
    eps=1e-8                     # 几乎永远不用改
)

# 使用方式和 SGD 完全一样：
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
```

### 8.3.3 Adam vs SGD

```
┌────────────┬───────────────────┬──────────────────────┐
│            │ Adam              │ SGD                  │
├────────────┼───────────────────┼──────────────────────┤
│ 收敛速度    │ 快（自适应学习率）  │ 慢                   │
│ 对 lr 敏感度│ 低（0.001 基本通用)│ 高（需要仔细调）       │
│ 泛化能力    │ 有时不如 SGD       │ 通常更好              │
│ 适用场景    │ 大多数任务的默认选择 │ 需要极致性能时         │
│ 初学推荐    │ ✅ 用这个！        │ 了解原理后再用         │
└────────────┴───────────────────┴──────────────────────┘

初学者的策略：永远从 Adam(lr=0.001) 开始。
如果效果不好，再考虑调 lr 或换 SGD。
```

---

## 8.4 学习率 —— 最重要的超参数

### 8.4.1 学习率影响的可视化

```
lr 太大（如 1.0）：
    x 的轨迹：4.0 → -4.0 → 4.0 → -4.0...（震荡，不收敛）
    类比：下山步子太大，一步跨到对面山上，然后又跨回来

lr 太小（如 0.00001）：
    x 的轨迹：4.0 → 3.9999 → 3.9998...（收敛太慢）
    类比：每步走 1 毫米，到天黑也走不到

lr 合适（如 0.1）：
    x 的轨迹：4.0 → 3.2 → 2.56 → 2.048 → ... → 0
    类比：步子合适，稳步下降
```

```python
# %%
# 不同学习率的效果对比
import torch

for lr in [0.001, 0.01, 0.1, 1.0, 2.0]:
    x = torch.tensor(4.0, requires_grad=True)
    for step in range(20):
        y = x ** 2
        y.backward()
        with torch.no_grad():
            x -= lr * x.grad
        x.grad.zero_()
    print(f"lr={lr:.3f}: x最终 = {x.item():+.4f}")
```

### 8.4.2 选择建议

```
┌──────────┬──────────────┐
│ 优化器    │ 推荐起始 lr   │
├──────────┼──────────────┤
│ Adam     │ 0.001        │
│ SGD      │ 0.01 或 0.1  │
└──────────┴──────────────┘

调试策略：
    loss 震荡/爆炸  → lr 除以 10
    loss 下降太慢   → lr 乘以 3
    训练到一半      → lr 再除以 10（学习率衰减）
```

---

## 8.5 训练步骤的"六字真言"

现在你有了所有的零件。每个训练步骤 = 6 个操作：

```
┌─────────────────────────────────────────────────┐
│          每个 batch 的训练步骤 = 6 步            │
│                                                 │
│  ① optimizer.zero_grad()   ← 清空旧梯度         │
│  ② output = model(x)       ← 前向传播           │
│  ③ loss = criterion(o, y)  ← 计算损失           │
│  ④ loss.backward()         ← 反向传播           │
│  ⑤ optimizer.step()        ← 更新参数           │
│                                                 │
│  口诀：zero → forward → loss → backward → step  │
└─────────────────────────────────────────────────┘
```

```python
# 完整示例
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()                    # ①
        outputs = model(x_batch)                 # ②
        loss = criterion(outputs, y_batch)       # ③
        loss.backward()                          # ④
        optimizer.step()                         # ⑤
```

---

## 8.6 本章总结

```
    优化器 = 自动参数更新器

    SGD:  简单、需要调 lr、配合 momentum 效果好
    Adam: 自适应 lr、收敛快、初学首选

    两个关键操作：
        zero_grad()  → 清空前一次迭代的梯度（必须在 backward 前调用）
        step()       → 用梯度更新参数（在 backward 之后调用）

    学习率：
        最重要的超参数
        Adam 从 0.001 开始
        太大 → 震荡；太小 → 太慢
```

---

## 8.7 本章练习

### 练习 8-1：手动梯度下降

```python
# 用 autograd + 手动更新，最小化 f(x,y) = x² + y²
# 从 (x=3, y=4)，lr=0.1，迭代 20 步
```

### 练习 8-2：不同学习率对比

```python
# 优化 y=x²，从 x=4 开始
# 分别用 lr=[0.001, 0.01, 0.1, 1.0, 2.0]
# 记录每个 lr 的最终 x 值
```

### 练习 8-3：zero_grad bug 复现

```python
# 用 SGD 优化器，故意不调用 zero_grad
# 观察 loss 的变化模式（和正常训练的对比）
```

### 练习 8-4：用 SGD 做线性回归

```python
# 数据：X=[1,2,3,4,5], y=[3,5,7,9,11]  (y=2x+1)
# 模型：nn.Linear(1,1)
# 优化器：SGD(lr=0.01)
# 训练 100 轮，每 20 轮打印 loss 和参数
```

### 练习 8-5：SGD vs Adam 对比

```python
# 同样的任务，分别用 SGD 和 Adam
# 观察收敛速度的差异
```

### 练习 8-6：不看答案——独立完成

> 关闭所有文档，独立完成：
> 1. 模型：Linear(1,1)
> 2. 优化器：Adam(lr=0.01)
> 3. 数据：X=[[1],[2],[3],[4]], y=[[2],[4],[6],[8]]  (y=2x)
> 4. 完整训练循环 20 轮（zero_grad → forward → MSELoss → backward → step）
> 5. 每 5 轮打印 loss
> 6. 训练后验证 W≈2, b≈0

---

## 答案与提示

<details>
<summary>练习 8-4 参考答案</summary>

```python
import torch
import torch.nn as nn

X = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
y = torch.tensor([[3.], [5.], [7.], [9.], [11.]])

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        w, b = model.weight.item(), model.bias.item()
        print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, W={w:.3f}, b={b:.3f}")
```
</details>

---

> **下一步**：六步循环记住了吗？进入[第九章：训练循环](09_train_loop.md)——把一切正式组装。
