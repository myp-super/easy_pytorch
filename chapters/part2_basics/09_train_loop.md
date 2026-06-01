# 第九章：训练循环 — 把一切组装起来

## 9.0 本章导引

前八章你学了所有零件：

```
第一章：Tensor → 数据怎么存
第二章：autograd → 梯度怎么算
第三章：OOP → 为什么网络是类
第四章：nn.Module → 网络的骨架
第五章：nn.Linear → 数据怎么流
第六章：激活函数 → 非线性在哪
第七章：Loss → 预测好不好
第八章：Optimizer → 参数怎么更新
```

这一章**没有新 API**。但这是最重要的一章——你把八个零件组装成一台能运转的机器。

---

## 9.1 训练的本质 —— 一句话概括

```
训练 = 反复执行：前向 → 算损失 → 反向 → 更新参数

for epoch in range(N):
    for batch_x, batch_y in data:
        1. optimizer.zero_grad()
        2. output = model(batch_x)
        3. loss = criterion(output, batch_y)
        4. loss.backward()
        5. optimizer.step()
```

### 9.1.1 几个关键术语

```
Epoch（轮次） = 把全部训练数据都看了一遍
    5 epochs = 每张图片被网络看了 5 遍

Batch（批次） = 一次前向+反向传播用了多少样本
    batch_size=64 → 每次取 64 张图一起处理

Iteration（迭代）= 一个 batch 的完整六步循环
    60,000 样本 ÷ batch_size 64 ≈ 938 iterations/epoch
```

---

## 9.2 第一个训练循环 —— 极简版本

让我们用最简单的数据：`y = 3x + 2`（带一点噪声）。

```python
# %%
import torch
import torch.nn as nn

# === 1. 准备数据 ===
# 真实关系：y = 3x + 2，加一点噪声模拟现实
X = torch.tensor([
    [1.0], [2.0], [3.0], [4.0], [5.0],
    [6.0], [7.0], [8.0], [9.0], [10.0]
])
y = torch.tensor([
    [5.1], [7.8], [11.2], [14.0], [16.9],
    [20.1], [23.0], [26.2], [29.1], [32.0]
])

# === 2. 创建模型 ===
model = nn.Linear(1, 1)    # 1 输入 → 1 输出（线性回归）
criterion = nn.MSELoss()   # 均方误差（回归任务）
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# === 3. 训练 ===
num_epochs = 100

for epoch in range(num_epochs):
    # 六步循环
    optimizer.zero_grad()          # ① 清零
    predictions = model(X)         # ② 前向（所有 10 个样本一起）
    loss = criterion(predictions, y)  # ③ 损失
    loss.backward()                # ④ 反向
    optimizer.step()               # ⑤ 更新
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: loss = {loss.item():.4f}")

print(f"\n训练完成！")
print(f"W = {model.weight.item():.3f} (期望 ≈ 3.0)")
print(f"b = {model.bias.item():.3f} (期望 ≈ 2.0)")
```

**逐行对照——每个操作来自哪一章：**

```
optimizer.zero_grad()      ← 第八章（清空梯度）
predictions = model(X)     ← 第四章（model(x) 触发 forward）
                           ← 第五章（nn.Linear 做 y=Wx+b）
loss = criterion(...)      ← 第七章（MSELoss 计算误差）
loss.backward()            ← 第二章（反向传播计算梯度）
optimizer.step()           ← 第八章（用梯度更新参数）
```

---

## 9.3 训练循环的扩展

### 9.3.1 记录 loss 历史

```python
# %%
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

loss_history = []   # ← 记录每轮的 loss

for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())

# 验证 loss 是否在下降
print(f"第 1 轮 loss:    {loss_history[0]:.4f}")
print(f"第 50 轮 loss:   {loss_history[49]:.4f}")
print(f"第 100 轮 loss:  {loss_history[99]:.4f}")
print(f"loss 下降了吗？   {loss_history[0] > loss_history[-1]}")  # 应该是 True
```

### 9.3.2 加上验证环节

在真实项目中，你需要知道模型在**没见过的数据**上表现如何：

```python
# %%
# 验证数据（训练时没见过的）
X_val = torch.tensor([[1.5], [3.5], [5.5]])
y_val = torch.tensor([[6.5], [12.5], [18.5]])  # y ≈ 3x+2

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    # === 训练阶段 ===
    model.train()              # 切换到训练模式
    optimizer.zero_grad()
    train_pred = model(X)
    train_loss = criterion(train_pred, y)
    train_loss.backward()
    optimizer.step()
    
    # === 验证阶段（每 20 轮）===
    if epoch % 20 == 0:
        model.eval()           # 切换到评估模式
        with torch.no_grad():  # 验证不需要梯度
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
        print(f"Epoch {epoch:3d}: train_loss={train_loss.item():.4f}, val_loss={val_loss.item():.4f}")
```

### 9.3.3 train() vs eval() —— 为什么重要

```
model.train()  → training = True
    - Dropout: 随机丢弃神经元（正则化）
    - BatchNorm: 用当前 batch 的统计量

model.eval()   → training = False
    - Dropout: 不丢弃（所有神经元都参与）
    - BatchNorm: 用全局统计量

虽然你现在还没学到 Dropout/BatchNorm，但切换的习惯要从现在养成。
```

### 9.3.4 验证时必须用 no_grad()

```
❌ 不用的后果：
    model.eval()
    output = model(X_val)   # 还是会构建计算图！
                             # 虽然因为 eval() 不会更新参数
                             # 但白白浪费了显存和计算时间

✅ 正确：
    model.eval()
    with torch.no_grad():
        output = model(X_val)   # 不构建计算图，省内存，快
```

---

## 9.4 过拟合——训练太多反而不好的直觉

```python
# %%
import torch
import torch.nn as nn

# 只有 5 个训练样本——故意创造过拟合条件
torch.manual_seed(42)
X_train = torch.randn(5, 1)
y_train = 2 * X_train + 1 + 0.05 * torch.randn(5, 1)

# 20 个验证样本
X_val = torch.randn(20, 1)
y_val = 2 * X_val + 1 + 0.05 * torch.randn(20, 1)

# 用一个大网络 + 很多 epoch
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

train_losses = []
val_losses = []

for epoch in range(500):
    # 训练
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val)
        val_losses.append(val_loss.item())

print(f"最终训练 loss: {train_losses[-1]:.6f}")
print(f"最终验证 loss: {val_losses[-1]:.6f}")
print(f"验证 loss / 训练 loss = {val_losses[-1]/train_losses[-1]:.1f}x")
# 过拟合时，验证 loss 远大于训练 loss
```

**过拟合的直觉**：网络"背"下了训练数据，而不是"理解"了规律。就像学生背了习题答案，但考试换一道题就不会了。

---

## 9.5 本章总结

```
    ┌──────────────────────────────────────────┐
    │           训练循环 = 六步反复              │
    │                                          │
    │  ① optimizer.zero_grad()                 │
    │  ② output = model(batch_x)               │
    │  ③ loss = criterion(output, batch_y)     │
    │  ④ loss.backward()                       │
    │  ⑤ optimizer.step()                      │
    │                                          │
    │  训练时：model.train()                    │
    │  验证时：model.eval() + torch.no_grad()   │
    └──────────────────────────────────────────┘
```

---

## 9.6 本章练习

### 练习 9-1：最简单的训练循环

```python
# X=[[1],[2],[3]], y=[[2],[4],[6]]  (y=2x)
# Linear(1,1), SGD(lr=0.01)
# 训练 50 轮，每 10 轮打印 loss
```

### 练习 9-2：加上 loss 记录

```python
# 记录每轮 loss 到列表
# 验证 loss[0] > loss[-1]
```

### 练习 9-3：加上验证循环

```python
# 创建验证集 X_val, y_val
# 每 10 轮评估，使用 model.eval() + torch.no_grad()
```

### 练习 9-4：过拟合实验

```python
# 3 个训练样本 + 大网络(Linear(1,32)→ReLU→Linear(32,32)→ReLU→Linear(32,1))
# 训练 1000 轮
# 观察 train loss 趋于 0 但 val loss 不降
```

### 练习 9-5：不看答案——独立完成训练循环

> 关闭所有文档，独立写出完整训练流程：
> 1. 数据：y = 0.5x - 1（10 个样本，加噪声）
> 2. 模型：Linear(1,1)
> 3. 优化器：Adam(lr=0.01)
> 4. 训练 100 轮
> 5. 记录并打印 loss（验证下降）
> 6. 验证 W≈0.5, b≈-1

---

> **下一步**：真实数据怎么加载？进入[第十章：Dataset 与 DataLoader](10_dataloader.md)。
