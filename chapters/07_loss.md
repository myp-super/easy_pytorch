# 第七章：损失函数 — 告诉网络"你错得有多离谱"

## 7.0 本章导引

现在网络能输出预测了。但关键问题是：**这个预测好不好？**

损失函数（Loss Function）就是回答这个问题的裁判。它给网络一个数字分数——"你的输出离正确答案差了多少"。

这个数字是所有后续学习的起点。**没有 loss，就没有 backward，就没有学习。**

---

## 7.1 损失函数的直觉

### 7.1.1 现实世界类比

```
你在练习射箭：

    靶心           = 真实标签（正确答案）
    箭的落点       = 网络预测
    离靶心的距离   = 损失（Loss）

    距离越小 → 损失越小 → 预测越准
    距离越大 → 损失越大 → 预测越差

训练目标 = 通过调整射箭姿势（参数），让平均距离越来越小
```

### 7.1.2 监督学习的核心

```
监督学习 = 你有正确答案。

    X → [神经网络] → 预测 ŷ
                         │
                    和正确答案 y 比较
                         │
                    差异 = Loss(ŷ, y)
                         │
                    Loss 越小越好
```

---

## 7.2 MSELoss — 回归任务的损失函数

### 7.2.1 什么时候用 MSE

**回归任务**——你预测的是一个连续数值：
- 房价：300 万、450 万、280 万……
- 温度：23.5°C、18.2°C……
- 年龄：25 岁、67 岁……

### 7.2.2 MSE 公式

```
MSE = (1/n) × Σ(predictionᵢ - targetᵢ)²

1. 每个样本：算预测值和真实值的差
2. 平方（让正负差都变成正的）
3. 取平均
```

```python
# %%
import torch
import torch.nn as nn

# 手动计算 MSE
predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
targets     = torch.tensor([1.5, 2.5, 2.5, 3.5])

# 一步步手算
diff = predictions - targets
print(f"差值:     {diff}")         # [-0.5, -0.5, 0.5, 0.5]

sq_diff = diff ** 2
print(f"平方:     {sq_diff}")      # [0.25, 0.25, 0.25, 0.25]

mse_manual = sq_diff.mean()
print(f"手动 MSE: {mse_manual}")   # 0.25

# 用 PyTorch
criterion = nn.MSELoss()
mse_torch = criterion(predictions, targets)
print(f"PyTorch MSE: {mse_torch}")  # 0.25
```

### 7.2.3 为什么平方而不是绝对值

```
    绝对值误差 |pred - target|：
        - 在最小值处不可导（梯度不连续）

    平方误差 (pred - target)²：
        - 处处可导（梯度连续）
        - 对大误差惩罚更重（平方放大离群值的影响）
        - 数学性质好
```

### 7.2.4 reduction 参数

```python
# %%
pred = torch.tensor([1.0, 3.0, 5.0])
targ = torch.tensor([2.0, 2.0, 2.0])

# 'mean'（默认）：取平均
print(nn.MSELoss(reduction='mean')(pred, targ))   # 2.0 (=(1+1+9)/3 → 11/3 ≈ 3.67.. 其实：(1²+1²+3²)/3 = 11/3)

# 'sum'：求和
print(nn.MSELoss(reduction='sum')(pred, targ))    # 11.0 (= 1+1+9)

# 'none'：不聚合，每个样本的损失单独返回
print(nn.MSELoss(reduction='none')(pred, targ))   # [1., 1., 9.]
```

---

## 7.3 CrossEntropyLoss — 分类任务的损失函数

### 7.3.1 为什么分类不能用 MSE

这是初学者最容易犯的概念错误。先看一个例子：

```python
# %%
# 分类任务：3 个类别，真实答案是"类别 1"
logits = torch.tensor([2.0, 1.0, 0.1])    # 网络输出的原始分数
target = torch.tensor(1)                    # 真实标签：第 1 类（0-indexed）

# 如果用 MSE：
# [2.0, 1.0, 0.1] 和 [0, 1, 0]（one-hot）的 MSE
# = ((2-0)² + (1-1)² + (0.1-0)²) / 3
# = (4 + 0 + 0.01) / 3 ≈ 1.34

# 问题在这：MSE 只关心"数值差"
# 但分类任务关心的是：哪个类别的分数最大？
# logits [0.1, 5.0, 3.0] → 类别 1 最大（正确）
# logits [0.1, 3.0, 5.0] → 类别 2 最大（错误！）
# MSE 不能很好地捕捉"排序是否正确"
```

**分类的核心是"概率分布的比较"，不是"数值差"。**

### 7.3.2 从 Softmax 到 CrossEntropy

CrossEntropyLoss 内部做了两件事：

```
Step 1: Softmax → 把 logits 变成概率分布（每行和为 1）
Step 2: NLLLoss → 取真实标签对应的概率，取 -log
```

**Step 1 — Softmax：**

```python
# %%
logits = torch.tensor([2.0, 1.0, 0.1])
softmax_probs = torch.softmax(logits, dim=0)
print(f"Softmax 结果: {softmax_probs}")
# tensor([0.6590, 0.2424, 0.0986])
print(f"总和: {softmax_probs.sum():.4f}")  # 1.0 ✓

# 物理含义：模型认为每个类别的概率是：
#   P(类别0) = 65.9%
#   P(类别1) = 24.2%
#   P(类别2) = 9.9%
```

**Step 2 — 取真实标签对应的概率并取负对数：**

```python
# %%
target = 1  # 真实标签是"类别 1"
prob_of_target = softmax_probs[target]   # 0.2424
loss = -torch.log(prob_of_target)
print(f"真实标签的概率: {prob_of_target:.4f}")
print(f"-log(概率):     {loss:.4f}")
# 如果模型对正确答案的概率很高 → -log(接近1) ≈ 0 → loss 小 ✅
# 如果模型对正确答案的概率很低 → -log(接近0) ≈ 大 → loss 大 ❌
```

### 7.3.3 直接用 CrossEntropyLoss

```python
# %%
criterion = nn.CrossEntropyLoss()

# batch=4, 类别数=3
logits = torch.randn(4, 3)          # [4, 3]
labels = torch.tensor([0, 2, 1, 1]) # [4] —— 整数索引

loss = criterion(logits, labels)
print(f"CrossEntropyLoss: {loss:.4f}")
```

### 7.3.4 ⚠️ 最重要的陷阱：CrossEntropyLoss 内置 Softmax

```python
# %%
# ❌ 错误！网络最后一层加了 Softmax
class BadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
    
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)  # 别加 Softmax！

# ✅ 正确！网络输出 raw logits
class GoodNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
    
    def forward(self, x):
        return self.fc(x)  # 返回原始分数（logits）

# 原因：CrossEntropyLoss = LogSoftmax + NLLLoss
# 如果网络做了 Softmax → 两次 Softmax → 数值错误
```

### 7.3.5 label 格式：整数索引，不是 one-hot

```python
# %%
# ✅ 正确：整数索引
labels = torch.tensor([0, 2, 1, 0, 1])  # [batch_size] —— 整数

# ❌ 错误：不要手动做 one-hot
# labels = torch.tensor([
#     [1, 0, 0],
#     [0, 0, 1],
#     [0, 1, 0],
# ])  # CrossEntropyLoss 不接受这种格式
```

---

## 7.4 损失函数与输出层——配对规则

```
┌──────────────────┬──────────────────┬──────────────────────┐
│ 任务              │ 输出层            │ 损失函数              │
├──────────────────┼──────────────────┼──────────────────────┤
│ 回归（连续值）    │ 1 个输出，无激活   │ MSELoss              │
│ 二分类            │ 1 个输出，无激活   │ BCEWithLogitsLoss    │
│ 多分类            │ N 个输出，无激活   │ CrossEntropyLoss     │
│ 多标签分类        │ N 个输出，Sigmoid │ BCEWithLogitsLoss    │
└──────────────────┴──────────────────┴──────────────────────┘

核心原则：如果损失函数名字里有 "Logits" → 网络不要加 Sigmoid/Softmax
          CrossEntropyLoss 虽然名字里没有 Logits，但它内置了 Softmax
```

---

## 7.5 loss.backward() —— 从损失反向传播

**这是连接第七章和第二章的关键桥梁。**

```python
# %%
import torch
import torch.nn as nn

# 创建一个简单网络
model = nn.Linear(4, 3)
x = torch.randn(2, 4)
target = torch.tensor([0, 2])

# 1. Forward
output = model(x)       # [2, 3]

# 2. Loss
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
print(f"loss = {loss.item():.4f}")
print(f"loss.requires_grad = {loss.requires_grad}")  # True

# 3. Backward —— 从这里，梯度反向传播流经整个网络！
loss.backward()

# 4. 参数现在有了梯度
for name, param in model.named_parameters():
    print(f"{name}.grad: {param.grad is not None}")  # 都是 True！
```

**`loss.backward()` 触发了什么——一张图看懂：**

```
    loss（标量）
      │ .backward()
      ▼
    梯度沿着计算图反向流动
      │
    output ←  model(x)
      │         │
      ▼         ▼
    model 的每个参数都被计算出了 .grad
```

---

## 7.6 本章总结

```
    ┌──────────┐     ┌──────────────┐     ┌──────────┐
    │ 网络输出   │ ──→ │ Loss Function │ ──→ │ 一个标量  │
    │ (logits)  │     │ 预测 vs 真实   │     │ loss=2.35 │
    └──────────┘     └──────────────┘     └────┬─────┘
                                               │
                                        loss.backward()
                                               │
                                        梯度反向传播到全网
```

**铁律**：
1. 回归 → MSE；分类 → CrossEntropyLoss
2. 最后一层**不要**加 Softmax/Sigmoid（除非你完全理解为什么）
3. label 传**整数索引**，不要手动做 one-hot
4. `loss.backward()` 是全书最重要的调用了之一

---

## 7.7 本章练习

### 练习 7-1：手动计算 MSE

```python
# pred = [1., 2., 3.], target = [2., 2., 2.]
# 手算 MSE，然后用 nn.MSELoss 验证
```

### 练习 7-2：手写 Softmax

```python
# 用 torch.exp 和 torch.sum 实现 Softmax
# 输入 [2.0, 1.0, 0.1]
# 对比 torch.softmax 的结果
```

### 练习 7-3：手写 CrossEntropyLoss

```python
# logits = [2.0, 1.0, 0.1], target = 0
# 一步一步算：
# 1. Softmax
# 2. 取 target 对应的概率
# 3. -log(概率)
# 4. 和 nn.CrossEntropyLoss 对比
```

### 练习 7-4：forward → loss → backward 完整流程

```python
# 网络：Linear(10, 3)
# 输入：torch.randn(4, 10)
# 标签：torch.tensor([0, 2, 1, 1])
# 完整 Forward → Loss → Backward
# 打印 loss，检查每个参数是否有 .grad
```

### 练习 7-5：MSE 做分类的实验

```python
# 同一个分类问题，分别用 MSE 和 CrossEntropyLoss
# 比较 loss 的数值
# 思考：为什么 MSE 不适合分类
```

### 练习 7-6：不看答案——独立完成

> 关闭所有文档，独立完成：
> 1. TwoLayerNet：Linear(8, 4) → ReLU → Linear(4, 2)
> 2. 输入 [3, 8]，标签 [0, 1, 0]（3 个样本的二分类）
> 3. Forward → CrossEntropyLoss → Backward
> 4. 打印 loss 和参数是否有 grad
> 5. 解释为什么最后一层没有加 Softmax

---

## 答案与提示

<details>
<summary>练习 7-3 步骤</summary>

```python
logits = torch.tensor([2.0, 1.0, 0.1])
target = torch.tensor(0)

# Step 1: Softmax
probs = torch.softmax(logits, dim=0)
# [0.6590, 0.2424, 0.0986]

# Step 2: 取 target=0 的概率
p = probs[0]  # 0.6590

# Step 3: -log
loss = -torch.log(p)  # ≈ 0.4170

# Step 4: 验证
ce = nn.CrossEntropyLoss()
print(ce(logits.unsqueeze(0), target.unsqueeze(0)))  # ≈ 0.4170
```
</details>

---

> **下一步**：loss 有了，梯度有了，怎么用？进入[第八章：优化器](./08_optimizer.md)。
