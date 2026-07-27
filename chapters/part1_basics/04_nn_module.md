# 第四章：nn.Module — 神经网络类的骨架

## 4.0 本章导引

第三章你学了 Python 类、self、继承、super()。现在我们把它们全部用到 PyTorch 上。

**`nn.Module` 是所有 PyTorch 神经网络的父类。** 你写的每一个网络都是它的子类。

这意味什么？这意味着你不需要从零写参数管理、设备迁移、模式切换、保存加载——PyTorch 已经在 `nn.Module` 里帮你写好了。你只需要**继承**它。

本章结束时，你应该能：
1. 独立写出继承 `nn.Module` 的网络类（不看模板）
2. 理解为什么是 `model(x)` 而不是 `model.forward(x)`
3. 知道 `nn.Module` 帮你做了哪些事——以及为什么你不需要管它们

---

## 4.1 先体验一下"没有 nn.Module 的世界"

在 PyTorch 中写神经网络，你不一定非得用 `nn.Module`。先看看没有它是什么样子，你就理解它解决了什么问题。

```python
# %%
import torch

# === 手动管理参数的"网络" ===
# 一个简单线性层：输入 3 维 → 输出 1 维，即 y = Wx + b

W = torch.randn(1, 3, requires_grad=True)   # 权重
b = torch.zeros(1, requires_grad=True)       # 偏置

def my_forward(x):
    return x @ W.T + b

# 使用
x = torch.randn(5, 3)   # batch=5, 特征=3
y = my_forward(x)
print(f"输出: {y.shape}")  # [5, 1]

# 这能工作。但问题马上来了：
```

**当网络变成 10 层时**：

```python
# === 10 层的手动参数管理 —— 噩梦 ===
W1 = torch.randn(128, 784, requires_grad=True)
b1 = torch.zeros(128, requires_grad=True)
W2 = torch.randn(64, 128, requires_grad=True)
b2 = torch.zeros(64, requires_grad=True)
W3 = torch.randn(32, 64, requires_grad=True)
b3 = torch.zeros(32, requires_grad=True)
# ... 还有 7 层 ...
# 参数散落各处——你不小心就会漏掉一个

def my_deep_forward(x):
    x = x @ W1.T + b1
    x = torch.relu(x)
    x = x @ W2.T + b2
    x = torch.relu(x)
    # ... 忘了哪一层有 relu、哪层没有？
    return x

# 痛点清单：
# ❌ 参数散落各处，容易遗漏
# ❌ 移到 GPU 要手动：W1 = W1.to('cuda') ... 做 20 遍
# ❌ 保存模型：要手动收集 20 个参数
# ❌ 优化器：要手动传 20 个参数
# ❌ 换个网络结构 → 全部重写
```

这就是 `nn.Module` 要解决的问题。它是 PyTorch 提供给你的"自动驾驶系统"：

```
┌──────────────────────────────────────────────────┐
│                  nn.Module 帮你做                  │
│                                                   │
│  ✅ 自动收集所有参数（parameters()）                │
│  ✅ 一键迁移设备（.to(device)）                    │
│  ✅ 模式切换（.train() / .eval()）                 │
│  ✅ 保存/加载（.state_dict()）                    │
│  ✅ 子模块自动发现                                  │
│                                                   │
│  你只需要关注：定义什么层 + 数据怎么流动              │
└──────────────────────────────────────────────────┘
```

---

## 4.2 第一个 nn.Module 子类 —— 5 行代码

```python
# %%
import torch
import torch.nn as nn

class SimplestNet(nn.Module):
    def __init__(self):
        super().__init__()                    # ← 必须！
        self.fc = nn.Linear(3, 1)             # ← 定义一个线性层
    
    def forward(self, x):                     # ← 定义前向传播
        return self.fc(x)

# 创建网络实例
model = SimplestNet()
print(model)
```

**输出**：
```
SimplestNet(
  (fc): Linear(in_features=3, out_features=1, bias=True)
)
```

**现在逐行深度解释：**

```
第 1 行：class SimplestNet(nn.Module):
    ↑
    第三章学的继承！SimplestNet 继承了 nn.Module。
    这意味着 SimplestNet 自动获得了 nn.Module 的全部能力。
    
    类比：你注册了一个公司 → 自动获得了开票、报税等"公司基础设施"。
    你不用自己搭建税务局。

第 3 行：super().__init__()
    ↑
    第三章学的 super()！调用 nn.Module.__init__()。
    这一行激活了 nn.Module 的内部机制：
    - 参数管理系统上线
    - 训练模式设为 True
    - 设备追踪器启动
    - 子模块扫描器就绪
    
    ⚠️ 如果不写这行，nn.Module 的内部机制不会启动。
    你的网络看起来正常，但 parameters()、to()、train() 全部失效。

第 4 行：self.fc = nn.Linear(3, 1)
    ↑
    第三章学的 self！self.fc 是一个实例属性。
    nn.Module 会自动扫描所有 self.xxx 属性，
    如果 xxx 是 nn.Module 的子类，就注册为"子模块"。
    
    用 self. 而不用局部变量的原因：
    - self.fc → nn.Module 能发现它 → 参数被自动收集
    - fc = ...   → 局部变量 → nn.Module 不知道它的存在 → 参数丢失

第 6-7 行：def forward(self, x): return self.fc(x)
    ↑
    第三章学的方法重写！nn.Module 有一个默认的 forward()（什么都不做）。
    你重写它来定义你的前向传播逻辑。
```

---

## 4.3 为什么是 model(x) 而不是 model.forward(x)

```python
# %%
x = torch.randn(2, 3)

output1 = model(x)              # ✅ 推荐
output2 = model.forward(x)      # ⚠️ 能工作，但不推荐

print(torch.allclose(output1, output2))   # True（结果一样）
```

**为什么推荐 `model(x)`？**

因为 `nn.Module` 实现了 `__call__` 方法。这是 Python 的一个特殊方法——当你把对象当函数调用时，Python 自动调用 `__call__`。

```python
# %%
# Python 原生演示 __call__
class Adder:
    def __init__(self, n):
        self.n = n
    
    def __call__(self, x):
        print(f"__call__ 被调用了！")
        return x + self.n

add5 = Adder(5)
print(add5(10))   # 15 ← add5 被当成了函数来用！
# 等价于 add5.__call__(10)
```

`nn.Module` 的 `__call__` 做了什么：

```
model(x) 触发：
    1. model.__call__(x) 被调用
    2. __call__ 做一些"家务"（如触发 hooks）
    3. __call__ 调用 model.forward(x)
    4. 返回 forward 的结果

如果你直接调 model.forward(x)，就跳过了步骤 2 的"家务"。
大多数情况下看不出区别，但某些高级功能会失效。
```

**记住**：永远用 `model(x)`，不用 `model.forward(x)`。

---

## 4.4 nn.Module 的四大核心能力

### 4.4.1 parameters() —— 自动收集所有参数

```python
# %%
class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = TwoLayerNet()

print("模型的所有参数：")
for name, param in model.named_parameters():
    print(f"  {name:20s} shape: {list(param.shape)}")
```

**输出**：
```
模型的所有参数：
  fc1.weight           shape: [3, 4]
  fc1.bias             shape: [3]
  fc2.weight           shape: [2, 3]
  fc2.bias             shape: [2]
```

**`named_parameters()` 做了什么**：它递归扫描所有 `self.xxx` 中定义的 `nn.Module` 子类，收集它们的参数，并给出名字。

```
    model
    ├── fc1 (nn.Linear)
    │   ├── fc1.weight  [3, 4]
    │   └── fc1.bias    [3]
    └── fc2 (nn.Linear)
        ├── fc2.weight  [2, 3]
        └── fc2.bias    [2]

    你不需要手动维护这个列表。
    nn.Module 自动完成了这一切。
```

### 4.4.2 to(device) —— 一键迁移设备

```python
# %%
# 如果有 GPU
if torch.cuda.is_available():
    model = model.to('cuda')     # 整个网络移到 GPU
    # 等价于：把每个参数都 .to('cuda')
    # 你不需要写 20 行重复代码

# 回到 CPU
model = model.to('cpu')

# 检查当前设备
param = next(model.parameters())
print(f"模型在: {param.device}")    # cpu 或 cuda:0
```

### 4.4.3 train() / eval() —— 模式切换

```python
# %%
model = TwoLayerNet()
print(f"初始模式: training = {model.training}")  # True（默认）

model.eval()
print(f"eval() 后: training = {model.training}")  # False

model.train()
print(f"train()后: training = {model.training}")  # True
```

**为什么要区分？**

有些层在训练和推理时行为不同：
- **Dropout**：训练时随机丢弃神经元 → 推理时不动
- **BatchNorm**：训练时用 batch 的统计量 → 推理时用全局统计量

虽然这两个你还没学到，但**切换 train/eval 的习惯要从现在养成**。

### 4.4.4 state_dict() —— 保存和加载

```python
# %%
model = TwoLayerNet()

# state_dict 是一个普通的 Python 字典
state = model.state_dict()
print(state.keys())
# dict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])

print(f"fc1.weight:\n{state['fc1.weight']}")

# 保存
torch.save(state, 'my_model.pth')

# 加载（第十二章详细讲）
# new_model = TwoLayerNet()
# new_model.load_state_dict(torch.load('my_model.pth'))
```

---

## 4.5 在 __init__ 中定义层

### 4.5.1 什么是"层"

**层 = 一个包含（或不包含）可学习参数的 nn.Module 子类。**

```
nn.Linear(3, 2)   → 是一个层，有参数（W 和 b）
nn.ReLU()         → 也是一个"层"，但没有参数（纯函数）
nn.Sequential()   → 是层，它把多个层串在一起
```

每一层做三件事：
```
输入 → [变换] → 输出
```

### 4.5.2 为什么用 self.xxx = ... 而不用局部变量

```python
# %%
# ✅ 正确
class GoodNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)    # self. → nn.Module 能发现
    
    def forward(self, x):
        return self.fc(x)

# ❌ 错误
class BadNet(nn.Module):
    def __init__(self):
        super().__init__()
        fc = nn.Linear(10, 5)         # 局部变量 → nn.Module 找不到！
    
    def forward(self, x):
        return fc(x)                  # 而且 forward 中也访问不到（作用域问题）

# 如果你运行 BadNet：
# bad = BadNet()
# list(bad.parameters())  # 空列表！参数丢失了！
```

### 4.5.3 nn.Module 的"子模块自动发现"

```python
# %%
class NestedNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Sequential 本身也是 nn.Module，它的内部也有子模块
        self.block = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
    
    def forward(self, x):
        return self.block(x)

model = NestedNet()
print("嵌套网络的参数：")
for name, param in model.named_parameters():
    print(f"  {name:30s} shape: {list(param.shape)}")
```

**输出**：
```
嵌套网络的参数：
  block.0.weight                shape: [5, 10]
  block.0.bias                  shape: [5]
  block.2.weight                shape: [2, 5]
  block.2.bias                  shape: [2]
```

注意参数名：`block.0.weight`。`block` 是 Sequential 的名字，`0` 是它内部第一层的索引。nn.Module 递归扫描了整个嵌套结构。

---

## 4.6 在 forward 中追踪数据流动

```python
# %%
class TraceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        print(f"输入:           {tuple(x.shape)}")
        x = self.fc1(x)
        print(f"fc1 后:         {tuple(x.shape)}")
        x = self.relu(x)
        print(f"ReLU 后:        {tuple(x.shape)}   ← 注意：shape 没变！")
        x = self.fc2(x)
        print(f"fc2 后(输出):   {tuple(x.shape)}")
        return x

model = TraceNet()
x = torch.randn(32, 784)     # batch=32, 特征=784（28×28 展平）
output = model(x)
```

**输出**：
```
输入:           (32, 784)
fc1 后:         (32, 128)
ReLU 后:        (32, 128)   ← 注意：shape 没变！
fc2 后(输出):   (32, 10)
```

**这种 `print(x.shape)` 的调试方式是每个深度学习工程师的日常。** 在写新网络时，你几乎总是先用 print 验证 shape 变化是否符合预期。

---

## 4.7 本章总结：nn.Module 心智模型

```
    ┌─────────────────────────────────────────┐
    │            nn.Module（父类）              │
    │                                         │
    │  ✅ 参数自动管理（parameters()）           │
    │  ✅ 设备一键迁移（.to(device)）           │
    │  ✅ 模式切换（.train() / .eval()）        │
    │  ✅ 保存/加载（.state_dict()）            │
    │  ✅ 子模块自动发现                        │
    │  ✅ __call__ → forward                  │
    │                                         │
    │  这些都是你不需要重写的"基础设施"           │
    └──────────────┬──────────────────────────┘
                   │ 继承
    ┌──────────────▼──────────────────────────┐
    │          你的网络（子类）                  │
    │                                         │
    │  __init__:                              │
    │    super().__init__()        ← 必须！     │
    │    self.fc1 = nn.Linear()    ← 必须用 self.│
    │    self.fc2 = nn.Linear()              │
    │                                         │
    │  forward(self, x):                      │
    │    x = self.fc1(x)           ← 数据流动    │
    │    x = self.fc2(x)                      │
    │    return x                             │
    └─────────────────────────────────────────┘
```

**五个铁律**：

| # | 铁律 | 为什么 |
|---|------|--------|
| 1 | `class X(nn.Module):` | 必须继承 |
| 2 | `super().__init__()` | 激活 nn.Module 的内部机制 |
| 3 | `self.xxx = ...` | 用 self. 才能被 nn.Module 发现 |
| 4 | `def forward(self, x):` | 重写 nn.Module 的 forward |
| 5 | `model(x)` 不用 `model.forward(x)` | 使用 __call__ 确保家务被执行 |

---

## 4.8 本章练习

### 练习 4-1：空网络

```python
# 写一个 IdentityNet：forward 直接返回输入
# 验证 model(torch.randn(2, 5)) 的输出等于输入
```

### 练习 4-2：打印参数

```python
# 创建以下网络，打印所有参数名和 shape：
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.fc2(self.fc1(x))
```

### 练习 4-3：单层网络 + shape 追踪

```python
# 写一个 OneLayerNet：Linear(8, 4)
# 用 print 追踪 forward 中的 shape
# 输入 torch.randn(3, 8)，验证输出是 [3, 4]
```

### 练习 4-4：三层网络 + 完整 shape 追踪

```python
# 写一个网络：10 → 6 → 3 → 1
# 在 forward 中打印每层后的 shape
# 输入 torch.randn(4, 10)
```

### 练习 4-5：验证 train/eval

```python
# 不运行代码，回答：
# model = nn.Linear(3, 2)
# print(model.training) → ?
# model.eval()
# print(model.training) → ?
# 然后运行验证
```

### 练习 4-6：不看答案——独立写两层网络

> 关闭所有文档，独立写出：

```python
# TwoLayerNet：Linear(6, 4) + Linear(4, 2)
# 输入 torch.randn(3, 6)
# 打印所有参数
# forward 中用 print 追踪 shape
```

### 练习 4-7：不看答案——独立写三层网络（自选维度）

> 关闭所有文档，独立写出一个三层网络。自己决定每层的维度。
> 确保维度匹配（上一层的 out_features = 下一层的 in_features）。

---

## 答案与提示

<details>
<summary>练习 4-1 答案</summary>

```python
class IdentityNet(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

model = IdentityNet()
x = torch.randn(2, 5)
print(torch.allclose(model(x), x))  # True
```
</details>

<details>
<summary>练习 4-6 参考答案</summary>

```python
class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 4)
        self.fc2 = nn.Linear(4, 2)
    
    def forward(self, x):
        print(f"输入:   {tuple(x.shape)}")
        x = self.fc1(x)
        print(f"fc1后:  {tuple(x.shape)}")
        x = self.fc2(x)
        print(f"fc2后:  {tuple(x.shape)}")
        return x

model = TwoLayerNet()
x = torch.randn(3, 6)
output = model(x)
for name, p in model.named_parameters():
    print(f"{name}: {list(p.shape)}")
```
</details>

---

> **下一步**：骨架能独立写了。现在填入血肉——[第五章：单层网络](05_linear_layer.md)，理解数据如何流动。
