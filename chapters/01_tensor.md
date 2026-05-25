# 第一章：Tensor — 深度学习的基本语言

## 1.0 本章导引

### 为什么这一章如此重要？

如果你问一个有经验的深度学习工程师："如果让我重新学 PyTorch，我最希望自己在哪个基础上下更多功夫？"

90% 的人会回答：**Tensor 操作**。

不是网络结构，不是损失函数，不是优化器——而是最基础的 Tensor。

为什么？因为：

- 你在调试维度不匹配的 bug 时，用到的是 Tensor 知识
- 你在设计网络结构时，脑中推算的是 shape 的变化
- 你在看论文复现代码时，第一件事就是推算每一层的 shape
- 你写的每一个 `forward` 函数，本质上都是在做 Tensor 变换

**把 Tensor 玩熟，后面的学习会轻松 10 倍。把 Tensor 糊弄过去，后面每一步都会踩坑。**

### 本章目标

学完本章后，你不需要查任何资料，就能：

1. 用 6 种方法创建任意形状的 Tensor
2. 看懂任何 Tensor 的 shape、dtype、device
3. 用 `view` / `reshape` / `squeeze` / `unsqueeze` / `transpose` / `permute` 自由变换形状
4. 判断两个 Tensor 能不能广播、广播后的 shape 是什么
5. 区分逐元素乘法和矩阵乘法，知道什么时候用哪一个
6. **不看答案，10 分钟内完成所有基础操作**

### 本章学习地图

```
1.1 什么是 Tensor？         ← 直觉理解
1.2 Tensor 的创建            ← 6 种创建方法
1.3 Tensor 的核心属性        ← shape / dtype / device / requires_grad
1.4 Tensor 的基本操作        ← 索引、四则运算、矩阵乘法
1.5 shape 操作               ← view / reshape / squeeze / unsqueeze / transpose / permute
1.6 Broadcast 广播机制       ← 最重要的隐式操作
1.7 综合实战                 ← 把本章所有知识串联起来
1.8 练习                     ← 9 类阶梯练习
```

---

## 1.1 什么是 Tensor

### 1.1.1 从直觉出发：数据如何变得越来越"复杂"

让我们从一个最最简单的场景开始——你在用温度计量体温。

```python
# 场景 1：你只量了一次体温
temperature = 36.5                # 一个单独的数字
```

这就是最简单的数据——**一个数**。

现在，你连续量了一周的体温：

```python
# 场景 2：一周的体温记录
temperatures = [36.5, 36.8, 37.1, 36.9, 37.2, 36.6, 36.7]
#              周一   周二   周三   周四   周五   周六   周日
```

你有了**一组数**——一个列表。每个数都有自己的位置（索引）。

现在更复杂一点——你不仅记录了体温，还记录了血压和心率：

```python
# 场景 3：每天三个指标 [体温, 收缩压, 心率]
monday    = [36.5, 120, 72]
tuesday   = [36.8, 125, 75]
wednesday = [37.1, 118, 70]
# ... 等等

health_data = [
    [36.5, 120, 72],   # 周一
    [36.8, 125, 75],   # 周二
    [37.1, 118, 70],   # 周三
]
```

现在你有了一个**二维结构**——像一个 Excel 表格。每一行是一天，每一列是一个指标。

再进一步——你在三个城市同时记录这些数据：

```python
# 场景 4：三个城市 × 七天 × 三个指标
# 这是三维数据！
# 城市1: [[36.5, 120, 72], ...]
# 城市2: [[36.3, 115, 68], ...]
# 城市3: [[36.9, 122, 74], ...]
```

**数据每多一层"分类"，就多一个维度。**

```
1 维：一周的体温          → [7]
2 维：一周 × 三项指标      → [7, 3]
3 维：三城市 × 一周 × 三项  → [3, 7, 3]
```

### 1.1.2 Tensor 到底是什么

**Tensor = 多维数组。**

就这么简单。如果你理解"列表的列表的列表"，你就理解了 Tensor 的基本概念。

但 Tensor 比 Python 列表多了三个关键能力：

```
┌──────────────────────────────────────────────┐
│                  Tensor                        │
│                                               │
│  ┌─────────────┐  ┌──────────┐  ┌─────────┐  │
│  │  多维数组    │  │ GPU 加速  │  │自动求导  │  │
│  │  (存数据)    │  │ (算得快)  │  │(学得会)  │  │
│  └─────────────┘  └──────────┘  └─────────┘  │
│                                               │
│  Python 列表只能做前两件中的第 0.5 件            │
└──────────────────────────────────────────────┘
```

| 能力 | 解释 | 为什么重要 |
|------|------|-----------|
| 多维数组 | 存储任意维度的数据 | 图片是 3D（H×W×C），视频是 4D（T×H×W×C），batch 是 4D/5D |
| GPU 加速 | 在显卡上并行计算 | 同样的矩阵乘法，GPU 比 CPU 快 10-100 倍 |
| 自动求导 | 自动计算梯度 | 神经网络学习的引擎（第二章详讲） |

### 1.1.3 现实世界中的 Tensor —— 建立直觉

**场景 A：一张图片**

```
一张 28×28 的灰度手写数字图片：

    ┌───────────┐
    │ 0 0 0 ... │  28 行
    │ 0 1 0 ... │
    │ ...       │
    │ ...    0  │
    └───────────┘
       28 列

    shape = [28, 28]         ← 2 维
    每个位置存一个 0-255 的像素值
```

**场景 B：一批彩色图片**

```
16 张 × 3 通道(RGB) × 28 高 × 28 宽：

    shape = [16, 3, 28, 28]  ← 4 维

    第 0 维：batch（16 张）
    第 1 维：通道（R、G、B）
    第 2 维：高度（28 像素）
    第 3 维：宽度（28 像素）
```

**场景 C：一段文本（token 序列）**

```
32 句话 × 每句最多 50 个 token × 每个 token 用 768 维向量表示：

    shape = [32, 50, 768]    ← 3 维
```

**场景 D：一个视频片段**

```
8 帧 × 3 通道 × 224 高 × 224 宽：

    shape = [8, 3, 224, 224]  ← 4 维
```

### 1.1.4 维度命名的约定

不同领域对维度有约定俗成的命名：

| 维度 | 名称 | 含义 | 示例 shape 值 |
|------|------|------|-------------|
| 第 0 维 | batch_size | 一次处理多少个样本 | 32 |
| 第 1 维 | channels / features | 通道数或特征数 | 3 (RGB) |
| 第 2 维 | height / sequence_length | 高度或序列长度 | 224 |
| 第 3 维 | width | 宽度 | 224 |

```
一个典型的图片 batch：
    [batch_size, channels, height, width]
       32           3        224     224

一个典型的文本 batch：
    [batch_size, sequence_length, hidden_dim]
       32           50               768
```

> **暂时不用背这些命名。** 你只需要知道 shape 的每个位置代表什么含义，这在你调试维度 bug 时至关重要。

### 1.1.5 一个快速的思维测试

在你继续往下读之前，试试看能不能回答：

```
shape = [4, 3, 28, 28] 的数据最可能是什么？

A. 4 句话，每句 3 个词
B. 4 张 RGB 图片，每张 28×28
C. 4 个视频，每个 3 帧

答案：B。
4 = batch，3 = RGB 通道，28 = 高，28 = 宽
```

---

## 1.2 Tensor 的创建 —— 6 种方法

现在你知道了 Tensor 是什么。让我们动手创建它们。

在 VS Code 中新建一个 `.py` 文件，用 `# %%` 分隔单元格（需要 Jupyter 插件）：

```python
# %%
import torch
print(f"PyTorch 版本: {torch.__version__}")
```

### 方法 1：从 Python 列表创建 —— `torch.tensor()`

这是最直接的方式——你把一个 Python 列表（或嵌套列表）交给 PyTorch，它帮你转成 Tensor。

```python
# %%
# 1 维 —— 从一维列表
t1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("1 维 Tensor:")
print(t1)
print(f"shape: {t1.shape}")    # torch.Size([4])
print(f"dtype: {t1.dtype}")    # torch.float32
```

**停下来，仔细看这段输出：**

```
tensor([1., 2., 3., 4.])
shape: torch.Size([4])
dtype: torch.float32
```

三个信息：
- `tensor([1., 2., 3., 4.])` —— 这是 Tensor 的内容。注意数字后面的 `.`，表示它们是浮点数。
- `torch.Size([4])` —— **shape 永远是一个 `torch.Size` 对象**，它本质上是一个元组。`[4]` 意思是"这是一个一维 Tensor，有 4 个元素"。
- `torch.float32` —— 每个元素占 32 位（4 字节），使用浮点格式存储。

> **关键细节**：为什么 `shape` 是 `torch.Size([4])` 而不是直接 `4`？
>
> 因为 shape 描述的是**所有维度的大小**。一个三维 Tensor 的 shape 是 `[2, 3, 4]`，一个标量的 shape 是 `[]`。为了保持一致性，一维 Tensor 的 shape 也是列表形式：`[4]`。

```python
# %%
# 2 维 —— 从嵌套列表
t2 = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
print("\n2 维 Tensor:")
print(t2)
print(f"shape: {t2.shape}")    # torch.Size([2, 3])
```

**shape `[2, 3]` 怎么理解？**

```
           列0  列1  列2
    行0:  [1.0, 2.0, 3.0]
    行1:  [4.0, 5.0, 6.0]

    2 行，3 列 → shape = [2, 3]
```

**shape 的"从外到内"规则：**

```
shape = [2, 3]

最外层有 2 个元素（2 行）
   每个元素（每行）里面有 3 个元素（3 列）

shape = [2, 3, 4]

最外层有 2 个元素
   每个元素里面有 3 个元素
       每个元素里面有 4 个元素
```

> **类比**：shape 就像俄罗斯套娃的层级描述。`[2, 3, 4]` = 2 个大套娃，每个大套娃里面有 3 个中套娃，每个中套娃里面有 4 个小套娃。

```python
# %%
# 3 维 —— 更深的嵌套
t3 = torch.tensor([
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [7, 8, 9],
        [10, 11, 12]
    ]
])
print(f"3 维 shape: {t3.shape}")  # torch.Size([2, 2, 3])
# 2 个"块"，每个块 2 行 3 列
```

### 方法 2：全零 / 全一 —— `torch.zeros()` / `torch.ones()`

当你需要初始化偏置为 0，或者需要一个占位 Tensor 时：

```python
# %%
# 全零 Tensor
zeros = torch.zeros(3, 4)     # 3 行 4 列，全是 0
print(f"zeros shape: {zeros.shape}")
print(zeros)

# 全一 Tensor
ones = torch.ones(2, 3)       # 2 行 3 列，全是 1
print(f"\nones:\n{ones}")

# 全指定值
filled = torch.full((2, 3), 7.0)  # 2 行 3 列，全是 7
print(f"\n全 7:\n{filled}")
```

**注意参数形式**：
- `torch.zeros(3, 4)` —— 直接传维度
- `torch.full((2, 3), 7.0)` —— 第一个参数是元组，第二个是填充值

### 方法 3：随机生成 —— `torch.rand()` / `torch.randn()`

这是最常用的创建方式——神经网络参数初始化就是随机生成的。

```python
# %%
# torch.rand: 均匀分布 U(0, 1)
rand_uniform = torch.rand(2, 4)
print("rand (0到1均匀):")
print(rand_uniform)
print(f"最小值: {rand_uniform.min():.3f}, 最大值: {rand_uniform.max():.3f}")
# 值永远在 [0, 1) 之间

# torch.randn: 标准正态分布 N(0, 1)
rand_normal = torch.randn(2, 4)
print("\nrandn (标准正态):")
print(rand_normal)
print(f"均值: {rand_normal.mean():.3f}, 标准差: {rand_normal.std():.3f}")
# 大部分值在 -2 到 +2 之间，均值为 0
```

**什么时候用哪个？**

```
torch.rand:  输出 [0, 1)，适合需要非负值的场景（比如概率）
torch.randn: 输出以 0 为中心，可正可负，适合权重初始化

在神经网络中，权重通常用 randn（或它的变体）初始化。
这是几十年的经验总结——以 0 为中心的随机初始化通常效果最好。
```

### 方法 4：等差序列 —— `torch.arange()` / `torch.linspace()`

```python
# %%
# arange(start, end, step) —— 和 Python 的 range() 一摸一样
seq1 = torch.arange(0, 10, 2)    # 从 0 到 10(不含)，步长 2
print(f"arange(0, 10, 2): {seq1}")  # [0, 2, 4, 6, 8]

seq2 = torch.arange(5)            # 只给 end：默认 start=0, step=1
print(f"arange(5): {seq2}")       # [0, 1, 2, 3, 4]

# linspace(start, end, steps) —— 固定数量，均匀分割
seq3 = torch.linspace(0, 1, 5)    # 0 到 1 之间，均匀取 5 个点
print(f"linspace(0, 1, 5): {seq3}")  # [0.00, 0.25, 0.50, 0.75, 1.00]
```

**arange vs linspace —— 一张图理解：**

```
arange(start=0, end=10, step=2):
    关注"步长"
    |---|---|---|---|---|---|---|---|---|---|
    0   2   4   6   8
    ↑           ↑
  start       end（不含）

linspace(start=0, end=1, steps=5):
    关注"等分数"
    |---|---|---|---|---|---|---|---|---|---|
    0  0.25 0.5 0.75 1.0
    ↑                   ↑
  start               end（含）
```

### 方法 5：单位矩阵 —— `torch.eye()`

```python
# %%
I = torch.eye(3)   # 3×3 单位矩阵
print("3×3 单位矩阵:")
print(I)
# [[1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.]]

I_rect = torch.eye(3, 4)  # 3×4 矩形"单位矩阵"
print(f"\n3×4:\n{I_rect}")
# 对角线上是 1，其余是 0
```

### 方法 6：从 NumPy 互相转换

```python
# %%
import numpy as np

# NumPy → Tensor
np_array = np.array([[1.0, 2.0], [3.0, 4.0]])
tensor_from_np = torch.from_numpy(np_array)
print(f"从 NumPy 创建:\n{tensor_from_np}")

# Tensor → NumPy
t = torch.tensor([1.0, 2.0, 3.0])
np_from_tensor = t.numpy()
print(f"从 Tensor 转 NumPy: {np_from_tensor}")

# ⚠️ 注意：torch.from_numpy() 创建的 Tensor 和原 NumPy 数组共享内存！
# 修改一个会影响另一个。如果需要独立拷贝，用 torch.tensor(np_array)
```

### 创建方法总览表

```
┌─────────────────┬──────────────────────┬──────────────────────┐
│ 方法             │ 用途                  │ 示例                  │
├─────────────────┼──────────────────────┼──────────────────────┤
│ torch.tensor()  │ 从列表创建             │ torch.tensor([1.,2.]) │
│ torch.zeros()   │ 全零（初始化偏置）      │ torch.zeros(2, 3)    │
│ torch.ones()    │ 全一                  │ torch.ones(2, 3)     │
│ torch.rand()    │ [0,1) 均匀随机        │ torch.rand(2, 3)     │
│ torch.randn()   │ 标准正态随机（初始化权重）│ torch.randn(2, 3)   │
│ torch.arange()  │ 等差序列（按步长）      │ torch.arange(0,10,2)│
│ torch.linspace()│ 等差序列（按数量）      │ torch.linspace(0,1,5)│
│ torch.eye()     │ 单位矩阵               │ torch.eye(3)        │
│ torch.full()    │ 全指定值               │ torch.full((2,3),7.)│
└─────────────────┴──────────────────────┴──────────────────────┘
```

> **现在停一下。** 不往下看，你能在脑中默念出这 9 种创建方法吗？如果可以，继续。如果不行，回上去再看一遍——后面的内容建立在你能熟练创建 Tensor 的基础上。

---

## 1.3 Tensor 的四个核心属性

每个 Tensor 都有四个你必须烂熟于心的属性。它们回答了关于数据的四个基本问题：

```
┌───────────────┬─────────────────────┬────────────────────┐
│ 属性           │ 回答的问题            │ 示例值             │
├───────────────┼─────────────────────┼────────────────────┤
│ .shape        │ 数据长什么样？        │ torch.Size([2,3,4])│
│ .dtype        │ 数据是什么类型？      │ torch.float32     │
│ .device       │ 数据存在哪里？        │ cpu / cuda:0      │
│ .requires_grad│ 需要为它算梯度吗？    │ True / False      │
└───────────────┴─────────────────────┴────────────────────┘
```

```python
# %%
import torch

# 创建一个有代表性的 Tensor
x = torch.randn(2, 3, 4)

# 四个属性一次性查看
print(f"shape:         {x.shape}")           # torch.Size([2, 3, 4])
print(f"dtype:         {x.dtype}")           # torch.float32
print(f"device:        {x.device}")          # cpu
print(f"requires_grad: {x.requires_grad}")   # False
```

### 1.3.1 shape —— 最重要的属性

`shape` 的使用频率是其他三个属性之和的 10 倍。

```python
# %%
# 不同维度的 shape
scalar = torch.tensor(3.14)
print(f"标量 shape:    {scalar.shape}")       # torch.Size([])

vector = torch.tensor([1.0, 2.0, 3.0])
print(f"向量 shape:    {vector.shape}")       # torch.Size([3])

matrix = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
print(f"矩阵 shape:    {matrix.shape}")       # torch.Size([3, 2])

t3d = torch.randn(2, 3, 4)
print(f"3D Tensor:     {t3d.shape}")          # torch.Size([2, 3, 4])
```

**shape 的"总元素数"：**

```python
# %%
print(f"[2, 3, 4] 总元素数: {2 * 3 * 4}")    # 24

# 也可以用 .numel()
print(f".numel(): {t3d.numel()}")              # 24

# 或者 len() —— 返回第 0 维的大小
print(f"len(t3d): {len(t3d)}")                 # 2
```

### 1.3.2 dtype —— 数据类型

```python
# %%
# PyTorch 会根据输入自动推断 dtype
f32 = torch.tensor([1.0, 2.0])        # 浮点数 → float32（默认）
f64 = torch.tensor([1.0, 2.0], dtype=torch.float64)
i64 = torch.tensor([1, 2])            # 整数 → int64（默认）
i32 = torch.tensor([1, 2], dtype=torch.int32)

print(f"浮点默认: {f32.dtype}")       # torch.float32
print(f"指定 f64: {f64.dtype}")       # torch.float64
print(f"整数默认: {i64.dtype}")       # torch.int64

# 类型转换
f32_to_f64 = f32.to(torch.float64)
f32_to_i64 = f32.to(torch.int64)      # 注意：小数部分被截断！
print(f"float32 → int64: {f32_to_i64}")  # [1, 2]
```

**为什么神经网络几乎总是用 `float32`？**

```
float64（双精度）：更精确，但占用内存翻倍，计算更慢
float32（单精度）：精度足够，速度快，GPU 原生支持最好
float16（半精度）：更快更省内存，但精度可能不够，需要特殊处理

结论：除非你有明确的理由，否则永远用 float32。
```

**dtype 对照表：**

| PyTorch dtype | 含义 | 占用字节 | 常用场景 |
|--------------|------|---------|---------|
| `torch.float32` / `torch.float` | 32 位浮点 | 4 | 神经网络默认 |
| `torch.float64` / `torch.double` | 64 位浮点 | 8 | 科学计算 |
| `torch.float16` / `torch.half` | 16 位浮点 | 2 | 混合精度训练 |
| `torch.int64` / `torch.long` | 64 位整数 | 8 | 标签/索引 |
| `torch.int32` / `torch.int` | 32 位整数 | 4 | 一般整数 |
| `torch.bool` | 布尔值 | 1 | 掩码（mask） |

### 1.3.3 device —— 数据在哪儿

```python
# %%
cpu_tensor = torch.tensor([1.0, 2.0])
print(f"默认设备: {cpu_tensor.device}")   # cpu

# 如果有 GPU：
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to('cuda')
    print(f"转移到: {gpu_tensor.device}")  # cuda:0

    # 回到 CPU
    back_to_cpu = gpu_tensor.to('cpu')
    print(f"转回: {back_to_cpu.device}")   # cpu
```

**现在只需要记住**：Tensor 默认在 CPU 上。第十三章会详细讲 GPU。

**device 的常见值**：
- `device(type='cpu')` → CPU
- `device(type='cuda', index=0)` → 第 0 块 GPU
- `device(type='cuda', index=1)` → 第 1 块 GPU

### 1.3.4 requires_grad —— 需要梯度吗

```python
# %%
# 默认是 False
x = torch.tensor([1.0, 2.0])
print(f"默认: {x.requires_grad}")  # False

# 创建时设置
w = torch.tensor([1.0, 2.0], requires_grad=True)
print(f"需要梯度: {w.requires_grad}")  # True

# 创建后修改
x.requires_grad_(True)  # _ 后缀表示原地修改
print(f"修改后: {x.requires_grad}")  # True
```

> 第二章会详细讲 `requires_grad`。现在只需要知道：**神经网络参数设为 True，输入数据设为 False。**

### 1.3.5 四个属性——一张图总结

```
    当你看到一个 Tensor x 时，在脑中问自己四个问题：

    ┌──────────────┐
    │ x.shape      │ → 它长什么样？   → [2, 3, 4]
    │ x.dtype      │ → 它是什么类型？ → float32
    │ x.device     │ → 它在哪儿？     → cpu
    │ x.requires_grad│→ 要追踪梯度吗？→ False
    └──────────────┘

    这四个属性是 Tensor 的"身份证"
```

---

## 1.4 Tensor 的基本操作

### 1.4.1 索引与切片 —— 和 NumPy/列表完全一样

如果你会 Python 列表的索引，你就会 Tensor 的索引。语法一摸一样。

```python
# %%
import torch

x = torch.tensor([
    [1.0,  2.0,  3.0,  4.0],
    [5.0,  6.0,  7.0,  8.0],
    [9.0, 10.0, 11.0, 12.0]
])
print(f"x shape: {x.shape}")  # [3, 4]
print(f"x:\n{x}\n")

# --- 基础索引 ---
print(f"x[0]:      {x[0]}")       # 第 0 行 → [1., 2., 3., 4.]
print(f"x[1, 2]:   {x[1, 2]}")    # 第 1 行，第 2 列 → 7.0
print(f"x[-1, -1]: {x[-1, -1]}")  # 最后一行，最后一列 → 12.0

# --- 切片 ---
print(f"\nx[:2]:\n{x[:2]}")       # 前两行
print(f"\nx[:, 1:3]:\n{x[:, 1:3]}")  # 所有行，第 1~2 列
print(f"\nx[1:, :2]:\n{x[1:, :2]}")  # 第 1 行起，前两列

# --- 花式索引 ---
print(f"\nx[[0, 2]]:\n{x[[0, 2]]}")  # 取第 0 行和第 2 行
```

**索引规则速查**：

```
x[start:end:step, start:end:step, ...]

- start 省略 → 从 0 开始
- end 省略 → 到末尾
- step 省略 → 步长为 1
- 负数 → 从末尾倒数

示例：
x[1:]    → 第 1 行到末尾
x[:, :2] → 所有行，前两列
x[::-1]  → 反转第 0 维
```

### 1.4.2 逐元素运算 —— 对应位置各自计算

```python
# %%
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a + b  = {a + b}")      # [5.,  7.,  9.]
print(f"a - b  = {a - b}")      # [-3., -3., -3.]
print(f"a * b  = {a * b}")      # [4., 10., 18.]   ← 注意！这是逐元素乘
print(f"a / b  = {a / b}")      # [0.25, 0.40, 0.50]
print(f"a ** 2 = {a ** 2}")     # [1., 4., 9.]
print(f"a > 2  = {a > 2}")      # [False, False, True]
```

**逐元素运算的规则**：两个 Tensor 的 shape 必须可以广播（见 1.6 节），结果 shape = 广播后的 shape。

### 1.4.3 矩阵乘法 —— ⚠️ 最容易搞混的地方！

这是初学 PyTorch 最容易踩的坑。请仔细看。

**逐元素乘法 (`*`) 和矩阵乘法 (`@`) 是完全不同的两个东西：**

```python
# %%
A = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])  # shape: [2, 3]

B = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])  # shape: [3, 2]

# ❌ A * B → 报错！[2,3] 和 [3,2] 不能做逐元素乘法
# print(A * B)   # RuntimeError!

# ✅ A @ B → 矩阵乘法
C = A @ B
print(f"A @ B:\n{C}")
print(f"结果 shape: {C.shape}")  # [2, 2]
```

**用图理解矩阵乘法的 shape 变化：**

```
    A          @       B         =        C
  [2, 3]      ×     [3, 2]      =     [2, 2]
    ↑  └─────────────┘  ↑                ↑
    └───────────────────┘                │
         内维必须相等                    结果取外维
```

**规则**：`[m, n] @ [n, p] → [m, p]`

- 第一个矩阵的**列数**必须等于第二个矩阵的**行数**
- 结果的行数 = 第一个矩阵的行数
- 结果的列数 = 第二个矩阵的列数

```python
# %%
# 更多矩阵乘法示例
A = torch.randn(2, 3)
B = torch.randn(3, 4)

C = A @ B              # [2, 3] @ [3, 4] → [2, 4]  ✅
C = torch.matmul(A, B) # 和 @ 完全等价

# 也可以用 torch.mm 但只适用于 2D 矩阵
C = torch.mm(A, B)     # 同 @，但输入必须是 2D
```

**三个乘法函数对比：**

| 函数 | 适用维度 | 说明 |
|------|---------|------|
| `A @ B` / `torch.matmul(A, B)` | 任意维 | **推荐使用这个。** 自动处理 batch 维度 |
| `torch.mm(A, B)` | 仅 2D | 严格的矩阵乘法，输入必须是 2 维 |
| `torch.bmm(A, B)` | 仅 3D | batch 矩阵乘法，输入必须是 3 维 |
| `A * B` | 相同 shape | 逐元素乘法，和矩阵乘法完全不同 |

> **一个口诀**：`*` 是"各管各"，`@` 是"行乘列"。

### 1.4.4 矩阵乘法 vs 逐元素乘法 —— 一张图彻底搞懂

```
假设 A = [[1, 2],       B = [[5, 6],
         [3, 4]]             [7, 8]]

逐元素乘法 A * B：
    [1×5, 2×6]   =   [5,  12]
    [3×7, 4×8]       [21, 32]
    ↑ 对应位置各自乘

矩阵乘法 A @ B：
    [1×5+2×7, 1×6+2×8]   =   [19, 22]
    [3×5+4×7, 3×6+4×8]       [43, 50]
    ↑ 行×列，求和
```

---

## 1.5 shape 操作 —— 重点中的重点

在深度学习中，你几乎每写一行代码都在改变 Tensor 的 shape。这一节是**整个第一章最重要的部分**。

### 1.5.0 先建立直觉：shape 变换到底在做什么

```
shape 变换 = 把同一批数据，用不同的"视角"来看

类比：你有 12 个苹果。你可以这样排列它们：
    - 1 排 12 列  → shape [12]
    - 3 排 4 列    → shape [3, 4]
    - 2 层 × 2 排 × 3 列 → shape [2, 2, 3]

苹果还是那 12 个苹果——只是排列方式变了。
Tensor 的 shape 变换也是同理：元素不变，排列变。
```

### 1.5.1 view() —— 改变视角

`view` 是最高效的 shape 变换——它不复制数据，只改变"怎么看"数据。

```python
# %%
import torch

# 创建 12 个元素
x = torch.arange(12)   # [0, 1, 2, ..., 11]   shape: [12]
print(f"原始: {x}")
print(f"shape: {x.shape}\n")

# 用不同方式"看"这 12 个元素
print(f"view(3, 4):\n{x.view(3, 4)}\n")     # 3行4列
print(f"view(4, 3):\n{x.view(4, 3)}\n")     # 4行3列
print(f"view(2, 2, 3):\n{x.view(2, 2, 3)}\n")  # 2×2×3
```

**view 的核心规则只有一个：元素总数不能变！**

```
[12] → view(3, 4)    ✅  3×4 = 12
[12] → view(2, 2, 3) ✅  2×2×3 = 12
[12] → view(3, 5)    ❌  3×5 = 15 ≠ 12
```

**自动推断维度：`-1`**

你不需要手动计算所有维度。用一个 `-1`，PyTorch 帮你算：

```python
# %%
x = torch.arange(24)

# "24 个元素，排成 3 行，列数你帮我算"
y = x.view(3, -1)
print(f"view(3, -1) → shape: {y.shape}")    # [3, 8]  ← 24/3 = 8

# "24 个元素，排成 2 个块，每个块 3 行，剩下的你算"
z = x.view(2, 3, -1)
print(f"view(2, 3, -1) → shape: {z.shape}")  # [2, 3, 4]  ← 24/(2×3) = 4

# ⚠️ 只能有一个 -1
# x.view(-1, -1)  # ❌ 报错！只有一个维度可以推断
```

### 1.5.2 reshape() —— 更灵活的变形

```python
# %%
x = torch.arange(12)

# reshape 和 view 用法几乎一样
y = x.reshape(3, 4)
print(f"reshape(3, 4):\n{y}")
```

**view vs reshape：什么时候用哪个？**

```
┌──────────┬─────────────────┬──────────────────────┐
│          │ view             │ reshape              │
├──────────┼─────────────────┼──────────────────────┤
│ 内存     │ 共享原数据       │ 可能复制数据          │
│ 速度     │ 更快（不复制）    │ 可能稍慢              │
│ 要求     │ 数据必须是连续的   │ 不要求连续            │
│ 安全性   │ 可能报错          │ 总是能成功            │
└──────────┴─────────────────┴──────────────────────┘

初学者的最佳策略：
    优先用 reshape —— 它总能工作，不会因为"数据不连续"而报错
    当你需要极致性能且确定数据连续时，用 view
```

```python
# %%
# transpose 后的 Tensor 可能不连续，view 会报错
x = torch.randn(2, 3)
y = x.transpose(0, 1)   # y 不连续
# y.view(-1)            # ❌ 报错！
print(y.reshape(-1))    # ✅ 正常工作
print(y.contiguous().view(-1))  # ✅ 先变连续再用 view
```

### 1.5.3 squeeze() / unsqueeze() —— 处理大小为 1 的维度

这是深度学习中最常用的两个 shape 操作之一。

**unsqueeze：添加一个大小为 1 的维度**

```python
# %%
x = torch.tensor([1.0, 2.0, 3.0])
print(f"原始 shape: {x.shape}")    # [3]

# 在第 0 维前面加一个维度
a = x.unsqueeze(0)
print(f"\nunsqueeze(0):")
print(f"  shape: {a.shape}")       # [1, 3]
print(f"  值: {a}")                # [[1., 2., 3.]]

# 在第 1 维前面加一个维度
b = x.unsqueeze(1)
print(f"\nunsqueeze(1):")
print(f"  shape: {b.shape}")       # [3, 1]
print(f"  值:\n{b}")               # [[1.], [2.], [3.]]
```

**用图理解 unsqueeze：**

```
原始：[1., 2., 3.]  →  shape = [3]

unsqueeze(0)：在"外面"加一层
    ┌─────────────────┐
    │ [1., 2., 3.]    │  ← 这整个东西被包在一个新维度里
    └─────────────────┘
    shape = [1, 3]   →  1 个样本，3 个特征

unsqueeze(1)：在每个元素外面加一层
    [1.] [2.] [3.]
     ↑    ↑    ↑
    每个元素单独被包起来
    shape = [3, 1]   →  3 个样本，每个 1 个特征
```

**为什么需要 unsqueeze？**

因为模型期望输入有 batch 维度：

```python
# %%
# 单张图片推理
image = torch.randn(28, 28)          # 一张 28×28 的图
# model(image)                        # ❌ model 期望 [batch, 28, 28]
image = image.unsqueeze(0)            # 加 batch 维度
print(f"加 batch 后: {image.shape}")  # [1, 28, 28]
# model(image)                        # ✅ 现在可以了
```

**squeeze：去掉大小为 1 的维度**

```python
# %%
x = torch.randn(1, 3, 1, 4)
print(f"原始 shape: {x.shape}")       # [1, 3, 1, 4]

# 去掉所有大小为 1 的维度
y = x.squeeze()
print(f"squeeze() 后: {y.shape}")     # [3, 4]

# 只去掉指定维度（如果该维度为 1）
z = x.squeeze(0)
print(f"squeeze(0) 后: {z.shape}")    # [3, 1, 4]

# 如果指定维度不为 1，不会报错，也不会变
w = x.squeeze(1)
print(f"squeeze(1) 后: {w.shape}")    # [1, 3, 1, 4]  ← 第1维=3，不变
```

### 1.5.4 transpose() / permute() —— 维度重排

**transpose：交换两个维度**

```python
# %%
x = torch.randn(2, 3, 4)
print(f"原始 shape: {x.shape}")

y = x.transpose(0, 2)
print(f"transpose(0, 2): {y.shape}")  # [4, 3, 2]

# 等同于
z = x.permute(2, 1, 0)
print(f"permute(2, 1, 0): {z.shape}") # [4, 3, 2]
```

**permute：同时重排多个维度**

```python
# %%
x = torch.randn(2, 3, 4)  # [batch, height, width]

# 图片常用的维度变换：[H, W, C] ↔ [C, H, W]
# 假设 x 是 [batch, height, width]
# 想变成 [batch, width, height]：
y = x.permute(0, 2, 1)
print(f"permute(0, 2, 1): {y.shape}")  # [2, 4, 3]
```

### 1.5.5 shape 操作全景流程图

```
                    你的 Tensor
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    改形状但不改维度  增删维度(1)    重排维度顺序
         │              │              │
    view/reshape   squeeze/        transpose/
                   unsqueeze       permute
         │              │              │
    元素总数不变    只在大小为1     维度顺序改变
                   的维度上操作    元素位置改变
                        │
                所有操作都不改变元素总数！
                （除非你刻意删除了元素）
```

---

## 1.6 Broadcast 广播机制 —— PyTorch 的隐形助手

广播是 PyTorch 中最优雅的设计之一，也是最容易被滥用的特性。

### 1.6.1 什么是广播？—— 用直觉理解

```python
# %%
import torch

# 你有一批数据，每行是一个样本
data = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])  # shape: [3, 3]  → 3 个样本，3 个特征

# 你想给每个特征加上不同的偏移
offsets = torch.tensor([10.0, 20.0, 30.0])  # shape: [3]

# 直接加？
result = data + offsets
print(result)
# tensor([[11., 22., 33.],
#         [14., 25., 36.],
#         [17., 28., 39.]])
```

**发生了什么？**

offsets 的 shape 是 `[3]`，data 的 shape 是 `[3, 3]`。按理说不能直接加。但 PyTorch 自动把 `[3]` "广播"成了 `[3, 3]`：

```
offsets 原始:    [10, 20, 30]          shape: [3]
                   ↓ 广播（复制 3 行）
offsets 广播后:  [[10, 20, 30],
                  [10, 20, 30],
                  [10, 20, 30]]        shape: [3, 3]

然后：
    data     +  offsets(广播后)  =  result
    [1,2,3]  +  [10,20,30]     =  [11,22,33]
    [4,5,6]  +  [10,20,30]     =  [14,25,36]
    [7,8,9]  +  [10,20,30]     =  [17,28,39]
```

**广播 = PyTorch 自动帮你把较小的 Tensor "扩展"到和较大的 Tensor 相同的 shape，然后再做逐元素运算。**

### 1.6.2 广播的规则 —— 三步判断法

判断两个 shape 能否广播，从**最后一个维度**往前比较：

```
规则：从后往前逐维比较，满足以下任一条件即可：
    1. 两个维度相等
    2. 其中一个维度是 1
    3. 其中一个维度不存在（短的 Tensor 缺少该维度）

如果以上三条都不满足 → 不能广播 → 报错。
```

**示例逐个分析：**

```
[8, 3] + [3]
         从后往前：
         第 -1 维：3 vs 3 → 相等 ✅
         第 -2 维：8 vs (无) → 规则 3，不存在  ✅
         结论：可以广播。结果 shape：[8, 3]

[8, 3] + [8]
         从后往前：
         第 -1 维：3 vs 8 → 不等，且都不为 1  ❌
         结论：不能广播。报错！

[8, 3] + [8, 1]
         从后往前：
         第 -1 维：3 vs 1 → 规则 2 ✅
         第 -2 维：8 vs 8 → 相等 ✅
         结论：可以广播。结果 shape：[8, 3]

[2, 3, 4] + [4]
         从后往前：
         第 -1 维：4 vs 4 → 相等 ✅
         第 -2 维：3 vs (无) → 规则 3 ✅
         第 -3 维：2 vs (无) → 规则 3 ✅
         结论：可以广播。结果 shape：[2, 3, 4]

[2, 3, 4] + [3, 4]
         从后往前：
         第 -1 维：4 vs 4 → 相等 ✅
         第 -2 维：3 vs 3 → 相等 ✅
         第 -3 维：2 vs (无) → 规则 3 ✅
         结论：可以广播。结果 shape：[2, 3, 4]

[2, 3, 4] + [2, 1, 4]
         从后往前：
         第 -1 维：4 vs 4 → 相等 ✅
         第 -2 维：3 vs 1 → 规则 2 ✅
         第 -3 维：2 vs 2 → 相等 ✅
         结论：可以广播。结果 shape：[2, 3, 4]
```

### 1.6.3 常见广播场景

```python
# %%
# 场景 1：给矩阵每列加不同值
X = torch.randn(4, 5)           # 4 样本，5 特征
col_offset = torch.randn(5)     # 5 个偏移
X_centered = X - col_offset     # 广播 [4,5] - [5] → [4,5]

# 场景 2：给矩阵每行加不同值
X = torch.randn(4, 5)
row_offset = torch.randn(4, 1)  # 4 个偏移，每个是 [1]
X_adjusted = X + row_offset     # 广播 [4,5] + [4,1] → [4,5]

# 场景 3：减去 batch 均值
batch = torch.randn(32, 10)     # 32 个样本，10 维
batch_mean = batch.mean(dim=0)  # [10] —— 每个特征的均值
batch_centered = batch - batch_mean  # [32,10] - [10] → [32,10]
```

### 1.6.4 广播的陷阱 —— 能广播 ≠ 计算有意义

**最大的陷阱：广播成功了，但你的意图是错的。**

```python
# %%
# 假设你在做分类任务
logits = torch.randn(32, 10)       # 32 个样本，10 个类别的分数
labels = torch.randint(0, 10, (32,))  # 32 个真实标签（整数 0-9）

# ❌ 错误！这会广播但完全没有意义
# result = logits + labels
# [32, 10] + [32] → 可以广播
# 但这只是把标签值加到了每列的分数上，毫无意义

# ✅ 正确的做法是用 CrossEntropyLoss（第七章）
```

> **检查清单**：每次你看到两个 shape 不同的 Tensor 在做运算，问自己："这个广播产生的结果是我想要的吗？"

---

## 1.7 综合实战 —— 把本章所有知识串联起来

让我们做一个完整的练习：模拟一批 MNIST 图片数据（32 张 28×28 灰度图），并进行常见的数据处理操作。

```python
# %%
import torch

# --- 1. 创建数据 ---
# 32 张 28×28 的灰度图
batch_size = 32
height, width = 28, 28
images = torch.randn(batch_size, height, width)
print(f"Step 1 - 原始图片 batch:")
print(f"  shape: {images.shape}")          # [32, 28, 28]
print(f"  dtype: {images.dtype}")          # torch.float32
print(f"  device: {images.device}")        # cpu
print(f"  requires_grad: {images.requires_grad}")  # False

# --- 2. 添加通道维度 ---
# 灰度图只有 1 个通道，但 PyTorch 通常期望 [batch, channels, height, width]
images = images.unsqueeze(1)               # 在第 1 维插入通道
print(f"\nStep 2 - 添加通道维度:")
print(f"  shape: {images.shape}")          # [32, 1, 28, 28]

# --- 3. 展平为向量 ---
# 全连接网络期望 [batch, features]
images_flat = images.view(batch_size, -1)  # 展平
print(f"\nStep 3 - 展平:")
print(f"  shape: {images_flat.shape}")     # [32, 784]
# 28×28×1 = 784

# --- 4. 模拟一个线性层的权重 ---
# W: [784, 128]（输入 784 → 输出 128）
W = torch.randn(784, 128)
# b: [128]（每个输出特征一个偏置）
b = torch.zeros(128)

# --- 5. 前向传播 ---
# y = X @ W + b
# [32, 784] @ [784, 128] + [128]
# → [32, 128] + [128]  (广播！)
# → [32, 128]
output = images_flat @ W + b
print(f"\nStep 5 - 模拟一层 Linear:")
print(f"  X @ W → [32, 784] @ [784, 128] = [32, 128]")
print(f"  + b [128] → (广播) → [32, 128]")
print(f"  最终 shape: {output.shape}")     # [32, 128]

# --- 6. 取第一个样本的前 5 个输出 ---
first_sample = output[0]                   # shape: [128]
first_5 = first_sample[:5]                # shape: [5]
print(f"\nStep 6 - 取第一个样本的前 5 个输出:")
print(f"  first_sample shape: {first_sample.shape}")
print(f"  first_5 shape: {first_5.shape}")
print(f"  first_5 values: {first_5}")

# --- 7. Reshape 回去 ---
# 把输出变回图片形状 [32, 1, 28, 28]...等等，输出是 [32, 128] 不是 [32, 784]
# 这不能 reshape 成图片。但如果可以：
images_reconstructed = images_flat.view(32, 1, 28, 28)
print(f"\nStep 7 - Reshape 回图片:")
print(f"  reconstructed shape: {images_reconstructed.shape}")
print(f"  和原始相同: {torch.equal(images, images_reconstructed)}")  # True

print("\n" + "=" * 50)
print("综合实战完成！你刚刚走完了一次完整的数据流。")
print("从 创建 → 改变shape → 矩阵乘法 → 广播 → 索引 → reshape")
print("=" * 50)
```

---

## 1.8 本章练习

### 练习 1-1：创建 Tensor（热身）

```python
# 不查文档，在 3 分钟内完成以下创建：
# 1. 创建一个值为 [0.5, 1.5, 2.5, 3.5, 4.5] 的 float32 Tensor
# 2. 创建一个 2×4 的全 1 Tensor
# 3. 创建一个 3×3 的 [0,1) 均匀分布随机 Tensor
# 4. 创建一个 2×5 的标准正态分布随机 Tensor
# 5. 创建一个 5×5 的单位矩阵
# 6. 创建从 3 到 15（不含），步长为 3 的等差 Tensor
# 7. 创建从 -1 到 1 之间均匀取 9 个点的 Tensor
```

### 练习 1-2：shape 操作

```python
# 对 x = torch.arange(24) 做以下操作：
# 1. reshape 成 [2, 3, 4]
# 2. reshape 成 [2, 12]
# 3. reshape 成 [4, -1]（自动推断）
# 4. 创建 y = torch.randn(3, 1, 5)，squeeze 掉所有大小为 1 的维度
# 5. 创建 z = torch.tensor([1., 2., 3., 4., 5.])，分别 unsqueeze(0) 和 unsqueeze(1)
# 6. 创建 w = torch.randn(2, 3, 4)，用 transpose 交换第 0 维和第 2 维
# 7. 用 permute 把 w 变成 [4, 3, 2]
```

### 练习 1-3：矩阵乘法

```python
# 不运行代码，先写出结果 shape，再验证：
# 1. [5, 3] @ [3, 2] → ?
# 2. [2, 4] @ [3, 4] → 合法吗？
# 3. [8, 2] @ [2, 1] → ?
# 4. [2, 3, 5] @ [5, 4] → ?
```

### 练习 1-4：广播判断

```python
# 判断以下每组运算能否广播，如果能，写出结果 shape：
# 1. [5, 3] + [3]
# 2. [5, 3] + [5]
# 3. [5, 3] + [5, 1]
# 4. [2, 3, 4] + [4]
# 5. [2, 3, 4] + [3, 4]
# 6. [2, 3, 4] + [2, 1, 1]
# 7. [2, 3, 4] + [2, 5, 4]
# 8. [16, 3, 32, 32] + [3, 1, 1]
```

### 练习 1-5：综合练习——模拟图片 batch

```python
# 创建并操作一个"假图片 batch"：
# 1. 创建 torch.randn(16, 3, 32, 32) —— 16 张 RGB 32×32 图片
# 2. 取第 5 张图片（索引 4），打印它的 shape
# 3. 取第 5 张图片的 G 通道，打印它的 shape
# 4. 将所有图片 resize（展平）为 [16, 3*32*32]
# 5. 模拟一个 Linear 层：输入 3072 → 输出 512
#    手动创建 W [3072, 512] 和 b [512]，计算 X @ W + b
# 6. 打印输出 shape，应该是 [16, 512]
```

### 练习 1-6：不看答案练习（核心检验）

> **这是最重要的练习。** 关闭所有文档、浏览器标签页，在 10 分钟内独立完成：

```python
# 1. 创建 shape 为 [3, 5] 的随机标准正态 Tensor
# 2. 打印它的 shape、dtype、device、requires_grad
# 3. 取前 2 行、后 3 列
# 4. reshape 为 [5, 3]
# 5. 创建一个 [3] 的 Tensor，加到原始 [3, 5] 的每一列上
# 6. 创建 [5, 2] 的随机矩阵 W，计算原始数据 @ W
# 7. 把 [3, 5] 的最后一个维度加一个大小为 1 的维度（变成 [3, 5, 1]）
```

### 练习 1-7：代码重构

```python
# 以下代码有大量重复的 print 语句。请写一个函数 print_info(tensor, name)
# 来消除重复。

x = torch.randn(2, 3)
print(f"x shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")

y = x.view(3, 2)
print(f"y shape: {y.shape}, dtype: {y.dtype}, device: {y.device}")

z = y.unsqueeze(0)
print(f"z shape: {z.shape}, dtype: {z.dtype}, device: {z.device}")

# 重构后应能这样调用：
# print_info(x, "x")
# print_info(y, "y")
# print_info(z, "z")
```

### 练习 1-8：Debug 练习

```python
# 找出以下每段代码的错误，不运行代码，用大脑推理：

# Bug 1:
x = torch.arange(10)          # 有 10 个元素
y = x.view(3, 3)              # 错在哪？

# Bug 2:
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = a @ b                     # 错在哪？

# Bug 3:
a = torch.randn(4, 5)
b = torch.randn(3, 5)
c = a + b                     # 错在哪？

# Bug 4:
x = torch.randn(2, 3)
y = x.transpose(0, 1)
z = y.view(-1)                # 错在哪？（提示：transpose 后的内存布局）

# Bug 5:
x = torch.randn(2, 1, 3)
y = x.squeeze()               # squeeze 会去掉哪些维度？结果 shape 是什么？
# 如果期望的结果是 [2, 3] 但实际得到了什么？
```

### 练习 1-9：思维导图

拿出一张纸（不是电脑，是纸和笔），画出本章的核心概念关系图，至少包含：

```
Tensor → 创建(6种方法) → 属性(4个) → 操作(索引/运算)
       → shape变换(view/reshape/squeeze/unsqueeze/transpose/permute)
       → 广播机制(3条规则)
       → 矩阵乘法 vs 逐元素乘法
```

---

## 答案与提示

> **⚠️ 请先独立完成所有练习，再看答案。** 如果你直接看答案，这一章就白学了。

<details>
<summary>练习 1-1 答案</summary>

```python
import torch

# 1
t1 = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])

# 2
t2 = torch.ones(2, 4)

# 3
t3 = torch.rand(3, 3)

# 4
t4 = torch.randn(2, 5)

# 5
t5 = torch.eye(5)

# 6
t6 = torch.arange(3, 15, 3)  # [3, 6, 9, 12]

# 7
t7 = torch.linspace(-1, 1, 9)
```
</details>

<details>
<summary>练习 1-3 答案</summary>

```
1. [5, 3] @ [3, 2] → [5, 2] ✅  (内维 3=3)
2. [2, 4] @ [3, 4] → ❌  (内维 4≠3，且 [3,4] 的转置是 [4,3]，需要用 .t())
3. [8, 2] @ [2, 1] → [8, 1] ✅  (内维 2=2)
4. [2, 3, 5] @ [5, 4] → [2, 3, 4] ✅  (batch matmul)
```
</details>

<details>
<summary>练习 1-4 答案</summary>

```
1. [5, 3] + [3] → ✅ [5, 3]
2. [5, 3] + [5] → ❌ (3 vs 5，不等且都不为 1)
3. [5, 3] + [5, 1] → ✅ [5, 3]
4. [2, 3, 4] + [4] → ✅ [2, 3, 4]
5. [2, 3, 4] + [3, 4] → ✅ [2, 3, 4]
6. [2, 3, 4] + [2, 1, 1] → ✅ [2, 3, 4]
7. [2, 3, 4] + [2, 5, 4] → ❌ (3 vs 5，不等且都不为 1)
8. [16, 3, 32, 32] + [3, 1, 1] → ✅ [16, 3, 32, 32]
```
</details>

<details>
<summary>练习 1-8 答案</summary>

```python
# Bug 1: torch.arange(10) 有 10 个元素，3×3=9 ≠ 10，元素数不匹配
# 修正：y = x[:9].view(3, 3) 或 y = x.view(2, 5)

# Bug 2: [3,4] @ [3,4] 不合法，矩阵乘法要求 a 的列数 = b 的行数
# 修正：b = torch.randn(4, 3)，然后 a @ b → [3, 3]

# Bug 3: [4,5] + [3,5] 不能广播，第0维 4 ≠ 3 且都不为 1
# 修正：b = torch.randn(4, 5) 或 b = torch.randn(1, 5)

# Bug 4: transpose 后的 Tensor 不连续，view 会报错
# 修正：y.contiguous().view(-1) 或 y.reshape(-1)

# Bug 5: x.squeeze() 会去掉第1维(大小为1)，结果 [2, 3]
# 但 squeeze() 去掉所有大小为1的维度，如果第0维也是1就也会去掉
# 安全做法：x.squeeze(1) 明确指定去哪个维度
```
</details>

---

> **下一步**：Tensor 的操作能闭眼写了吗？如果能，进入[第二章：自动求导 autograd](./02_autograd.md)。如果不能，把练习 1-6 再做一遍。
