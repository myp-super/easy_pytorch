# 第〇章：环境搭建与工具准备

## 0.0 本章导引

### 为什么从环境开始？

很多初学者跳过环境配置，结果第一个 `import torch` 就报错，然后花两小时在网上搜解决方案，学习热情直接凉了一半。

本章只有一个目标：**让你成功运行 `import torch` 并创建第一个 Tensor。**

本章不需要你理解任何深度学习概念。你只需要跟着操作，确保每一步都成功。

### 一个重要的心态

编程环境的配置本质上是"让不同软件之间能互相找到对方"。当你遇到报错时，不要慌——报错信息是计算机在告诉你"我哪里没找到"。学会读报错，比学会写代码更重要。

### 本章地图

```
0.1 理解虚拟环境          ← 为什么要"隔离"
0.2 创建虚拟环境            ← conda / venv 二选一
0.3 安装 PyTorch           ← CPU / GPU 二选一
0.4 配置 VS Code           ← 开发工具
0.5 验证安装              ← 第一个 Tensor
0.6 练习                  ← 制造错误 + 修复
```

---

## 0.1 为什么需要虚拟环境

### 0.1.1 一个故事帮你理解

小李在 2024 年做了一个项目，用了 PyTorch 1.8。
2025 年，他开始一个新项目，需要 PyTorch 2.5。

**如果他没有用虚拟环境：**
```
pip install torch==2.5
# PyTorch 1.8 被覆盖了！
# 回到 2024 年的项目 → 代码报错 → 因为 PyTorch API 变了
```

**如果他用了虚拟环境：**
```
项目 A 的虚拟环境：PyTorch 1.8  ← 完全独立
项目 B 的虚拟环境：PyTorch 2.5  ← 完全独立
两个项目互不影响
```

### 0.1.2 虚拟环境是什么

```
┌─────────────────────────────────────┐
│        你的操作系统                  │
│                                     │
│  ┌──────────────┐  ┌──────────────┐ │
│  │ 虚拟环境 A    │  │ 虚拟环境 B    │ │
│  │ PyTorch 1.8  │  │ PyTorch 2.5  │ │
│  │ numpy 1.21   │  │ numpy 1.26   │ │
│  │ python 3.8   │  │ python 3.10  │ │
│  └──────────────┘  └──────────────┘ │
│         ↑                 ↑         │
│    互不干扰            互不干扰       │
└─────────────────────────────────────┘
```

每个虚拟环境 = 一个独立的 Python 安装副本。库的版本、Python 的版本，都在这个"泡泡"里面。

### 0.1.3 什么时候不用虚拟环境？

**几乎没有。** 任何时候都应该用虚拟环境，即使你只做一个项目。这已经成了 Python 开发的基本规范。

---

## 0.2 创建虚拟环境

你有两个选择：**conda**（推荐初学者）和 **venv**（Python 自带）。

### 0.2.1 conda（推荐）

conda 比 pip 多做一件事：它能管理 Python 本身和非 Python 的底层库（比如 CUDA）。这对 PyTorch 特别重要，因为 GPU 版本的 PyTorch 依赖于很多底层 C++ 库。

**Step 1：安装 conda**

如果你还没有 conda，下载 Miniconda（最小安装版）：
- 官网：https://docs.conda.io/en/latest/miniconda.html
- 选 Python 3.10+ 版本
- 安装时勾选 "Add to PATH"

**Step 2：创建环境**

打开终端（Windows 按 Win+R，输入 `cmd`，回车）：

```bash
# 创建一个名为 pytorch_tutorial 的环境，Python 版本 3.10
conda create -n pytorch_tutorial python=3.10

# 你会看到 conda 列出将要安装的包
# 输入 y 确认
```

**Step 3：激活环境**

```bash
# 激活
conda activate pytorch_tutorial

# 验证：终端提示符前面应该出现 (pytorch_tutorial)
# 例如： (pytorch_tutorial) C:\Users\你的用户名>
```

**Step 4：验证 Python 版本**

```bash
python --version
# 应输出：Python 3.10.x
```

### 0.2.2 venv（Python 自带，备选）

如果你不想装 conda，Python 自带的 venv 也能用：

```bash
# Windows
python -m venv pytorch_tutorial
pytorch_tutorial\Scripts\activate

# Mac / Linux
python -m venv pytorch_tutorial
source pytorch_tutorial/bin/activate
```

### 0.2.3 关键提醒：每次都要激活

**虚拟环境不会自动激活。** 每次打开新终端，都需要重新 `conda activate` 或 `source activate`。

这是初学者最常见的"我的 import 怎么不行了"的原因——环境没激活。

---

## 0.3 安装 PyTorch

### 0.3.1 CPU 还是 GPU？

```
┌──────────────┬──────────────────────────────┐
│ CPU 版       │ GPU 版                        │
├──────────────┼──────────────────────────────┤
│ 任何电脑都能跑 │ 需要 NVIDIA 显卡               │
│ 安装简单      │ 安装复杂（需要 CUDA 驱动）       │
│ 训练较慢      │ 训练快 10-100 倍               │
│ 适合学习      │ 适合实际训练                    │
└──────────────┴──────────────────────────────┘

建议：先装 CPU 版，确定能用后再考虑 GPU 版。
第一到十二章在 CPU 上完全能跑（MNIST 也只需要几分钟）。
```

### 0.3.2 检查你的显卡

```bash
# Windows：打开终端，输入
nvidia-smi
```

如果显示显卡信息和驱动版本 → 有 NVIDIA 显卡 → 可以装 GPU 版。
如果显示 "不是内部或外部命令" → 没有 NVIDIA 显卡或没装驱动 → 装 CPU 版。

### 0.3.3 安装命令

**确保虚拟环境已激活**（提示符前有 `(pytorch_tutorial)`），然后：

```bash
# === CPU 版本（所有人适用）===
pip install torch torchvision

# === GPU 版本（有 NVIDIA 显卡的选这个）===
# CUDA 11.8（兼容性最好）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1（较新显卡）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**安装时间**：大约 5-10 分钟（取决于网速）。PyTorch 包比较大（~1-2GB）。

---

## 0.4 配置 VS Code

### 0.4.1 为什么是 VS Code

- 免费
- 有 Jupyter 插件，可以在 `.py` 文件中逐单元格运行代码（`# %%` 分隔）
- 比纯 Jupyter Notebook 更适合工程代码（文件更容易版本管理）
- 有强大的 Python 智能提示

### 0.4.2 安装和配置

**1. 安装 VS Code**

官网：https://code.visualstudio.com/ → 下载安装。

**2. 安装两个扩展**

打开 VS Code，点击左侧"扩展"图标（或按 Ctrl+Shift+X），搜索并安装：
- **Python**（Microsoft 出品）
- **Jupyter**（Microsoft 出品）

**3. 选择 Python 解释器**

```
按 Ctrl+Shift+P → 输入 "Python: Select Interpreter"
→ 选择你刚创建的 pytorch_tutorial 环境
```

如果用的是 conda，解释器路径类似：
```
C:\Users\你的用户名\miniconda3\envs\pytorch_tutorial\python.exe
```

### 0.4.3 用 `# %%` 分隔代码单元格

在 VS Code 中创建一个 `.py` 文件，用 `# %%` 可以分隔"单元格"：

```python
# %%  ← 这是一个单元格的标记
print("这是第一个单元格")

# %%  ← 另一个单元格
print("这是第二个单元格")
```

按 `Shift+Enter` 运行当前单元格并跳到下一个。这和 Jupyter Notebook 的体验完全一样，但你的代码是 `.py` 文件，方便用 git 管理。

---

## 0.5 验证安装

创建一个新文件 `verify_install.py`，写入以下内容并运行：

```python
# %%
import torch
print(f"PyTorch 版本: {torch.__version__}")

# %%
# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA 可用: {cuda_available}")
if cuda_available:
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

# %%
# 第一个 Tensor！
first_tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"第一个 Tensor: {first_tensor}")
print(f"shape:  {first_tensor.shape}")
print(f"dtype:  {first_tensor.dtype}")
print(f"device: {first_tensor.device}")

# %%
# 简单运算
a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
print(f"a + b = {a + b}")

# %%
# torchvision
import torchvision
print(f"torchvision 版本: {torchvision.__version__}")

print("\n✅ 环境搭建完成！可以开始学习第一章了。")
```

**期望输出**（版本号可能不同）：

```
PyTorch 版本: 2.x.x
CUDA 可用: True  (或 False)

第一个 Tensor: tensor([1., 2., 3.])
shape:  torch.Size([3])
dtype:  torch.float32
device: cpu

a + b = tensor([4., 6.])
torchvision 版本: 0.x.x

✅ 环境搭建完成！可以开始学习第一章了。
```

### 如果出错了？

```
┌──────────────────────────────────────────────┐
│ 常见错误                解决                   │
├──────────────────────────────────────────────┤
│ ModuleNotFoundError:    虚拟环境没激活          │
│ No module named 'torch' 或没装 PyTorch         │
│                                               │
│ import 后面什么都        Python 解释器没选对     │
│ 没发生，也不报错                              │
│                                               │
│ CUDA not available      没装 GPU 版 PyTorch    │
│                         或驱动太旧              │
│                                               │
│ pip 安装报错             检查网络，或用镜像源     │
└──────────────────────────────────────────────┘
```

> **遇到问题不要慌。** 把错误信息复制到搜索引擎，90% 的问题都有答案。这是程序员的日常。

---

## 0.6 本章练习

### 练习 0-1：环境自检

在终端中运行以下命令，确认每一步都成功：

```bash
conda activate pytorch_tutorial    # 激活环境
python --version                    # 确认 Python 版本
python -c "import torch; print(torch.__version__)"  # 确认 PyTorch 已安装
```

### 练习 0-2：修改验证脚本

在 `verify_install.py` 的基础上：
1. 额外创建一个值为 `[5.0, 6.0, 7.0, 8.0]` 的 Tensor
2. 打印它的 `shape`
3. 思考：为什么 shape 是 `torch.Size([4])` 而不是 `4`？（答案在第一章）

### 练习 0-3：故意制造错误

> 学会看报错是编程最重要的能力之一。

尝试以下操作，**认真读**每一条错误信息，尝试理解它在说什么：

```python
# 错误 1：模块名大小写错误
import torch
# import Pytorch  ← 取消注释运行，看报错

# 错误 2：属性 vs 方法
t = torch.tensor([1.0, 2.0])
print(t.shape)       # ✅ 这是属性
# print(t.shape())    # ← 取消注释运行，看报错

# 错误 3：维度不匹配
a = torch.tensor([1.0, 2.0])
b = torch.tensor([1.0, 2.0, 3.0])
# c = a + b           # ← 取消注释运行，看报错
```

### 练习 0-4：不看文档重装一遍

> 关闭所有文档，从零开始：

```bash
# 1. 删除 pytorch_tutorial 环境
conda remove -n pytorch_tutorial --all

# 2. 重新创建
conda create -n pytorch_test python=3.10

# 3. 激活
conda activate pytorch_test

# 4. 安装 PyTorch（CPU 版）
pip install torch torchvision

# 5. 验证
python -c "import torch; print(torch.__version__)"
```

如果你能不看文档独立完成以上步骤，这一章就真正学会了。

---

> **下一步**：环境就绪！进入[第一章：Tensor — 深度学习的基本语言](01_tensor.md)，开始真正的学习。
