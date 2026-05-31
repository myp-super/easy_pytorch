# 第十八章：RNN / LSTM — 序列建模

## 18.0 本章导引

前三章处理的是图片——固定大小的数据。但世界上很多数据是**序列**：
- 文本：一串词 "I love deep learning"
- 时间序列：每天的股价 [100, 102, 99, 105, ...]
- 语音：一段音频信号

序列的核心特征：**顺序重要。** "猫咬狗"和"狗咬猫"完全不同。

CNN 和全连接网络无法自然处理变长序列。本章学会如何处理它们。

---

## 18.1 为什么序列需要特殊架构

### 18.1.1 全连接网络处理序列的问题

```
文本："I love deep learning"（4 个词）

全连接：
    把 4 个词拼接成一个大向量 → 过 Linear
    问题：句子长度不同，向量大小不同，Linear 的 in_features 必须固定
    解决：截断或填充 → 信息丢失或浪费

    更严重的问题：全连接没有"记忆"——
    处理 "love" 时，不知道前面有个 "I"
```

### 18.1.2 RNN 的直觉：有"记忆"的网络

```
    RNN 处理序列：
    "I"     → 网络产生输出 + "记忆状态" h₁
    "love"  → 网络接收 "love" + 上一个状态 h₁ → 产生输出 + 新状态 h₂
    "deep"  → 接收 "deep" + h₂ → 输出 + h₃
    "learning" → 接收 "learning" + h₃ → 最终输出

    每个词的输出都"知道"前面发生了什么。
```

### 18.1.3 用图理解

```
      x₀="I"      x₁="love"    x₂="deep"    x₃="learning"
        │            │            │            │
        ▼            ▼            ▼            ▼
    ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐
    │ RNN   │───→│ RNN   │───→│ RNN   │───→│ RNN   │
    │ cell  │ h₁ │ cell  │ h₂ │ cell  │ h₃ │ cell  │
    └──┬────┘    └──┬────┘    └──┬────┘    └──┬────┘
       │            │            │            │
       ▼            ▼            ▼            ▼
      y₀           y₁           y₂           y₄（最终输出）
```

---

## 18.2 RNN 的问题：梯度消失

```
RNN 在反向传播时，梯度要穿过整个时间序列。
如果序列很长（100 步），梯度在传播中不断乘以权重：

    如果权重 < 1 → 梯度指数衰减 → 前面的词"学不到" → 梯度消失
    如果权重 > 1 → 梯度指数爆炸 → 数值溢出 → 梯度爆炸

    就像传话游戏：一句话传 100 个人，已经面目全非。
```

---

## 18.3 LSTM — 长短期记忆

LSTM 解决了 RNN 的梯度消失问题。它引入了"门控机制"和一条贯穿时间的"记忆高速公路"。

### 18.3.1 LSTM 的核心思想——三个门

```
    LSTM 有两条线在时间上流动：
    
    细胞状态 C（长期记忆）：
        像一条高速公路，信息可以直接流过整个序列
        梯度不需要经过复杂的计算就能回流
        
    隐藏状态 h（短期记忆/输出）：
        从 C 中"提取"当前需要的信息

    三个"门"控制信息的流动：
        遗忘门：从 C 中丢弃什么信息？
        输入门：往 C 中加入什么新信息？
        输出门：从 C 中取出什么作为当前输出？
```

### 18.3.2 在 PyTorch 中使用 LSTM

```python
# %%
import torch
import torch.nn as nn

# nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
lstm = nn.LSTM(
    input_size=100,       # 每个时间步的输入维度（如 word embedding 维度）
    hidden_size=256,      # 隐藏状态 h 的维度（"记忆"的容量）
    num_layers=2,         # 堆叠几层 LSTM
    batch_first=True      # 输入 shape：(batch, seq_len, input_size)
)

# 输入：4 个句子，每句 10 个词，每个词用 100 维向量表示
x = torch.randn(4, 10, 100)    # [batch, seq_len, input_size]

# 输出
output, (h_n, c_n) = lstm(x)

print(f"output shape: {output.shape}")   # [4, 10, 256]  ← 每个时间步的输出
print(f"h_n shape:    {h_n.shape}")       # [2, 4, 256]   ← 最后时间步的隐藏状态
print(f"c_n shape:    {c_n.shape}")       # [2, 4, 256]   ← 最后时间步的细胞状态
```

---

## 18.4 实战：文本情感分析

```python
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# === 极简 IMDb 风格情感分类 ===
# 假设已做预处理：句子 → 整数索引序列 → padding 到相同长度

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 词向量
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # ×2 因为双向
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: [batch, seq_len] —— 整数索引序列
        x = self.embedding(x)                    # [batch, seq_len, embedding_dim]
        output, (h_n, c_n) = self.lstm(x)        # output: [batch, seq_len, hidden*2]
        
        # 取最后一个时间步的输出（或取两个方向的最后状态拼接）
        last_output = output[:, -1, :]           # [batch, hidden*2]
        x = self.dropout(last_output)
        x = self.fc(x)                           # [batch, 2]
        return x

# 示例参数
# vocab_size = 10000   # 词汇量
# embedding_dim = 100   # 每个词的向量维度
# hidden_dim = 256      # LSTM 隐藏层大小
```

---

## 18.5 本章总结

```
    RNN：有"记忆"的网络，适合序列数据
    问题：长序列 → 梯度消失

    LSTM：用"门控机制"解决梯度消失
        三个门：遗忘、输入、输出
        细胞状态 C：高速公路，信息直接流过

    PyTorch：nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    输入：[batch, seq_len, input_size]
    输出：[batch, seq_len, hidden_size * (2 if bidirectional)]

    适用场景：文本分类、情感分析、时间序列预测、机器翻译
```

---

## 18.6 本章练习

### 练习 18-1：理解 LSTM 输入输出

```python
# 创建一个 LSTM(100, 256, num_layers=3, batch_first=True)
# 输入 torch.randn(8, 20, 100)（8 条，每条 20 个时间步，每步 100 维）
# 打印 output, h_n, c_n 的 shape
```

### 练习 18-2：简单序列分类

```python
# 生成模拟数据：10 类序列，每类有不同模式
# 训练 LSTM 分类
# 目标准确率 ≥ 80%
```

### 练习 18-3：双向 vs 单向 LSTM

```python
# 对比 bidirectional=True 和 False 的效果
```

### 练习 18-4：不看答案——独立实现 LSTM 分类器

> 关闭文档，独立写一个 LSTM 分类器（文本分类风格）。

---

> **下一步**：[第十九章：Transformer](./19_transformer.md)。
