# 第十九章：Transformer — 现代架构基础

## 19.0 本章导引

2017 年，Google 发表了一篇论文《Attention Is All You Need》，提出了 Transformer。

它彻底改变了深度学习——从 NLP 到 CV，几乎所有前沿模型都基于 Transformer：BERT、GPT、ViT、DALL·E……

LSTM 是"顺序处理"（第 1 个词 → 第 2 个词 → …），无法并行；Transformer 是"同时看所有词"，训练快得多。

本章从零实现一个 Transformer 编码器，理解 Self-Attention 的核心机制。

---

## 19.1 Self-Attention 的直觉

### 19.1.1 一句话解释

```
Self-Attention = 每个词"关注"句子中的所有其他词，计算"谁和我最相关"。

句子："The cat sat on the mat because it was tired."

"it" 指的是什么？猫？垫子？
Self-Attention 让 "it" 去"看"所有其他词，发现 "cat" 和自己最相关。
```

### 19.1.2 Query, Key, Value —— 三个核心概念

```
类比：你在图书馆找书。

    Query (Q) = 你的搜索需求："我想找关于深度学习的书"
    Key (K)   = 每本书的标签/标题
    Value (V) = 每本书的实际内容

    流程：
    1. 用你的 Query 和每本书的 Key 计算"匹配度"（点积）
    2. 匹配度 → Softmax → 注意力权重
    3. 用注意力权重对 Value 加权求和 → 最终输出

    翻译成 Self-Attention：
    每个词都"发出"一个 Query（我在找什么？）
    每个词都"提供"一个 Key（我能提供什么？）和一个 Value（我的实际内容）
```

### 19.1.3 一个具体的计算示例

```python
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设：3 个词，每个用 4 维向量表示
x = torch.randn(1, 3, 4)    # [batch, seq_len, d_model]

d_model = 4

# 创建 Q、K、V 的投影矩阵
W_q = nn.Linear(4, 4, bias=False)
W_k = nn.Linear(4, 4, bias=False)
W_v = nn.Linear(4, 4, bias=False)

Q = W_q(x)  # [1, 3, 4]  — 每个词的"查询"
K = W_k(x)  # [1, 3, 4]  — 每个词的"键"
V = W_v(x)  # [1, 3, 4]  — 每个词的"值"

# 计算注意力分数：Q 和 K 的点积
attn_scores = Q @ K.transpose(-2, -1) / (d_model ** 0.5)
# [1, 3, 3]  — 3×3 的"关系矩阵"
# attn_scores[i][j] = 词 i 对词 j 的"关注度"

print(f"注意力分数 shape: {attn_scores.shape}")
print(f"注意力分数:\n{attn_scores[0]}")

# Softmax → 注意力权重
attn_weights = F.softmax(attn_scores, dim=-1)

# 用权重对 V 加权求和
output = attn_weights @ V
print(f"输出 shape: {output.shape}")  # [1, 3, 4]
```

---

## 19.2 Multi-Head Attention

单头注意力可能只关注一种关系。Multi-Head 让模型同时从多个角度关注。

```python
# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads     # 每个头的维度
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        
        # 投影并拆分为多头
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # 每个 shape: [batch, num_heads, seq_len, d_k]
        
        # Scaled Dot-Product Attention
        attn_scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ V
        # [batch, num_heads, seq_len, d_k]
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.W_o(attn_output)
```

---

## 19.3 完整 Transformer 编码器

```python
# %%
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)         # LayerNorm（Transformer 中比 BatchNorm 更常用）
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(                   # Feed-Forward Network
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-Attention + 残差连接 + LayerNorm
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))   # ← 残差连接！
        
        # Feed-Forward + 残差连接 + LayerNorm
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4, 
                 num_classes=10, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # 分类 token
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x: [batch, seq_len] —— token 索引
        batch, seq_len = x.shape
        
        x = self.embedding(x)                           # [batch, seq_len, d_model]
        x = x + self.pos_encoding[:, :seq_len, :]       # 加入位置信息
        x = self.dropout(x)
        
        for block in self.encoder_blocks:
            x = block(x)
        
        x = x.mean(dim=1)                              # 全局平均池化
        return self.fc(x)
```

---

## 19.4 Transformer 的关键设计

```
1. Self-Attention   → 每个词关注所有其他词（核心创新）
2. Multi-Head       → 从多个角度关注（类比：多个专家同时分析）
3. 残差连接          → x + f(x)，让梯度可以直接流通（深网络必备）
4. LayerNorm        → 在特征维度做归一化（比 BatchNorm 更适合 NLP）
5. Position Encoding → 注意力本身不感知位置 → 手动注入位置信息
```

---

## 19.5 本章总结

```
    Transformer = Self-Attention + FFN + 残差 + LayerNorm

    QKV：
        Query: "我在找什么？"
        Key:   "我能提供什么？"
        Value: "我的实际内容是什么？"

    Multi-Head Attention：
        多组 QKV → 从多个角度关注 → 拼接

    残差连接：
        x = x + f(x) → 梯度高速公路 → 可以训练很深的网络
```

---

## 19.6 本章练习

### 练习 19-1：手算 Self-Attention

```python
# 创建 2 个词 × 3 维的输入，手动实现 Scaled Dot-Product Attention
# 验证和 PyTorch 的 F.scaled_dot_product_attention 结果一致
```

### 练习 19-2：理解 Multi-Head

```python
# 创建 MultiHeadAttention(d_model=16, num_heads=4)
# 输入 [2, 5, 16]，验证输出的 shape
```

### 练习 19-3：实现 Transformer 编码器

```python
# 实现一个单层的 TransformerEncoderBlock
# 验证残差连接不会改变 shape
```

### 练习 19-4：序列分类

```python
# 用 SimpleTransformer 做简单的序列分类
# 生成模拟数据，训练验证
```

### 练习 19-5：不看答案——独立实现

> 关闭文档，独立实现 MultiHeadAttention 和 TransformerEncoderBlock。

---

> **下一步**：[第二十章：PyTorch 工程化实践](20_engineering.md)。
