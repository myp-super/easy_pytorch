# 第二十一章：进阶实战项目

## 21.0 你现在的能力

完成前二十章后，你应该能：
- 独立书写全连接网络、CNN、RNN/LSTM、Transformer 的所有核心组件
- 使用 BN、Dropout、数据增强、学习率调度优化训练
- 使用 TensorBoard、AMP、梯度裁剪进行工程化训练
- 从论文中理解并实现网络架构

以下 3 个项目是你的"毕业设计"。**每个都是独立完成。**

---

## 21.1 项目 A：Kaggle 风格 CIFAR-10 分类（目标 ≥ 93%）

### 要求

使用所有学到的技术，在 CIFAR-10 上达到 ≥ 93% 准确率。

```
必须包含：
    1. 深度 CNN（至少 6 层卷积）
    2. BatchNorm + Dropout
    3. 数据增强（RandomFlip、RandomCrop、Cutout）
    4. 学习率调度（CosineAnnealing + Warmup）
    5. 混合精度训练（AMP）
    6. 训练/验证曲线（TensorBoard 或 matplotlib）
    7. 保存最佳模型
```

### 评分标准

| 标准 | 分数 |
|------|------|
| 网络架构合理（≥6 层 Conv） | 20 |
| 使用所有要求的优化技术 | 20 |
| 训练曲线合理（无过拟合） | 10 |
| 准确率 ≥ 90% | 20 |
| 准确率 ≥ 93% | 30 |

---

## 21.2 项目 B：IMDb 文本情感分析（目标 ≥ 85%）

### 要求

使用 LSTM 或 Transformer 对 IMDb 影评做情感分类。

```
数据集：torchtext.datasets.IMDB 或 HuggingFace datasets
模型：LSTM（双向，多层）或 Transformer
目标：≥ 85% 测试准确率

必须包含：
    1. 文本预处理（tokenization、截断/填充）
    2. 预训练词向量或学习 embedding
    3. 至少 2 种正则化技术
    4. 验证集上的准确率曲线
```

### 评分标准

| 标准 | 分数 |
|------|------|
| 文本预处理正确 | 20 |
| LSTM / Transformer 架构合理 | 25 |
| 正则化使用正确 | 15 |
| 准确率 ≥ 80% | 15 |
| 准确率 ≥ 85% | 25 |

---

## 21.3 项目 C：自选论文复现

### 要求

从以下选一篇论文，复现其核心网络结构并在对应数据集上训练：

```
选项 1：VGG 风格网络 → CIFAR-10 (≥ 93%)
    - 论文：Very Deep Convolutional Networks for Large-Scale Image Recognition

选项 2：ResNet-18 → CIFAR-10 (≥ 93%)
    - 论文：Deep Residual Learning for Image Recognition
    - 核心创新：残差连接

选项 3：Vision Transformer (ViT) 迷你版 → CIFAR-10 (≥ 80%)
    - 论文：An Image is Worth 16x16 Words
    - 核心创新：把图片切成 patch → 当作文本 token 处理
```

### 评分标准

| 标准 | 分数 |
|------|------|
| 正确理解论文核心思想 | 20 |
| 网络结构实现准确 | 30 |
| 训练+评估流程完整 | 20 |
| 达到基准准确率 | 30 |

---

## 21.4 最终自检清单

完成所有项目后，给自己做一次全面自检：

| 能力 | ✓ |
|------|---|
| 能独立设计网络架构（层数、维度、激活函数） | □ |
| 能根据任务选择正确的损失函数和优化器 | □ |
| 能诊断并解决过拟合/欠拟合 | □ |
| 能正确使用 train/eval 模式切换 | □ |
| 能写出生产级的训练循环（AMP + 调度器 + TensorBoard） | □ |
| 能从论文中理解并复现网络结构 | □ |
| 能调试维度不匹配、loss 不下降等常见问题 | □ |
| 能独立从零完成一个完整的 DL 项目 | □ |

> 如果全部勾选，你已经是**合格的 PyTorch 深度学习工程师**。

---

## 21.5 后续学习方向

```
1. 更深的架构：ResNet、DenseNet、EfficientNet
2. 生成模型：GAN、VAE、Diffusion Models
3. 目标检测：YOLO、Faster R-CNN
4. NLP 进阶：BERT、GPT 微调
5. 强化学习：DQN、PPO
6. 分布式训练：DDP、FSDP
7. MLOps：模型版本管理、CI/CD、监控
```

---

> **恭喜完成全部教程！** 从 "import torch" 到独立复现论文，你已经走了很远。深度学习的世界很深，但你现在有了一只坚固的船。
