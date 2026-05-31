# 第十七章：数据增强与学习率调度

## 17.0 本章导引

前两章你学了两个"放在网络里面"的技术（BN 和 Dropout）。本章学两个"放在网络外面"的技术。

数据增强 → 给模型看更多"变体"的数据，提升泛化能力
学习率调度 → 让学习率在训练过程中动态变化，更好收敛

这两个技术几乎零成本（不改网络结构），但能显著提升效果。

---

## 17.1 数据增强（Data Augmentation）

### 17.1.1 直觉

```
    你只有 50,000 张训练图片，但你想让模型"看"更多。

    把每张图片稍微变一下：
        翻转、旋转、裁剪、变色...
    → 模型看到的不只是"原图"，还有各种"变体"
    → 模型被迫学到"这是一只猫，不管它有没有被翻转"
    → 泛化能力提升
```

### 17.1.2 常用增强

```python
# %%
from torchvision import transforms

# 训练时的增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),   # 50% 概率水平翻转
    transforms.RandomCrop(32, padding=4),      # 先 pad 到 40×40，再随机裁 32×32
    transforms.ColorJitter(brightness=0.2,     # 亮度 ±20%
                          contrast=0.2,       # 对比度 ±20%
                          saturation=0.2),    # 饱和度 ±20%
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
])

# 验证时不要增强！只做必要的归一化
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
])
```

**RandomHorizontalFlip**：
```
    原图：🐱（猫脸朝左）
    翻转后：🐱（猫脸朝右）
    模型学到：猫不管朝左朝右，都是猫
```

**RandomCrop(padding=4)**：
```
    原图 32×32 → pad 到 40×40 → 随机裁回 32×32
    → 每次看到的"框框"不同 → 模型对位置不那么敏感
```

---

## 17.2 学习率调度（Learning Rate Scheduler）

### 17.2.1 为什么需要衰减学习率

```
    训练初期：参数离最优值远 → 需要大步子（大 lr）
    训练后期：参数接近最优值 → 需要小步子（小 lr），避免"跳过"最优值

    恒定 lr：
        → 前期收敛慢（lr 太小）
        → 后期震荡（lr 太大）
    
    衰减 lr：
        → 前期大步快跑 → 后期小步微调
```

### 17.2.2 StepLR — 阶梯式衰减

```python
# %%
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# 每 10 个 epoch，lr 乘以 0.1

for epoch in range(30):
    train_one_epoch(model, train_loader, criterion, optimizer)
    scheduler.step()    # ← 每个 epoch 结束后调用
    print(f"Epoch {epoch+1}: lr = {scheduler.get_last_lr()[0]:.6f}")
```

**lr 变化**：`0.01 → 0.01 (10轮) → 0.001 (10轮) → 0.0001 (10轮)`

### 17.2.3 CosineAnnealingLR — 余弦衰减

```python
# %%
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
# 50 个 epoch 内，lr 从初始值按余弦曲线衰减到接近 0

# lr 变化：
#  │
#  │  * * * *
#  │          * *
#  │             *
#  │              * *
#  │                 * * *
#  └────────────────────────→ epoch

# 比 StepLR 更平滑，通常效果更好
```

### 17.2.4 ReduceLROnPlateau — 自适应衰减

```python
# %%
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
# 监控 val_loss，如果 5 个 epoch 内没有改善 → lr 减半

for epoch in range(30):
    train_loss = train_one_epoch(...)
    val_loss = evaluate(...)
    scheduler.step(val_loss)   # ← 传入监控指标
    print(f"Epoch {epoch+1}: lr = {optimizer.param_groups[0]['lr']:.6f}")
```

---

## 17.3 完整训练脚本（CNN + BN + Dropout + 增强 + 调度）

```python
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# === 网络（第十五章 + 第十六章的结合）===
class ModernCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 32×32 → 16×16
            
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # → 8×8
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # → 4×4
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# === 数据（含增强）===
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_loader = DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=train_transform),
    batch_size=128, shuffle=True
)
test_loader = DataLoader(
    datasets.CIFAR10('./data', train=False, transform=test_transform),
    batch_size=128, shuffle=False
)

# === 训练配置 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModernCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# === 训练 ===
for epoch in range(50):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    scheduler.step()
    print(f"Epoch {epoch+1:2d}: Loss={train_loss/len(train_loader):.4f}, "
          f"Acc={100*correct/total:.2f}%, lr={scheduler.get_last_lr()[0]:.6f}")
```

**期望结果**：准确率可达 90%+。

---

## 17.4 本章总结

```
数据增强：
    训练时对图片做随机变换 → 变相增加数据量 → 提升泛化
    常用：RandomHorizontalFlip, RandomCrop, ColorJitter
    验证时只做归一化，不做增强

学习率调度：
    训练初期大步快跑，后期小步微调
    StepLR：阶梯式（简单有效）
    CosineAnnealingLR：余弦平滑衰减（通常最佳）
    ReduceLROnPlateau：自适应（稳但不一定快）
```

---

## 17.5 本章练习

### 练习 17-1：观察不同增强的效果

分别只用一种增强（Flip / Crop / ColorJitter）训练，比较准确率。

### 练习 17-2：对比不同的 lr 调度器

用 StepLR、CosineAnnealingLR、无调度各训练一次，画 lr 曲线 + 准确率曲线。

### 练习 17-3：完整训练 CIFAR-10

```python
# CNN + BN + Dropout + 数据增强 + CosineAnnealingLR
# 目标准确率 ≥ 90%
```

### 练习 17-4：不看答案——独立实现

> 关闭所有文档，独立写出带所有优化的 CIFAR-10 训练代码。

---

> **下一步**：[第十八章：RNN / LSTM](./18_rnn_lstm.md)。
