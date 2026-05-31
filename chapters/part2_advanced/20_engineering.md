# 第二十章：PyTorch 工程化实践

## 20.0 本章导引

前十九章你学会了"怎么写对"的 PyTorch 代码。这一章学"怎么写好"。

这些技术不改变模型结构，但让你从"能训练"变成"专业地训练"。

---

## 20.1 TensorBoard — 可视化一切

```python
# %%
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./runs/experiment_1')

# 训练中记录
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = evaluate(...)
    
    # 记录标量（loss、准确率）
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Acc/train', train_acc, epoch)
    writer.add_scalar('Acc/val', val_acc, epoch)
    
    # 记录模型图
    if epoch == 0:
        writer.add_graph(model, sample_input)
    
    # 记录直方图（参数分布、梯度分布）
    for name, param in model.named_parameters():
        writer.add_histogram(f'params/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'grads/{name}', param.grad, epoch)

writer.close()

# 启动 TensorBoard：终端运行
# tensorboard --logdir=./runs
# 浏览器打开 http://localhost:6006
```

---

## 20.2 混合精度训练（AMP）

用 float16 替代 float32 做前向和反向，速度翻倍、显存减半。

```python
# %%
scaler = torch.cuda.amp.GradScaler()    # 防止 float16 下溢

for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    
    # 前向：float16 自动混合
    with torch.cuda.amp.autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    # 反向：scaler 处理梯度缩放
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 只需加 3 行代码，训练速度和显存占用显著改善
```

---

## 20.3 梯度裁剪

防止梯度爆炸（尤其在 RNN/Transformer 训练中）：

```python
# 在 loss.backward() 之后，optimizer.step() 之前
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## 20.4 随机种子与可复现性

```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# 每次运行得到完全相同的结果
```

---

## 20.5 模型部署：TorchScript

```python
# 导出为 TorchScript
model.eval()
example = torch.randn(1, 3, 32, 32)
traced_model = torch.jit.trace(model, example)
traced_model.save('model.pt')

# 加载（不需要 Python 类定义！）
loaded_model = torch.jit.load('model.pt')
output = loaded_model(example)
```

---

## 20.6 生产级训练脚本模板

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random, numpy as np, time, os

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(f'./runs/{time.strftime("%Y%m%d_%H%M%S")}')
    
    # 数据、模型、优化器、调度器、损失函数...
    model = ... ; model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(100):
        # 训练
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        
        # 评估
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = 100 * correct / total
        
        scheduler.step()
        writer.add_scalar('Acc/val', val_acc, epoch)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch+1}: Acc={val_acc:.2f}% (best={best_acc:.2f}%)")
    
    writer.close()
    print(f"训练完成。最佳准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
```

---

## 20.7 本章总结

```
    TensorBoard   → 可视化 loss、准确率、梯度分布
    AMP           → float16 训练，速度↑ 显存↓
    梯度裁剪       → 防止梯度爆炸（RNN/Transformer 必备）
    随机种子       → 确保实验可复现
    TorchScript   → 导出模型，脱离 Python 部署
```

---

## 20.8 本章练习

### 练习 20-1：添加 TensorBoard

在 CIFAR-10 训练中添加 TensorBoard 记录，查看训练曲线。

### 练习 20-2：启用混合精度

在 CIFAR-10 CNN 训练中启用 AMP，对比训练速度和显存。

### 练习 20-3：可复现性实验

```python
# 不设种子，训练两次 → 结果不同
# 设种子，训练两次 → 结果完全相同
```

### 练习 20-4：不看答案——独立写出生产级训练脚本

> 包含：TensorBoard + AMP + 梯度裁剪 + 最佳模型保存 + 学习率调度。

---

> **下一步**：[第二十一章：进阶实战项目](./21_advanced_projects.md)。
