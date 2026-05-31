# 第十二章：推理、保存与加载 — 训练不是终点

## 12.0 本章导引

训练好的模型如果不用，就等于白训练了。本章教你三件事：

1. **推理**（Inference）：用训练好的模型对新的、没见过的数据做预测
2. **保存**：把模型参数存到硬盘，下次直接用
3. **加载**：从硬盘恢复模型，继续用或继续训练

---

## 12.1 推理 vs 训练

```
训练：
    有 X 也有 y（正确答案）
    前向 → loss → backward → 更新参数
    目标：让模型学会规律

推理：
    只有 X（没有答案）
    只有前向（不更新参数）
    目标：用学到的规律做预测
```

**推理时的两个必须**：

```python
model.eval()            # ① 切换到评估模式
with torch.no_grad():   # ② 不追踪梯度（省内存、加速）
    output = model(x)
```

---

## 12.2 单张图片推理

```python
# %%
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# 假设有训练好的模型
# model = MNISTNet()
# model.load_state_dict(torch.load('mnist_model.pth'))

# 1. 取一张测试图片
test_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
image, true_label = test_dataset[0]
print(f"真实标签: {true_label}")
print(f"图片 shape: {image.shape}")          # [1, 28, 28]

# 2. 加 batch 维度（模型期望 [batch, 1, 28, 28]）
image_batch = image.unsqueeze(0)            # [1, 1, 28, 28]
print(f"加 batch 后: {image_batch.shape}")

# 3. 推理
model.eval()
with torch.no_grad():
    output = model(image_batch)             # [1, 10]
    predicted = output.argmax(dim=1)        # 取最大分数的索引
    print(f"预测标签: {predicted.item()}")
    print(f"{'✅ 正确' if predicted.item() == true_label else '❌ 错误'}")
```

**argmax 的作用**：

```python
# output 是 10 个分数的向量：
# [0.05, 0.12, 8.30, 0.20, ...]  ← 第 2 个（索引2）的分数最大
# argmax → 2 → 模型预测"数字 2"
```

---

## 12.3 模型的保存与加载

### 12.3.1 保存参数（推荐）

```python
# 保存
torch.save(model.state_dict(), 'mnist_model.pth')

# state_dict 是一个普通字典：
# {'fc1.weight': tensor(...), 'fc1.bias': tensor(...), ...}
```

### 12.3.2 加载参数

```python
# ① 创建一个结构相同的新模型
new_model = MNISTNet()

# ② 加载参数
new_model.load_state_dict(torch.load('mnist_model.pth'))

# ③ 切到评估模式
new_model.eval()
```

### 12.3.3 保存参数 vs 保存整个模型

```
方式 A: torch.save(model.state_dict(), 'weights.pth')
    加载：需要重新创建模型结构 → 然后 load_state_dict
    推荐 ✅ 灵活，不会因代码变动出问题

方式 B: torch.save(model, 'full_model.pth')
    加载：model = torch.load('full_model.pth')
    不推荐 ❌ 换了目录或改了代码可能加载失败
```

### 12.3.4 保存 checkpoint（训练检查点）

```python
# 保存完整的训练状态（用于中断后恢复）
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}
torch.save(checkpoint, 'checkpoint.pth')

# 恢复训练
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1   # 下一轮继续
```

---

## 12.4 批量推理与评估

```python
def predict_all(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())
    
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return all_preds, accuracy
```

---

## 12.5 本章总结

```
训练 → 保存 state_dict → 加载 load_state_dict → 推理 eval + no_grad

铁律：
    1. 推理前必须 model.eval()
    2. 推理时用 torch.no_grad()
    3. 保存用 state_dict（不用整个模型）
    4. 单张图片推理记得 unsqueeze(0) 加 batch 维度
    5. argmax(dim=1) 取预测类别
```

---

## 12.6 本章练习

### 练习 12-1：训练 → 保存 → 加载 → 验证

```python
# 完整流程：训练 → 保存 → 新模型实例 → 加载 → 验证准确率一致
```

### 练习 12-2：单张推理

```python
# 取测试集一张图 → 推理 → 打印预测和真实标签
```

### 练习 12-3：保存 checkpoint

```python
# 训练 3 个 epoch 后保存 checkpoint → "模拟崩溃" → 加载恢复 → 继续训练
```

### 练习 12-4：不看答案——独立完成全流程

> 训练 → 保存 → 加载 → 单张推理 → 测试集准确率。
> 全流程独立完成。

---

> **下一步**：[第十三章：GPU 训练](./13_gpu.md)。
