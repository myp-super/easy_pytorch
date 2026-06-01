# 第十三章：GPU 训练 — 让训练快起来

## 13.0 本章导引

到目前为止，所有代码都在 CPU 上运行。对于 MNIST 这样的小数据集，CPU 足够（几分钟就训练完）。但对于更大的模型和数据集（比如 ImageNet，百万张图片），CPU 太慢了。

**好消息：切换到 GPU 只需要加几行代码。**

---

## 13.1 CPU vs GPU

```
CPU（中央处理器）：
    4-16 个强大核心
    擅长复杂逻辑、分支、多样的任务
    像几个博士——能做复杂思考但人少

GPU（图形处理器）：
    数千个简单核心
    擅长大量并行的简单计算
    像几千个小学生——每人只会加减乘除但人多力量大

神经网络的计算 = 大量矩阵乘法 = 大量并行的简单计算 = GPU 的完美场景
```

---

## 13.2 GPU 训练只需三处改动

### 13.2.1 改动对照

```python
# === CPU 版 ===
model = MNISTNet()
# ...
for images, labels in train_loader:
    # images, labels 在 CPU 上
    outputs = model(images)
    # ...

# === GPU 版（只需加三处）===
# ① 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ② 模型移到设备
model = MNISTNet().to(device)

# ③ 数据移到设备
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    # 其余完全一样！
    outputs = model(images)
    # ...
```

**除了这三行，训练循环完全不变。**

### 13.2.2 完整 GPU 训练代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ① 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用: {device}")

# 网络
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=64, shuffle=False
)

# ② 模型 → GPU
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        # ③ 数据 → GPU
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # ③
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch+1}: {100*correct/total:.2f}%")
```

---

## 13.3 GPU 常见 bug

```
Bug 1: 模型在 GPU，数据在 CPU
    RuntimeError: Expected all tensors to be on the same device
    修：images, labels = images.to(device), labels.to(device)

Bug 2: loss 相关的运算跨设备
    修：确保所有参与 loss 计算的 Tensor 都在同一设备

Bug 3: .numpy() 不能用于 GPU Tensor
    RuntimeError: can't convert CUDA tensor to numpy
    修：tensor.cpu().numpy()

Bug 4: 模型保存后加载到 CPU 设备
    修：model.load_state_dict(torch.load('model.pth', map_location='cpu'))
```

---

## 13.4 本章总结

```
CPU → GPU 只改三处：
    ① device = torch.device('cuda')
    ② model = model.to(device)
    ③ data = data.to(device)

铁律：模型和数据必须在同一个设备上！
```

---

## 13.5 本章练习

### 练习 13-1：加 GPU 支持

在 MNIST 训练代码中添加 GPU 支持。如果没有 GPU，代码应优雅回退到 CPU。

### 练习 13-2：速度对比

```python
import time
# 分别在 CPU 和 GPU 上训练 1 epoch，对比时间
```

### 练习 13-3：不看答案——独立写 GPU 训练

> 关闭文档，独立写出完整 GPU 训练代码。

---

> **下一步**：[第十四章：实战项目](14_projects.md)。
