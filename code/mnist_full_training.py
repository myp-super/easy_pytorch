"""
MNIST 完整训练代码 — 第十一章的独立可运行版本。

这是教程的"最终产物"——你能独立写出这段代码时，就真正学会了 PyTorch。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. 网络定义
# ============================================================
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)       # 展平: [batch, 1, 28, 28] → [batch, 784]
        x = torch.relu(self.fc1(x))     # [batch, 784] → [batch, 128]
        x = torch.relu(self.fc2(x))     # [batch, 128] → [batch, 64]
        x = self.fc3(x)                 # [batch, 64] → [batch, 10]  (logits)
        return x


# ============================================================
# 2. 数据准备
# ============================================================
def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ============================================================
# 3. 训练一个 epoch
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


# ============================================================
# 4. 评估
# ============================================================
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


# ============================================================
# 5. 主函数
# ============================================================
def main():
    # 超参数
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 准备
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n开始训练 ({NUM_EPOCHS} epochs)...")
    print("-" * 50)

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.2f}%")

    print("-" * 50)
    print("训练完成！")

    # 保存模型
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("模型已保存到 mnist_model.pth")


if __name__ == "__main__":
    main()
