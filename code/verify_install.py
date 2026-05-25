"""
验证 PyTorch 安装是否成功。

运行方式：
    python verify_install.py

期望输出：
    PyTorch 版本: 2.x.x
    CUDA 是否可用: True/False
    第一个 Tensor: tensor([1., 2., 3.])
    Tensor 的 shape: torch.Size([3])
    Tensor 的 dtype: torch.float32
    torchvision 版本: 0.x.x
    环境搭建完成！
"""

import torch
import torchvision


def main():
    print("=" * 50)
    print("PyTorch 环境验证")
    print("=" * 50)

    # 1. 检查 PyTorch 版本
    print(f"\nPyTorch 版本: {torch.__version__}")

    # 2. 检查 CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")
    if cuda_available:
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")

    # 3. 创建第一个 Tensor
    first_tensor = torch.tensor([1.0, 2.0, 3.0])
    print(f"\n第一个 Tensor: {first_tensor}")
    print(f"Tensor 的 shape: {first_tensor.shape}")
    print(f"Tensor 的 dtype: {first_tensor.dtype}")
    print(f"Tensor 的 device: {first_tensor.device}")

    # 4. 基本运算
    second_tensor = torch.tensor([4.0, 5.0, 6.0])
    result = first_tensor + second_tensor
    print(f"\n基本运算: {first_tensor} + {second_tensor} = {result}")

    # 5. 检查 torchvision
    print(f"\ntorchvision 版本: {torchvision.__version__}")

    # 6. 测试 nn.Module
    import torch.nn as nn

    class TestNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 2)

        def forward(self, x):
            return self.fc(x)

    model = TestNet()
    test_input = torch.randn(2, 3)
    test_output = model(test_input)
    print(f"\nnn.Module 测试通过!")
    print(f"输入 shape: {test_input.shape}")
    print(f"输出 shape: {test_output.shape}")

    print("\n" + "=" * 50)
    print("环境搭建完成！可以开始学习了。")
    print("=" * 50)


if __name__ == "__main__":
    main()
