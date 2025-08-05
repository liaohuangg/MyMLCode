import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True' #用于避免jupyter环境突然关闭
torch.backends.cudnn.benchmark=True #用于加速GPU运算的代码
import sys
# 将模块所在的目录添加到系统路径
sys.path.append("/root/workspace/MyMLCode/DLLearn/course_2")
# 导入另一个文件
import function
from function import *
#导入pytorch一个完整流程所需的可能全部的包
import torchvision
from torch import nn, optim
from torch.nn import functional
from torchvision import transforms
from torchvision import models
from torchvision import datasets
from torch.utils.data import DataLoader

#导入作为辅助工具的各类包
import matplotlib.pyplot as plt
from time import time
import random
import numpy as np
import pandas as pd
import datetime
import gc # 垃圾回收 删掉了，但是内存还存在

#设置全局的随机数种子，这些随机数种子只能提供有限的控制
#并不能完全令模型稳定下来
torch.manual_seed(1412)
random.seed(1412)
np.random.seed(1412)
# 如果有GPU可用，设置GPU的随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(1412)
    torch.cuda.manual_seed_all(1412)
# 数据预处理，使用以下流程处理数据
# RandomCrop → RandomRotation → ToTensor → Normalize
train_transform = transforms.Compose([
    transforms.RandomCrop(28),               # 随机裁剪为28×28大小
    transforms.RandomRotation(degrees=[-30, 30]),  # 在-30°到30°之间随机旋转
    transforms.ToTensor(),                   # 转换为Tensor并归一化到[0,1]
    transforms.Normalize(                    # 使用ImageNet的均值和标准差进行标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 测试集的变换通常只包括ToTensor和Normalize
test_transform = transforms.Compose([
    transforms.CenterCrop(28),             # 中心裁剪为28×28大小
    transforms.ToTensor(),                   # 转换为Tensor并归一化到[0,1]
    transforms.Normalize(                    # 使用ImageNet的均值和标准差进行标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 下载训练集
train_dataset = datasets.SVHN(
    root='/root/workspace/MyMLCode/DLLearn/dataset/SVHN', split='train', download=False, transform=train_transform
)

# 下载测试集
test_dataset = datasets.SVHN(
    root='/root/workspace/MyMLCode/DLLearn/dataset/SVHN', split='test', download=False, transform=test_transform
)

# 确认数据集大小和形状
# for x,y in train_dataset:
#     print(x.shape)
#     print(y)
#     break

# 数据加载器
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 主函数
def main():
    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 初始化18层ResNet模型（2+2+2+2=8个残差块，每个残差块2层卷积，共16层+初始卷积+全连接=18层）
    model = ResNet18(BasicBlock, [2, 2, 2, 2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)  # 添加L2正则化
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 学习率调度器
    
    # 训练模型
    epochs = 2
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        scheduler.step()

    # 保存模型
    # torch.save(model.state_dict(), 'results/resnet18_mnist.pth')
    # print('Model saved as results/resnet18_mnist.pth')
    
    # 绘制训练和测试的损失和准确率曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accs, label='Train Accuracy')
    plt.plot(range(1, epochs+1), test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("/root/workspace/MyMLCode/DLLearn/course_2/training_curves.png")
    # plt.show()
    
    # 展示一些预测结果
    model.eval()
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    with torch.no_grad():
        example_data = example_data.to(device)
        outputs = model(example_data)
        _, predicted = outputs.max(1)
    
if __name__ == '__main__':
    main()
