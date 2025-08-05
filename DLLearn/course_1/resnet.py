import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # MNIST是28x28，调整为32x32以适应ResNet
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='/root/workspace/MyMLCode/DLLearn/dataset', train=True, download=False, transform=transform
)
test_dataset = datasets.MNIST(
    root='/root/workspace/MyMLCode/DLLearn/dataset', train=False, download=False, transform=transform
)

# 数据加载器
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 定义残差块
class BasicBlock(nn.Module):
    expansion = 1  # 扩展系数，BasicBlock中为1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接，用于匹配输入输出维度
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存输入，用于跳跃连接

        # 第一个卷积操作
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积操作
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样（维度不匹配时）
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：输出 + 输入（跳跃连接）
        out += identity
        out = self.relu(out)

        return out

# 定义18层ResNet
class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64  # 初始输入通道数
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个残差块组，构成18层网络
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)   # 2层
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 2层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 2层
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 2层
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        # 如果步长不为1或输入输出通道数不匹配，需要下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 添加第一个残差块（可能包含下采样）
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # 添加剩余的残差块
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积和池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 通过四个残差块组
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化和全连接层
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 训练函数
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 计算准确率
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 打印训练进度
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(f'Train Epoch: {epoch} Average loss: {train_loss:.6f}, Accuracy: {correct}/{total} ({train_acc:.2f}%)')
    
    return train_loss, train_acc

# 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    print(f'Test set: Average loss: {test_loss:.6f}, Accuracy: {correct}/{total} ({test_acc:.2f}%)')
    
    return test_loss, test_acc

# 主函数
def main():
    # 创建保存结果的目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 初始化18层ResNet模型（2+2+2+2=8个残差块，每个残差块2层卷积，共16层+初始卷积+全连接=18层）
    model = ResNet18(BasicBlock, [2, 2, 2, 2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)  # 添加L2正则化
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 学习率调度器
    
    # 训练模型
    epochs = 15
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
    torch.save(model.state_dict(), 'results/resnet18_mnist.pth')
    print('Model saved as results/resnet18_mnist.pth')
    
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
    plt.savefig('results/training_curves.png')
    plt.show()
    
    # 展示一些预测结果
    model.eval()
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    with torch.no_grad():
        example_data = example_data.to(device)
        outputs = model(example_data)
        _, predicted = outputs.max(1)
    
    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i].cpu().squeeze(), cmap='gray', interpolation='none')
        plt.title(f'Predicted: {predicted[i]}, Actual: {example_targets[i]}')
        plt.xticks([])
        plt.yticks([])
    plt.savefig('results/predictions.png')
    plt.show()

if __name__ == '__main__':
    main()
    