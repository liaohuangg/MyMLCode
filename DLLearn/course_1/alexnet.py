import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# alexnet的输入形状
# 数据预处理：将MNIST转换为AlexNet兼容的输入格式
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 核心：将28×28放大到224×224
    transforms.Grayscale(num_output_channels=3),  # 可选：将1通道转为3通道（模拟RGB）
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 保持MNIST的归一化参数
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='/root/workspace/MyMLCode/DLLearn/dataset', train=True,
                               download=False, transform=transform)
test_dataset = datasets.MNIST(root='/root/workspace/MyMLCode/DLLearn/dataset', train=False,
                              download=False, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        """
        in_channels: 输入图像通道数（1 或 3，MNIST 用 1 或转 3 通道）
        num_classes: 分类类别数（MNIST 是 10）
        """
        super(AlexNet, self).__init__()
        
        # 核心卷积层（适配 224×224 输入）
        self.features = nn.Sequential(
            # 卷积层 1: 3→96（原版 AlexNet 是 3→96，这里可根据需求调整）
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # 池化层 1
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 卷积层 2: 96→256
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            # 池化层 2
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 卷积层 3: 256→384
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # 卷积层 4: 384→384
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # 卷积层 5: 384→256
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 池化层 3
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 全连接层（适配卷积后的特征维度）
        self.classifier = nn.Sequential(
            # 关键：计算卷积后的输出维度
            # 输入维度 = 256 * 6 * 6 （224→3次池化后：224/(4*2*2)=6）
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 展平（适配全连接层）
        x = x.view(x.size(0), 256 * 6 * 6)
        # 分类
        x = self.classifier(x)
        return x

# 训练模型
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 99:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    train_acc = 100. * correct / total
    print('Epoch {} Train Accuracy: {:.2f}%'.format(epoch, train_acc))
    return train_acc


# 测试模型
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / total
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    return test_acc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = AlexNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, num_epochs + 1):
        train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
        test_acc = test(model, test_loader, criterion, device)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    # 保存模型
    torch.save(model.state_dict(), 'alexnet_mnist.pth')
    print('Model saved as alexnet_mnist.pth')

    # 绘制准确率曲线
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('AlexNet Accuracy on MNIST')
    plt.legend()
    plt.savefig('alexnet_accuracy.png')
    plt.show()


if __name__ == '__main__':
    main()