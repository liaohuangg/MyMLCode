import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 确保数据集和数据加载器
def get_data_loaders(batch_size=64):
    # 数据预处理：转换为张量并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载MNIST数据集
    train_dataset = datasets.MNIST(
        root='/root/workspace/MyMLCode/DLLearn/dataset', train=True, download=False, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='/root/workspace/MyMLCode/DLLearn/dataset', train=False, download=False, transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader

# 定义LeNet-5模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层：1个输入通道，6个输出通道，5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 平均池化层：2x2池化核，步长为2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层：6个输入通道，16个输出通道，5x5卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 16个通道，每个特征图4x4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10个类别输出（0-9）
        
        # 使用Tanh激活函数
        self.tanh = nn.Tanh()

    def forward(self, x):
        # 第一层：卷积 -> 激活 -> 池化
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool1(x)
        
        # 第二层：卷积 -> 激活 -> 池化
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.pool2(x)
        
        # 展平特征图
        x = x.view(-1, 16 * 4 * 4)
        
        # 全连接层
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        
        return x

# 训练函数
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计训练信息
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 每100个batch打印一次信息
        if batch_idx % 100 == 99:
            print(f'[{epoch + 1}, {batch_idx + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    
    train_acc = 100 * correct / total
    print(f'Train Accuracy: {train_acc:.2f}%')
    return train_acc

# 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # 不计算梯度
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_acc = 100 * correct / total
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({test_acc:.2f}%)\n')
    return test_acc

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 超参数
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # 初始化模型、损失函数和优化器
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录准确率用于绘图
    train_accuracies = []
    test_accuracies = []
    
    # 训练和测试模型
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        
        train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
        test_acc = test(model, test_loader, criterion, device)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
    # 保存模型
    torch.save(model.state_dict(), 'lenet5_mnist.pth')
    print('Model saved as lenet5_mnist.pth')
    
    # 绘制准确率曲线
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('LeNet-5 Accuracy on MNIST')
    plt.legend()
    plt.savefig('lenet5_accuracy.png')
    plt.show()

if __name__ == '__main__':
    main()