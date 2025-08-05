import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # MNIST是28x28，调整为32x32以适应GoogLeNet
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

# Inception模块定义
class Inception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(Inception, self).__init__()
        
        # 1x1卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1卷积 + 3x3卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )
        
        # 1x1卷积 + 5x5卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )
        
        # 3x3池化 + 1x1卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # 拼接所有分支的输出
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# 辅助分类器，用于GoogLeNet的中间监督
class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 定义适用于MNIST的GoogLeNet模型
class GoogLeNetMNIST(nn.Module):
    def __init__(self, num_classes=10, aux_logits=True):
        super(GoogLeNetMNIST, self).__init__()
        self.aux_logits = aux_logits
        
        # 初始卷积层
        self.pre_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # 多个Inception模块
        self.a3 = Inception(64, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(128+32+32+64, 128, 128, 192, 32, 96, 64)
        
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.a4 = Inception(192+96+64+128, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(208+48+64+192, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(224+64+64+160, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(256+64+64+128, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(288+64+64+112, 256, 160, 320, 32, 128, 128)
        
        self.a5 = Inception(320+128+128+256, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(320+128+128+256, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(384+128+128+384, num_classes)
        
        # 辅助分类器
        if self.aux_logits:
            self.aux1 = AuxiliaryClassifier(208+48+64+192, num_classes)
            self.aux2 = AuxiliaryClassifier(288+64+64+112, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None
            
    def forward(self, x):
        x = self.pre_layers(x)
        
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        
        x = self.a4(x)
        
        # 辅助分类器1（训练时使用）
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        else:
            aux1 = None
            
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        
        # 辅助分类器2（训练时使用）
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        else:
            aux2 = None
            
        x = self.e4(x)
        x = self.maxpool(x)
        
        x = self.a5(x)
        x = self.b5(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x, aux1, aux2

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
        outputs, aux1, aux2 = model(inputs)
        
        # 计算损失（主损失 + 辅助损失）
        loss = criterion(outputs, targets)
        if model.aux_logits:
            loss += 0.3 * criterion(aux1, targets)
            loss += 0.3 * criterion(aux2, targets)
        
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
            
            outputs, _, _ = model(inputs)
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
    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 初始化模型、损失函数和优化器
    model = GoogLeNetMNIST(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
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
    torch.save(model.state_dict(), 'googlenet_mnist.pth')
    print('Model saved as googlenet_mnist.pth')
    
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
    plt.savefig('training_curves.png')
    plt.show()
    
    # 展示一些预测结果
    model.eval()
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    with torch.no_grad():
        example_data = example_data.to(device)
        outputs, _, _ = model(example_data)
        _, predicted = outputs.max(1)
    
    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i].cpu().squeeze(), cmap='gray', interpolation='none')
        plt.title(f'Predicted: {predicted[i]}, Actual: {example_targets[i]}')
        plt.xticks([])
        plt.yticks([])
    plt.savefig('predictions.png')
    plt.show()

if __name__ == '__main__':
    main()
