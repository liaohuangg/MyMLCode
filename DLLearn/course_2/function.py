import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True' #用于避免jupyter环境突然关闭
torch.backends.cudnn.benchmark=True #用于加速GPU运算的代码

#导入pytorch一个完整流程所需的可能全部的包
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import models as m
from torch.utils.data import DataLoader

#导入作为辅助工具的各类包
import matplotlib.pyplot as plt
from time import time
import random
import numpy as np
import pandas as pd
import datetime
import gc # 垃圾回收 删掉了，但是内存还存在

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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
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
        # x = self.maxpool(x)

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
    
    print("before torch.cuda.memory_allocated ", torch.cuda.memory_allocated())  # 张量内存的占用情况（现状）
    print("before torch.cuda.memory_reserved ", torch.cuda.memory_reserved())  # 缓存分配器占用的所有内存（现状）
    print("before torch.cuda.max_memory_allocated ", torch.cuda.max_memory_allocated()) # 自GPU运行以来占用过的最大张量内存（峰值）
    # 清理内存
    del inputs, targets, outputs, loss, correct, total # 删除数据与变量
    gc.collect() #清除数据与变量相关的缓存
    torch.cuda.empty_cache() 

    '''
    80 - 张量内存
    20 - 缓存分配器指定的更多缓存

    GPU - 使用100字节的资源

    del 张量 因为GPU不会删除张量,需要手动进行
    torch.cuda.empty_cache() 缓存分配器指定的那些缓存
    '''
    print("torch.cuda.memory_allocated ", torch.cuda.memory_allocated())  # 张量内存的占用情况（现状）
    print("torch.cuda.memory_reserved ", torch.cuda.memory_reserved())  # 缓存分配器占用的所有内存（现状）
    print("torch.cuda.max_memory_allocated ", torch.cuda.max_memory_allocated()) # 自GPU运行以来占用过的最大张量内存（峰值）
    # 返回训练损失和准确率
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
