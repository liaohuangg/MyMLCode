import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True' #用于避免jupyter环境突然关闭
torch.backends.cudnn.benchmark=True #用于加速GPU运算的代码

#导入pytorch一个完整流程所需的可能全部的包
import torchvision
from torch import nn, optim
from torch.nn import functional
from torchsummary import summary
from torchvision import transforms
from torchvision import models
from torchvision import datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader

#导入作为辅助工具的各类包
import matplotlib.pyplot as plt
from time import time
import random
import numpy as np
import pandas as pd
import datetime
import gc # 垃圾回收 删掉了，但是内存还存在

'''
torch.random.manual_seed(0) #设置随机种子

x = torch.randn(5, 1, 28, 28) #随机生成一个张量
labels = torch.tensor([0, 1, 2, 0, 1]) #随机生成标签
embed = nn.Embedding(num_embeddings = 3 # 标签类别数量
                    ,embedding_dim  = 28*28 #要投射的维度
                    ) 
print(embed(labels).shape)
y = embed(labels)
y = y.view(-1, 1, 28, 28) #将标签嵌入的结果变形为与x相同的形状
inputs = torch.cat([x, y], dim=1) #将x和y在通道维度上拼接

print(inputs.shape)
'''

# 判别器的结构
class cDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 转化标签
        self.label_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=10, embedding_dim=50),  # 假设标签类别为10
            nn.Linear(50, 784),  # 将标签嵌入映射到100维
            nn.ReLU(True)
        )

        # 核心结构
        self.main = nn.Sequential(
            # 使原图像尺寸减半的卷积核使 3 2 1
            nn.Conv2d(2, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 输出
        self.out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128 * 7 * 7, 1), 
            nn.Sigmoid()  # 输出一个概率值
        )
    def forward(self, lable, real_data):
        # 处理标签
        label = self.label_embedding(lable)
        lable = label.view(-1, 1, 28, 28)  # 将标签嵌入的结果变形为与输入数据相同的形状

        # 拼接标签和真实数据
        inputs = torch.cat([real_data, lable], dim=1) 

        # 将合并数据输入核心架构
        features = self.main(inputs)

        # 将特征展平并输出
        features = features.view(-1, 7*7*128)

        outputs = self.out(features)
        return outputs

# 生成器的结构
class cGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # 标签的上采样, 输入的标签是整体的标签类别
        self.label_upsample = nn.Sequential(nn.Embedding(10, 50)
                                            , nn.Linear(50, 49)
                                            , nn.ReLU(True)
                                            )
        # 噪音的上采样
        self.noise_upsample = nn.Sequential(nn.Linear(100, 6272)
                                            , nn.LeakyReLU(0.2, True)
                                            )
        # 核心架构
        self.main = nn.Sequential(nn.ConvTranspose2d(129, 128, kernel_size=4, stride=2, padding=1)
                                    , nn.LeakyReLU(0.2, True)
                                    , nn.ConvTranspose2d(128, 128, 4, 2, 1)
                                    , nn.LeakyReLU(0.2, True)
                                    , nn.Conv2d(128, 1, kernel_size=3, padding=1)
                                    )
    def forward(self, label, noise):
        # 转化标签
        label = self.label_upsample(label)
        label = label.view(-1, 1, 7, 7)
        
        # 转化噪音
        noise = self.noise_upsample(noise)
        noise = noise.view(-1, 128, 7, 7)
        
        # 合并数据
        inputs = torch.cat((noise, label), dim=1)
        
        # 将数据输入核心架构，生成假数据
        fakedata = self.main(inputs)
        
        return fakedata

z = torch.ones((10,100))
realimage = torch.ones((10,1,28,28))
y = torch.tensor([0,1,2,3,4,5,6,7,8,9])
cgen = cGenerator()
cdisc = cDiscriminator()