import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True' #用于避免jupyter环境突然关闭
torch.backends.cudnn.benchmark=True #用于加速GPU运算的代码
#导入pytorch一个完整流程所需的可能全部的包
import torchvision
from torch import nn, optim
from torch.nn import functional
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

class Discriminator(nn.Module):
    def __init__(self, in_features):
        """in_features : 真实数据的维度、同时也是生成的假数据的维度，对于普通神经网络而言就是特征数量"""
        super().__init__()
        self.disc = nn.Sequential(nn.Linear(in_features, 128),
                                  # nn.BatchNorm1d(128)
                                  nn.LeakyReLU(0.1),  # 由于生成对抗网络的损失非常容易梯度消失，因此使用
                                  nn.Linear(128, 1),
                                  nn.Sigmoid()
                                  )

    def forward(self, data):
        """输入的data可以是真实数据时，Disc输出dx。输入的data是gz时，Disc输出dgz"""
        return self.disc(data)

class Generator(nn.Module):
    def __init__(self, in_features, out_features):
        """
        in_features:生成器的in_features，一般输入z的维度z_dim，该值可自定义
        out_features:生成器的out_features，需要与真实数据的维度一致
        """
        super().__init__()
        self.gen = nn.Sequential(nn.Linear(in_features, 256),
                                 #,nn.BatchNorm1d(256)
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(256, out_features),
                                 nn.Tanh()  # 用于归一化数据,sigmiod的效果放在生成器效果不太好
                                 )

    def forward(self, z):
        gz = self.gen(z)
        return gz

# # 检测生成器、判别器能否顺利跑通
# # 假设真实数据结构为28*28*1 = 784
# z = torch.ones((10, 64))
# gen = Generator(64, 784)
# # 生成数据并查看形状
# print(gen(z).shape)  # 输出 torch.Size([10, 784])

# disc = Discriminator(784)
# # 输入生成器生成的数据，查看判别器输出形状
# print(disc(gen(z)).shape)  # 输出唯一值概率（这里看形状是 torch.Size([10, 1]))

# 超参数及配置设置
batch_size = 32
lr = 3e-4
num_epochs = 15
realdata_dim = 28 * 28 * 1  # 784
z_dim = 64

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 真实数据加载（MNIST 数据集）
transform = transforms.Compose([
    transforms.ToTensor(),
    #, transforms.Normalize((0.1307,), (0.3081,))
    transforms.Normalize((0.5,), (0.5,))
])
# 修正 dest 为 datasets
dataset = datasets.MNIST(
    root="/root/workspace/MyMLCode/DLLearn/dataset",
    transform=transform,
    download=False
)

dataloader = DataLoader(  # 修正 DataLoader 首字母大写
    dataset,
    batch_size=batch_size,
    shuffle=True
)

# 设置一组固定的噪音数据，用于在训练中不断验证生成器的结果
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# 实例化判别器与生成器
gen = Generator(in_features=z_dim, out_features=realdata_dim).to(device)
disc = Discriminator(in_features=realdata_dim).to(device)

# 定义判别器与生成器所使用的优化算法
optim_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.9, 0.999))
optim_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.9, 0.999))

# 定义损失函数
criterion = nn.BCELoss(reduction="mean")

# 在训练完毕之后，如果需要继续训练，则千万要避免运行这一个cell
img_list = []
G_losses = []
D_real_loss = []
D_fake_loss = []
D_losses = []
iters = 0

# 开始训练
for epoch in range(num_epochs):
    for batch_idx, (x, _) in enumerate(dataloader):
        x = x.view(-1, 784).to(device)
        batch_size = x.shape[0]

        ####################################
        # (1) 判别器的反向传播：最小化 -[logD(x) + log(1 - D(G(z)))]
        ####################################

        # -logdx
        dx = disc(x).view(-1)
        loss_real = criterion(dx, torch.ones_like(dx))
        loss_real.backward()
        D_x = dx.mean().item()

        # -log(1-dgz)
        noise = torch.randn((batch_size, z_dim)).to(device)
        gz = gen(noise)
        dgz1 = disc(gz.detach())
        loss_fake = criterion(dgz1, torch.zeros_like(dgz1))
        loss_fake.backward()
        D_G_z1 = dgz1.mean().item()

        # 计算errorD
        errorD = (loss_real + loss_fake) / 2
        optim_disc.step()
        disc.zero_grad()

        ####################################
        # (2) 生成器的反向传播：最小化 -log(D(G(z)))
        ####################################
        
        # 生成需要输入criterion的真实标签1与预测概率dgz
        # 注意，由于在此时判别器上的权重已经被更新过了，所以dgz的值会变化，需要重新生成
        dgz2 = disc(gz)
        # 计算errorG
        errorG = criterion(dgz2, torch.ones_like(dgz2))
        errorG.backward()  # 反向传播
        optim_gen.step()  # 更新生成器上的权重
        gen.zero_grad()  # 清零生成器更新后梯度
        D_G_z2 = dgz2.mean().item()

        if batch_idx % 500 == 0 or batch_idx == 0:
            print('[{}%/{}%][{}%/{}%]\tLoss_D: {:.4f}\tloss_G: {:.4f}\tD(x): {:.4f}\tD(G(z)): {:.4f} / {:.4f}'
                .format(epoch + 1, num_epochs, batch_idx, len(dataloader),
                        errorD.item(), errorG.item(), D_x, D_G_z1, D_G_z2))

        # 保存errorG和errorD，以便后续绘图用
        G_losses.append(errorG.item())
        D_real_loss.append(loss_real.item())
        D_fake_loss.append(loss_fake.item())
        D_losses.append(errorD.item())

        # 将固定噪音fixed_noise输入生成器，查看输出的结果变化
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (batch_idx == len(dataloader) - 1)):
            with torch.no_grad():
                fake = gen(fixed_noise).cpu().detach()
            img_list.append(fake)
        iters += 1

plt.figure(figsize=(10,5),dpi=300)
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_real_loss,label="Dreal")
plt.plot(D_fake_loss,label="Dfake")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig('/root/workspace/MyMLCode/DLLearn/course_3/gan1.png')
plt.pause(3)
plt.close()

# 绘制原始图像与生成图像
real_batch = next(iter(dataloader))

# 原始图像中抽取一个batch
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.rot90(np.transpose(vutils.make_grid(real_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1, 2, 0)), k=1))
# 绘制保存的最后一组生成图像
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(img_list[-1].reshape(-1, 1, 28, 28)[:batch_size], padding=2)))  # 括号匹配可能需微调，按实际场景修正
plt.show()
plt.savefig('/root/workspace/MyMLCode/DLLearn/course_3/gan2.png')
plt.pause(3)
plt.close()

