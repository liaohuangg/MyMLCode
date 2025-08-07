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

# 实现 DCGAN 中的 GENERATOR
def BasicTransConv2d(in_channels, out_channels, ks, s, p):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, ks, s, p, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

class DCGen(nn.Module):
    def __init__(self):
        super(DCGen, self).__init__()
        self.main = nn.Sequential(
            BasicTransConv2d(100, 1024, 4, 1, 0),
            BasicTransConv2d(1024, 512, 4, 2, 1),
            BasicTransConv2d(512, 256, 4, 2, 1),
            BasicTransConv2d(256, 128, 4, 2, 1),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            #nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, z):
        return self.main(z)

# 实现 DCGAN 中的 DISCRIMINATOR
def BasicConv2d(in_channels, out_channels, ks, s, p):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, ks, s, p, bias=False),
        nn.BatchNorm2d(out_channels),  # 注意原代码里是 BatchNorm2d，图片中拼写有误（BatchNomr2d 是错误写法 ）
        nn.LeakyReLU(0.2, inplace=True)
    )

class DCDisc(nn.Module):
    def __init__(self):
        super(DCDisc, self).__init__()
        self.main = nn.Sequential(
            # BasicConv2d(3, 128, 3, 2, 1),
            BasicConv2d(1, 128, 4, 2, 1),
            BasicConv2d(128, 256, 4, 2, 1),
            BasicConv2d(256, 512, 4, 2, 1),
            BasicConv2d(512, 1024, 4, 2, 1),
            nn.Conv2d(1024, 1, 4, 1, 0),
            nn.Sigmoid()
        )
        
    def forward(self, gz):
        return self.main(gz)
# 测试该生成器能不能正确运行
# z = torch.ones((10,100,1,1))
gen = DCGen()
summary(gen, input_size=(100, 1, 1), device="cpu")

# data = torch.ones((10,3,64,64))
disc = DCDisc()
summary(disc, input_size = (1,64,64), device="cpu")

# 超参数及配置设置
batch_size = 128
lr = 2e-4
num_epochs = 10
z_dim = 100

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 真实数据加载（MNIST 数据集）
transform = transforms.Compose([
                                       transforms.Resize(64),
                                       transforms.CenterCrop(64),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                             ])
# 修正 dest 为 datasets
dataset = datasets.MNIST(
    root="/root/workspace/MyMLCode/DLLearn/dataset",
    transform=transform,
    download=False
)
print("数据集长度：", len(dataset))
print(dataset[0][0].shape)  # 打印第一个样本的形状
dataloader = DataLoader(  # 修正 DataLoader 首字母大写
    dataset,
    batch_size=batch_size,
    shuffle=True
)

#权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 设置一组固定的噪音数据，用于在训练中不断验证生成器的结果
fixed_noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
# 实例化判别器与生成器
gen = DCGen().apply(weights_init).to(device)
disc = DCDisc().apply(weights_init).to(device)

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
        x = x.to(device)
        batch_size = x.shape[0]
        ####################################
        # (1) 判别器的反向传播：最小化 -[logD(x) + log(1 - D(G(z)))]
        ####################################

        # -logdx
        dx = disc(x).view(-1)  # 展平输出
        loss_real = criterion(dx, torch.ones_like(dx))
        loss_real.backward()
        D_x = dx.mean().item()

        # -log(1-dgz)
        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
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
plt.savefig('/root/workspace/MyMLCode/DLLearn/course_3/dcgan1.png')
plt.pause(3)
plt.close()

# 绘制原始图像与生成图像
real_batch = next(iter(dataloader))

# 原始图像中抽取一个batch
plt.figure(figsize=(15, 15))
# Plot the real images
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
# plt.imshow(utils.make_grid(real_batch[0][:100] * 0.5 + 0.5, nrow=10).permute(1, 2, 0))
plt.imshow(np.rot90(np.transpose(vutils.make_grid(real_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1, 2, 0)), k=1))


plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(img_list[-1].reshape(-1, 1, 64, 64)[:batch_size], padding=2)))  # 括号匹配可能需微调，按实际场景修正
plt.show()
plt.savefig('/root/workspace/MyMLCode/DLLearn/course_3/dcgan2.png')
plt.pause(3)
plt.close()

