import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
# 包含对数据集本身数字进行修改的类
# dataloader tensordataset对数据结构和归纳方式进行变换
minist = torchvision.datasets.MNIST(
    root='/root/workspace/MyMLCode/DLLearn/dataset', #目录地址
    train=True, #是否是训练集
    download=False, #是否下载数据集，已经下好了
    transform=transforms.ToTensor() #对数据进行变换
    )
# 读取数据集
# print(len(minist)) #打印数据集长度
# print(minist.data.shape)
# print(minist[0][0].shape) #打印第一张图片的形状
# # 当前数据集缺少颜色通道维度(C=1)，实际应为[60000,1,28,28] 
# # 神经网络通常需要三维输入(包含通道维度)，因此需要调整结构
# plt.imshow(minist[0][0].view(28,28).numpy()) #显示第一张图片
# plt.show() #显示图片

# print("end")
lr = 0.15 #学习率
gamma = 0.8 #动量参数
epochs = 3 #训练轮数
batch_size = 128 #批处理大小

batchdata = DataLoader(
    dataset=minist, #数据集
    batch_size=batch_size, #批处理大小
    shuffle=True#是否打乱数据
)

input_ = minist.data[0].numel()
output_ = len(minist.targets.unique()) #输出类别数量

class Model(nn.Module):
    def __init__(self, in_features = 10, out_features = 2):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(in_features, 128, bias = False)
        self.output = nn.Linear(128, out_features, bias = False)

    def forward(self, x):
        # 整理 x的结构, -1是占位符，请pytorch帮助我们自动计算-1对应的维度是多少
        x = x.view(-1, 28*28)
        sigma1 = torch.relu(self.linear1(x)) #激活函数
        sigma2 = F.log_softmax(self.output(sigma1), dim=1) #输出层
        return sigma2
    
# 定义一个训练函数
def fit(net, batchdata, lr=0.01, gamma=0.9, epochs=10):
    criterion = nn.NLLLoss()
    opt = optim.SGD(net.parameters(), lr=lr, momentum=gamma)
    correct = 0
    samples = 0
    for epoch in range(epochs):  # 全数据被训练几次
        for batch_idx, (x, y) in enumerate(batchdata):
            y = y.view(x.shape[0])  # 降维
            sigma = net.forward(x)  # 正向传播
            loss = criterion(sigma, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            # 求解准确率， 全部判断正确的样本数量/已经看过的总样本量
            y_hat= torch.max(sigma, 1)[1] # 取出预测结果
            correct += torch.sum(y_hat == y)# 统计预测正确的样本数量
            
            samples += x.shape[0]  # 统计样本数
            if(batch_idx + 1) % 100 == 0:
                print("epoch {} : [{}/{}({:.0f}%)] loss:{:.6f} accuracy:{:.3f}".format(epoch+1, samples,
                                                                                     epochs*len(batchdata.dataset), 
                                                                                     100. * samples / (epochs*len(batchdata.dataset)+1),
                                                                                     loss.data.item(),
                                                                                     float(100. *(correct / samples))))

# torch.manual_seed(1) #设置随机种子
# model = Model(in_features=input_, out_features=output_) #创建模型
# fit(model, batchdata, lr=lr, gamma=gamma, epochs=epochs) #训练模型

import torch
print(torch.__version__)
print(torch.cuda.is_available())
