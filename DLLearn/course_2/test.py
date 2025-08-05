import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
torch.backends.cudnn.benchmark = True
# 包含对数据集本身数字进行修改的类
# dataloader tensordataset对数据结构和归纳方式进行变换

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

#设置全局的随机数种子，这些随机数种子只能提供有限的控制
#并不能完全令模型稳定下来
torch.manual_seed(1412)
random.seed(1412)
np.random.seed(1412)

# 下载训练集
train_dataset = datasets.SVHN(
    root='/root/workspace/MyMLCode/DLLearn/dataset', split='train', download=True
)

# 下载测试集
test_dataset = datasets.SVHN(
    root='/root/workspace/MyMLCode/DLLearn/dataset', split='test', download=True
)