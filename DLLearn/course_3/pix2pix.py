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
Stain-to-Stain Translation with Generative Adversarial Networks
'''