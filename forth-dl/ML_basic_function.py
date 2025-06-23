import numpy as np
import pandas as pd


np.random.seed(24)

# 生成0到4之间的20个等间距数，并重塑为列向量
x = np.linspace(0, 4, 20).reshape(-1, 1)

# 添加全1的偏置项列（构造设计矩阵）
x = np.concatenate((x, np.ones_like(x)), axis=1)

# print(x)
# [4]: 数据集标签生成代码
# 对第一列特征进行指数运算（exp(x+1）)并重塑为列向量
y = np.exp(x[:, 0] + 1).reshape(-1, 1)

np.linalg.lstsq(x,y,rcond=-1)[0]