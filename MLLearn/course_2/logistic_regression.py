import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import sys
import os
# 获取当前文件的父目录的父目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.ML_basic_function import *

# 创建初始标记值
num_inputs = 2      # 数据集特征维度
num_examples = 500  # 每一类样本数
num_class = 2
# 设置随机种子（保证可复现性）
np.random.seed(24)

features, labels = arrayGenCla(num_examples,num_inputs, num_class, deg_dispersion=[4, 2], bias=True)
# 划分数据集
Xtrain, Xtest, ytrain, ytest = array_split(features, labels, 0.2, 24, True)

# 数据归一化
# 计算训练集特征（除最后一列）的均值和标准差
# [:, :-1] 表示排除最后一列， 最后一列是bias
mean_ = Xtrain[:, :-1].mean(axis=0)  # 按列计算均值
std_ = Xtrain[:, :-1].std(axis=0)    # 按列计算标准差

# 标准化处理（Z-Score标准化）
Xtrain[:, :-1] = (Xtrain[:, :-1] - mean_) / std_  # 训练集标准化
Xtest[:, :-1] = (Xtest[:, :-1] - mean_) / std_    # 测试集使用训练集的统计量

# 设置参数
n = features.shape[1]
w = np.random.randn(n, 1) # 生成形状为(n, 1)的随机矩阵

batch_size = 50
num_epoch = 200
lr_init = 0.2

# print(w)
for i in range(num_epoch) :
    sgd_cal(Xtrain, w, ytrain, batch_size, lr_init, i)
# print(w)

# 计算准确率
acc_t = logit_acc(Xtrain,w,ytrain,thr=0.5)
print(acc_t)

acc = logit_acc(Xtest,w,ytest,thr=0.5)
print(acc)

# print(finally_yhat[:10])
# 生成两类正态分布数据
# data0 = np.random.normal(4, 2, size=(num_examples, num_inputs))  # 均值4，标准差2
# data1 = np.random.normal(-2, 2, size=(num_examples, num_inputs)) # 均值-2，标准差2

# label0 = np.zeros(500)
# label1 = np.ones(500)

# feature 含有两类特征
# features = np.concatenate((data0, data1), 0)
# labels = np.concatenate((label0, label1), 0)

# c 着色, 显示样本点的分类情况
# plt.scatter(features[:, 0], features[:, 1], c = labels)
# plt.savefig("/root/workspace/MyMLCode/MLLearn/course_2/output2.png")
