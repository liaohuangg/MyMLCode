import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import *

'''
num_inputs = 2 #两个特征
num_example = 1000 #1000条数据

np.random.seed(24)

x = np.random.randn(2,2)

w_true = np.array([2, -1]).reshape(-1, 1)
b_true = np.array(1)

# 扰动项
delta = 0.01

# 创建数据集的特征和标签取值
featrue = np.random.randn(num_example, num_inputs)
label_true = featrue.dot(w_true) + b_true

label = label_true + np.random.normal(size = label_true.shape) * delta #加上一个扰动项
#print(featrue)

plt.subplot(121)
plt.scatter(featrue[:,0], label) #第一个特征和label的关系
plt.subplot(122)
plt.scatter(featrue[:,1], label) #第二个特征和label的关系
# plt.savefig("/root/workspace/MyMLCode/MLLearn/course-1/output.png")
'''

# 数据生成器
def arrayGenReg(num_examples = 1000, w = [2, -1, 1], bias = True, delta = 0.01, deg = 1):
    """
    同归类数据集创建函数。

    :param num_examples: 创建数据集的数据量
    :param w: 包括截距的（如果存在）特征系数向量
    :param bias: 是否需要截距
    :param delta: 扰动项取值
    :param deg: 方程最高项次数
    :return: 生成的特征张量和标签张量
    """
    
    if bias == True:
        num_inputs = len(w)-1
        features_true = np.random.randn(num_examples, num_inputs)  # 数据集特征个数
        w_true = np.array(w[:-1]).reshape(-1, 1)  # 原始特征
        b_true = np.array(w[-1])  # 自变量系数
        labels_true = np.power(features_true, deg).dot(w_true) + b_true  # 严格满足人造规律的标签
        features = np.concatenate((features_true, np.ones_like(labels_true)), axis=1)  # 加上全为1的一列之后的特征
    else:
        num_inputs = len(w)
        features = np.random.randn(num_examples, num_inputs)
        w_true = np.array(w).reshape(-1, 1)
        labels_true = np.power(features_true, deg).dot(w_true)
    labels = labels_true + np.random.normal(size = labels_true.shape) * delta
    return features, labels

def SSELoss(X, w, y):
    """
    SSE计算函数
    :param X: 输入数据的特征矩阵
    :param w: 线性方程参数
    :param y: 输入数据的标签数组
    :return SSE: 返回对应数据集预测结果和真实结果的误差平方和
    """
    y_hat = X.dot(w)
    SSE = (y - y_hat).T.dot(y - y_hat)
    return SSE