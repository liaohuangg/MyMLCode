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

# 数据生成器，生成回归问题的数据集
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


# 数据生成器， 生成分类问题的数据集
def arrayGenCla(num_examples=500, num_inputs=2, num_class=3, deg_dispersion=[4, 2], bias=False):
    """
    分类数据集创建函数。
    
    :param num_examples: 每个类别的数据数量
    :param num_inputs: 数据集特征数量
    :param num_class: 数据集标签类别总数
    :param deg_dispersion: 数据分布离散程度参数，列表格式：
                          [0] 类别均值的参考值，[1] 随机数组标准差
    :param bias: 是否添加全1列作为截距项
    :return: 特征张量float型二维数组 标签张量int型二维数组
    """
    cluster_1 = np.empty([num_examples, 1])  # 每一类标签数组的形状
    mean_ = deg_dispersion[0]                # 类别均值的参考值
    std_ = deg_dispersion[1]                 # 标准差
    lf = []                                  # 存储特征的列表
    ll = []                                  # 存储标签的列表
    k = mean_ * (num_class - 1) / 2          # 特征均值的惩罚因子
    
    # 此处应补充数据生成逻辑（图片未显示完整部分）
    # 示例：生成多类正态分布数据并合并
    for i in range(num_class):
        data = np.random.normal(
            loc=mean_ * i - k,               # 均值按类别偏移， 让随机生成的点簇在原点附近
            scale=std_,
            size=(num_examples, num_inputs)
        )
        labels = np.full_like(cluster_1, i)
        lf.append(data)
        ll.append(labels)
    
     # 合并所有类别数据
    features = np.concatenate(lf)
    labels = np.concatenate(ll)
    
    if bias == True:
        features = np.concatenate((features, np.ones(labels.shape)), 1) #添加一列全为1
    
    return features, labels


def array_split(features, labels, test_size = 0.2, random_state = None, shuffle = True):
    # test_size 训练集和测试集的比例
    # random_state 随机种子
    # 验证输入
    if len(features) != len(labels):
        raise ValueError(f"特征样本数({len(features)})与标签数({len(labels)})不匹配")
    
    # 生成随机索引
    n_samples = len(features)
    indices = np.arange(n_samples)
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(indices) #打乱索引
    
    # 计算分割点
    split_idx = int(n_samples * (1 - test_size))
    
    # 划分数据
    X_train = features[indices[:split_idx]]
    X_test = features[indices[split_idx:]]
    y_train = labels[indices[:split_idx]]
    y_test = labels[indices[split_idx:]]
    
    return X_train, X_test, y_train, y_test

def logit_cla(yhat, thr = 0.5):
    for i in range(len(yhat)):
        if yhat[i] >= thr:
            yhat[i] = 1
        else :
            yhat[i] = 0
    return yhat

def sigmoid(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    z = X.dot(w)
    S = 1 / (1 + np.exp(-z))
    return S

def grad_func(X, w, y) :
    '''
    logistic regression gradient descent
    '''
    print(w)
    print(X[:10])
    m = X.shape[0]
    grad = X.T.dot(sigmoid(X, w) - y) / m
    return grad


def lr_lambda(i) :
    return 0.95

def sgd_cal(Xtrain, w, ytrain, batch_size, lr_init, epoch = 1) :
    # 分批训练
    n_samples = Xtrain.shape[0]
    # for i in range(epoch):
    # 随机打乱数据
    indices = np.random.permutation(n_samples)
    X_shuffled = Xtrain[indices]
    y_shuffled = ytrain[indices]
    
    times = (int)(n_samples / batch_size)
    # 分批训练
    for i in range(times):
        start = i * batch_size
        end = start + batch_size
        if end > n_samples:
            end = n_samples
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        # 计算梯度并更新权重
        m = X_batch.shape[0]
        # print(w)
        # print(X_batch[:10])
        gradient = X_batch.T.dot(sigmoid(X_batch, w) - y_batch) / m
        # w -= lr_init * lr_lambda(epoch) * gradient
        w -= lr_init *  gradient
            
            
def logit_acc(
    X: np.ndarray, 
    w: np.ndarray, 
    y: np.ndarray, 
    thr: float = 0.5
) -> float:
    """
    计算逻辑回归模型的分类准确率
    
    参数:
        X : 特征矩阵 (n_samples, n_features)
        w : 权重向量 (n_features, 1)
        y : 真实标签 (n_samples, 1) 或 (n_samples,)
        thr : 分类阈值 (默认0.5)
    
    返回:
        预测准确率 (0.0 ~ 1.0)
    """
    # 计算预测概率
    yhat = sigmoid(X, w)  # sigmoid(Xw)
    
    # 二值化预测结果
    y_cal = logit_cla(yhat, thr)
    
    # 确保y是列向量
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # 计算准确率
    return (y_cal == y).mean()