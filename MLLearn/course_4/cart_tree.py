from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.ML_basic_function import *

# cart分类树 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# 准备数据集
X = np.array([[1, 1], [2, 2], [2, 1], [1, 2], [1, 1], [1, 2], [1, 2], [2, 1]])
y = np.array([0, 0, 0, 1, 0, 1, 1, 0])

# 调用决策树评估器并进行训练
clf = DecisionTreeClassifier().fit(X, y)

acc = clf.score(X, y)  # 输出模型的准确率
print("模型的准确率: ", acc)

# 绘制树状图
# from sklearn import tree
# plt.figure(figsize=(6, 2), dpi=150)
# tree.plot_tree(clf, filled=True)
# plt.show()
# plt.pause(3)
# plt.close()

# cart回归树

# 创建二维坐标数据, 一个特征值与一个连续型标签
data = np.array([[1, 1], [2, 3], [3, 3], [4, 6], [5, 6]])

# 绘制散点图（x: 第一列数据，y: 第二列数据）
# plt.scatter(data[:, 0], data[:, 1])
# plt.show()  # 显示图形

impurity_decrease = []
mse_a = np.power(data[:, 1] - data[:, 1].mean(), 2).sum() / data[:, 1].size  # 另一种计算方式
for i in range(4):
    # 寻找切分点
    splitting_point = data[i: i+2, 0].mean()
    
    # 进行数据集切分
    data_b1 = data[data[:, 0] <= splitting_point]
    data_b2 = data[data[:, 0] > splitting_point]
    
    # 分别计算两个子数据集的MSE
    mse_b1 = np.power(data_b1[:, 1] - data_b1[:, 1].mean(), 2).sum() / data_b1[:, 1].size
    mse_b2 = np.power(data_b2[:, 1] - data_b2[:, 1].mean(), 2).sum() / data_b2[:, 1].size
    
    # 计算两个子数据集整体的MSE
    mse_b = data_b1[:, 1].size/data[:, 1].size * mse_b1 + data_b2[:, 1].size/data[:, 1].size * mse_b2
    # mse_b = mse_b1 + mse_b2  # 这是另一种计算方式
    
    # 计算当前划分情况下MSE下降结果
    impurity_decrease.append(mse_a - mse_b)
print("每次划分的MSE下降结果: ", impurity_decrease)
# 选择MSE下降最大的划分点
best_split = np.argmax(impurity_decrease)

# y_range= np.arange(1, 6, 0.1)   
# # 创建2x2网格中的第一个子图(左上)
# plt.subplot(221)
# plt.scatter(data[:, 0], data[:, 1])  # 绘制第一列和第二列的散点图
# plt.plot(np.full_like(y_range, 1.5), y_range, 'r--')  # 添加x=1.5的红色虚线

# # 创建第二个子图(右上)
# plt.subplot(222)
# plt.scatter(data[:, 0], data[:, 1])
# plt.plot(np.full_like(y_range, 2.5), y_range, 'r--')  # 添加x=2.5的红色虚线

# # 创建第三个子图(左下)
# plt.subplot(223)
# plt.scatter(data[:, 0], data[:, 1])
# plt.plot(np.full_like(y_range, 3.5), y_range, 'r--')  # 添加x=3.5的红色虚线

# # 创建第四个子图(右下)
# plt.subplot(224)
# plt.scatter(data[:, 0], data[:, 1])
# plt.plot(np.full_like(y_range, 4.5), y_range, 'r--')  # 添加x=4.5的红色虚线

# # 显示图形
# plt.tight_layout()  # 自动调整子图间距
# plt.show()
# plt.pause(3)
# plt.close()

# 进行多轮迭代， 都是以一个均值对一个区间内的数据进行预测，实际上是一种"分区定值"