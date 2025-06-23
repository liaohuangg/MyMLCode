import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import sys
import os
from sklearn.preprocessing import PolynomialFeatures
# 获取当前文件的父目录的父目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.ML_basic_function import *

# 二阶特征衍生
np.random.seed(24)

n_dots = 20
# 等距分布
x = np.linspace(0, 1, n_dots)
# 加入扰动项
y = np.sqrt(x) + 0.2 * np.random.rand(n_dots) - 0.1
# y = PolynomialFeatures(degree=2).fit_transform(x.reshape(-1,1))
# s = np.polyfit(x, y, 2)
# p = np.poly1d(s)
# print(p)
y0 = x**2
s = np.polyfit(x, y0, 2)
p = np.poly1d(s)

# plot_polynomial_fit(x, y, 3)
# 创建画布（高分辨率大图）
# plt.figure(figsize=(18, 4), dpi=200)  # 宽度18英寸，高度4英寸，DPI=200

# # 定义三个子图的标题
# titles = ['Under Fitting', 'Fitting', 'Over Fitting']

# # 循环绘制三种不同阶数的多项式拟合
# for index, deg in enumerate([1, 3, 10]):  # 分别对应1阶、3阶、10阶多项式
#     plt.subplot(1, 3, index + 1)  # 创建1行3列的子图，当前绘制第index+1个   
#     # 调用自定义的拟合绘图函数（需提前定义plot_polynomial_fit）
#     plot_polynomial_fit(x, y, deg)  # 参数：x特征, y标签, 多项式阶数  
#     # 设置子图标题（大字号）
#     plt.title(titles[index], fontsize=20)
# plt.show()
'''
x:
[[0.30223321 0.93447759]]
y:
[[1.         0.30223321 0.09134491]
 [1.         0.93447759 0.87324836]]
x^0    x^1    x^2
'''
# 二阶特征衍生 只包含交叉项
# y1 = PolynomialFeatures(degree=10, interaction_only=True).fit_transform(x.reshape(-1,1))
# 二阶特征衍生 只包含交叉项
Xtrain, Xtest, ytrain, ytest = array_split(x, y, 0.05, 24, True)
# 数据增强
Xtrain1 = PolynomialFeatures(degree=10, include_bias=False).fit_transform(Xtrain.reshape(-1, 1))
Xtest1 = PolynomialFeatures(degree=10, include_bias=False).fit_transform(Xtest.reshape(-1, 1))
print(Xtrain1.shape)
# print(x1.shape)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(Xtrain1, ytrain)
print("\ncoef\n")
print(lr.coef_)

# 查看过拟合时MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(lr.predict(Xtest1), ytest)
#均方误差
print(mse)


# 观察建模结果
# 生成测试数据点 (假设x, y, lr已提前定义)
# x = ... (原始特征数据，形状应为 (n_samples, n_features))
# y = ... (对应标签数据)
# lr = LinearRegression().fit(x, y) (已训练的线性回归模型)

# 创建0到1之间均匀分布的100个点用于可视化
t = np.linspace(0, 1, 200)

# 绘制图形：
# 1. 红色圆点显示原始数据 (x,y)
# 2. 实线显示线性回归预测结果
# 3. 红色虚线显示t与sqrt(t)的参考曲线

x_l = []
for i in range(10):
    x_l.append(np.power(t, i+1).reshape(-1, 1))  # 10次多项式特征
X = np.concatenate(x_l, 1)  # 合并特征


# 绘制图形：
# 1. 红色圆点显示原始数据 (x,y)
# 2. 实线显示线性回归预测结果
# 3. 红色虚线显示t与sqrt(t)的参考曲线
plt.plot(Xtrain, ytrain, 'ro',          # 原始数据(红色圆点)
         t, lr.predict(X), '-',  # 回归预测(实线)
         t, np.sqrt(t), 'r--')   # t vs sqrt(t)(红色虚线)

# # # 设置标题
plt.title('10-degree')
print("end")
# # 建议补充显示图形
plt.show()
plt.pause(3)
plt.close()
# 过拟合体现为复杂折线