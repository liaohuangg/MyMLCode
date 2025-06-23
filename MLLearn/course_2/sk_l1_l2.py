import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import sys
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
# 获取当前文件的父目录的父目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.ML_basic_function import *


# 解决过拟合风险
# 二阶特征衍生
np.random.seed(24)

n_dots = 20
# 等距分布
x = np.linspace(0, 1, n_dots)
# 加入扰动项
y = np.sqrt(x) + 0.2 * np.random.rand(n_dots) - 0.1

Xtrain, Xtest, ytrain, ytest = array_split(x, y, 0, 24, True)
# 数据增强
Xtrain1 = PolynomialFeatures(degree=10, include_bias=False).fit_transform(Xtrain.reshape(-1, 1))
# Xtest1 = PolynomialFeatures(degree=10, include_bias=False).fit_transform(Xtest.reshape(-1, 1))
# 岭回归
reg_rid = Ridge(alpha=0.005)
reg_rid.fit(Xtrain1, ytrain)

# L1正则化
# reg_las = Lasso(alpha=0.001)

print(reg_rid.coef_)
# 查看过拟合时MSE
from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(reg_rid.predict(Xtest1), ytest)
# #均方误差
# print(mse)

t = np.linspace(0, 1, 200)

# 绘制图形：
# 1. 红色圆点显示原始数据 (x,y)
# 2. 实线显示线性回归预测结果
# 3. 红色虚线显示t与sqrt(t)的参考曲线

x_l = []
for i in range(10):
    x_l.append(np.power(t, i+1).reshape(-1, 1))  # 10次多项式特征
X = np.concatenate(x_l, 1)  # 合并特征


plt.plot(Xtrain, ytrain, 'ro',          # 原始数据(红色圆点)
         t, reg_rid.predict(X), '-',  # 回归预测(实线)
         t, np.sqrt(t), 'r--')   # t vs sqrt(t)(红色虚线)

# # # 设置标题
plt.title('10-degree-Ridge')
print("end")
# # 建议补充显示图形
plt.show()
plt.pause(3)
plt.close()
# ridge 生成的曲线，过拟合得到抑制
#