import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.ML_basic_function import *

from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

data = load_diabetes()
# 回归问题
X, y = load_diabetes(return_X_y=True) 
# print("特征名称:", data.feature_names) 
# print(X[:1,:])
# print(y[:10])

# 实例化随机森林
reg_random_forest = RandomForestRegressor()
# 实例化决策树
reg_decision_tree = DecisionTreeRegressor()
# 实例化交叉验证方式
cv = KFold(n_splits=5, shuffle=True, random_state=42)
'''
核心参数
    estimator:要进行交叉验证的评估器(如随机森林或决策树)
    X,y:输入数据和标签
    cv:交叉验证模式(如之前实例化的KFold)
    scoring:评估指标(如"neg_mean_squared_error")
特殊参数:
    return_train_score:
    默认False,不返回训练分数以节省计算资源
    监督过拟合时需要设为True
verbose:
    默认False不打印进程
    集成算法训练时间长时建议设为True
n_jobs:
    控制调用线程数,正整数表示具体线程数
    -1表示调用全部线程(需谨慎使用)
'''
result_t = cross_validate(reg_decision_tree, X, y, cv=cv, scoring='neg_mean_squared_error',
                          return_train_score=True, verbose=True, n_jobs=8)
# print("决策树交叉验证结果:", result_t)
'''
Using backend LokyBackend with 8 concurrent workers
8个线程
{
'fit_time': array([0.00254536, 0.0018518 , 0.00197625, 0.00179505, 0.00195026]),
'score_time': array([0.00054955, 0.00041485, 0.00044489, 0.00044203, 0.000386  ]), 
'test_score': array([-4950.94382022, -7387.40449438, -8405.92045455, -6178.76136364, -7659.        ]), 
'train_score': array([-0., -0., -0., -0., -0.])
}
'''

result_f = cross_validate(reg_random_forest, X, y, cv=cv, scoring='neg_mean_squared_error',
                          return_train_score=True, verbose=True, n_jobs=8)
# print("随机森林交叉验证结果:", result_f)
'''
fit_time, 每次训练模型的时间（秒），例如第一次训练耗时0.118秒。
​​score_time, 每次验证模型的时间（秒），例如第一次验证耗时0.004秒。
​​test_score, 模型在测试集上的负均方误差（Negative MSE），例如第一次测试得分为-2969.59。
​​train_score, 模型在训练集上的负均方误差（Negative MSE），例如第一次训练得分为-466.65。


{'fit_time': array([0.11843348, 0.12874246, 0.11954737, 0.1269908 , 0.13389683]), 
'score_time': array([0.00405693, 0.00365829, 0.00387764, 0.00420928, 0.00431013]), 
'test_score': array([-2969.58939438, -2950.05582697, -4088.75861932, -3581.57666818,-3294.36374205]), 
'train_score': array([-466.64541785, -481.12368017, -436.63737994, -466.66093927,-484.58010282])}
'''

# 计算随机森林模型的训练集和测试集RMSE（均方根误差）
trainRMSE_f = abs(result_f["train_score"])**0.5  # 训练集RMSE（随机森林）
testRMSE_f = abs(result_f["test_score"])**0.5    # 测试集RMSE（随机森林）
trainRMSE_t = abs(result_t["train_score"])**0.5  # 训练集RMSE（其他模型，如决策树）
testRMSE_t = abs(result_t["test_score"])**0.5    # 测试集RMSE（其他模型）

# 输出随机森林的RMSE统计量
print("训练集RMSE均值:", trainRMSE_f.mean())      # 输出: 11115.70
print("测试集RMSE均值:", testRMSE_f.mean())        # 输出: 30277.45
print("训练集RMSE标准差:", trainRMSE_f.std())      # 输出: 475.69

# 定义横轴（5折交叉验证）
xaxis = range(1, 6)

# 设置画布
plt.figure(figsize=(8, 6), dpi=80)

# 绘制随机森林的RMSE曲线
plt.plot(xaxis, trainRMSE_f, color="green", label="RandomForestTrain")          # 训练集（实线）
plt.plot(xaxis, testRMSE_f, color="green", linestyle="--", label="RandomForestTest")  # 测试集（虚线）

# 绘制决策树的RMSE曲线
plt.plot(xaxis, trainRMSE_t, color="orange", label="DecisionTreeTrain")        # 训练集（实线）
plt.plot(xaxis, testRMSE_t, color="orange", linestyle="--", label="DecisionTreeTest")  # 测试集（虚线）

# 设置坐标轴
plt.xticks([1, 2, 3, 4, 5])  # 固定x轴刻度为整数
plt.xlabel("CVcounts", fontsize=16)  # x轴标签
plt.ylabel("RMSE", fontsize=16)      # y轴标签

# 添加图例并显示
plt.legend()
plt.show()
plt.pause(3)
plt.close()

'''
随机森林参数
弱评估器结构

criterion, 弱评估器分枝时的不纯度衡量指标（如Gini、熵、MSE等）
max_depth, 弱评估器被允许的最大深度（默认None，即不限制）
min_samples_split, 分枝时父节点上最少需要的样本个数
min_samples_leaf, 叶子节点上最少需拥有的样本个数
min_weight_fraction_leaf, 样本权重调整时，叶子节点最少需拥有的样本权重
max_leaf_nodes, 弱评估器最多可有的叶子节点数量
min_impurity_decrease, 分枝时允许的最小不纯度下降量
'''