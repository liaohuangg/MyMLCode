import time
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
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# load 数据集
data = load_diabetes()
# 回归问题
X, y = load_diabetes(return_X_y=True) 

trainRMSE = np.array([])
testRMSE = np.array([])
trainSTD = np.array([])
testSTD = np.array([])
# 参数搜索
Option = [1, *range(5, 105, 5)]
# 在参数取值中进行循环
for n_estimators in Option:  # Option应为预定义的参数列表，如[1, 5, 10,...,100]

    # 按照当下的参数，实例化模型
    reg_f = RandomForestRegressor(n_estimators=n_estimators, random_state=1412)

    # 实例化交叉验证方式，输出交叉验证结果
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    result_f = cross_validate(reg_f, X, y,  # X,Y需提前定义
                            cv=cv,
                            scoring="neg_mean_squared_error",
                            return_train_score=True,
                            n_jobs=8)  # 使用所有CPU核心并行计算

    # 根据输出的MSE进行RMSE计算（取绝对值的平方根）
    train = abs(result_f["train_score"])**0.5
    test = abs(result_f["test_score"])**0.5

    # 将本次交叉验证中RMSE的均值、标准差保存到数组中
    trainRMSE = np.append(trainRMSE, train.mean())
    testRMSE = np.append(testRMSE, test.mean())
    trainSTD = np.append(trainSTD, train.std())
    testSTD = np.append(testSTD, test.std())

'''
转折点识别:
寻找泛化误差开始上升或转为平稳的转折点
示例：当n_estimators=20时测试误差趋于平稳
激进方案：range(20,100,5)
保守方案：range(15,25,5)
'''
# plot_show_grid_tuning(Option, trainRMSE, testRMSE, trainSTD, testSTD)

# #查看所有的树
# reg_f.estimators_
# reg_f.estimators_[n] 单独查看第n棵树
# #查看单棵树的结构 from sklearn.tree._tree import Tree
# reg_f.estimators_[0].tree_

# 建立benchmark 基线
reg_f_baseline = RandomForestRegressor(random_state=1412)
# 实例化交叉验证方式，输出交叉验证结果
cv_baseline = KFold(n_splits=5, shuffle=True, random_state=1412)
result_f_baseline = cross_validate(reg_f_baseline, X, y,  # X,Y需提前定义
                            cv=cv_baseline,
                            scoring="neg_mean_squared_error",
                            return_train_score=True,
                            verbose=True,
                            n_jobs=8)  # 使用所有CPU核心并行计算
print("基线模型训练RMSE和测试RMES: ", RMES(result_f_baseline))
'''
基线模型训练RMSE和测试RMES:  (np.float64(21.54633065201584), np.float64(57.64272416782055))
'''

# 建立网格参数搜索
# 参数网格定义
param_grid_simple = {
    "criterion": ["squared_error", "poisson"],  # 分裂标准：均方误差或泊松偏差
    "n_estimators": [*range(20, 100, 5)],      # 树数量：20到95，步长5
    "max_depth": [*range(10, 25, 2)],           # 最大深度：10到24，步长2
    "max_features": ["log2", "sqrt", 3, 5, 7, None],  # 特征选择方式
    "min_impurity_decrease": [*np.arange(0, 5, 10)]        # 最小不纯度减少量
}

# 实例化随机森林模型
reg_grid = RandomForestRegressor(random_state=1412, verbose=True, n_jobs=-1)

# 设置5折交叉验证（打乱数据）
cv_grid = KFold(n_splits=5, shuffle=True, random_state=1412)

# 实例化网格搜索
search = GridSearchCV(
    estimator=reg_grid,
    param_grid=param_grid_simple,          # 使用上述参数网格
    scoring="neg_mean_squared_error",      # 评估指标为负均方误差
    verbose=True,                          # 打印详细日志
    cv=cv_grid,                                 # 使用定义好的交叉验证
    n_jobs=8                              # 并行使用所有CPU核心
)

start = time.time()
search.fit(X, y)  # 执行网格搜索
end = time.time()
print("网格搜索耗时: ", end - start)

# 查看结果
print(search.best_params_)
'''
网格搜索耗时:  90.13978624343872
{'criterion': 'squared_error', 'max_depth': 10, 'max_features': 3, 'min_impurity_decrease': np.int64(0), 'n_estimators': 45}
'''