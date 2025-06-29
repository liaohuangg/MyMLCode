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
reg_grid = RandomForestRegressor(random_state=1412, verbose=False, n_jobs=-1)

# 设置5折交叉验证（打乱数据）
cv_grid = KFold(n_splits=5, shuffle=True, random_state=1412)

# 实例化网格搜索
# search = GridSearchCV(
#     estimator=reg_grid,
#     param_grid=param_grid_simple,          # 使用上述参数网格
#     scoring="neg_mean_squared_error",      # 评估指标为负均方误差
#     verbose=True,                          # 打印详细日志
#     cv=cv_grid,                                 # 使用定义好的交叉验证
#     n_jobs=8                              # 并行使用所有CPU核心
# )

# start = time.time()
# search.fit(X, y)  # 执行网格搜索
# end = time.time()
# print("网格搜索耗时: ", end - start)

# # 查看结果
# print(search.best_params_)
'''
网格搜索耗时:  90.13978624343872
{'criterion': 'squared_error', 'max_depth': 10, 'max_features': 3, 'min_impurity_decrease': np.int64(0), 'n_estimators': 45}
'''
# 打包成函数供后续使用
# 评估指标RMSE
def RMSE(cvresult, key):
    return (abs(cvresult[key])**0.5).mean()

# 计算参数空间大小
def count_space(param):
    no_option = 1
    for i in param_grid_simple:
        no_option *= len(param_grid_simple[i])
    print(no_option)

# 在最优参数上进行重新建模验证结果
def rebuild_on_best_param(ad_reg):
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    result_post_adjusted = cross_validate(ad_reg, X, y, cv=cv, scoring="neg_mean_squared_error",
                                          return_train_score=True,
                                          verbose=True,
                                          n_jobs=-1)
    print("训练RMSE:{:.3f}".format(RMSE(result_post_adjusted, "train_score")))
    print("测试RMSE:{:.3f}".format(RMSE(result_post_adjusted, "test_score")))

# rebuild_on_best_param(search.best_estimator_)

# 随机网格搜索
from sklearn.model_selection import RandomizedSearchCV
# 定义随机搜索
# search = RandomizedSearchCV(estimator=reg_grid
#                            ,param_distributions=param_grid_simple
#                            ,n_iter = 800  # 子空间的大小是全域空间的一半左右
#                            ,scoring = "neg_mean_squared_error"
#                            ,verbose = True
#                            ,cv = cv
#                            ,random_state=1412
#                            ,n_jobs=-1)
# 训练随机搜索评估器
#======【TIME WARNING: 5~10min】======#
# start = time.time()
# search.fit(X,y)
# print(time.time() - start)

# 随机网格搜索29.885427474975586
# 随机网格搜索建议取全域空间大小的一半

# 对半网格搜索 
'''
优化枚举网格搜索
1 调整搜索空间
2 调整每次训练的数据
比起每次使用全部数据来验证一组参数，可以考虑只带入训练数据的子集对超参数进行筛选
条件
任意子集的分布与全数据集分布类似
1、首先从全数据集中无放回随机抽样出一个很小的子集 d₀，并在 d₀上验证全部参数组合的性能。根据 d₀上的验证结果，淘汰评分排在后 1/2 的那一半参数组合
2、然后，从全数据集中再无放回抽样出一个比 d₀大一倍的子集 d₁，并在 d₁上验证剩下的那一半参数组合的性能。根据 d₁上的验证结果，淘汰评分排在后 1/2 的参数组合
3、再从全数据集中无放回抽样出一个比 d₁大一倍的子集 d₂，并在 d₂上验证剩下 1/4 的参数组合的性能。根据 d₂上的验证结果，淘汰评分排在后 1/2 的参数组合……
如此迭代，当参数组合只剩下一组或者剩余可用的数据不足时，停下

这种模式，选出的参数，泛化能力好，并且效率高
但是局限性在于，在最开始迭代的时候就用最小的数据集筛掉了最多的参数组合
整体数据量必须很大

aggressive_elimination参数的作用 
双重停止机制：迭代终止条件包括：① 剩余参数组合≤1 ② 所需资源超过max_resources
激进淘汰模式：当aggressive_elimination=True时，强制用完所有可用资源才停止，可能增加计算时间但结果更可靠
默认策略：参数默认为False，允许提前终止（当任一停止条件触发时），适合大型参数空间的快速筛选

样本量不够了，使用开始用的样本重复，来筛选参数
'''
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
# 初始化 HalvingGridSearchCV
halving_search = HalvingGridSearchCV(
    estimator=reg_grid,
    param_grid=param_grid_simple
    ,scoring = "neg_mean_squared_error"
    ,verbose = False
    ,cv = cv
    ,random_state=1412
    ,n_jobs=-1
    ,factor=1.5 # 每次迭代 样本量倍数
    ,min_resources=500 # 开局使用的样本量
)
print("\nHalvingGridSearchCV\n")
start = time.time()
halving_search.fit(X,y)
print(time.time() - start)
rebuild_on_best_param(halving_search.best_estimator_)