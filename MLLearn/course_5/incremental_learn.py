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
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import seaborn as sns

### 使用增量学习时，如果数据量巨大，可以用deque库导入csv最后几行看索引确定数据量
from collections import deque
from io import StringIO


'''
增量学习允许算法不断接入新的数据来拓展当前模型
即允许巨量数据被分成若干子集，分别输入模型进行训练

核心思想：通过分批次增加新树实现持续学习，每批新树对应新数据特征
'''
# X ,y = fetch_california_housing(return_X_y=True)
# # 新数据集
# X_new, y_new = load_diabetes(return_X_y=True)
# # print(X.shape, y.shape)
# # 关闭增量学习
# model = RandomForestRegressor(n_estimators=3, warm_start=True)
# model1 = model.fit(X, y)
# print(mean_squared_error(y, model1.predict(X))**0.5)
# print(model1.estimators_)
# model.set_params(n_estimators=6)  # 增加3棵树
# model2 = model.fit(X_new[:,:8], y_new)
# print(mean_squared_error(y, model1.predict(X))**0.5)
# print(model2.estimators_)

# 模型没有保留之前训练的树
'''
warm_start=False时，模型会重新训练所有的树
0.30417018089162245
[DecisionTreeRegressor(max_features=1.0, random_state=134643941), 
DecisionTreeRegressor(max_features=1.0, random_state=1539752309), 
DecisionTreeRegressor(max_features=1.0, random_state=1502550471)]
202.26806673951856
[DecisionTreeRegressor(max_features=1.0, random_state=1319664467), 
DecisionTreeRegressor(max_features=1.0, random_state=1797709805), 
DecisionTreeRegressor(max_features=1.0, random_state=676535773)]
'''

# 开启增量学习, 模型之前学习的数据不会改变
'''
不增加新的树n_estimators，就不会训练新的模型
0.2922750446376418
[DecisionTreeRegressor(max_features=1.0, random_state=1406190302),
DecisionTreeRegressor(max_features=1.0, random_state=365477204), 
DecisionTreeRegressor(max_features=1.0, random_state=1256793119)]
0.2922750446376418
[DecisionTreeRegressor(max_features=1.0, random_state=1406190302), 
DecisionTreeRegressor(max_features=1.0, random_state=365477204), 
DecisionTreeRegressor(max_features=1.0, random_state=1256793119)]

model.set_params(n_estimators=6)  # 增加3棵树
0.29345138683599353
[DecisionTreeRegressor(max_features=1.0, random_state=1308199249), 
DecisionTreeRegressor(max_features=1.0, random_state=896489028), 
DecisionTreeRegressor(max_features=1.0, random_state=1623988066)]
123.301225255362
[DecisionTreeRegressor(max_features=1.0, random_state=1308199249), 
DecisionTreeRegressor(max_features=1.0, random_state=896489028), 
DecisionTreeRegressor(max_features=1.0, random_state=1623988066), 
DecisionTreeRegressor(max_features=1.0, random_state=873925024), 
DecisionTreeRegressor(max_features=1.0, random_state=1064999809), 
DecisionTreeRegressor(max_features=1.0, random_state=195180312)]
'''

'''
训练大量数据
'''
# 定义测试集
# df = pd.read_csv('/root/scikit_learn_data/data_train_five_personality.csv')
# X_five = df.to_numpy()  # 或 df.values（旧版）
# df = pd.read_csv('/root/scikit_learn_data/data_label_five_personality.csv')
# y_five = df.to_numpy()  # 或 df.values（旧版）
# print(X_five.shape)
# print(y_five.shape)

# with open('/root/scikit_learn_data/data_train_five_personality.csv', 'r') as data:
#     q = deque(data, 5)
#     print(q)

# 如果数据没有索引，使用pandas的skiprow与nrows进行尝试
# for i in range(0, 10**7, 1000000):
#     df = pd.read_csv('/root/scikit_learn_data/data_train_five_personality.csv', skiprows=i, nrows=1)
#     print(i)
xtrainpath = '/root/scikit_learn_data/data_train_five_personality.csv'
ytrainpath = '/root/scikit_learn_data/data_label_five_personality.csv'
# 建立增量学习使用的模型，定义测试集
reg_incre = RandomForestRegressor(n_estimators=10, random_state=1412, warm_start=True, verbose=True, n_jobs=8)
looprange = range(0, 10**6, 50000)
for line in looprange:
    if line == 0:
        # 首次读取时，保留列名，并且不增加树的数量
        header = "infer"
        newtree = 0
    else:
        # 非首次读取时，不要列名，每次增加10棵树
        header = None
        newtree = 10

    xtrainsubset = pd.read_csv(xtrainpath, header = header, skiprows=line, nrows=50000)
    ytrainsubset = pd.read_csv(ytrainpath, header = header, skiprows=line, nrows=50000)
    Xtrain = xtrainsubset.iloc[:,:]
    Ytrain = ytrainsubset.iloc[:]
    print(Xtrain.shape, Ytrain.shape)
    reg_incre.n_estimators += newtree
    reg_incre = reg_incre.fit(Xtrain,Ytrain)
    print("DONE",line+50000)

    # 当训练集的数据量小于50000时，打断循环
    if Xtrain.shape[0] < 50000:
        break
print(reg_incre.estimators_)
# test_data = pd.read_csv(input_file, skiprows=range(1, skip_rows+1))
print(f"正在从第 {line+1} 行开始读取 {20000} 行数据作为测试集...")
xtest = pd.read_csv(xtrainpath, header = None, skiprows=line, nrows=20000)
ytest = pd.read_csv(ytrainpath, header = None, skiprows=line, nrows=20000)
s = reg_incre.score(xtest, ytest)
print("\n分数\n",s)