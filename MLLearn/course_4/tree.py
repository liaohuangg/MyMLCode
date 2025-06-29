from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True) 
# 进行数据切分
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24)
print(X_train.shape)
# 实例化模型，使用默认参数
# SAGA的核心是通过​​梯度历史信息的平均​​来减少随机梯度的方差，从而加速收敛
clf = LogisticRegression(max_iter=(int(1e6)), solver='saga')
'''
print(clf.get_params())
可调整参数
{'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 
'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 1000000, 
'multi_class': 'deprecated', 'n_jobs': None, 'penalty': 'l2', 
'random_state': None, 'solver': 'saga', 'tol': 0.0001, 
'verbose': 0, 'warm_start': False}
'''

# 构造参数空间, 可以为一个字典
param_grif_simple = {'penalty': ['l1', 'l2'], 'C': [1, 0.5, 0.1, 0.05, 0.01]}

# 如果考虑弹性网，那么需要考虑l1和l2的比例， 划分两类，构造两个参数空间，封装到一个list中
# param_grid_ra = [
#     {'penalty': ['l1', 'l2'], 'C': [1, 0.5, 0.1, 0.05, 0.01]},
#     {'penalty': ['elasticnet'], 'C': [1, 0.5, 0.1, 0.05, 0.01], 'l1_ratio': [0.3, 0.6, 0.9]}
# ]
search = GridSearchCV(estimator=clf, param_grid=param_grif_simple)
search.fit(X_train, y_train)

# 返回最佳结果
print(search.best_estimator_)
'''
最佳参数
LogisticRegression(C=1, max_iter=1000000, penalty='l1', solver='saga')
'''

# 查看训练误差和测试误差
print(search.best_estimator_.score(X_train, y_train), search.best_estimator_.score(X_test, y_test))
### 三分类问题，第二个分类模型的权重为0 
print(search.best_estimator_.coef_)

'''
[[ 0.          0.         -3.47333853  0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.55513244 -0.3424083   3.03225411  4.12148703]]

可以观察到 1-2,3 分类的权重，只有第三个特征值是有效的
2-3,1 分类的权重为0，说明该特征对分类没有贡献
'''
# 决策边界 当花瓣长度时判定为第一类鸢尾花，否则归为二三类
t = np.array(y)
# print(t[:])

# 将 2 3类划分为一类
t[50:] = 1
# print(t)

# 提取待分类的子数据集
X = np.array(X[t==1])
y = np.array(y[t==1])

# 接下来构建一个包含L1正则化的逻辑回归模型，并通过段调整C的取值，通过观察参数系数的变化情况，挑选重要特征
# C_l 取值1-0.1 取值100个点
C_l = np.linspace(1, 0.1, 100)
coef_l = []
print("\n子数据集分类\n")
for c in C_l:
    clf = LogisticRegression(C=c, penalty='l1', solver='saga', max_iter=int(1e6))
    clf.fit(X, y)
    coef_l.append(clf.coef_.flatten())

# 观察C_l从1-0.1变化时，参数系数的变化情况
# ax = plt.gca()
# ax.plot(C_l, coef_l)
# ax.set_xlim(ax.get_xlim()[::-1])
# plt.xlabel('C')
# plt.ylabel('weights')
# plt.show()
# print(coef_l)

'''
0.        , 0.        , 1.80172824, 0.        ]
可以看到，当模型的约束越大，weight稀疏性强，仍然是第三个特征比较重要
'''
# 故，构建一个C = 0.2的逻辑回归进行训练，此时除第三个特征外，其他参数都归0
clf = LogisticRegression(C=0.2, penalty='l1', solver='saga', max_iter=int(1e6))
clf.fit(X, y)
# print(clf.coef_, clf.intercept_)
# print(clf.score(X, y))
b = - clf.intercept_/clf.coef_[0, 2]
print(b)
plt.plot(X[:, 2][y==1], X[:, 3][y==1], 'ro')      # 红色圆圈标记类别1
plt.plot(X[:, 2][y==2], X[:, 3][y==2], 'bo')      # 蓝色圆圈标记类别2
plt.plot(np.array([b]*20), np.arange(0.5, 2.5, 0.1), 'r--')  # 红色虚线
plt.show()
plt.pause(3)
plt.close()