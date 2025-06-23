''''
在sklearn中使用网格搜索调参
'''
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV

X, y = load_iris(return_X_y=True) 
# 进行数据切分
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24)

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

