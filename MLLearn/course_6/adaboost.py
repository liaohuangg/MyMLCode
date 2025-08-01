from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_digits
from sklearn.datasets import load_diabetes
# import sklearn
# print(sklearn.__version__)
# 用于分类的数据
X_c, y_c = load_digits(return_X_y=True)
'''
adaboost分类器默认的弱评估器是最大深度为1的树桩
adaboost分类器默认的弱评估器是最大深度为3的树苗
'''
# clf = AdaBoostClassifier(n_estimators=3).fit(X_c, y_c)

# 自建弱评估器
base_e = DecisionTreeClassifier(max_depth=10, max_features=30)
clf = AdaBoostClassifier(estimator=base_e, n_estimators=3).fit(X_c, y_c)
print(clf.estimator_)
# 用于回归的数据集
X_r, y_r = load_diabetes(return_X_y=True) 
reg = AdaBoostRegressor(n_estimators=3).fit(X_r, y_r)
print(reg.estimator_)