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

# cart分类树  cart回归树
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

'''
很多超参数
'''
print(DecisionTreeClassifier?)