## 科学计算模块
import numpy as np
import pandas as pd

## 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

## 自定义模块
import sys
import os
# 获取当前文件的父目录的父目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.ML_basic_function import *

## Scikit-Learn相关模块
## 评估器类
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

## 实用函数
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## 数据准备
from sklearn.datasets import load_iris

## 加入评估指标函数
from sklearn.metrics import precision_score, recall_score, f1_score
## 都是基于混淆矩阵
# 真实标签和预测标签
y_true = np.array([1, 0, 0, 1, 0, 1])
y_pred = np.array([1, 1, 0, 1, 0, 1])

# 计算评估指标
precision = precision_score(y_true, y_pred)  # 精确率
recall = recall_score(y_true, y_pred)        # 召回率
f1 = f1_score(y_true, y_pred)               # F1分数

print(f"精确率: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"F1分数: {f1:.2f}")

### 三分类问题
# 真实标签
y_true = np.array([0, 1, 2, 2, 0, 1, 1, 2, 0, 2])

# 预测标签
y_pred = np.array([0, 1, 0, 2, 2, 1, 2, 2, 0, 2])

# 对​​每个类别单独计算召回率​​，然后求所有类别的​​算术平均值
ma = recall_score(y_true, y_pred, average='macro')

# 对​​每个类别单独计算召回率​​，根据每个类别的数量进行加权求和
we = recall_score(y_true, y_pred, average='weighted')

print(ma, we)

# 第一种情况：输入概率值
y_true = np.array([1, 0, 0, 1, 0, 1])
y_pred_proba = np.array([0.9, 0.7, 0.2, 0.7, 0.4, 0.8])
auc_score_proba = roc_auc_score(y_true, y_pred_proba)
print(f"概率输入的AUC分数: {auc_score_proba:.4f}")  # 输出: 0.9444

# 第二种情况：输入分类结果
y_pred_label = np.array([1, 1, 0, 1, 0, 1])
auc_score_label = roc_auc_score(y_true, y_pred_label)
print(f"分类输入的AUC分数: {auc_score_label:.4f}")