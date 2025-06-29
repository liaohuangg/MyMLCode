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

data_path="/root/scikit_learn_data/data-final.csv"
data_raw=pd.read_csv(data_path, delimiter='\t')

data = data_raw.copy()
pd.options.display.max_columns = 150
# 缺失值处理
data.drop(data.columns[50:107], axis=1, inplace=True)
data.drop(data.columns[51:], axis=1, inplace=True)
print('参与者数量: ', len(data))
data.head()

print('是否缺少任何值? ', data.isnull().values.any())
print('缺少多少值? ', data.isnull().values.sum())
data.dropna(inplace=True)
print('消除缺失值后的参与者数量： ', len(data))

# 归一化
from sklearn.preprocessing import MinMaxScaler
# df = data.drop('country', axis=1)
# columns = list(df.columns)
# scaler = MinMaxScaler(feature_range=(0,1))
# df = scaler.fit_transform(df)
# df = pd.DataFrame(df, columns=columns)

# 创建 K-means Cluster Model
from sklearn.cluster import KMeans
# 使用未归一化数据
df_model = data.drop('country', axis=1)
# 拟合模型
kmeans = KMeans(n_clusters=5, random_state=1412)
k_fit = kmeans.fit(df_model)

# 预测聚类
pd.options.display.max_columns = 10
predictions = k_fit.labels_
print(predictions.shape)
df_model['Clusters'] = predictions
df_model.head()

print("\n每个cluster有多少人\n",df_model.Clusters.value_counts())
# 得到通过K聚类后的 特征与标签
X = df_model.copy()
y = df_model['Clusters'].copy()

from sklearn.decomposition import PCA
'''
PCA是一种无监督的线性降维方法，通过正交变换将高维数据投影到低维空间，保留数据中方差最大的方向（即主成分）。其核心目标是：
​​降维​​：减少特征数量，提高计算效率。
​​去冗余​​：消除特征间的线性相关性。
​​可视化​​：将高维数据压缩到2D/3D便于可视化
'''
# pca = PCA(n_components=2)
# pca_fit = pca.fit_transform(df_model)
# df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])
# df_pca['Clusters'] = predictions
# df_pca.head()

# plt.figure(figsize=(10,10))
# sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Clusters', palette='Set2', alpha=0.8)
# plt.title('Personality Clusters after PCA')
# plt.show()
# plt.pause(3)
# plt.close()

print(X.shape)
print(y.shape)
data_train = X
data_label = y
df_data_train = pd.DataFrame(data_train)
df_data_train.to_csv('/root/scikit_learn_data/data_train_five_personality.csv', index=False)
df_data_label = pd.DataFrame(data_label)
df_data_label.to_csv('/root/scikit_learn_data/data_label_five_personality.csv', index=False)