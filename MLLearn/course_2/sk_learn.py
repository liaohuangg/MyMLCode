import sys
import os
# 获取当前文件的父目录的父目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.ML_basic_function import *

# 获取数据集
from sklearn.datasets import load_iris
'''
.data
.target
feature_names
target_names
'''
X, y = load_iris(return_X_y=True) 
# print(iris_data.data[:10])
# iris_data_frame = load_iris(as_frame=True) 
# iris_data_frame.__format__

# 数据集切分
num_inputs = 2
from sklearn.model_selection import train_test_split
data0 = np.random.normal(4, 2, size=(10, num_inputs))  # 均值4，标准差2
label0 = np.zeros(10)
# 默认切分为0.25
newX, newTX, newY, newTY = train_test_split(data0, label0, random_state=42)
#print(train_test_split(data0, label0, random_state=42))

# 数据归一化与标准化
from sklearn import preprocessing
preprocessing.scale(data0)

# 更多的采用评估器进行标准化
from sklearn.preprocessing import StandardScaler 
scalar = StandardScaler()
'''
StandardScaler.fit() 会遍历输入数据(虽然图中未显示数据参数，实际需传入 X)，计算：
每列的均值(mean_)
每列的标准差(std_)
每列的方差(var_)
总共有效的训练数据条数(n_samples_seen_)
可通过scaler.data_min_和scaler.data_max_属性查看各列的最小值和最大值
MaxAbsScaler：针对稀疏矩阵的标准化
RobustScaler：针对存在异常值点的特征矩阵标准化
Non-linear transformation：非线性变化的标准化方法
'''
scalar.fit(newX)
# print(scalar.mean_)

# 使用逻辑回归评估器
from sklearn.linear_model import LogisticRegression
# 实例化模型，使用默认参数
clf_test = LogisticRegression(max_iter=1000)
# 代入所有数据进行训练
clf_test.fit(X, y)
# 查看线性方程系数
# print(clf_test.coef_)
#查看准确率
# print(clf_test.score(X, y))
from sklearn.metrics import accuracy_score
accuracyScore = accuracy_score(y, clf_test.predict(X))
print(accuracyScore)
