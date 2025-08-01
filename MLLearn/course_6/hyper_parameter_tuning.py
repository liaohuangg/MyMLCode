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

# 枚举网格搜索 经典 基础
# 贝叶斯优化调参 最成熟
# 基于梯度的各类优化方法 新型方向
# 基于种群的优化方法 进化，遗传