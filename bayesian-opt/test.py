# from bayes_opt import BayesianOptimization #导入库
# 比较适合连续性变量的调参
# 调节树的最大深度
# 学习率
# 基评估器的数量
# 只能找目标函数的最大值
# 基本工具
import numpy as np
import pandas as pd
import time
import os

# 算法/损失/评估指标等
import sklearn
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import KFold, cross_validate

# 优化器
import bayes_opt
from bayes_opt import BayesianOptimization

import hyperopt
from hyperopt import hp, fmin, tpe, Trials, partial
from hyperopt.early_stop import no_progress_loss

import optuna
# print(optuna.__version__)
# print(hyperopt.__version__)


# 基于bayes_opt实现GP优化
# 算法空间中带有大量连续性参数是 ,才考虑bayes_opt库

'''
目标函数的值即 f(x) 的值 贝叶斯优化会计算 f(x) 在不同x上的观测值 因此 f(x) 的计算方式需要被明确
在HPO过程中 希望筛选令模型泛华能力最大的参数组合 因此 f(x) 应该是损失函数的交叉验证值或者某种评估指标的交叉验证值

1. 目标函数的输入必须是具体的超参数 ,而不能是整个超参数空间 ,
更不能是数据、算法等超参数以外的元素 ,因此在定义目标函数时 ,
我们需要让超参数作为目标函数的输入。

2. 超参数的输入值只能是浮点数 ,不支持整数与字符串。
因此当算法的实际参数需要输入字符串时 ,该参数不能使用bayes_opt进行调整 ,
当算法的实际参数需要输入整数时 ,则需要在目标函数中规定参数的类型。

3. bayes_opt只支持寻找f(x)的最大值 ,不支持寻找最小值。
因此当我们定义的目标函数是某种损失时 ,目标函数的输出需要取负（即 ,如果使用RMSE ,
则应该让目标函数输出负RMSE ,这样最大化负RMSE后 ,才是最小化真正的RMSE。 )当我们定义的目标函数是准确率 ,
或者auc等指标 ,则可以让目标函数的输出保持原样
'''

# 定义需要的目标函数
def bayesopt_objective(n_estimators, max_depth, max_features, min_impurity_decrease):
    # 定义评估器
    # 需要调整的超参数等于目标函数的输入 ,不需要调整的超参数则直接等于固定值
    # 默认参数输入一定是浮点数 ,因此需要套上 int 函数处理成整数
    reg = RFR(n_estimators=int(n_estimators),
              max_depth=int(max_depth),
              max_features=int(max_features),
              min_impurity_decrease=min_impurity_decrease,
              random_state=1412,
              verbose=False,  # 可自行决定是否开启森林建树的 verbose
              n_jobs=-1)

    # 定义损失的输出 ,5折交叉验证下的结果 ,输出负根均方误差(-RMSE)
    # 注意 ,交叉验证需要使用数据 ,但我们不能让数据X,y成为目标函数的输入
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    validation_loss = cross_validate(reg, X, y,
                                     scoring="neg_root_mean_squared_error",
                                     cv=cv,
                                     verbose=False,
                                     n_jobs=-1,
                                     error_score='raise'
                                     # 如果交叉验证中的算法执行报错 ,则告诉我们错误的理由
                                     )

    # 交叉验证输出的评估指标是负根均方误差 ,因此本来就是负的损失
    # 目标函数可直接输出该损失的均值
    return np.mean(validation_loss["test_score"])

param_grid_simple = {
    'n_estimators': (80, 100),
    'max_depth': (10, 25),
    "max_features": (10, 20),
    "min_impurity_decrease": (0, 1)
}

def param_bayes_opt(init_points, n_iter):
    # 定义优化器，先实例化优化器
    opt = BayesianOptimization(bayesopt_objective  # 需要优化的目标函数
                             , param_grid_simple  # 备选参数空间
                             , random_state=1412  # 随机数种子，虽然无法控制住
                             )

    # 使用优化器，记住bayes_opt只支持最大化
    opt.maximize(init_points=init_points  # 抽取多少个初始观测值
                , n_iter=n_iter  # 一共观测/迭代多少次
                )

    # 优化完成，取出最佳参数与最佳分数
    params_best = opt.max["params"]
    score_best = opt.max["target"]

    # 打印最佳参数与最佳分数
    print("\n", "\n", "best params: ", params_best,
          "\n", "\n", "best cvscore: ", score_best)

    # 返回最佳参数与最佳分数
    return params_best, score_best

# 定义参数空间

if __name__ == "__main__":
    # start = time.time()
    # params_best, score_best = param_bayes_opt(20, 280)  # 初始看20个观测值，后面迭代280次
    # print('It takes %s minutes' % ((time.time() - start)/60))
    # validation_score = bayes_opt_validation(params_best)
    # print("\n", "\n", "validation_score: ", validation_score)
    data = pd.read_csv(r"/root/workspace/MyMLCode/dataset/test.csv")

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    # print(X.shape)
    print(X)
    print(y)
    A = np.array([[1, 2], [1, 1]])
    AM = np.mat(A) #将数组转化为矩阵
    A.reshape(1,4)
    A.T#转置
    np.eye(2)#生成n×n单位矩阵
    np.diag(A)#将一维数组置于对角线上
    np.triu(A)
    
    #逐元素，点积np.dot(a,a)或a.dot(a)
    # np.vdot(a,a)
    # np.inner(a,a)
    '''
    偏移控制：np.triu(a1,k)中k控制清零区域
    k=0：默认情况，严格上三角
    k=-1：左下偏移，保留对角线下一层
    k=1：右上偏移，对角线也被清零
    '''
    '''
    matrix可直接用*进行矩阵乘法
    array需使用.dot()方法或@运算符进行矩阵乘法
    新版支持: 新版NumPy支持使用@符号进行数组的矩阵乘法运算
    
    matrix乘法: AM * AM 直接得到矩阵乘法结果
    array乘法:
    传统方法: A.dot(A)
    新方法: A @ A
    '''
    
    np.trace(A) #矩阵的迹 非方阵取行号和列号相同的位置元素求和，直到无法继续为止
    np.linalg.matrix_rank(A)# 矩阵的秩 矩阵中行或列的极大线性无关组的个数
    np.linalg.matrix_inv(A)# 矩阵的逆 若矩阵和矩阵相乘得到单位矩阵，即，则称为的逆矩阵，记作A^(-1)
    np.linalg.det(A) #求行列式 必须为方阵