from ML_basic_function import *
'''
线性回归模型
'''
np.random.seed(24)
# 创建一个扰动不大的数据集
f, l = arrayGenReg(delta = 0.01)

# 随机取一组参数
w = np.random.randn(3).reshape(-1,1)

# 计算SSE
SSE = SSELoss(f, w, l)
# print("random")
# print(SSE)
hls = np.linalg.det(f.T.dot(f))
# if hls != 0:
#     print("can")
# else :
#     print("can't")
w = np.linalg.inv(f.T.dot(f)).dot(f.T).dot(l)
# print(w)
now_SSE = SSELoss(f, w, l)
# print("now")
# print(now_SSE)
# 最小二乘法求解
neww = np.linalg.lstsq(f,l,rcond = -1)
# print(neww)

f1, l1 = arrayGenReg(w = [2, 1], deg = 3, delta = 0.01)
# plt.plot(f1[:,0], l1, 'o')
# plt.savefig("/root/workspace/MyMLCode/MLLearn/course-1/output1.png")

w1 = np.linalg.lstsq(f1,l1,rcond = -1)[0] #最小二乘法获得模型参数
# 进行评估
sse1 = SSELoss(f1,w1,l1)
# 很大的数值，
print(sse1)