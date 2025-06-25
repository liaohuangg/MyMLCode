from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

# 生成月亮形状数据集
X, y = make_moons(n_samples=200, noise=0.05, random_state=24)

# 绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

# 添加图表修饰
plt.title('Moon-shaped Dataset Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, linestyle='--', alpha=0.3)

# 显示图形
plt.show()