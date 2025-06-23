from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子确保可重复性
np.random.seed(42)

# 创建两个类别的数据点
X_red = np.random.normal(loc=[4, 0], scale=[4, 2], size=(50, 2))
X_blue = np.random.normal(loc=[6, 0], scale=[1, 0.5], size=(50, 2))
X = np.concatenate((X_red, X_blue), 0)

# 定义中心点
center = np.array([[4, 0], [6, 0]])

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制红色类别的数据点和中心点
plt.plot(X_red[:, 0], X_red[:, 1], 'o', c='lightcoral', label='Class A')
plt.plot(center[0, 0], center[0, 1], 'o', c='red', markersize=10, label='Center A')

# 绘制蓝色类别的数据点和中心点
plt.plot(X_blue[:, 0], X_blue[:, 1], 'o', c='cyan', label='Class B')
plt.plot(center[1, 0], center[1, 1], 'o', c='blue', markersize=10, label='Center B')

# 设置坐标轴范围和标签
plt.xlim(-2, 15)
plt.ylim(-2, 2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Binary Classification Dataset Visualization')

# 添加图例
plt.legend()

# 显示图形
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
plt.pause(3)
plt.close()

km =  KMeans(n_clusters=2, random_state=42)
km.fit(X)

print(km.cluster_centers_)

print(km.labels_)

# 绘制数据点（按聚类标签着色）
plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='viridis', s=100)

# 标记聚类中心
plt.plot(km.cluster_centers_[0, 0], km.cluster_centers_[0, 1], 'o', 
         markersize=10, c='red', label='Cluster Center 1')
plt.plot(km.cluster_centers_[1, 0], km.cluster_centers_[1, 1], 'o', 
         markersize=10, c='cyan', label='Cluster Center 2')
# 添加图例和标题
plt.legend()
plt.title('K-Means Clustering Results (n_clusters=2)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, linestyle='--', alpha=0.5)

# 显示图形
plt.show()
plt.pause(3)
plt.close()