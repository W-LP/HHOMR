import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 创建示例数据
np.random.seed(0)
X = np.dot(np.random.random(size=(2, 2)), np.random.normal(size=(2, 200))).T

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 执行 PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 绘制原始数据和主成分
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.5)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    plt.quiver(pca.mean_[0], pca.mean_[1], v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('PCA of dataset')
plt.grid()
plt.show()
