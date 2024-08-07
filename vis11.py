import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# 原始曲线点
original_points = np.array([
    [0, 0], [1, 2], [2, 3], [3, 5], [4, 7], [5, 8],[6,10]
])

# 变形约束
constraints = {
    0: original_points[0],        # 固定第一个点
    len(original_points)-1: original_points[-1],  # 固定最后一个点
    2: np.array([2, 6]),          # 变形中间点，使其更明显
    3: np.array([3, 7])
}

# 计算拉普拉斯矩阵（增加权重）
n = len(original_points)
alpha = 0.5  # 拉普拉斯矩阵的权重
diagonals = [-2 * alpha * np.ones(n), alpha * np.ones(n-1), alpha * np.ones(n-1)]
L = diags(diagonals, [0, -1, 1], format='csc')

# 拉普拉斯变形
B = L.dot(original_points)

# 施加约束
for idx, pos in constraints.items():
    B[idx] = pos
    L[idx] = 0
    L[idx, idx] = 1

# 求解线性系统
deformed_points = spsolve(L, B)

# 绘制原始和变形后的曲线
plt.figure()
plt.plot(original_points[:, 0], original_points[:, 1], 'bo-', label='Original Curve')
plt.plot(deformed_points[:, 0], deformed_points[:, 1], 'ro-', label='Deformed Curve')
plt.legend()
plt.title('Laplacian Curve Deformation')
plt.show()
