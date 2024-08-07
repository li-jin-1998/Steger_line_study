import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 生成示例数据（例如，一段带角点的曲线）
t = np.linspace(0, 4 * np.pi, 200)
x = np.cos(t)
y = np.sin(3*t)  # 可以调整这个表达式来生成不同的曲线

# 计算一阶导数（切线方向）
dx = np.gradient(x)
dy = np.gradient(y)

# 计算法线方向
norm = np.sqrt(dx**2 + dy**2)
nx = -dy / norm
ny = dx / norm

# 计算相邻点之间的法线方向变化率
dnx = np.gradient(nx)
dny = np.gradient(ny)
curvature = np.sqrt(dnx**2 + dny**2)

# 找到法线方向变化剧烈的点（即角点）
peaks, _ = find_peaks(curvature, height=np.mean(curvature)*1.5)

# 可视化结果
plt.figure(figsize=(10, 8))
plt.plot(x, y, label='Curve')
plt.quiver(x, y, nx, ny, color='red', angles='xy', scale_units='xy', scale=5, label='Normals')
plt.scatter(x[peaks], y[peaks], color='blue', s=100, label='Corner Points')
plt.legend()
plt.title('Corner Detection using Normal Direction Change')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()
