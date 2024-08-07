import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# 定义一组二维散点
# points = np.array([
#     [0, 0], [1, 1], [2, 0.5], [3, 1.5], [4, 0],
#     [5, -1], [6, -0.5], [7, -1.5], [8, 0]
# ])
points = np.array([
    [0, 0], [1, 0], [2, 1], [3, 2], [4, 3],
    [5, 3], [6, 3.1], [7, 2.9], [8, 3]
])

# 将数据点拆分为x和y坐标
x = points[:, 0]
y = points[:, 1]

# 使用二次多项式插值
f = interp1d(x, y, kind='quadratic')
x_new = np.linspace(x.min(), x.max(), 9)
y_new = f(x_new)

# 计算插值后的导数
dx = np.gradient(x_new)
dy = np.gradient(y_new)
d2x = np.gradient(dx)
d2y = np.gradient(dy)

# 计算曲率
curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5

# 找到曲率的局部最大值
peaks, _ = find_peaks(curvature)
peak_points = np.array([x_new[peaks], y_new[peaks]])

# 找到原始数据中最接近这些峰值点的点
def find_nearest_points(peak_points, original_points):
    nearest_points = []
    for peak in peak_points.T:
        distances = np.linalg.norm(original_points - peak, axis=1)
        nearest_index = np.argmin(distances)
        nearest_points.append(original_points[nearest_index])
    return np.array(nearest_points)

nearest_points = find_nearest_points(peak_points, points)

# 打印结果
print("Coordinates of curvature peaks in interpolated data:", peak_points)
print("Nearest original points:", nearest_points)

# 可视化结果
plt.figure()
plt.plot(x, y, 'o', label='Original Points')
plt.plot(x_new, y_new, '-', label='Quadratic Interpolation')
plt.plot(peak_points[0], peak_points[1], 'ro', label='Curvature Peaks (Interpolated)')
plt.plot(nearest_points[:, 0], nearest_points[:, 1], 'go', label='Nearest Original Points')
plt.legend()
plt.title("Curvature Peaks and Nearest Original Points")
plt.show()

# 可视化曲率变化
plt.figure()
plt.plot(x_new, curvature, label='Curvature')
plt.plot(x_new[peaks], curvature[peaks], 'ro', label='Curvature Peaks')
plt.legend()
plt.title("Curvature Plot")
plt.show()

