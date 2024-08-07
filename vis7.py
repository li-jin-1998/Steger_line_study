import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks


def calculate_curvature(points):
    x = points[:, 0]
    y = points[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
    curvature2 = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
    return curvature,curvature2


def find_local_maxima(curvature):
    local_maxima = []
    for i in range(1, len(curvature) - 1):
        if curvature[i] > curvature[i - 1] and curvature[i] > curvature[i + 1]:
            if curvature[i] - curvature[i - 1] > 0.01 and curvature[i] - curvature[i + 1] > 0.01:
                local_maxima.append(i)
    return local_maxima


def enhance_points(points,curvature, maxima_indices, enhancement_factor=0.5):
    enhanced_points = np.copy(points)
    for idx in maxima_indices:
        if 0 < idx < len(points) - 1:
            # 增加局部最大值点的Y值来增强效果
            print(curvature[idx])
            if curvature[idx] > 0:
                enhanced_points[idx, 1] -= enhancement_factor
            else:
                enhanced_points[idx,1] += enhancement_factor
    return enhanced_points


def interpolate_points(points, num_points=100):
    x = points[:, 0]
    y = points[:, 1]
    cs = CubicSpline(x, y, bc_type='clamped')
    x_interp = np.linspace(x.min(), x.max(), num_points)
    y_interp = cs(x_interp)
    return np.vstack((x_interp, y_interp)).T


# 生成示例数据
points = np.array([
    [0, 0], [1, 1], [2, 2], [3, 2.2], [4, 2.],
    [5, -1], [6, -0.5], [7, -1.5], [8, 0]
])
# 原始曲线插值
interpolated_points = interpolate_points(points)
# 计算曲率
curvature,curvature2 = calculate_curvature(points)

# 找到局部最大值
local_maxima_indices = find_local_maxima(curvature)
# 找到曲率的局部最大值
# local_maxima_indices, _ = find_peaks(curvature)
# peak_points = np.array([x_new[peaks], y_new[peaks]])

# 增强原始点数据
enhanced_points = enhance_points(points, curvature2,local_maxima_indices, enhancement_factor=0.2)



# 增强曲线插值
enhanced_interpolated_points = interpolate_points(enhanced_points)

# 可视化
plt.figure(figsize=(12, 6))

# 原始数据和曲线
plt.subplot(1, 2, 1)
plt.plot(points[:, 0], points[:, 1], 'o', label='Original Points')
plt.plot(interpolated_points[:, 0], interpolated_points[:, 1], '-', label='Original Interpolated Curve')
plt.plot(points[local_maxima_indices, 0], points[local_maxima_indices, 1], 'rx', label='Curvature Peaks')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Curve')
plt.grid()

# 增强曲率后的曲线
plt.subplot(1, 2, 2)
plt.plot(points[:, 0], points[:, 1], 'o', label='Original Points')
plt.plot(interpolated_points[:, 0], interpolated_points[:, 1], '--', label='Original Interpolated Curve')
plt.plot(enhanced_interpolated_points[:, 0], enhanced_interpolated_points[:, 1], '-',
         label='Enhanced Interpolated Curve')
plt.plot(enhanced_points[local_maxima_indices, 0], enhanced_points[local_maxima_indices, 1], 'rx',
         label='Enhanced Curvature Peaks')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Enhanced Curvature Curve')
plt.grid()

plt.tight_layout()
plt.show()
