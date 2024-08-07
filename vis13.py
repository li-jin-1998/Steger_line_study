import numpy as np
import matplotlib.pyplot as plt

# 示例数据
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, len(x))

# 移动窗口大小
window_size = 10

# 初始化平滑后的数据
x_smooth = []
y_smooth = []

# 遍历数据点
for i in range(len(x) - window_size + 1):
    # 当前窗口的数据
    x_window = x[i:i+window_size]
    y_window = y[i:i+window_size]

    # 对窗口内的数据进行拟合（使用一阶多项式，即线性拟合）
    coefficients = np.polyfit(x_window, y_window, 1)
    poly = np.poly1d(coefficients)

    # 计算拟合曲线上的值
    x_smooth.append(x_window[window_size // 2])
    y_smooth.append(poly(x_window[window_size // 2]))

# 转换为NumPy数组
x_smooth = np.array(x_smooth)
y_smooth = np.array(y_smooth)

# 可视化原始数据和平滑后的数据
plt.scatter(x, y, label='Original Data', color='blue', s=10)
plt.plot(x_smooth, y_smooth, label='Smoothed Data', color='red')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot Smoothing with Moving Window Fitting')
plt.show()
