import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 生成示例数据
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.1 * np.random.randn(100)  # 加入一些噪声


def find_peaks_simple(data, height=None, threshold=None, distance=10):
    """
    简化版 find_peaks 实现
    :param data: 输入数据
    :param height: 峰值高度的阈值
    :param threshold: 峰值的阈值
    :param distance: 峰值之间的最小距离
    :return: 峰值的索引
    """
    data = np.asarray(data)
    if len(data) < 3:
        return []

    peaks = []
    n = len(data)

    # 检测局部极值
    for i in range(1, n - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:  # 局部最大值
            if (height is None or data[i] >= height) and (threshold is None or data[i] > threshold):
                if not peaks or (i - peaks[-1] >= distance):
                    peaks.append(i)

    return peaks

# 找到极值点
peaks= find_peaks_simple(y)  # 局部最大值
troughs= find_peaks_simple(-y)  # 局部最小值

# 设置窗口范围
window_size = 5

# 复制原始数据
y_scaled_up = y.copy()
y_scaled_down = y.copy()

# 对极值点周围区域进行放大或缩小
for peak in peaks:
    start = max(0, peak - window_size)
    end = min(len(x), peak + window_size)
    y_scaled_up[start:end] = y[start:end] * 2  # 放大2倍

for trough in troughs:
    start = max(0, trough - window_size)
    end = min(len(x), trough + window_size)
    y_scaled_down[start:end] = y[start:end] * 0.5  # 缩小0.5倍

# 绘图
plt.figure(figsize=(12, 8))
plt.plot(x, y, label='Original Data', color='blue')
plt.plot(x, y_scaled_up, label='Scaled Up Around Peaks', color='red', linestyle='--')
plt.plot(x, y_scaled_down, label='Scaled Down Around Troughs', color='green', linestyle='--')

# 标记极值点
plt.scatter(x[peaks], y[peaks], color='red', zorder=5, label='Peaks')
plt.scatter(x[troughs], y[troughs], color='green', zorder=5, label='Troughs')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Effect of Scaling Around Extrema')
plt.legend()
plt.grid(True)
plt.show()
