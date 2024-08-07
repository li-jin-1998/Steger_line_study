import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# 示例数据
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)

# 高斯平滑
sigma = 1  # 标准差
y_smooth = gaussian_filter1d(y, sigma=sigma)

# 找出峰值和谷值
peaks = (np.diff(np.sign(np.diff(y_smooth))) < 0).nonzero()[0] + 1
valleys = (np.diff(np.sign(np.diff(y_smooth))) > 0).nonzero()[0] + 1

# 设定阈值，过滤掉较小的峰值和谷值
peak_threshold = 0.5 * np.max(y_smooth)
valley_threshold = 0.5 * np.min(y_smooth)
filtered_peaks = peaks[y_smooth[peaks] > peak_threshold]
filtered_valleys = valleys[y_smooth[valleys] < valley_threshold]

# 可视化
plt.plot(x, y, label='Original data')
plt.plot(x, y_smooth, label='Gaussian smoothed data')
plt.scatter(x[filtered_peaks], y_smooth[filtered_peaks], color='red', label='Filtered Peaks')
plt.scatter(x[filtered_valleys], y_smooth[filtered_valleys], color='blue', label='Filtered Valleys')
plt.legend()
plt.show()
