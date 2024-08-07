import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 控制点
control_points = np.array([[0, 0], [0.7, 2], [2, 4], [3, 3], [4, 1]])

# Catmull-Rom 样条插值
t = np.arange(len(control_points))
spl = CubicSpline(t, control_points, bc_type='clamped')

# 生成曲线
tt = np.linspace(0, t.max(), 100)
curve = spl(tt)

# 绘制曲线和控制点
plt.plot(curve[:, 0], curve[:, 1], label='Catmull-Rom Spline Curve')
plt.plot(control_points[:, 0], control_points[:, 1], 'o-', label='Control Points')
plt.legend()
plt.show()
