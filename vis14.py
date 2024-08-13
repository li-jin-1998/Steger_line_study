import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 定义爱心曲线的参数方程
def heart(t):
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    return x, y

# 定义时间变量
t = np.linspace(0, 2 * np.pi, 1000)

# 初始化图形
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)

line, = ax.plot([], [], 'r-')

# 动画初始化函数
def init():
    line.set_data([], [])
    return line,

# 动画更新函数
def update(frame):
    scale = 1 + 0.3 * np.sin(2 * np.pi * frame / 100)
    x, y = heart(t)
    x *= scale
    y *= scale
    line.set_data(x, y)
    return line,

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=50)

# 显示动画
plt.show()
