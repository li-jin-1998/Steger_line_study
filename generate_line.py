import cv2
import numpy as np


def create_stripe_image(width, height, stripe_width, stripe_color, angle):
    # 创建一个空的灰度图像
    img = np.zeros((height, width), dtype=np.uint8)

    # 计算中心点
    center_x, center_y = width // 2, height // 2

    # 创建一个旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # 创建条纹图像
    stripe = np.zeros((height, width), dtype=np.uint8)
    cv2.line(stripe, (center_x - stripe_width // 2, 0), (center_x - stripe_width // 2, height), stripe_color,
             stripe_width)

    # 旋转条纹图像
    stripe_rotated = cv2.warpAffine(stripe, rotation_matrix, (width, height))

    # 将条纹图像添加到空图像中
    img = cv2.add(img, stripe_rotated)

    # 创建一个高斯模糊的条纹以实现边缘渐变
    # blurred_stripe = cv2.GaussianBlur(stripe_rotated, (21, 21), 0)

    return stripe_rotated


# 图像参数
width = 100
height = 100
stripe_width = 5
stripe_color = 255  # 灰度值 (0-255)
angle = 135  # 条纹角度

# 生成条纹图像
stripe_image = create_stripe_image(width, height, stripe_width, stripe_color, angle)

# 保存图像
cv2.imwrite('stripe_image.png', stripe_image)

