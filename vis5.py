import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_subpixel_centroid(gray, x, y):
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]

    sum_weights = 0
    sum_x = 0
    sum_y = 0

    for i in range(8):
        nx = x + dx[i]
        ny = y + dy[i]

        if 0 <= nx < gray.shape[1] and 0 <= ny < gray.shape[0]:
            weight = gray[ny, nx]
            sum_weights += weight
            sum_x += nx * weight
            sum_y += ny * weight

    # Include the center pixel
    center_weight = gray[y, x]
    sum_weights += center_weight
    sum_x += x * center_weight
    sum_y += y * center_weight

    return sum_x / sum_weights, sum_y / sum_weights


def visualize_centroid(image, x, y, centroid_x, centroid_y):
    # Create a copy of the image for visualization
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    vis_image = cv2.resize(vis_image, (33, 33), interpolation=cv2.INTER_NEAREST)
    # Draw the original point
    # cv2.circle(vis_image, (x, y), 1, (0, 0, 255), -1)

    # Draw the centroid
    print(vis_image.shape)
    vis_image[int(centroid_y * 11 + 5), int(centroid_x * 11 + 5), :] = (255, 0, 0)

    # Display the image
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Original Point: ({x}, {y})\nCentroid: ({centroid_x:.2f}, {centroid_y:.2f})')
    plt.show()


# 创建一个3x3的示例灰度图像
image = np.array([[191, 23, 223],
                  [210, 229, 199],
                  [167, 205, 208]], dtype=np.uint8)

# 设置示例像素坐标（中心像素）
x = 1
y = 1

# 计算灰度重心
centroid_x, centroid_y = compute_subpixel_centroid(image, x, y)

# 打印结果
print(f"Subpixel centroid: ({centroid_x}, {centroid_y})")

# 可视化结果
visualize_centroid(image, x, y, centroid_x, centroid_y)
