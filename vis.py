import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_hessian(image):
    # 确保图像是浮点型
    image_float = np.float64(image)

    # 计算一阶导数
    grad_x = cv2.Sobel(image_float, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_float, cv2.CV_64F, 0, 1, ksize=3)

    # 计算二阶导数
    grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
    grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
    grad_xy = cv2.Sobel(grad_x, cv2.CV_64F, 0, 1, ksize=3)

    return grad_xx, grad_xy, grad_yy


def compute_hessian_eigenvectors(grad_xx, grad_xy, grad_yy):
    rows, cols = grad_xx.shape
    normals_x = np.zeros((rows, cols))
    normals_y = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            H = np.array([[grad_xx[i, j], grad_xy[i, j]], [grad_xy[i, j], grad_yy[i, j]]])
            eigenvalues, eigenvectors = np.linalg.eig(H)

            # 选择特征值最大的特征向量
            if eigenvalues[0] > eigenvalues[1]:
                normal_x = eigenvectors[0, 0]
                normal_y = eigenvectors[0, 1]
            else:
                normal_x = eigenvectors[1, 0]
                normal_y = eigenvectors[1, 1]

            normals_x[i, j] = normal_x
            normals_y[i, j] = normal_y

    return normals_x, normals_y


# 读取图像
image = cv2.imread("stripe_image.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found or unable to load.")

# 计算Hessian矩阵的分量
grad_xx, grad_xy, grad_yy = compute_hessian(image)

# 计算Hessian矩阵的特征向量（法向量）
normal_x, normal_y = compute_hessian_eigenvectors(grad_xx, grad_xy, grad_yy)
alpha = np.arctan2(normal_y, normal_x) / np.pi * 180
print(alpha)
# 显示结果
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(alpha, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Hessian XX')
plt.imshow(grad_xx, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Hessian XY')
plt.imshow(grad_xy, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Hessian YY')
plt.imshow(grad_yy, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
