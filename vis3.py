import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义生成一维高斯核函数
def gaussian_1d_kernel(size, sigma):
    kernel = cv2.getGaussianKernel(size, sigma)
    return kernel.flatten()

# 定义生成二维高斯核函数
def gaussian_2d_kernel(size, sigma):
    kx = cv2.getGaussianKernel(size, sigma)
    ky = cv2.getGaussianKernel(size, sigma)
    kernel = np.outer(kx, ky)
    return kernel

# 计算一维高斯核的一阶导数
def gaussian_1d_derivative(size, sigma):
    kernel = gaussian_1d_kernel(size, sigma)
    x = np.linspace(-(size // 2), size // 2, size)
    derivative = -x / (sigma ** 2) * kernel
    return derivative

# 计算二维高斯核的一阶导数
def gaussian_2d_derivative(size, sigma):
    kernel = gaussian_2d_kernel(size, sigma)
    x = np.linspace(-(size // 2), size // 2, size)
    y = np.linspace(-(size // 2), size // 2, size)
    X, Y = np.meshgrid(x, y)
    Gx = -X / (sigma ** 2) * kernel
    Gy = -Y / (sigma ** 2) * kernel
    return Gx, Gy

# 对图像应用卷积操作
def apply_convolution(image, kernel):
    return cv2.filter2D(image,cv2.CV_64F, kernel)

# 参数设置
size = 3
sigma = 1.0

# 读取图像并转换为灰度图
image = cv2.imread("stripe_image.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Unable to load image.")
    exit()

# 生成二维高斯核及其导数
gaussian_2d = gaussian_2d_kernel(size, sigma)
gaussian_2d_deriv_x, gaussian_2d_deriv_y = gaussian_2d_derivative(size, sigma)

# 对图像应用高斯模糊
smoothed_image = apply_convolution(image, gaussian_2d)

# 对图像应用高斯导数
gradient_x = apply_convolution(image, gaussian_2d_deriv_x)
gradient_y = apply_convolution(image, gaussian_2d_deriv_y)

grad_xx = apply_convolution(gradient_x, gaussian_2d_deriv_x)
grad_yy = apply_convolution(gradient_y, gaussian_2d_deriv_y)
grad_xy = apply_convolution(gradient_x, gaussian_2d_deriv_y)
# 初始化方向矩阵
orientation = np.zeros(image.shape, dtype=np.float64)

# 计算 Hessian 矩阵
rows, cols = image.shape
for i in range(rows):
    for j in range(cols):
        if image[i, j] <50:
            continue
        H = np.array([[grad_xx[i, j], grad_xy[i, j]], [grad_xy[i, j], grad_yy[i, j]]])
        # eigenvalues, eigenvectors = np.linalg.eig(H)
        # # 选择特征值最大的特征向量，亮条纹特征值较大
        # if eigenvalues[0] > eigenvalues[1]:
        #     orientation[i, j] = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])/ np.pi * 180
        # else:
        #     orientation[i, j] = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])/ np.pi * 180

        ret, eigenvalues, eigenvectors = cv2.eigen(H)
        # if np.abs(eigenvalues[0]) > np.abs(eigenvalues[1]):
        if eigenvalues[0] > eigenvalues[1]:
            orientation[i, j] = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) / np.pi * 180
        else:
            orientation[i, j] = np.arctan2(eigenvectors[1, 1], eigenvectors[1, 0]) / np.pi * 180
print(orientation)
# 计算梯度幅值

# 显示结果
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 2, 2)
plt.imshow(smoothed_image, cmap='gray')
plt.title('Smoothed Image')
plt.subplot(2, 2, 3)
plt.imshow(gradient_x, cmap='gray')
plt.title('Gradient X')
plt.subplot(2, 2, 4)
plt.imshow(gradient_y, cmap='gray')
plt.title('Gradient Y')
plt.figure(figsize=(6, 6))
plt.imshow(orientation, cmap='gray')
plt.title('orientation')
plt.show()
