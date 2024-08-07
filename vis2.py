import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_bright_stripes(image):
    # 确保图像是浮点型
    image_float = np.float64(image)
    # image_float = cv2.GaussianBlur(image_float, ksize=(0, 0), sigmaX=2, sigmaY=2)

    # dyFilter = np.array([[1],[0],[-1]])
    # dxFilter = np.array([[1,0,-1]])
    # dxxFilter = np.array([[1],[-2],[1]])
    # dyyFilter = np.array([[1,-2,1]])
    # dxyFilter = np.array([[1,-1],[-1,1]])
    # # compute derivative
    # grad_x = cv2.filter2D(image_float,-1, dxFilter)
    # grad_y = cv2.filter2D(image_float,-1, dyFilter)
    # grad_xx = cv2.filter2D(image_float,-1, dxxFilter)
    # grad_yy = cv2.filter2D(image_float,-1, dyyFilter)
    # grad_xy = cv2.filter2D(image_float,-1, dxyFilter)
    # 计算图像的一阶导数
    grad_x = cv2.Sobel(image_float, cv2.CV_64F, 1, 0, ksize=3)

    grad_y = cv2.Sobel(image_float, cv2.CV_64F, 0, 1, ksize=3)

    # 计算二阶导数
    grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
    grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
    grad_xy = cv2.Sobel(grad_x, cv2.CV_64F, 0, 1, ksize=3)

    # 初始化方向矩阵
    orientation = np.zeros(image.shape, dtype=np.float64)

    # 计算 Hessian 矩阵
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
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
                orientation[i, j] = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])/ np.pi * 180
            else:
                orientation[i, j] = np.arctan2(eigenvectors[1, 1], eigenvectors[1, 0])/ np.pi * 180


    return orientation


# 读取图像并转换为灰度图
image = cv2.imread("stripe_image.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found or unable to load.")

# 检测亮条纹
bright_stripes_orientation = detect_bright_stripes(image)

# 显示结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Bright Stripes Orientation')
plt.imshow(bright_stripes_orientation, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
