import cv2
import numpy as np

def harris_corner_detection(image_path):
    # 读取图像并转换为灰度图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行 Harris 角点检测
    dst = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.04)

    # 结果进行膨胀操作
    # dst = cv2.dilate(dst, None)

    # 设定阈值并标记角点
    img[dst > 0.1 * dst.max()] = [0, 0, 255]

    # 显示结果
    cv2.imwrite('test_corner.png', img)

if __name__ == "__main__":
    harris_corner_detection("test.png")
