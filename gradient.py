import cv2
import numpy as np

hessian = np.zeros((2, 2))
hessian[0, 0] = 931
hessian[0, 1] = 303
hessian[1, 0] = 303
hessian[1, 1] = -353

# eigenVal, eigenVect=np.linalg.eig(hessian)
ret, eigenVal, eigenVect = cv2.eigen(hessian)
print(eigenVal)
# print(eigenVal[0,0],eigenVal[1,0])
if np.abs(eigenVal[0]) >= np.abs(eigenVal[1]):
    nx = eigenVect[0, 0]
    ny = eigenVect[0, 1]
else:
    nx = eigenVect[1, 0]
    ny = eigenVect[1, 1]

print(nx, ny)


# 创建一个测试矩阵
matrix = np.array([[931, 303], [303, -353]], dtype=np.float32)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# 找到最大特征值的索引
max_index = np.argmax(eigenvalues)

# 输出最大特征值和对应的特征向量
print("-"*100)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
print("Maximum Eigenvalue:", eigenvalues[max_index])
print("Corresponding Eigenvector:\n", eigenvectors[ :,max_index])

# 创建一个测试矩阵
matrix = np.array([[931, 303], [303, -353]], dtype=np.float32)

# 转换为 cv::Mat
matrix_cv = cv2.Mat(matrix)

# 初始化存储特征值和特征向量的矩阵
eigenvalues_cv = np.zeros(2, dtype=np.float32)
eigenvectors_cv = np.zeros((2, 2), dtype=np.float32)

# 计算特征值和特征向量
ret = cv2.eigen(matrix_cv, eigenvalues_cv, eigenvectors_cv)

# 找到最大特征值的索引
max_index = np.argmax(eigenvalues_cv)

# 输出最大特征值和对应的特征向量
print("-"*100)
print("Eigenvalues (cv2.eigen):", eigenvalues_cv)
print("Eigenvectors (cv2.eigen):\n", eigenvectors_cv)
print("Maximum Eigenvalue (cv2.eigen):", eigenvalues_cv[max_index])
print("Corresponding Eigenvector (cv2.eigen):\n", eigenvectors_cv[max_index])
