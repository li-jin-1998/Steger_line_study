import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance

def fit_line_3d(points):
    """Fit a line to 3D points and return the direction vector and a point on the line."""
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)
    # Compute the covariance matrix
    cov_matrix = np.cov(points.T)
    # Perform eigen decomposition
    _, eigvecs = np.linalg.eigh(cov_matrix)
    # The direction vector is the eigenvector corresponding to the largest eigenvalue
    direction = eigvecs[:, -1]
    return centroid, direction

def compute_angle_3d(dir1, dir2):
    """Compute the angle between two direction vectors in 3D."""
    cos_theta = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical issues
    theta_degrees = np.degrees(theta)
    supplementary_angle = 180 - theta_degrees
    return theta_degrees, supplementary_angle

def plot_lines_and_points_3d(points, line1_point, line1_dir, line2_point, line2_dir, angle_degrees, supplementary_angle, frame):
    """Plot the 3D points and fitted lines."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points in blue
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='Points')

    # Plot the fitted lines
    t = np.linspace(-10, 10, 100)
    line1 = line1_point[:, np.newaxis] + t * line1_dir[:, np.newaxis]
    line2 = line2_point[:, np.newaxis] + t * line2_dir[:, np.newaxis]

    ax.plot(line1[0, :], line1[1, :], line1[2, :], 'r--', label='Fit Line 1')
    ax.plot(line2[0, :], line2[1, :], line2[2, :], 'g--', label='Fit Line 2')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Two Lines and Their Angle: {angle_degrees:.2f} Degrees\nSupplementary Angle: {supplementary_angle:.2f} Degrees')
    ax.legend()
    plt.show()

# Generate random 3D points around two linear trends
np.random.seed(42)
x1 = np.linspace(1, 10, 10)
y1 = 2 * x1 + np.random.normal(0, 1, len(x1))
z1 = 0.5 * x1 + np.random.normal(0, 1, len(x1))

x2 = np.linspace(11, 20, 10)
y2 = 5 * x2 - 30 + np.random.normal(0, 1, len(x2))
z2 = -0.2 * x2 + np.random.normal(0, 1, len(x2))

points = np.vstack((np.column_stack((x1, y1, z1)), np.column_stack((x2, y2, z2))))

max_angle = 0
best_split = None
best_fit = None

# Traverse all possible split points
for pt in range(4, len(points) - 4):
    line1_points = points[pt - 4:pt + 1]
    line2_points = points[pt:pt + 5]

    line1_point, line1_dir = fit_line_3d(line1_points)
    line2_point, line2_dir = fit_line_3d(line2_points)

    angle_degrees, supplementary_angle = compute_angle_3d(line1_dir, line2_dir)

    if angle_degrees > max_angle:
        max_angle = angle_degrees
        best_split = pt
        best_fit = (line1_point, line1_dir, line2_point, line2_dir)

    # Plot the lines and points
    plot_lines_and_points_3d(points, line1_point, line1_dir, line2_point, line2_dir, angle_degrees, supplementary_angle, pt)

print(f'Best split point: {best_split}, Max angle: {max_angle:.2f} degrees')
