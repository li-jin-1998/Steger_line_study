import os

import imageio
import numpy as np
from matplotlib import pyplot as plt


def fit_line(points):
    """Fit a line to the given points and return the slope and intercept."""
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


def compute_intersection(m1, c1, m2, c2):
    """Compute the intersection point of two lines."""
    x_intersect = (c2 - c1) / (m1 - m2)
    y_intersect = m1 * x_intersect + c1
    return x_intersect, y_intersect


def compute_angle(m1, m2):
    """Compute the angle between two lines given their slopes."""
    v1 = np.array([1, m1])
    v2 = np.array([1, m2])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(cos_theta)
    theta_degrees = np.degrees(theta)
    supplementary_angle = 180 - theta_degrees
    return theta_degrees, supplementary_angle


def perpendicular_distance(m, c, x, y):
    """Compute the perpendicular distance from a point (x, y) to the line y = mx + c."""
    return np.abs(m * x - y + c) / np.sqrt(m ** 2 + 1)


def midline_between_lines(m1, c1, m2, c2, x_range):
    """Compute the midline between two lines y = m1*x + c1 and y = m2*x + c2 over a given x range."""
    x1, x2 = x_range
    y1_line1 = m1 * x1 + c1
    y2_line1 = m1 * x2 + c1
    y1_line2 = m2 * x1 + c2
    y2_line2 = m2 * x2 + c2

    mid_y1 = (y1_line1 + y1_line2) / 2
    mid_y2 = (y2_line1 + y2_line2) / 2

    m_mid = (mid_y2 - mid_y1) / (x2 - x1)
    c_mid = mid_y1 - m_mid * x1

    return m_mid, c_mid


def plot_lines_and_points(points, m1, c1, m2, c2, intersection, theta_degrees, supplementary_angle, split_index, frame,
                          m_mid, c_mid):
    """Plot the points, fitted lines, intersection point, and the midline with additional visual elements."""
    plt.figure(figsize=(10, 6))

    # Plot all points in blue
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Points')

    # Highlight the points used for the best fit in red and green
    plt.scatter(points[split_index - 4:split_index + 1, 0], points[split_index - 4:split_index + 1, 1], color='red',
                label='Fit Points 1')
    plt.scatter(points[split_index:split_index + 5, 0], points[split_index:split_index + 5, 1], color='green',
                label='Fit Points 2')

    # Plot the fitted lines
    x_fit = np.linspace(min(points[:, 0]) - 1, max(points[:, 0]) + 1, 100)
    y_fit1 = m1 * x_fit + c1
    y_fit2 = m2 * x_fit + c2
    y_mid = m_mid * x_fit + c_mid

    plt.plot(x_fit, y_fit1, 'r--', label=f'Fit Line 1: y = {m1:.2f}x + {c1:.2f}')
    plt.plot(x_fit, y_fit2, 'g--', label=f'Fit Line 2: y = {m2:.2f}x + {c2:.2f}')
    plt.plot(x_fit, y_mid, 'b--', label=f'Midline: y = {m_mid:.2f}x + {c_mid:.2f}')

    # Plot the intersection point
    plt.scatter(*intersection, color='purple', label='Intersection')

    # Add annotations for lines
    plt.annotate('Line 1', xy=(x_fit[0], y_fit1[0]), xytext=(x_fit[0] + 1, y_fit1[0] + 5),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    plt.annotate('Line 2', xy=(x_fit[0], y_fit2[0]), xytext=(x_fit[0] + 1, y_fit2[0] - 5),
                 arrowprops=dict(facecolor='green', shrink=0.05))
    plt.annotate('Midline', xy=(x_fit[0], y_mid[0]), xytext=(x_fit[0] + 1, y_mid[0]),
                 arrowprops=dict(facecolor='blue', shrink=0.05))

    # Draw perpendiculars from intersection to lines
    for x in np.linspace(min(points[:, 0]), max(points[:, 0]), num=10):
        y_line1 = m1 * x + c1
        y_line2 = m2 * x + c2
        y_mid = m_mid * x + c_mid

        plt.plot([x, x], [y_line1, y_mid], 'r:', linestyle='--', color='red', alpha=0.5)
        plt.plot([x, x], [y_line2, y_mid], 'g:', linestyle='--', color='green', alpha=0.5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(
        f'Two Lines and Their Angle: {theta_degrees:.2f} Degrees\nSupplementary Angle: {supplementary_angle:.2f} Degrees')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    filename = f'frame_{frame:02d}.png'
    plt.savefig(filename)
    plt.close()
    return filename


# Generate random points around two linear trends
np.random.seed(42)
x1 = np.linspace(1, 20, 10)
y1 = 2 * x1 + np.random.normal(0, 1, len(x1))

x2 = np.linspace(21, 40, 10)
y2 = 5 * x2 - 60 + np.random.normal(0, 1, len(x2))

points = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2))))

max_angle = 0
best_split = None
best_fit = None
filenames = []

# Traverse all possible split points, using only 5 points before and 5 points after each split
for pt in range(4, len(points) - 4):
    m1, c1 = fit_line(points[pt - 4:pt + 1])
    m2, c2 = fit_line(points[pt:pt + 5])

    theta_degrees, supplementary_angle = compute_angle(m1, m2)

    if theta_degrees > max_angle:
        max_angle = theta_degrees
        best_split = pt
        best_fit = (m1, c1, m2, c2)

    intersection = compute_intersection(m1, c1, m2, c2)
    m_mid, c_mid = midline_between_lines(m1, c1, m2, c2, (min(points[:, 0]), max(points[:, 0])))
    filename = plot_lines_and_points(points, m1, c1, m2, c2, intersection, theta_degrees, supplementary_angle, pt,
                                     len(filenames), m_mid, c_mid)
    filenames.append(filename)

# Print the best fit line details
if best_fit:
    m1, c1, m2, c2 = best_fit
    print(f'Best split point: {best_split}, Max angle: {max_angle:.2f} degrees')
    print(f'Line 1: y = {m1:.2f}x + {c1:.2f}')
    print(f'Line 2: y = {m2:.2f}x + {c2:.2f}')
else:
    print('No suitable split found.')

# Set the duration for each frame (in seconds)
frame_duration = 300

# Create a gif
with imageio.get_writer('fit_lines.gif', mode='I', duration=frame_duration, loop=0) as writer:
    for filename in filenames:
        image = imageio.imread_v2(filename)
        writer.append_data(image)

# Remove the images
for filename in filenames:
    os.remove(filename)
