import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QSlider, QPushButton, QFileDialog, \
    QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent
from PyQt5.QtCore import Qt, QPoint
import os
import json


class DraggableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._dragging = False
        self._drag_start_pos = QPoint()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start_pos = event.pos()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            new_pos = self.mapToParent(event.pos() - self._drag_start_pos)
            self.move(new_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            event.accept()


class CannyEdgeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Canny Edge Detection')
        self.setGeometry(100, 100, 1200, 600)

        # Initialize current directory
        self.current_dir = self.load_last_directory()

        # Initialize image paths and index
        self.image_paths = []
        self.current_image_index = -1

        # Load last window position
        self.load_window_geometry()

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Layout for images
        self.image_layout = QHBoxLayout()
        self.layout.addLayout(self.image_layout)

        # Spacer to center the images
        self.image_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.image_layout.addItem(self.image_spacer)

        # Image label for original image
        self.original_image_label = DraggableLabel()
        self.image_layout.addWidget(self.original_image_label)

        # Image label for Canny edges
        self.edges_image_label = DraggableLabel()
        self.image_layout.addWidget(self.edges_image_label)

        # Spacer to center the images
        self.image_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.image_layout.addItem(self.image_spacer)

        # Layout for sliders and buttons (horizontal layout)
        self.control_layout = QHBoxLayout()
        self.layout.addLayout(self.control_layout)

        # Slider for low threshold
        self.slider_low = QSlider(Qt.Horizontal)  # Change to horizontal orientation
        self.slider_low.setRange(0, 255)
        self.slider_low.setValue(100)
        self.slider_low.valueChanged.connect(self.update_edges)
        self.control_layout.addWidget(self.slider_low)

        # Slider for high threshold
        self.slider_high = QSlider(Qt.Horizontal)  # Change to horizontal orientation
        self.slider_high.setRange(0, 255)
        self.slider_high.setValue(200)
        self.slider_high.valueChanged.connect(self.update_edges)
        self.control_layout.addWidget(self.slider_high)

        # Button to open image file
        self.open_button = QPushButton('Open Image')
        self.open_button.clicked.connect(self.open_image)
        self.control_layout.addWidget(self.open_button)

        # Button to switch to next image
        self.next_button = QPushButton('Next Image')
        self.next_button.clicked.connect(self.next_image)
        self.control_layout.addWidget(self.next_button)

        # Initialize image variables
        self.image = None
        self.edges = None

    def open_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", self.current_dir,
                                                   "Images (*.png *.xpm *.jpg);;All Files (*)", options=options)

        if file_path:
            self.current_dir = os.path.dirname(file_path)
            self.image_paths = [file_path]  # For simplicity, reset to the selected image
            self.current_image_index = 0
            self.save_last_directory(self.current_dir)
            self.load_image(file_path)

    def load_image(self, file_path):
        self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        self.update_display()

    def update_display(self):
        if self.image is None:
            return

        # Display original image
        q_image = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0],
                         QImage.Format_Grayscale8)
        self.original_image_label.setPixmap(QPixmap.fromImage(q_image))

        # Update and display edges
        self.update_edges()

    def update_edges(self):
        if self.image is None:
            return

        low_threshold = self.slider_low.value()
        high_threshold = self.slider_high.value()

        blurred_image = cv2.GaussianBlur(self.image, (5, 5), 1.5)
        self.edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

        q_image = QImage(self.edges.data, self.edges.shape[1], self.edges.shape[0], self.edges.strides[0],
                         QImage.Format_Grayscale8)
        self.edges_image_label.setPixmap(QPixmap.fromImage(q_image))

    def next_image(self):
        if not self.image_paths:
            return

        # Try to get image files from the current directory
        if self.current_image_index == -1:
            return

        # List all image files in the current directory
        all_files = [f for f in os.listdir(self.current_dir) if
                     f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if not all_files:
            return

        # Update the list of image paths
        self.image_paths = [os.path.join(self.current_dir, f) for f in sorted(all_files)]

        # Update the index
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self.load_image(self.image_paths[self.current_image_index])

    def save_last_directory(self, directory):
        # Save the last opened directory to a JSON file
        with open('last_directory.json', 'w') as file:
            json.dump({'last_directory': directory}, file)

    def load_last_directory(self):
        # Load the last opened directory from a JSON file
        if os.path.exists('last_directory.json'):
            with open('last_directory.json', 'r') as file:
                data = json.load(file)
                return data.get('last_directory', '')
        return ''

    def save_window_geometry(self):
        # Save the window geometry to a JSON file
        with open('window_geometry.json', 'w') as file:
            data = {
                'x': self.x(),
                'y': self.y(),
                'width': self.width(),
                'height': self.height()
            }
            json.dump(data, file)

    def load_window_geometry(self):
        # Load the window geometry from a JSON file
        if os.path.exists('window_geometry.json'):
            with open('window_geometry.json', 'r') as file:
                data = json.load(file)
                self.setGeometry(
                    data.get('x', 100),
                    data.get('y', 100),
                    data.get('width', 800),
                    data.get('height', 400)
                )
        else:
            # Center the window on the screen
            screen = QApplication.primaryScreen().geometry()
            self.setGeometry(
                (screen.width() - 800) // 2,
                (screen.height() - 400) // 2,
                800,
                400
            )

    def closeEvent(self, event):
        # Save the window geometry when the application is closed
        self.save_window_geometry()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CannyEdgeApp()
    window.show()
    sys.exit(app.exec_())
