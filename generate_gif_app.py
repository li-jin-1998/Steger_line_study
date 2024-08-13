import sys
import os
import json
import datetime
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QVBoxLayout,
    QWidget, QLabel, QScrollArea, QGridLayout, QSlider
)
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import Qt
from PIL import Image


class GifCreatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('GIF Creator')
        self.resize(800, 600)
        self.center()

        self.label = QLabel('Select images to create a GIF', self)

        self.selectButton = QPushButton('Select Images', self)
        self.selectButton.clicked.connect(self.selectImages)

        self.createButton = QPushButton('Create GIF', self)
        self.createButton.clicked.connect(self.createGIF)
        self.createButton.setEnabled(False)

        self.openButton = QPushButton('Open GIF', self)
        self.openButton.clicked.connect(self.openGIF)
        self.openButton.setEnabled(False)

        self.exitButton = QPushButton('Exit', self)
        self.exitButton.clicked.connect(self.close)  # Connect to close method

        self.speedLabel = QLabel('Speed:', self)
        self.speedSlider = QSlider(Qt.Horizontal, self)
        self.speedSlider.setMinimum(1)
        self.speedSlider.setMaximum(100)
        self.speedSlider.setValue(10)  # Default value for speed
        self.speedSlider.setTickInterval(1)
        self.speedSlider.setTickPosition(QSlider.TicksBelow)
        self.speedSlider.setToolTip('Adjust GIF playback speed (lower is faster)')
        self.speedSlider.valueChanged.connect(self.updateSpeedLabel)

        self.imagePreviewArea = QScrollArea(self)
        self.imagePreviewWidget = QWidget()
        self.imagePreviewLayout = QGridLayout(self.imagePreviewWidget)
        self.imagePreviewWidget.setLayout(self.imagePreviewLayout)
        self.imagePreviewArea.setWidget(self.imagePreviewWidget)
        self.imagePreviewArea.setWidgetResizable(True)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.selectButton)
        self.layout.addWidget(self.imagePreviewArea)
        self.layout.addWidget(self.createButton)
        self.layout.addWidget(self.openButton)
        self.layout.addWidget(self.speedLabel)
        self.layout.addWidget(self.speedSlider)
        self.layout.addWidget(self.exitButton)

        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

        self.image_files = []
        self.gif_path = None
        self.last_directory = self.loadLastDirectory()
        self.speed = 500  # Default speed (duration in milliseconds)

    def loadLastDirectory(self):
        if os.path.exists('config.json'):
            with open('config.json', 'r') as file:
                config = json.load(file)
                return config.get('last_directory', os.path.expanduser('~'))
        return os.path.expanduser('~')

    def saveLastDirectory(self, directory):
        with open('config.json', 'w') as file:
            json.dump({'last_directory': directory}, file)

    def center(self):
        screen = QApplication.primaryScreen().availableGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )

    def selectImages(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", self.last_directory, "Images (*.png *.jpg *.bmp)", options=options)
        if files:
            self.image_files = files
            self.label.setText(f'Selected {len(files)} images')
            self.createButton.setEnabled(True)
            self.last_directory = os.path.dirname(files[0])
            self.saveLastDirectory(self.last_directory)
            self.updateImagePreview()

    def updateImagePreview(self):
        for i in reversed(range(self.imagePreviewLayout.count())):
            widget_to_remove = self.imagePreviewLayout.itemAt(i).widget()
            self.imagePreviewLayout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

        for i, image_file in enumerate(self.image_files):
            pixmap = QPixmap(image_file).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label = QLabel(self)
            label.setPixmap(pixmap)
            self.imagePreviewLayout.addWidget(label, i // 5, i % 5)

    def updateSpeedLabel(self):
        speed = self.speedSlider.value()
        self.speed = max(1, 100 - speed) * 10  # Map slider value to duration
        self.speedLabel.setText(f'Speed: {self.speed} ms per frame')

    def createGIF(self):
        if not self.image_files:
            return

        images = [Image.open(image) for image in self.image_files]

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        gif_directory = os.path.join(self.last_directory, 'GIF')
        os.makedirs(gif_directory, exist_ok=True)
        default_name = f"output_{timestamp}.gif"
        self.gif_path = os.path.join(gif_directory, default_name)

        images[0].save(
            self.gif_path,
            save_all=True,
            append_images=images[1:],
            duration=self.speed,
            loop=0
        )
        self.label.setText(f'GIF saved to {self.gif_path}')
        self.openButton.setEnabled(True)
        self.showGIFPreview()

    def showGIFPreview(self):
        self.gifLabel = QLabel(self)
        movie = QMovie(self.gif_path)
        self.gifLabel.setMovie(movie)
        self.imagePreviewLayout.addWidget(self.gifLabel, 0, 5)
        movie.start()

    def openGIF(self):
        if self.gif_path:
            if sys.platform == "win32":
                os.startfile(self.gif_path)
            elif sys.platform == "darwin":
                subprocess.call(("open", self.gif_path))
            else:
                subprocess.call(("xdg-open", self.gif_path))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GifCreatorApp()
    ex.show()
    sys.exit(app.exec_())
