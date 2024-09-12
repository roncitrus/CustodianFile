import os.path
from cProfile import label

import cv2
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QFileDialog, QMenu
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt



class ResultWindow(QWidget):
    def __init__(self, image, video_path, parent=None):
        super().__init__(parent)
        self.image = image
        self.video_path = video_path
        self.setWindowTitle('Interpolation result')

        # Display the QImage in QLabel
        self.label = QLabel(self)
        self.label.setScaledContents(True)
        self.init_ui()


    def init_ui(self):
        # Convert the numpy array image from BGR to RGB
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Convert the numpy array image to QPixmap
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        if self.label is not None:
            # Display the QImage in QLabel
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)

            # set layout
            layout = QVBoxLayout(self)
            layout.addWidget(self.label)
            self.setLayout(layout)

            #Resize image
            self.adjust_window_size(width, height)
            self.center()

            # Enable context menu
            self.label.setContextMenuPolicy(Qt.CustomContextMenu)
            self.label.customContextMenuRequested.connect(self.show_context_menu)
        else:
            raise ValueError("Label is not initialized properly")

    def adjust_window_size(self, image_width, image_height):
        # Set the maximum window size based on screen resolution or other criteria
        screen_geometry = self.screen().geometry()
        max_width = screen_geometry.width() * 0.8  # 80% of the screen width
        max_height = screen_geometry.height() * 0.8  # 80% of the screen height

        # Scale down if necessary
        if image_width > max_width or image_height > max_height:
            aspect_ratio = image_width / image_height
            if image_width > max_width:
                image_width = max_width
                image_height = max_width / aspect_ratio
            if image_height > max_height:
                image_height = max_height
                image_width = max_height * aspect_ratio

        # Convert dimensions to integers
        image_width = int(image_width)
        image_height = int(image_height)

        # #resize image for display purposes
        # resized_image = cv2.resize(self.image, (image_width, image_height), interpolation=cv2.INTER_AREA)
        #
        # #update QLabel and QPixmap with resized image
        # bytes_per_line = 3 * image_width
        # q_img = QImage(resized_image.data, image_width, image_height, bytes_per_line, QImage.Format_RGB888)
        # pixmap = QPixmap.fromImage(q_img)
        # self.label.setPixmap(pixmap)

        # resize window to mathc image size
        self.resize(image_width, image_height)


    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)
        self.resize(pixmap.size())
        self.adjust_window_size(pixmap.width(), pixmap.height())

    def center(self):
        # Centers the window on the screen
        frame_geometry = self.frameGeometry()
        screen_center = self.screen().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())

    def show_context_menu(self, position):
        context_menu = QMenu(self)
        save_action = context_menu.addAction("Save As...")
        save_action.triggered.connect(self.save_image_as)
        context_menu.exec_(self.mapToGlobal(position))

    def save_image_as(self):
        options = QFileDialog.Options()

        base_name = os.path.basename(self.video_path)
        suggested_name = os.path.splitext(base_name)[0] + "_interp.png"

        save_path, _ = QFileDialog.getSaveFileName(self, "Save image as", suggested_name, "PNG Files (*.png);;All Files (*)", options=options)

        if save_path:
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            QImage(image_rgb.data, image_rgb.shape[1], image_rgb.shape[0], image_rgb.strides[0], QImage.Format_RGB888).save(save_path)
