from PyQt5.QtCore import Qt
from PyQt5.QtCore import QObject, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtWidgets import QLabel, QApplication
import cv2
import numpy as np
from jinja2 import TemplateRuntimeError


class CircleDrawer(QObject):
    def __init__(self, label):
        super().__init__()
        self.circle_start = None
        self.circle_radius_start = 0
        self.drawing = False
        self.circle_confirmed = False
        self.label = label
        self.image = None


        # Connect the QLabel's mouse events to custom event handlers
        self.label.mousePressEvent = self.mouse_press_event
        self.label.mouseMoveEvent = self.mouse_move_event
        self.label.mouseReleaseEvent = self.mouse_release_event
        self.label.setFocusPolicy(Qt.StrongFocus) # ensure QLabel can receive keyboard events
        self.label.keyPressEvent = self.key_press_event

    def set_circle(self, start, radius):
        self.circle_start = start
        self.circle_radius_start = radius
        print(f"Set circle_start: {self.circle_start}, circle_radius_start: {self.circle_radius_start}")


    def set_image(self, image):
        self.image = image
        self.update_label()


    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.circle_start = self.get_scaled_position(event.x(), event.y())
            self.circle_radius_start = 0
            print(f"Mouse pressed at: {self.circle_start}")

    def mouse_move_event(self, event):
        if self.drawing:
            end_pos = self.get_scaled_position(event.x(), event.y())
            self.circle_radius_start = int(np.sqrt(
                (end_pos[0] - self.circle_start[0]) ** 2 +
                (end_pos[1] - self.circle_start[1]) ** 2)
            )

            self.update_label()


    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.circle_confirmed = True
            print(f"Mouse released. circle confirmed at {self.circle_start} with radius {self.circle_radius_start}")
            #self.set_circle((event.x(), event.y()), self.circle_radius_start)
            self.update_label()

    def get_scaled_position(self, x, y):
        # Calculate scaling factor between QLabel size and image size
        label_width = self.label.width()
        label_height = self.label.height()
        image_height, image_width, _ = self.image.shape

        scale_x = image_width / label_width
        scale_y = image_height / label_height

        # Scale the cursor position back to the image's coordinate system
        return int(x * scale_x), int(y * scale_y)


    def key_press_event(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.circle_confirmed = True  # Confirm the circle when Enter is pressed


    def update_label(self):
        if self.image is not None and self.circle_start is not None:
            temp_image = self.image.copy()
            cv2.circle(temp_image, self.circle_start, self.circle_radius_start, (0, 255, 0), 2)
            self.show_image(temp_image)


    def show_image(self, image):
        # Convert the image to a QPixmap and display it in the label
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_img))
        self.label.setScaledContents(True)


    def get_circle_positions(self, frames, num_keyframes):

        keyframe_idxs = np.linspace(0, len(frames)  - 1, num=num_keyframes, dtype=int)
        keyframe_pns = []

        for idx in keyframe_idxs:
            self.circle_start = None
            self.circle_radius_start = 0
            # self.circle_confirmed = False  # Reset the confirmation flag
            self.set_image(frames[idx].copy())

            # Wait for the user to finish drawing the circle and confirm it
            while not self.circle_confirmed:
                QApplication.processEvents()  # Keep the UI responsive

            if self.circle_start is not None and self.circle_radius_start > 0:
                keyframe_pns.append((self.circle_start, self.circle_radius_start))
                print(f"Circle captured at position {self.circle_start} with radius {self.circle_radius_start}")
            else:
                print(f"No valid circle drawn for keyframe {idx}")

        print(f"Generated keyframe_indices: {keyframe_idxs}")
        print(f"Generated keyframe_positions: {keyframe_pns}")

        return keyframe_idxs, keyframe_pns