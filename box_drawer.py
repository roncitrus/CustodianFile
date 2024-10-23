from PyQt5.QtCore import Qt
from PyQt5.QtCore import QObject, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtWidgets import QLabel, QApplication
import cv2
import numpy as np


class BoxDrawer(QObject):
    def __init__(self, label):
        super().__init__()


