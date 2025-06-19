import os
import sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent

# Ensure the project root is on the Python path so gui can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gui import EraserLabel


def test_mouse_move_calls_parent_handle():
    app = QApplication.instance() or QApplication([])

    class DummyParent(QWidget):
        def __init__(self):
            super().__init__()
            self.called = []
            self.eraser_mode = True

        def handle_eraser_click(self, x, y):
            self.called.append((x, y))

    parent = DummyParent()
    label = EraserLabel(parent)

    event = QMouseEvent(
        QMouseEvent.MouseMove,
        QPoint(5, 7),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )

    label.mouseMoveEvent(event)
    assert parent.called == [(5, 7)]
