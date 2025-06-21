from PyQt5.QtCore import QThread, pyqtSignal


class VideoProcessingThread(QThread):
    """Run video processing operations in a separate thread."""

    finished = pyqtSignal(object)
    progress = pyqtSignal(int)

    def __init__(self, processor, mode='process', green_boxes=None, red_boxes=None):
        super().__init__()
        self.processor = processor
        self.mode = mode
        self.green_boxes = green_boxes
        self.red_boxes = red_boxes

    def run(self):
        cancel = self.isInterruptionRequested
        if self.mode == 'preprocess':
            result_image = self.processor.preprocess_all_frames(should_cancel=cancel)
        else:
            result_image = self.processor.process_with_squares(self.green_boxes, self.red_boxes, should_cancel=cancel)

        self.finished.emit(result_image)
