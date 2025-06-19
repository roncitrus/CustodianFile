import os
import sys
import cv2

from Result import ResultWindow
from PyQt5.QtWidgets import QSizePolicy, QApplication, QMainWindow, QFileDialog, QLabel, QSlider, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QProgressBar
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
import video_processing
from video_thread import VideoProcessingThread


class EraserLabel(QLabel):
    """QLabel that notifies its parent of left-clicks when erasing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent

    def mousePressEvent(self, event):
        if (
            self.parent_window
            and getattr(self.parent_window, "eraser_mode", False)
            and event.button() == Qt.LeftButton
        ):
            self.parent_window.handle_eraser_click(event.pos().x(), event.pos().y())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self.parent_window
            and getattr(self.parent_window, "eraser_mode", False)
            and (event.buttons() & Qt.LeftButton)
        ):
            self.parent_window.handle_eraser_click(event.pos().x(), event.pos().y())
            event.accept()
            return
        super().mouseMoveEvent(event)

class CustodianApp(QMainWindow):
    DEFAULT_PREVIEW_WIDTH = 720
    DEFAULT_PREVIEW_HEIGHT = 405  # 16:9 aspect ratio
    def __init__(self):
        super().__init__()
        self.threshold_value = 25
        self.interpolate_button = None
        self.progress_bar = None
        self.result_window = None
        self.processor = video_processing.VideoProcessor(None, self.threshold_value, None, None)
        self.frames = []
        self.current_frame_index = 0
        self.video_path = None
        self.video_preview_label = None
        self.video_preview_scroll_area = None
        self.upload_button = None
        self.threshold_slider = None
        self.video_label = None
        self.threshold_label = None
        self.thread = None
        self.info_text_panel = None
        self.progress = None
        self.preprocess_button = None
        self.process_button = None
        self.slider_timer = QTimer(self)
        self.slider_timer.setSingleShot(True)
        self.slider_timer.timeout.connect(self.process_slider_change)
        self.eraser_mode = False

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Dragon Interpolation Generator')
        self.setGeometry(100, 100, 1280, 720)

        # Central widget to hold all other widgets
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        button_and_preview_layout = QHBoxLayout()

        # Add video preview label
        self.video_preview_label = EraserLabel(self)
        self.video_preview_label.setAlignment(Qt.AlignCenter)
        self.video_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_preview_label.setScaledContents(False)
        self.video_preview_label.setMinimumSize(320, 180)
        self.video_preview_label.setMaximumSize(1920, 1080)

        button_and_preview_layout.addWidget(self.video_preview_label)

        button_layout = QVBoxLayout()

        # upload button
        self.upload_button = QPushButton('Upload Video', self)
        self.upload_button.clicked.connect(self.upload_video)
        self.upload_button.setFixedHeight(40)
        button_layout.addWidget(self.upload_button)

        # Preprocess button
        self.preprocess_button = QPushButton('Preprocess', self)
        self.preprocess_button.clicked.connect(self.preprocess_video)
        self.preprocess_button.setFixedHeight(40)
        button_layout.addWidget(self.preprocess_button)

        # Process button
        self.process_button = QPushButton('Process', self)
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setFixedHeight(40)
        self.process_button.setEnabled(False)  # Disable initially
        button_layout.addWidget(self.process_button)

        # Eraser toggle button
        self.eraser_button = QPushButton('Eraser: Off', self)
        self.eraser_button.setCheckable(True)
        self.eraser_button.clicked.connect(self.toggle_eraser)
        self.eraser_button.setFixedHeight(40)
        self.eraser_button.setEnabled(False)
        button_layout.addWidget(self.eraser_button)

        button_layout.addStretch(1)
        button_and_preview_layout.addLayout(button_layout)
        main_layout.addLayout(button_and_preview_layout)

        sliders_container = QWidget()
        sliders_layout = QVBoxLayout(sliders_container)

        # Add threshold label
        self.threshold_label = QLabel(f'Threshold: {self.threshold_value}', self)
        #self.threshold_label.setFixedHeight(20)
        sliders_layout.addWidget(self.threshold_label)

        #add slider for threshold control
        self.threshold_slider = QSlider(Qt.Horizontal, self)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(250)
        self.threshold_slider.setValue(self.threshold_value)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        sliders_layout.addWidget(self.threshold_slider)

        #add min_speed label
        self.min_speed_label = QLabel(f'Min Speed: {self.processor.min_speed}', self)
        #self.threshold_label.setFixedHeight(20)

        sliders_layout.addWidget(self.min_speed_label)

        #add slider for minimum  speed control
        self.min_speed_slider = QSlider(Qt.Horizontal, self)
        self.min_speed_slider.setMinimum(0)
        self.min_speed_slider.setMaximum(1500)
        self.min_speed_slider.setValue(self.processor.min_speed)
        self.min_speed_slider.valueChanged.connect(self.update_min_speed)
        #self.threshold_label.setFixedHeight(20)
        sliders_layout.addWidget(self.min_speed_slider)

        # add max_size label
        self.max_size_label = QLabel(f'Max Size: {self.processor.max_size}', self)
        #self.threshold_label.setFixedHeight(20)
        sliders_layout.addWidget(self.max_size_label)

        #add slider for max size control
        self.max_size_slider = QSlider(Qt.Horizontal, self)
        self.max_size_slider.setMinimum(0)
        self.max_size_slider.setMaximum(1000)
        self.max_size_slider.setValue(self.processor.max_size)
        self.max_size_slider.valueChanged.connect(self.update_max_size)
        #self.threshold_label.setFixedHeight(20)
        sliders_layout.addWidget(self.max_size_slider)

        # Add info text panel to display print statements
        self.info_text_panel = QTextEdit(self)
        self.info_text_panel.setReadOnly(True)  # Make it read-only
        self.info_text_panel.setFixedHeight(60)
        sliders_layout.addWidget(self.info_text_panel)

        # Progress bar for processing
        self.progress_bar = QProgressBar(self)
        sliders_layout.addWidget(self.progress_bar)

        sliders_container.setFixedHeight(250)
        main_layout.addWidget(sliders_container)

        self.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.frames and self.video_preview_label:
            self.display_frame(self.current_frame_index)

    def upload_video(self):
        if hasattr(self, 'result_window') and self.result_window is not None:
            self.result_window.close()
            self.result_window = None

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload Video", "", "Video Files (*.mp4);;All Files (*)", options=options)

        if not file_name:
            return

        # Check if the same video is already loaded
        if self.video_path == file_name:
            self.append_text("Video is already loaded.")
            return

        self.video_path = file_name
        print(f"Selected video path: {file_name}")
        base_name = os.path.basename(file_name)

        self.setWindowTitle(f"Dragon Interpolation Generator - {base_name}")
        self.append_text(f"selected filename = {base_name}")

        try:
            self.processor = video_processing.VideoProcessor(self.video_path, self.threshold_value, self.video_preview_label)
            self.processor.load_video()
            self.frames = self.processor.frames
            if self.frames:
                self.current_frame_index = 0
                self.display_frame(0)
                self.append_text("Video loaded successfully.")
                self.preprocess_button.setEnabled(True)  # Enable preprocess button
                self.preprocess_video()
            else:
                self.append_text("Failed to load frames. Please select a different video.")
                self.video_path = None
                self.processor = None
        except Exception as e:
            if self.frames:
                self.append_text(f"Error loading video: {str(e)}")
                self.video_path = None
                self.processor = None

            self.frames = self.processor.frames
            self.preprocess_video()

    def display_frame(self, frame_index, rgb_frame=None):
        if rgb_frame is None:
            if 0 <= frame_index < len(self.frames):
                rgb_frame = cv2.cvtColor(self.frames[frame_index], cv2.COLOR_BGR2RGB)
        if rgb_frame is not None:
            if 0 <= frame_index < len(self.frames):
                rgb_frame = cv2.cvtColor(self.frames[frame_index], cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_frame.shape
            bytes_per_line = channel * width
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Dynamically fetch the current size of the label
            label_width = self.video_preview_label.width()
            label_height = self.video_preview_label.height()

            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_preview_label.setPixmap(scaled_pixmap)
            self.video_preview_label.update()

    def update_threshold(self, value):
        self.threshold_value = value
        self.threshold_label.setText(f'Threshold: {value}')
        self.processor.threshold_value = value
        self.processor.create_background_subtractor()
        self.display_frame(self.current_frame_index)
        self.slider_timer.start(300)

    def update_min_speed(self, value):
        if self.processor:
            self.processor.min_speed = value
            self.min_speed_label.setText(f'Min Speed: {value}')
            self.display_frame(self.current_frame_index)
            self.slider_timer.start(300)

    def update_max_size(self, value):
        if self.processor:
            self.processor.max_size = value
            self.max_size_label.setText(f'Max Size: {value}')
            self.display_frame(self.current_frame_index)
            self.slider_timer.start(300)

    def toggle_eraser(self):
        self.eraser_mode = self.eraser_button.isChecked()
        state = "On" if self.eraser_mode else "Off"
        self.eraser_button.setText(f"Eraser: {state}")

    def label_to_frame_coordinates(self, x, y):
        if not self.processor or not self.processor.frames:
            return 0, 0
        frame_h, frame_w = self.processor.frames[0].shape[:2]
        label_w = self.video_preview_label.width()
        label_h = self.video_preview_label.height()
        scale = min(label_w / frame_w, label_h / frame_h)
        offset_x = (label_w - frame_w * scale) / 2
        offset_y = (label_h - frame_h * scale) / 2
        frame_x = int((x - offset_x) / scale)
        frame_y = int((y - offset_y) / scale)
        frame_x = max(0, min(frame_w - 1, frame_x))
        frame_y = max(0, min(frame_h - 1, frame_y))
        return frame_x, frame_y

    def handle_eraser_click(self, x, y):
        fx, fy = self.label_to_frame_coordinates(x, y)
        if self.processor:
            self.processor.remove_boxes_at(fx, fy)

    def process_slider_change(self):
        print("slider changed - preprocess_video called")
        if self.processor and self.frames:
            self.processor.prev_fast_positions = []
            self.preprocess_video()  # Rerun preprocessing with new threshold
        else:
            print("Processor or frames not initialized. Please upload a video.")

    def append_text(self, text):
        self.info_text_panel.append(text)
        self.info_text_panel.ensureCursorVisible()  # Scrolls to the latest line


    def start_processing(self):
        if not self.processor or not self.processor.frames:
            self.append_text("Please preprocess the video first.")
            return

        self.append_text("Processing video...")
        self.preprocess_button.setEnabled(False)  # Disable preprocess button
        self.process_button.setEnabled(False)  # Disable process button
        self.eraser_button.setChecked(False)
        self.toggle_eraser()
        self.start_processing_thread(mode='process')


    def start_processing_thread(self, mode='process', green_boxes=None, red_boxes=None):
        if self.thread is not None and self.thread.isRunning():
            print("Previous process still running...")
            return
        print(f"Starting {mode} thread...")
        self.thread = VideoProcessingThread(
            self.processor, mode, green_boxes, red_boxes)  # pass in self.processor here
        self.processor.progress_signal = self.thread.progress
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.start()

    def on_processing_finished(self, result_images):
        if self.thread.mode == 'preprocess':
            print("Preprocessing finished successfully.")
            self.frames = result_images
            self.display_frame(self.current_frame_index)

            self.append_text("Preprocessing complete. You can now adjust the threshold if needed.")
            self.append_text("Click 'Process' when ready to generate the final image.")
            self.process_button.setEnabled(True)
            print(f"Process button enabled: {self.process_button.isEnabled()}")
            self.preprocess_button.setEnabled(True)  # Re-enable preprocess button
            self.eraser_button.setEnabled(True)
        else:
            print("Processing final image.")
            if hasattr(self, 'result_window') and self.result_window is not None:
                self.result_window.close()
            result_image = result_images[0]
            self.result_window = ResultWindow(image=result_image, video_path=self.video_path)
            self.result_window.show()
            self.process_button.setEnabled(True)  # Re-enable process button
            self.preprocess_button.setEnabled(True)  # Re-enable preprocess button

    def preprocess_video(self):
        if not self.processor:
            self.append_text("Please upload a video first.")
            return
        if not self.frames:
            self.append_text("No frames found in the video. Please re-upload.")
            print(f"Processor frames: {self.processor.frames}")
            return

        self.append_text("Preprocessing video...")
        print("Starting preprocessing...")
        self.preprocess_button.setEnabled(False)  # Disable preprocess button
        self.process_button.setEnabled(False)  # Disable process button
        self.eraser_button.setChecked(False)
        self.toggle_eraser()
        self.eraser_button.setEnabled(False)
        self.start_processing_thread(mode='preprocess')



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CustodianApp()
    sys.exit(app.exec_())
