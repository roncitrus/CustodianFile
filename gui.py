import os
import sys
import numpy as np
from Result import ResultWindow
from circle_drawer import CircleDrawer
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QSlider, QPushButton, QWidget, QVBoxLayout, \
    QHBoxLayout, QTextEdit, QProgressBar, QCheckBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage

from video_processing import VideoProcessor
import video_processing

class VideoProcessingThread(QThread):
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)

    def __init__(self, video_path, threshold, preview_label, progress, mode='process'):
        super().__init__()
        self.video_path = video_path
        self.threshold = threshold
        self.preview_label = preview_label
        self.progress_bar = progress
        self.mode = mode

    def run(self):
        processor = VideoProcessor(
            video_path=self.video_path,
            threshold_value=self.threshold,
            preview_label=self.preview_label,
            progress_bar=self.progress.emit)
        processor.load_video()

        if self.mode == 'preprocess':
            result_image = processor.preprocess_all_frames(self.threshold)
        else:
            result_image = processor.process_with_squares()

        self.finished.emit(result_image)

class CustodianApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.interpolate_button = None
        self.progress_bar = None
        self.result_window = None
        self.threshold_value = 5
        self.frames = []
        self.processor = None
        self.current_frame_index = 0
        self.video_path = None
        self.video_preview_label = None
        self.upload_button = None
        self.threshold_slider = None
        self.video_label = None
        self.threshold_label = None
        self.thread = None
        self.info_text_panel = None
        self.progress = None
        self.preprocess_button = None
        self.process_button = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Dragon Interpolation Generator')
        self.setGeometry(100, 100, 1280, 720)

        # Central widget to hold all other widgets
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # upload button
        self.upload_button = QPushButton('Upload Video', self)
        self.upload_button.clicked.connect(self.upload_video)
        layout.addWidget(self.upload_button)
        self.upload_button.setFixedHeight(40)

        # Preprocess button
        self.preprocess_button = QPushButton('Preprocess', self)
        self.preprocess_button.clicked.connect(self.preprocess_video)
        layout.addWidget(self.preprocess_button)
        self.preprocess_button.setFixedHeight(40)

        # Process button
        self.process_button = QPushButton('Process', self)
        self.process_button.clicked.connect(self.start_processing)
        layout.addWidget(self.process_button)
        self.process_button.setFixedHeight(40)
        self.process_button.setEnabled(False)  # Disable initially

        # Add video preview label
        self.video_preview_label = QLabel(self)
        layout.addWidget(self.video_preview_label)

        # Calculate appropriate size maintaining the 16:9 aspect ratio
        max_width = 720
        aspect_ratio = 16 / 9
        preview_height = int(max_width / aspect_ratio)

        self.video_preview_label.setMaximumSize(max_width, preview_height)
        self.video_preview_label.setMinimumSize(max_width, preview_height)
        self.video_preview_label.setScaledContents(True)
        layout.addWidget(self.video_preview_label)

        # Horizontal layout for the slider and label
        slider_layout = QHBoxLayout()
        # Add threshold label
        self.threshold_label = QLabel(f'Threshold: {self.threshold_value}', self)
        slider_layout.addWidget(self.threshold_label)
        slider_layout.setContentsMargins(10, 10, 10, 10)

        #add slider for threshold control
        self.threshold_slider = QSlider(Qt.Horizontal, self)
        self.threshold_slider.setGeometry(50, 100, 200, 30)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(self.threshold_value)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        slider_layout.addWidget(self.threshold_slider)
        layout.addLayout(slider_layout)

        #display label for video
        self.video_label = QLabel(self)
        self.video_label.setMaximumSize(1280, 720)  # Set a maximum size
        self.video_label.setScaledContents(True)  # Scale the image to fit the label
        layout.addWidget(self.video_label)

        # Add info text panel to display print statements
        self.info_text_panel = QTextEdit(self)
        self.info_text_panel.setReadOnly(True)  # Make it read-only
        layout.addWidget(self.info_text_panel)

        # Progress bar for processing
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        # Display label for final video
        self.video_label = QLabel(self)
        self.video_label.setMaximumSize(max_width, preview_height)  # Set the same size as the preview
        self.video_label.setScaledContents(False)  # Ensure aspect ratio is preserved
        layout.addWidget(self.video_label)

        self.show()


    def upload_video(self):
        if hasattr(self, 'result_window') and self.result_window is not None:
            self.result_window.close()
            self.result_window = None

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload Video", "", "Video Files (*.mp4);;All Files (*)", options=options)

        if file_name:
            self.video_path = file_name
            print(f"Selected video path: {file_name}")
            base_name = os.path.basename(file_name)
            print(f"selected filename = {base_name}")
            self.video_path = file_name
            self.processor = video_processing.VideoProcessor(self.video_path, self.threshold_value, self.video_preview_label, self.progress_bar.setValue)
            self.processor.load_video()

            # display the filename
            self.setWindowTitle(f"Dragon Interpolation Generator - {base_name}")
            self.frames = self.processor.frames
            self.current_frame_index = 0

            # Automatically start preprocessing
            self.preprocess_video()


    def display_frame(self, frame_index):
        if not self.video_preview_label:
            raise ValueError("Video_preview_label is not initialised")

        if not self.processor or frame_index >= len(self.processor.frames):
            raise ValueError("Invalid frame indexor video not loaded")

        frame = self.processor.frames[frame_index]


        if frame_index > 0:
            prev_frame = self.processor.detect_fast_objects(frame, prev_frame)


    def update_threshold(self, value):
        self.threshold_value = value
        self.threshold_label.setText(f'Threshold: {value}')
        if self.processor and self.frames:
            self.preprocess_video()  # Rerun preprocessing with new threshold

    def append_text(self, text):
        self.info_text_panel.append(text)
        self.info_text_panel.ensureCursorVisible()  # Scrolls to the latest line

    def toggle_interpolation_button(self, state):
        if state == Qt.Checked:
            self.confirm_button.setVisible(False)
            self.interpolate_button.setVisible(True)
        else:
            self.confirm_button.setVisible(True)
            self.interpolate_button.setVisible(False)
            
    def start_processing(self):
        if not self.processor or not self.frames:
            self.append_text("Please preprocess the video first.")
            return

        self.append_text("Processing video...")
        self.preprocess_button.setEnabled(False)  # Disable preprocess button
        self.process_button.setEnabled(False)  # Disable process button
        self.start_processing_thread(self.video_path, self.threshold_value, mode='process')

    def start_processing_thread(self, video_path, threshold_value, mode='process'):
        self.thread = VideoProcessingThread(
            video_path, threshold_value, self.video_preview_label, self.progress_bar.setValue, mode)

        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.start()

    def on_processing_finished(self, result_image):
        if self.thread.mode == 'preprocess':
            self.processor.update_preview(result_image)
            self.append_text("Preprocessing complete. You can now adjust the threshold if needed.")
            self.append_text("Click 'Process' when ready to generate the final image.")
            self.process_button.setEnabled(True)
            self.preprocess_button.setEnabled(True)  # Re-enable preprocess button
        else:
            if hasattr(self, 'result_window') and self.result_window is not None:
                self.result_window.close()
            self.result_window = ResultWindow(image=result_image, video_path=self.video_path)
            self.result_window.show()
            self.process_button.setEnabled(True)  # Re-enable process button
            self.preprocess_button.setEnabled(True)  # Re-enable preprocess button

    def preprocess_video(self):
        if not self.processor or not self.frames:
            self.append_text("Please upload a video first.")
            return

        self.append_text("Preprocessing video...")
        self.preprocess_button.setEnabled(False)  # Disable preprocess button
        self.process_button.setEnabled(False)  # Disable process button
        self.start_processing_thread(self.video_path, self.threshold_value, mode='preprocess')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CustodianApp()
    sys.exit(app.exec_())
