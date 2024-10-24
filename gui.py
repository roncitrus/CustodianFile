import os
import sys
import cv2
import numpy as np
from Result import ResultWindow
from PyQt5.QtWidgets import QSizePolicy, QApplication, QMainWindow, QFileDialog, QLabel, QSlider, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QProgressBar, QScrollArea, QCheckBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QPixmap, QImage


from video_processing import VideoProcessor
import video_processing

class VideoProcessingThread(QThread):
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)

    def __init__(self, video_path, threshold, preview_label, progress, mode='process', green_boxes=None, red_boxes=None):
        super().__init__()
        self.video_path = video_path
        self.threshold = threshold
        self.preview_label = preview_label
        self.progress_bar = progress
        self.mode = mode
        self.green_boxes = green_boxes
        self.red_boxes = red_boxes


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
    DEFAULT_PREVIEW_WIDTH = 720
    DEFAULT_PREVIEW_HEIGHT = 405 # 16:9 aspect ratio
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
        self.video_preview_label.setAlignment(Qt.AlignCenter)
        self.video_preview_label.setScaledContents(False)
        self.video_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.video_preview_scroll_area = QScrollArea(self)
        self.video_preview_scroll_area.setWidget(self.video_preview_label)
        self.video_preview_scroll_area.setWidgetResizable(True)
        self.video_preview_scroll_area.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.video_preview_scroll_area)

        self.resizeEvent = self.onResize

        # Add video preview label
        self.video_preview_label = QLabel(self)
        self.video_preview_label.setMinimumSize(self.DEFAULT_PREVIEW_WIDTH, self.DEFAULT_PREVIEW_HEIGHT)
        self.video_preview_label.setScaledContents(True)
        self.video_preview_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.video_preview_label)

        # Horizontal layout for the slider and label
        slider_layout = QHBoxLayout()
        # Add threshold label
        self.threshold_label = QLabel(f'Threshold: {self.threshold_value}', self)
        slider_layout.addWidget(self.threshold_label)
        slider_layout.setContentsMargins(10, 10, 10, 10)

        #add slider for threshold control
        self.threshold_slider = QSlider(Qt.Horizontal, self)
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

                # Display label for final video - do I need this?
        self.final_video_label = QLabel(self)
        self.final_video_label.setMinimumSize(self.DEFAULT_PREVIEW_WIDTH, self.DEFAULT_PREVIEW_HEIGHT)  # Set the same size as the preview
        self.final_video_label.setScaledContents(False)  # Ensure aspect ratio is preserved
        layout.addWidget(self.final_video_label)

        self.show()


    def upload_video(self):
        if hasattr(self, 'result_window') and self.result_window is not None:
            self.result_window.close()
            self.result_window = None

        if self.video_path is not None:
            print("video is already loaded.")
            return

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

            if self.processor.frames:
                height, width = self.processor.frames[0].shape[:2]
                aspect_ratio = width / height

                preview_width = min(width, self.DEFAULT_PREVIEW_WIDTH)
                preview_height = int(preview_width / aspect_ratio)

                self.video_preview_label.setFixedSize(preview_width, preview_height)

            # display the filename
            self.setWindowTitle(f"Dragon Interpolation Generator - {base_name}")
            self.frames = self.processor.frames
            self.current_frame_index = 0

            self.display_frame(0)

            # Automatically start preprocessing
            self.preprocess_video()


    def display_frame(self, frame_index):
        if self.processor and self.frames:
            frame = self.frames[frame_index]
            if frame_index > 0:
                prev_frame = self.frames[frame_index - 1]
                fast_positions, slow_positions = self.processor.detect_fast_objects(frame, prev_frame, self.threshold_value)
                display_frame = self.processor.draw_object_rectangles(frame, fast_positions, slow_positions)
            else:
                display_frame = frame

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            scaled_pixmap = pixmap.scaled(self.video_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            print(f"Label size: {self.video_preview_label.size().width()}x{self.video_preview_label.size().height()}")

            self.video_preview_label.setPixmap(scaled_pixmap)
            self.updatePreviewSize()

    def update_threshold(self, value):
        self.threshold_value = value
        self.threshold_label.setText(f'Threshold: {value}')
        self.slider_timer.start(300)
        if self.processor and self.frames:
            self.preprocess_video()  # Rerun preprocessing with new threshold
    
    def process_slider_change(self):
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

    def start_processing_thread(self, video_path, threshold_value, mode='process', green_boxes=None, red_boxes=None):
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        self.thread = VideoProcessingThread(
            video_path, threshold_value, self.video_preview_label, self.progress_bar.setValue, mode, green_boxes, red_boxes)

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

    def onResize(self, event):
        if self.video_preview_label.pixmap():
            self.updatePreviewSize()

    def updatePreviewSize(self):
        if self.video_preview_label.pixmap():
            available_size = self.video_preview_scroll_area.viewport().size()
            pixmap = self.video_preview_label.pixmap()
            scaled_pixmap = pixmap.scaled(available_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_preview_label.setPixmap(scaled_pixmap)
            
            # center the label if it's smaller than the available space
            if scaled_pixmap.width() < available_size.width() or scaled_pixmap.height() < available_size.height():
                self.video_preview_label.setAlignment(Qt.AlignCenter)
            else:
                self.video_preview_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

            # Scale the pixmap to fit the available size while maintaining aspect ratio

            print(f"Scroll area size: {self.video_preview_scroll_area.size().width()}x{self.video_preview_scroll_area.size().height()}")
            print(f"Scaled pixmap size: {scaled_pixmap.width()}x{scaled_pixmap.height()}")
            print(f"Label size: {self.video_preview_label.size().width()}x{self.video_preview_label.size().height()}")
            if self.video_preview_label.pixmap():
                print(f"Pixmap size: {self.video_preview_label.pixmap().size().width()}x{self.video_preview_label.pixmap().size().height()}")
            # Set the scaled pixmap to the label


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CustodianApp()
    sys.exit(app.exec_())
