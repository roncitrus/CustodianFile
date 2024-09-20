import os
import sys
import numpy as np
from Result import ResultWindow
from circle_drawer import CircleDrawer
import cv2
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QSlider, QPushButton, QWidget, QVBoxLayout, \
    QHBoxLayout, QTextEdit, QProgressBar, QCheckBox, QSizePolicy
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import video_processing
from video_processing import VideoProcessor


class VideoProcessingThread(QThread):
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)

    def __init__(self, video_path, threshold, keyframe_indices, keyframe_positions, preview_label, progress):
        super().__init__()
        self.video_path = video_path
        self.threshold = threshold
        self.preview_label = preview_label
        self.progress_bar = progress
        self.keyframe_indices = keyframe_indices
        self.keyframe_positions = keyframe_positions


    def run(self):
        # process video in a separate thread
        processor = VideoProcessor(
            video_path=self.video_path,
            threshold_value=self.threshold,
            preview_label=self.preview_label,
            progress_callback=self.progress,
            keyframe_indices=self.keyframe_indices,
            keyframe_positions=self.keyframe_positions)
        processor.load_video()
        result_image = processor.process_with_circles(self.keyframe_indices, self.keyframe_positions)
        # signal when done
        self.finished.emit(result_image)

class CustodianApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.interpolate_button = None
        self.progress_bar = None
        self.skip_circles_checkbox = None
        self.result_window = None
        self.threshold_value = 5
        self.frames = []
        self.processor = None
        self.circle_drawer = None
        self.current_frame_index = 0
        self.video_path = None
        self.video_preview_label = None
        self.keyframe_indices = []
        self.keyframe_positions = []
        self.current_keyframe_index =0
        self.upload_button = None
        self.threshold_slider = None
        self.video_label = None
        self.threshold_label = None
        self.thread = None
        self.confirm_button = None
        self.info_text_panel = None
        self.progress = None
        self.final_video_label = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Dragon Interpolation Generator')
        self.setGeometry(100, 100, 800, 600)

        # Central widget to hold all other widgets
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # upload button
        self.upload_button = QPushButton('Upload Video', self)
        self.upload_button.clicked.connect(self.upload_video)
        layout.addWidget(self.upload_button)
        self.upload_button.setFixedHeight(40)

        # Skip circle drawing checkbox
        self.skip_circles_checkbox = QCheckBox("Skip circle drawing", self)
        self.skip_circles_checkbox.stateChanged.connect(self.toggle_interpolation_button)
        layout.addWidget(self.skip_circles_checkbox)

        # Interpolate button (hidden initially)
        self.interpolate_button = QPushButton('Interpolate', self)
        self.interpolate_button.clicked.connect(self.start_processing)
        self.interpolate_button.setVisible(False)  # Hide initially
        layout.addWidget(self.interpolate_button)

        # Confirm button to confirm the drawn circle
        self.confirm_button = QPushButton('Confirm Circle', self)
        self.confirm_button.clicked.connect(self.confirm_circle)
        self.confirm_button.setFixedHeight(40)
        layout.addWidget(self.confirm_button)

        # Calculate appropriate size maintaining the 16:9 aspect ratio
        max_width = 720
        aspect_ratio = 16 / 9
        preview_height = int(max_width / aspect_ratio)

        # Add video preview label
        self.video_preview_label = QLabel(self)
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
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(self.threshold_value)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        slider_layout.addWidget(self.threshold_slider)
        layout.addLayout(slider_layout)

        #display label for video
        self.video_label = QLabel(self)
        self.video_label.setMaximumSize(640, 480)  # Set a maximum size
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
        self.final_video_label.setMaximumSize(max_width, preview_height)  # Set the same size as the preview
        self.final_video_label.setScaledContents(False)  # Ensure aspect ratio is preserved
        layout.addWidget(self.final_video_label)

        self.show()


    def upload_video(self):
        if hasattr(self, 'result_window') and self.result_window is not None:
            self.result_window.close()
            self.result_window = None

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload Video", "", "Video Files (*.mp4);;All Files (*)", options=options)

        if file_name:
            print(f"Selected video path: {file_name}")
            base_name = os.path.basename(file_name)
            print(f"selected filename = {base_name}")
            self.video_path = file_name
            self.processor = video_processing.VideoProcessor(self.video_path, self.threshold_value, self.video_preview_label)
            self.processor.load_video()

            self.keyframe_indices = self.processor.keyframe_indices
            self.frames = self.processor.frames
            #self.keyframe_positions = []
            self.current_keyframe_index = 0

            self.circle_drawer = None
            self.current_frame_index = 0

            self.video_preview_label.clear()
            self.video_label.clear()

            # display the filename
            self.setWindowTitle(f"Dragon Interpolation Generator - {base_name}")
            self.frames = self.processor.frames
            self.current_frame_index = 0

            self.display_frame(0)  # Display the first frame for circle drawing


    def display_frame(self, frame_index):
        if not self.video_preview_label:
            raise ValueError("Video_preview_label is not initialised")

        if not self.processor or frame_index >= len(self.processor.frames):
            raise ValueError("Invalid frame indexor video not loaded")

        frame = self.processor.frames[frame_index]

        if not self.circle_drawer:
            self.circle_drawer = CircleDrawer(self.video_preview_label)

        # Initialize the CircleDrawer and set the callback to the UI label
        self.circle_drawer.set_image(frame.copy())
        self.circle_drawer.show_image(frame)


    def update_threshold(self, value):
        self.threshold_value = value
        self.threshold_label.setText(f'Threshold: {value}')


    def confirm_circle(self):
        self.append_text("Confirm circle button clicked")
        try:
            # skip circle drawing
            if self.skip_circles_checkbox.isChecked():
                print("Skipping circle drawing, proceeding to processing.")
                self.start_processing_thread(self.video_path, self.threshold_value, self.keyframe_indices, [])

            if self.circle_drawer:
                circle_data = (self.circle_drawer.circle_start, self.circle_drawer.circle_radius_start)
                self.append_text(f"Circle data captured: {circle_data}")

                if circle_data[0] is None or circle_data[1] <= 0:
                    self.append_text("No valid circle drawn.")
                    return

                # Store the circle data in keyframe_positions
                self.keyframe_positions.append(circle_data)
                self.append_text(f"Stored circle data: {self.keyframe_positions}")
                self.append_text(f"")
                self.circle_drawer.circle_confirmed = True
                self.append_text("circle confirmed")
                self.append_text(f"Keyframe indices in confirm_circle: {self.keyframe_indices}")

                # Move to the next keyframe or start processing if all keyframes are done
                if self.current_keyframe_index < len(self.keyframe_indices) -1:
                    # Move to the next keyframe
                    self.current_keyframe_index += 1
                    self.append_text(f"current_frame_index: {self.current_keyframe_index}, keyframe indices: {self.keyframe_indices}")
                    next_frame_index = self.keyframe_indices[self.current_keyframe_index]
                    self.display_frame(next_frame_index) # move to next frame
                    self.circle_drawer.circle_confirmed = False
                else:
                    self.append_text(f"Starting processing with keyframe_indices: {self.keyframe_indices}")
                    self.append_text(f"Starting processing with keyframe_positions: {self.keyframe_indices}")
                    self.start_processing_thread(self.video_path, self.threshold_value, self.keyframe_indices, self.keyframe_positions)

        except Exception as e:
            self.append_text(f"Error in confirm_circle: {e}")

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
        print("Interpolate button clicked")
        self.keyframe_positions = []
        self.start_processing_thread(self.video_path, self.threshold_value, self.keyframe_indices,self.keyframe_positions)

    def start_processing_thread(self, video_path, threshold_value, keyframe_indices, keyframe_positions):
        self.thread = VideoProcessingThread(
            video_path, threshold_value, keyframe_indices, keyframe_positions,  self.video_preview_label, self.progress)

        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.start()

    def on_processing_finished(self, result_image):

        # open the result in a new window:
        self.result_window = ResultWindow(image=result_image, video_path=self.video_path)
        self.result_window.show()


        # pixmap = QPixmap(result_image_path)

        # # Calculate a reasonable maximum size based on the window's dimensions
        # max_width = self.width() - 40  # Subtract some margin
        # max_height = int(max_width * (9 / 16))  # Maintain 16:9 aspect ratio
        #
        # # Scale the pixmap while maintaining the aspect ratio
        # scaled_pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        #
        # self.video_label.setPixmap(scaled_pixmap)
        # self.video_label.setFixedSize(scaled_pixmap.size())  # Adjust QLabel size to fit the pixmap
        # self.video_label.setScaledContents(False)
        #
        #
        # # Resize the main window to fit the new QLabel size
        # self.resize(self.video_label.width() + 40, self.video_label.height() + 100)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CustodianApp()
    sys.exit(app.exec_())
