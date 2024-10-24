import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

class VideoProcessor:

    def __init__( self, video_path, threshold_value, preview_label, progress_bar = None):
        self.video_path = video_path
        self.threshold_value = threshold_value
        self.preview_label = preview_label
        self.progress_bar = progress_bar
        self.frames = []
        self.min_speed = 1
        self.max_size = 1000
        self.object_positions = []
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=threshold_value, detectShadows=False)

    def load_video(self):
        print(f"Loading video from: {self.video_path}")
        if not isinstance(self.video_path, str) or not self.video_path:
            raise ValueError("Invalid video path provided")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Error opening video file {self.video_path}")

        self.frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        success, frame = cap.read()
        while success:
            self.frames.append(frame)
            success, frame = cap.read()
        cap.release()


    def process_with_squares(self):
        frames = self.frames
        frame_count = len(frames)
        all_positions = []

        for i in range(1, frame_count):
            frame = frames[i]
            prev_frame = frames[i-1]

            fast_positions, slow_positions = self.detect_fast_objects(frame, prev_frame, self.threshold_value)
            
            # Filter out positions that overlap with slow positions
            filtered_positions = [pos for pos in fast_positions if not self.overlaps_with_slow(pos, slow_positions)]
            
            all_positions.append((frame, filtered_positions))

            if self.progress_bar:
                progress = int((i + 1) / frame_count * 100)
                self.progress_bar(progress)

        return self.create_final_image(frames[0], all_positions)

    def create_final_image(self, first_frame, all_positions):
        final_image = first_frame.copy()

        for frame, positions in all_positions:
            for(x, y, w, h) in positions:
                object_region = frame[y:y+h, x:x+w] # clip the object region
                #insert cropped object
                final_image[y:y+h, x:x+w] = object_region

        return final_image

    def detect_fast_objects(self, frame, prev_frame, threshold):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_grey = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Compute frame difference for fast movers detection
        frame_diff = cv2.absdiff(prev_grey, grey)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

        # Apply background subtractor for detecting larger, slower movers
        fgmask = self.fgbg.apply(frame)

        # Detect contours for both fast and slow movers
        contours_fast, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_slow, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fast_positions = []
        slow_positions = []

        for contour in contours_fast:
            area = cv2.contourArea(contour)
            if 5 < area < self.max_size:  # Adjust these values as needed
                (x, y, w, h) = cv2.boundingRect(contour)
                speed = np.linalg.norm(np.array([x+w/2, y+h/2]) - np.array([0, 0]))
                if speed > self.min_speed:
                    fast_positions.append((x, y, w, h))

        for contour in contours_slow:
            area = cv2.contourArea(contour)
            if area > 50:  # Adjust this value as needed
                (x, y, w, h) = cv2.boundingRect(contour)
                slow_positions.append((x, y, w, h))

        return fast_positions, slow_positions

    def update_preview(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Scale the pixmap to fit the label size while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        print(f"Label size: {self.preview_label.size().width()}x{self.preview_label.size().height()}")

        self.preview_label.setPixmap(scaled_pixmap)
        QApplication.processEvents()

    def preprocess_all_frames(self, threshold):
        all_fast_positions = []
        all_slow_positions = []
        frame_count = len(self.frames)

        for i in range(1, frame_count):
            frame = self.frames[i]
            prev_frame = self.frames[i-1]

            fast_positions, slow_positions = self.detect_fast_objects(frame, prev_frame, threshold)
            all_fast_positions.extend(fast_positions)
            all_slow_positions.extend(slow_positions)

            if self.progress_bar:
                progress = int((i + 1) / frame_count * 100)
                self.progress_bar(progress)

        return self.create_preprocessed_image(all_fast_positions, all_slow_positions)

    def create_preprocessed_image(self, fast_positions, slow_positions):
        return self.draw_object_rectangles(self.frames[0], fast_positions, slow_positions)

    def overlaps_with_slow(self, fast_pos, slow_positions):
        x1, y1, w1, h1 = fast_pos
        for x2, y2, w2, h2 in slow_positions:
            if (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2):
                return True
        return False
    
    def draw_object_rectangles(self, frame, fast_positions, slow_positions):
        result_image = frame.copy()
        for (x, y, w, h) in fast_positions:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y, w, h) in slow_positions:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return result_image

