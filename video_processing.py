from weakref import finalize

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

class VideoProcessor:

    def __init__( self, video_path, threshold_value, preview_label, progress_signal = None):
        self.prev_fast_positions = []
        self.video_path = video_path
        self.threshold_value = threshold_value
        self.preview_label = preview_label
        self.progress_bar = progress_signal
        self.frames = []
        self.min_speed = 600
        self.max_size = 1000
        self.object_positions = []
        self.all_positions = []
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=threshold_value, detectShadows=False)

    def load_video(self):
        print(f"Loading video from: {self.video_path}")
        if not isinstance(self.video_path, str) or not self.video_path:
            raise ValueError("Invalid video path provided")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Error opening video file {self.video_path}")

        self.frames = []
        #fps = cap.get(cv2.CAP_PROP_FPS)
        success, frame = cap.read()
        while success:
            self.frames.append(frame)
            success, frame = cap.read()
        cap.release()
        if not self.frames:
            raise ValueError("No frames were loaded from the video. Check the file format or codec.")
        print(f"Loaded {len(self.frames)} frames successfully.")

    def process_with_squares(self, green_boxes=None, red_boxes=None):
        final_image = self.frames[0].copy()

        for i in range(len(self.frames) -1):
            current_frame = self.frames[i].copy()
            #temp_image = final_image.copy()

            # Process all detected positions for this frame
            for x, y, w, h in self.all_positions[i]:
                try:
                    # Get the ROI (region of interest)
                    roi = self.frames[i + 1][y:y + h, x:x + w]

                    # Convert ROI to grayscale if required for contour detection
                    if len(roi.shape) == 3:  # Check if ROI is in color
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    else:
                        roi_gray = roi

                    # Find contours in the binary ROI
                    _, binary_roi = cv2.threshold(roi_gray, self.threshold_value, 255, cv2.THRESH_BINARY)

                    # Check that binary_roi is single-channel and has type CV_8UC1
                    assert binary_roi.dtype == np.uint8, f"binary_roi dtype is {binary_roi.dtype}, expected uint8"
                    assert len(binary_roi.shape) == 2, f"binary_roi shape is {binary_roi.shape}, expected single-channel"

                    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Create a mask for the current object
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(mask, [contours[0]], -1, 255, -1)

                        object_region = cv2.bitwise_and(current_frame[y:y + h, x:x + w], current_frame[y:y + h, x:x + w], mask=mask[:, :, None])

                        # Place the extracted object region into the composite image
                        final_image[y:y + h, x:x + w] = object_region
                except Exception as e:
                    print(f"Error processing object at frame {i}, box ({x}, {y}, {w}, {h}): {e}")

        return [final_image]


    def detect_fast_objects(self, frame, prev_frame):
        if self.prev_fast_positions is None:
            self.prev_fast_positions = []

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_grey = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Compute frame difference for fast movers detection
        frame_diff = cv2.absdiff(prev_grey, grey)
        _, thresh = cv2.threshold(frame_diff, self.threshold_value, 255, cv2.THRESH_BINARY)

        # Apply background subtractor for detecting larger, slower movers
        fgmask = self.fgbg.apply(frame)

        # Detect contours for both fast and slow movers
        contours_fast, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours_slow, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Contours detected (fast): {len(contours_fast)}")
        #print(f"Contours detected (slow): {len(contours_slow)}")

        fast_positions = []
        #slow_positions = []
        current_fast_positions = []  # Store positions in the current frame

        for i, contour in enumerate(contours_fast):
            area = cv2.contourArea(contour)
            if 5 < area < self.max_size:  # Ensure reasonable area range
                print(f"Fast contour area: {area}, max size: {self.max_size}")
                x, y, w, h = cv2.boundingRect(contour)
                current_fast_positions.append(((x + w / 2, y + h / 2), (w, h), i))  # Store center and dimensions

        # Calculate speeds based on previous frame data
        for (cx, cy), (w, h), i in current_fast_positions:
            if self.prev_fast_positions:
                for prev_cx, prev_cy in self.prev_fast_positions:
                    speed = np.linalg.norm(np.array([cx, cy]) - np.array([prev_cx, prev_cy]))
                    if speed > self.min_speed:
                        print(f"Object speed: {speed}, min speed: {self.min_speed}")
                        x, y, w, h = cv2.boundingRect(contours_fast[i])
                        fast_positions.append((x, y, w, h))
        self.prev_fast_positions = [pos[0] for pos in current_fast_positions] # update positions for next frame

        # for contour in contours_slow:
        #     area = cv2.contourArea(contour)
        #     if area > 50:  # Adjust this value as needed
        #         (x, y, w, h) = cv2.boundingRect(contour)
        #         slow_positions.append((x, y, w, h))

        return fast_positions, [], frame

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

    def preprocess_all_frames(self):
        self.preprocessed_frames = []
        self.all_positions = []
        frame_count = len(self.frames)
        frame_0 = self.frames[0].copy()  # Start with the first frame

        for i in range(1, frame_count):
            print(f"Processing frame {i}/{frame_count}")
            self.create_background_subtractor()
            frame = self.frames[i].copy()
            prev_frame = self.frames[i-1].copy() if i > 0 else None
            fast_positions, _, _ = self.detect_fast_objects(frame, prev_frame)

            #filtered_fast = [pos for pos in fast_positions if not self.overlaps_with_slow(pos, slow_positions)]

            filtered_fast = fast_positions
            print(f"Filtered fast objects: {len(filtered_fast)}")

            self.all_positions.append(filtered_fast)

            for (x, y, w, h) in filtered_fast:
                print(f"Drawing fast box at: x={x}, y={y}, w={w}, h={h}")
                cv2.rectangle(frame_0, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for fast objects
            # for (x, y, w, h) in slow_positions:
            #     cv2.rectangle(frame_0, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for slow objects



            if self.progress_bar:
                progress = int((i + 1) / frame_count * 100)
                self.progress_bar.emit(progress)

        self.preprocessed_frames.append(frame_0)
        print("Preprocessing completed.")
        print(f"Frames after preprocessing: {len(self.frames)}")
        print(f"Preprocessed frames: {len(self.preprocessed_frames)}")
        self.update_preview(frame_0)
        return self.preprocessed_frames


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

    def create_background_subtractor(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=self.threshold_value, detectShadows=False)
