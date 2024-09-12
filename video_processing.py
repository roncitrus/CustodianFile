import os.path
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from Result import ResultWindow
from circle_drawer import CircleDrawer



class VideoProcessor:

    def __init__( self, video_path, threshold_value, preview_label, progress_bar = None, keyframe_indices = None, keyframe_positions = None):
        self.video_path = video_path
        self.threshold_value = threshold_value
        self.preview_label = preview_label
        self.progress_bar = progress_bar
        self.keyframe_indices = keyframe_indices if isinstance(keyframe_indices, (list, np.ndarray)) else []
        self.keyframe_positions = keyframe_positions if isinstance(keyframe_positions, list) and len(keyframe_positions) > 0 else []
        self.current_keyframe_index = 0
        self.frames = []


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

        # calculate keyframes for circle drawing
        frame_count = len(self.frames)
        video_length_seconds = frame_count / fps

        if video_length_seconds < 1:
            num_keyframes = 2
        elif video_length_seconds < 2:
            num_keyframes = 3
        else: num_keyframes = max(3, int(video_length_seconds / 0.5))

        self.keyframe_indices = np.linspace(0, frame_count -1, num=num_keyframes, dtype=int)
        self.current_keyframe_index = 0

        print(f"Calculated keyframe_indices: {self.keyframe_indices}")

    def process_with_circles(self, keyframe_indices, keyframe_positions):
        print(f"Processing with keyframe_indices: {keyframe_indices}")
        print(f"Processing with keyframe_positions: {keyframe_positions}")

        if len(self.frames) == 0:
            print("Error: No frames loaded.")
            return None

        frame_count = len(self.frames)
        action_seq = self.frames[0].copy()
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=self.threshold_value, detectShadows=False)

        if keyframe_positions:
            circle_centers, circle_radii = self.interpolate_circle_positions(
                keyframe_indices, keyframe_positions, frame_count
            )
        else:
            print("No circles provided, proceeding without exclusions.")
            circle_centers = [None] * frame_count
            circle_radii = [None] * frame_count

        for i in range(frame_count):
            current_center = circle_centers[i]
            current_radius = circle_radii[i]

            if current_center is not None and current_radius is not None:
                moving_mask = np.zeros_like(self.frames[i][:, :, 0])
                cv2.circle(moving_mask, current_center, current_radius, 255, -1)
                moving_mask_inv = cv2.bitwise_not(moving_mask)
            else:
                moving_mask_inv = None

            fg_mask = fgbg.apply(self.frames[i], learningRate=0.001)
            _, fg_mask = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            if moving_mask_inv is not None:
                fg_mask = cv2.bitwise_and(fg_mask, moving_mask_inv)

            action_seq = cv2.bitwise_or(action_seq, cv2.bitwise_and(self.frames[i], self.frames[i], mask=fg_mask))

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5:  # Assume min_size is 5
                    mask = np.zeros_like(fg_mask)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    original_part = cv2.bitwise_and(self.frames[i], self.frames[i], mask=mask)
                    mask_inv = cv2.bitwise_not(mask)
                    action_seq = cv2.bitwise_and(action_seq, action_seq, mask=mask_inv)
                    action_seq = cv2.add(action_seq, original_part)

            # Update progress bar
            if self.progress_bar:
                progress_value = int((i + 1) / frame_count * 100)
                self.progress_bar.emit(progress_value)

        return action_seq

    def interpolate_circle_positions(self, keyframe_indices, keyframe_positions, total_frames):
        circle_centers = []
        circle_radii = []

        for i in range(total_frames):
            if i <= keyframe_indices[0]:
                circle_centers.append(keyframe_positions[0][0])
                circle_radii.append(keyframe_positions[0][1])
            elif i >= keyframe_indices[-1]:
                circle_centers.append(keyframe_positions[-1][0])
                circle_radii.append(keyframe_positions[-1][1])
            else:
                for j in range(1, len(keyframe_indices)):
                    if keyframe_indices[j - 1] <= i <= keyframe_indices[j]:
                        ratio = (i - keyframe_indices[j - 1]) / (keyframe_indices[j] - keyframe_indices[j - 1])
                        center_x = int(keyframe_positions[j - 1][0][0] * (1 - ratio) + keyframe_positions[j][0][0] * ratio)
                        center_y = int(keyframe_positions[j - 1][0][1] * (1 - ratio) + keyframe_positions[j][0][1] * ratio)
                        radius = int(keyframe_positions[j - 1][1] * (1 - ratio) + keyframe_positions[j][1] * ratio)
                        circle_centers.append((center_x, center_y))
                        circle_radii.append(radius)
                        break

        return circle_centers, circle_radii
