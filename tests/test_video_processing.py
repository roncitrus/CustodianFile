import os
import sys
import numpy as np
import cv2
import types

# Ensure the project root is on the Python path so video_processing can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from video_processing import VideoProcessor

class DummyProcessor(VideoProcessor):
    """Subclass that disables preview updates for testing."""
    def update_preview(self, frame):
        # Mark that the function was called without invoking Qt
        self.preview_updated = True


def make_frame_with_rect(start, end, size=(100, 100)):
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.rectangle(frame, start, end, (255, 255, 255), -1)
    return frame


def test_extract_object_region_returns_region():
    frame = make_frame_with_rect((10, 10), (19, 19))
    proc = VideoProcessor(None, threshold_value=10, preview_label=None)
    region = proc.extract_object_region(frame, 10, 10, 10, 10)
    assert region is not None
    assert region.shape[0] == 10 and region.shape[1] == 10


def test_detect_fast_objects_identifies_movement():
    prev_frame = make_frame_with_rect((10, 10), (19, 19))
    curr_frame = make_frame_with_rect((30, 10), (39, 19))
    proc = VideoProcessor(None, threshold_value=5, preview_label=None)
    proc.min_speed = 1
    proc.max_size = 200
    proc.prev_fast_positions = [(15, 15)]
    boxes, _, _ = proc.detect_fast_objects(curr_frame, prev_frame)
    assert boxes, "No fast-moving objects detected"


def test_preprocess_all_frames_updates_preview_and_returns_frame():
    frame1 = make_frame_with_rect((2, 2), (5, 5), size=(20, 20))
    frame2 = make_frame_with_rect((5, 2), (8, 5), size=(20, 20))
    proc = DummyProcessor(None, threshold_value=5, preview_label=None)
    proc.min_speed = 1
    proc.max_size = 200
    proc.frames = [frame1, frame2]
    result = proc.preprocess_all_frames()
    assert hasattr(proc, "preview_updated")
    assert isinstance(result, list) and result


def test_overlaps_with_slow_returns_true_when_overlap():
    proc = VideoProcessor(None, threshold_value=10, preview_label=None)
    fast_box = (10, 10, 10, 10)
    slow_boxes = [(15, 15, 10, 10)]
    assert proc.overlaps_with_slow(fast_box, slow_boxes)
