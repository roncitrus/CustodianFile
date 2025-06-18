import os
import sys
import pytest

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from video_processing import VideoProcessor


def test_load_video_raises_for_invalid_path(tmp_path):
    """VideoProcessor.load_video should raise an error for bad paths."""
    bad_path = tmp_path / "nonexistent.mp4"
    proc = VideoProcessor(str(bad_path), threshold_value=10, preview_label=None)
    with pytest.raises((ValueError, IOError)):
        proc.load_video()

