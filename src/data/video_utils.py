"""
BioViT3R-Beta Video Utility Module
Processes videos into frames and time-series batch inputs.
"""

import cv2
from pathlib import Path
from typing import List
import numpy as np


def extract_frames(video_path: str, output_dir: str, every_n: int = 30) -> List[Path]:
    """
    Extract frames from video at specified interval.

    Args:
        video_path: Video file path
        output_dir: Directory to save frames
        every_n: Extract every n-th frame
    Returns:
        List of saved frame paths
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    saved_paths = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    while success:
        if count % every_n == 0:
            frame_path = output_dir / f"frame_{count:06d}.png"
            cv2.imwrite(str(frame_path), image)
            saved_paths.append(frame_path)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    return saved_paths
