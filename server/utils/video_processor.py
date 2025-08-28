# server/utils/video_processor.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

def ken_burns_from_image(image_path: Path, out_path: Path, duration_sec: float = 5.0, fps: int = 30,
                         out_w: int = 1280, out_h: int = 720, zoom_start: float = 1.0, zoom_end: float = 1.2) -> Tuple[Path, int]:
    """
    Simple Ken Burns (pan & zoom) effect using OpenCV.
    Returns (out_path, frames_written).
    """
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    ih, iw = img.shape[:2]

    # Determine video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter for output")

    frames = int(duration_sec * fps)
    frames_written = 0

    # Calculate pan path (center moves slightly)
    cx0, cy0 = iw / 2, ih / 2
    cx1, cy1 = iw / 2 * 1.02, ih / 2 * 0.98  # small drift
    for i in range(frames):
        t = i / max(1, frames - 1)
        zoom = zoom_start * (1 - t) + zoom_end * t
        cx = cx0 * (1 - t) + cx1 * t
        cy = cy0 * (1 - t) + cy1 * t

        # Compute crop window to maintain aspect ratio
        target_aspect = out_w / out_h
        crop_w = iw / zoom
        crop_h = crop_w / target_aspect
        if crop_h > ih / zoom:
            crop_h = ih / zoom
            crop_w = crop_h * target_aspect

        x1 = int(np.clip(cx - crop_w / 2, 0, iw - crop_w))
        y1 = int(np.clip(cy - crop_h / 2, 0, ih - crop_h))
        x2 = int(x1 + crop_w)
        y2 = int(y1 + crop_h)

        crop = img[y1:y2, x1:x2]
        frame = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        writer.write(frame)
        frames_written += 1

    writer.release()
    return out_path, frames_written
