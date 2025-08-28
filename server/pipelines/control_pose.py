from __future__ import annotations
from pathlib import Path
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

def extract_pose(image: Image.Image):
    mp_pose = mp.solutions.pose
    arr = np.array(image.convert("RGB"))
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        res = pose.process(arr)
    return res.pose_landmarks if res else None

def draw_pose(image: Image.Image, landmarks) -> Image.Image:
    im = image.copy().convert("RGB")
    d = ImageDraw.Draw(im)
    if not landmarks: return im
    w, h = im.size
    for lm in landmarks.landmark:
        x, y = lm.x * w, lm.y * h
        d.ellipse((x-3,y-3,x+3,y+3), fill=(0,255,0))
    return im
