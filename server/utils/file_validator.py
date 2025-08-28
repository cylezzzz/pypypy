# server/utils/file_validator.py
from __future__ import annotations

import imghdr
import os
from pathlib import Path
from typing import Tuple

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_MB = 48

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def validate_image_upload(filename: str, size_bytes: int) -> Tuple[bool, str]:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        return False, f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXT))}"
    if size_bytes > MAX_IMAGE_MB * 1024 * 1024:
        return False, f"File too large: {size_bytes} bytes (limit {MAX_IMAGE_MB} MB)"
    return True, "ok"

def sniff_image(path: Path) -> bool:
    try:
        kind = imghdr.what(path)
        return kind in {"jpeg", "png", "webp"}
    except Exception:
        return False
