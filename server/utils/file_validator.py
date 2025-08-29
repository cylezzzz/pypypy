# server/utils/file_validator.py
from __future__ import annotations

import imghdr
import mimetypes
from pathlib import Path
from typing import Optional, Tuple

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def is_image(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    kind = imghdr.what(path)
    return kind in {"jpeg", "png", "gif", "bmp", "tiff", "webp"}

def sniff_image(path: Path) -> Tuple[bool, Optional[str]]:
    """Prüft schnell, ob Datei ein Bild ist, und gibt (groben) MIME-Typ zurück."""
    if not path:
        return (False, None)
    p = Path(path)
    if not p.exists() or not p.is_file():
        return (False, None)
    if is_image(p):
        mime, _ = mimetypes.guess_type(str(p))
        if not mime:
            suffix = p.suffix.lower()
            mime = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
                ".tif": "image/tiff",
                ".tiff": "image/tiff",
                ".webp": "image/webp",
            }.get(suffix, "application/octet-stream")
        return (True, mime)
    return (False, None)

def validate_image_upload(path: Path) -> Tuple[bool, str]:
    ok, mime = sniff_image(path)
    if not ok:
        return False, "Unsupported or corrupt image file"
    return True, mime or "application/octet-stream"
