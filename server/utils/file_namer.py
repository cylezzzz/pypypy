from __future__ import annotations
from pathlib import Path
import re

__all__ = ["next_filename"]

_num_re_cache: dict[str, re.Pattern[str]] = {}

def _num_re(base: str) -> re.Pattern[str]:
    # Build and cache a regex like: ^{base}(\d+)$, applied on the STEM (no suffix)
    if base not in _num_re_cache:
        _num_re_cache[base] = re.compile(rf"^{re.escape(base)}(\d+)$")
    return _num_re_cache[base]

def next_filename(base: str, ext: str, folder: Path) -> str:
    """
    Returns the next available filename like '{base}1{ext}', '{base}2{ext}', … in 'folder'.
    - base: e.g. 'wandrobe', 'editor', 'motion'
    - ext:  file extension WITH dot (e.g., '.png', '.jpg'), case-insensitive match
    - folder: target directory
    The function scans existing files that match ^{base}(\\d+){ext}$ (case-insensitive on ext)
    and returns the next number (max+1). If collision occurs, it increments until free.
    """
    if not ext.startswith("."):
        ext = "." + ext
    folder.mkdir(parents=True, exist_ok=True)

    # Collect existing numbers
    nums: list[int] = []
    rx = _num_re(base)
    lower_ext = ext.lower()
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != lower_ext:
            continue
        m = rx.match(p.stem)
        if m:
            try:
                nums.append(int(m.group(1)))
            except ValueError:
                pass

    next_num = (max(nums) + 1) if nums else 1

    # Ensure no collision (paranoid)
    while True:
        candidate = f"{base}{next_num}{ext}"
        if not (folder / candidate).exists():
            return candidate
        next_num += 1