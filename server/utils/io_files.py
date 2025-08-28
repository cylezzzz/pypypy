from __future__ import annotations
from pathlib import Path
from PIL import Image
import time, os, json, hashlib

def ensure_dir(d: Path):
    Path(d).mkdir(parents=True, exist_ok=True)

def safe_name(name: str) -> str:
    keep = "-_.() []"
    s = "".join([c for c in name or "" if c.isalnum() or c in keep]).strip()
    return s or f"file_{int(time.time())}"

def list_media(root: Path, kind: str):
    items = []
    if kind == "image":
        exts = {".png",".jpg",".jpeg",".webp"}
    else:
        exts = {".mp4",".webm",".mov",".mkv"}
    for p in Path(root).glob("**/*"):
        if p.is_file() and p.suffix.lower() in exts:
            st = p.stat()
            items.append({
                "path": str(p),
                "url": str(p.relative_to(Path(__file__).resolve().parents[2])).replace("\","/"),
                "size": st.st_size,
                "mtime": int(st.st_mtime*1000)
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items

def hash_path(p: Path) -> str:
    return hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:10]
