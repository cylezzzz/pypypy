# server/utils/gallery_indexer.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import json
import datetime

def _iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")

def _derive_model_from_name(name: str) -> str:
    n = name.lower()
    if n.startswith("wandrobe"): return "wardrobe"
    if n.startswith("editor"):   return "editor"
    if n.startswith("motion"):   return "motion"
    if n.startswith("txt2img"):  return "txt2img"
    return "unknown"

def _read_sidecar(p: Path) -> Dict:
    j = p.with_suffix(p.suffix + ".json")
    if j.exists():
        try:
            return json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def list_gallery(project_root: Path) -> Dict[str, List[Dict]]:
    """
    Scannt:
      - workspace/uploads
      - outputs/images
    und baut Einträge:
      { name, url, folder, date, model, size }
    Falls Sidecar JSON existiert (gleicher Dateiname + '.json'): model + prompt etc. werden übernommen.
    """
    uploads = []
    images = []

    base = Path(project_root)
    up_dir = base / "workspace" / "uploads"
    img_dir = base / "outputs" / "images"

    def collect(dir_path: Path, url_prefix: str, target_list: List[Dict]):
        if not dir_path.exists():
            return
        for p in sorted(dir_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            stat = p.stat()
            meta = _read_sidecar(p)
            model = meta.get("model") or _derive_model_from_name(p.stem)
            target_list.append({
                "name": p.name,
                "url": f"{url_prefix}/{p.name}",
                "folder": str(dir_path.relative_to(base)),
                "date": _iso(stat.st_mtime),
                "model": model,
                "size": stat.st_size,
                "meta": meta or None
            })

    collect(up_dir,  "/workspace/uploads", uploads)
    collect(img_dir, "/outputs/images",   images)

    return {"uploads": uploads, "images": images, "total": len(uploads) + len(images)}
