from pathlib import Path
from typing import Dict, List
def list_outputs(root: Path) -> Dict[str, List[str]]:
    root = Path(root)
    images_root = root / "images"
    videos_root = root / "videos"
    out = {"images": [], "videos": []}
    if images_root.exists():
        for p in sorted(images_root.rglob("*")):
            if p.suffix.lower() in {".png",".jpg",".jpeg",".webp"}:
                out["images"].append(str(p.relative_to(root)).replace('\\','/'))
    if videos_root.exists():
        for p in sorted(videos_root.rglob("*")):
            if p.suffix.lower() in {".mp4",".webm",".mov",".mkv"}:
                out["videos"].append(str(p.relative_to(root)).replace('\\','/'))
    return out
