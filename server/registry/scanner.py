from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

"""
Model-Scanner für AndioMediaStudio

- Unterstützt **flaches** models/ Layout (alle Modelle direkt unter models/)
- Unterstützt weiterhin Legacy-Unterordner: models/image, models/video
- Erkennt:
  • Diffusers-Modelle (Ordner mit model_index.json)
  • Single-File Checkpoints (.safetensors/.ckpt/.bin/.pt)
- Kennzeichnet Typ (image/video/voice/other) heuristisch am Namen
"""

CHECKPOINT_EXT = {".safetensors", ".ckpt", ".bin", ".pt"}

HEURISTIC_TYPES = [
    ("video", ["video", "svd", "img2vid", "animatediff"]),
    ("voice", ["wav2lip", "sadtalker", "talker", "tts"]),
    ("image", ["sd", "stable-diffusion", "sdxl", "flux", "realistic", "vision", "unet", "text_encoder", "vae"]),
]

def _guess_type(name: str) -> str:
    lower = name.lower()
    for t, keys in HEURISTIC_TYPES:
        if any(k in lower for k in keys):
            return t
    return "image"  # default

def _scan_root(root: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not root.exists():
        return items

    for p in root.iterdir():
        try:
            if p.is_dir():
                if (p / "model_index.json").exists():
                    items.append({
                        "name": p.name,
                        "path": str(p),
                        "kind": "diffusers",
                        "type": _guess_type(p.name),
                        "size": None,
                    })
                else:
                    # Ordner kann Teil eines größeren HF-Dumps sein – optional aufnehmen
                    items.append({
                        "name": p.name,
                        "path": str(p),
                        "kind": "folder",
                        "type": _guess_type(p.name),
                        "size": None,
                    })
            elif p.is_file() and p.suffix.lower() in CHECKPOINT_EXT:
                items.append({
                    "name": p.name,
                    "path": str(p),
                    "kind": "checkpoint",
                    "type": _guess_type(p.name),
                    "size": p.stat().st_size,
                })
        except Exception:
            # Skip kaputte Einträge still
            continue

    return items

def scan_models(base_dir: Path) -> List[Dict[str, Any]]:
    """
    Scannt Modelle an folgenden Orten (in dieser Reihenfolge):
      1) base/models            (flach)
      2) base/models/image      (legacy)
      3) base/models/video      (legacy)
    """
    base = Path(base_dir)
    locations = [
        base / "models",
        base / "models" / "image",
        base / "models" / "video",
    ]

    seen = set()
    results: List[Dict[str, Any]] = []

    for loc in locations:
        for item in _scan_root(loc):
            key = (item["name"], item["path"])
            if key in seen:
                continue
            seen.add(key)
            results.append(item)

    # Stabil sortieren
    results.sort(key=lambda x: (x["type"], x["name"].lower()))
    return results

def find_model(base_dir: Path, name: str) -> Dict[str, Any] | None:
    """
    Sucht ein Modell nach Name (exakte oder fuzzy Übereinstimmung)
    """
    items = scan_models(base_dir)
    # exakte Treffer (Name oder Dateiname ohne Endung)
    for it in items:
        if it["name"] == name or it["name"].split(".")[0] == name:
            return it
    # fuzzy
    nl = name.lower()
    for it in items:
        if nl in it["name"].lower():
            return it
    return None
