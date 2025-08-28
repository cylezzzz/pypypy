from pathlib import Path

MODEL_EXTS = {".safetensors",".ckpt",".bin",".pt",".onnx",".gguf",".json"}

def tag_by_name(name: str):
    n = name.lower()
    tags = []
    if any(k in n for k in ["flux","sdxl","sd3","stable-diffusion"]):
        tags += ["txt2img","img2img","inpaint","edit_ok","avatar_ok","nsfw_ok"]
    if "control" in n or "pose" in n:
        tags += ["pose"]
    if "svd" in n or "video" in n:
        tags += ["txt2video","img2video"]
    if "animatediff" in n:
        tags += ["img2video"]
    if "wav2lip" in n:
        tags += ["lipsync"]
    if "sadtalker" in n:
        tags += ["talking"]
    if "sam" in n:
        tags += ["mask"]
    return sorted(set(tags))

def scan_models(root: Path):
    root = Path(root)
    models = []
    # Prefer subfolders image/ and video/
    for sub in ["image","video"]:
        base = root/sub
        if not base.exists(): 
            continue
        for f in base.rglob("*"):
            if f.is_file() and f.suffix.lower() in MODEL_EXTS:
                rel = f.relative_to(root).as_posix()
                models.append({
                    "name": f.stem,
                    "path": rel,
                    "type": f.suffix.lower().lstrip("."),
                    "tags": tag_by_name(f.stem),
                    "group": sub,
                    "license": "unknown"
                })
    return {"models": models}
