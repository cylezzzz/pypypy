# server/api/unified_endpoints.py
# -----------------------------------------------------------------------------
# AndioMediaStudio – dynamischer, modell-freier Router
#
# Features:
# - /api/models, /api/models_slim     → dynamische Modell-Liste (ohne Code-Pins)
# - /api/outputs, /api/outputs/images, /api/outputs/videos → Galerie-Feeds
# - /api/generate  (task = txt2img | img2img | img2vid) → generische Erzeugung
# - /api/txt2img, /api/img2img, /api/video/img2vid, /api/motion/animate → Bequem
#
# Lädt Pipelines LAZY je Modellpfad; Caches sie, bis Server stoppt.
# Keine fest verdrahteten Modellnamen; alles kommt aus models/* Ordnern.
# -----------------------------------------------------------------------------

from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from datetime import datetime
import io
import os

import torch
from PIL import Image, ImageFilter, ImageChops

# Diffusers Pipelines (wir wählen passend zur Modellstruktur)
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableVideoDiffusionPipeline,
)

router = APIRouter(prefix="/api", tags=["unified"])

# -----------------------------------------------------------------------------#
# Pfade / Runtime-Cache
# -----------------------------------------------------------------------------#
ROOT = Path(__file__).resolve().parents[2]
IMG_ROOT = ROOT / "models" / "image"
VID_ROOT = ROOT / "models" / "video"
OUT_IMG = ROOT / "outputs" / "images"
OUT_VID = ROOT / "outputs" / "videos"
for p in (OUT_IMG, OUT_VID):
    p.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Pipeline-Cache je Modellordner
_TXT2IMG: Dict[str, Any] = {}
_IMG2IMG: Dict[str, Any] = {}
_IMG2VID: Dict[str, Any] = {}

# -----------------------------------------------------------------------------#
# Utils
# -----------------------------------------------------------------------------#
def _today_dir(base: Path) -> Path:
    d = base / datetime.now().strftime("%Y%m%d")
    d.mkdir(parents=True, exist_ok=True)
    return d

def _save_image(img: Image.Image, suffix="png") -> str:
    d = _today_dir(OUT_IMG)
    name = f"gen_{datetime.now().strftime('%H%M%S')}.{suffix}"
    p = d / name
    img.save(p)
    return "/" + p.relative_to(ROOT).as_posix()

def _save_frames_as_video(frames: List[Image.Image], fps: int = 8) -> str:
    d = _today_dir(OUT_VID)
    base = d / f"vid_{datetime.now().strftime('%H%M%S')}"
    mp4 = base.with_suffix(".mp4")
    gif = base.with_suffix(".gif")
    # Versuche MP4 mit imageio (falls ffmpeg vorhanden). Fallback: GIF.
    try:
        import imageio
        imageio.mimsave(mp4, [f.convert("RGB") for f in frames], fps=int(fps))
        return "/" + mp4.relative_to(ROOT).as_posix()
    except Exception:
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=int(1000/int(fps)), loop=0)
        return "/" + gif.relative_to(ROOT).as_posix()

def _letterbox(im: Image.Image, target_wh: Tuple[int,int]) -> Image.Image:
    tw, th = target_wh
    w, h = im.size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    im2 = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas.paste(im2, (ox, oy))
    return canvas

def _looks_like_sdxl(model_dir: Path) -> bool:
    n = model_dir.name.lower()
    if "xl" in n or "sdxl" in n:
        return True
    # verlässlicher: model_index.json wird von SDXL-Ordnern i.d.R. mit "StableDiffusionXLPipeline" referenziert
    idx = model_dir / "model_index.json"
    if idx.exists():
        try:
            import json
            data = json.loads(idx.read_text(encoding="utf-8"))
            text = str(data)
            if "StableDiffusionXLPipeline" in text or "StableDiffusionXL" in text:
                return True
        except Exception:
            pass
    return False

def _has_model_index(model_dir: Path) -> bool:
    return (model_dir / "model_index.json").exists()

def _is_usable_image_model(d: Path) -> bool:
    if not d.is_dir(): return False
    name = d.name.lower()
    banned = {"components","checkpoints","loras",".cache","motion_modules"}
    if name in banned: return False
    # Ohne model_index.json ist Laden via from_pretrained (diffusers) nicht möglich
    return _has_model_index(d)

def _is_usable_video_model(d: Path) -> bool:
    return d.is_dir() and _has_model_index(d)

def _scan_models() -> Dict[str, Any]:
    image = []
    if IMG_ROOT.exists():
        for d in sorted([p for p in IMG_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            entry = {
                "name": d.name,
                "path": "/" + d.relative_to(ROOT).as_posix(),
                "loadable": _is_usable_image_model(d),
                "type_hint": "sdxl" if _looks_like_sdxl(d) else "sd15_or_other"
            }
            image.append(entry)
    video = []
    if VID_ROOT.exists():
        for d in sorted([p for p in VID_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            entry = {
                "name": d.name,
                "path": "/" + d.relative_to(ROOT).as_posix(),
                "loadable": _is_usable_video_model(d),
                "type_hint": "svd"  # generische Kennzeichnung
            }
            video.append(entry)
    return {"image": image, "video": video}

def _model_dir_from_name(name_or_path: str, domain: str) -> Path:
    """
    name_or_path kann 'stable-diffusion-xl-base-1.0' ODER '/models/image/...' sein.
    domain: 'image' | 'video'
    """
    base = IMG_ROOT if domain == "image" else VID_ROOT
    # absolut/relativ?
    p = Path(name_or_path)
    if p.is_absolute():
        return p
    # evtl. führender Slash kommt aus API
    name = name_or_path.strip("/").split("/")[-1]
    candidate = base / name
    if candidate.exists():
        return candidate
    # Fallback: genau so wie geliefert relativ zum ROOT
    guess = ROOT / name_or_path.strip("/")
    return guess

# -----------------------------------------------------------------------------#
# Models – zwei Varianten (für Kompat. mit bestehender UI)
# -----------------------------------------------------------------------------#
@router.get("/models")
def models():
    return {"ok": True, **_scan_models()}

@router.get("/models_slim")
def models_slim():
    data = _scan_models()
    return {
        "ok": True,
        "image": [m["name"] for m in data["image"] if m["loadable"]],
        "video": [m["name"] for m in data["video"] if m["loadable"]],
    }

# -----------------------------------------------------------------------------#
# Outputs – zum Schließen der 404-Lücken der Web-UI
# -----------------------------------------------------------------------------#
def _list_media(dir_path: Path, exts: Tuple[str,...]) -> List[Dict[str,str]]:
    out = []
    if not dir_path.exists(): return out
    for day in sorted(dir_path.iterdir(), reverse=True):
        if not day.is_dir(): continue
        for f in sorted(day.iterdir(), reverse=True):
            if f.suffix.lower().lstrip(".") in exts:
                out.append({
                    "name": f.name,
                    "url": "/" + f.relative_to(ROOT).as_posix(),
                    "date": day.name
                })
    return out

@router.get("/outputs")
def outputs_root():
    return {"ok": True, "images": len(_list_media(OUT_IMG, ("png","jpg","jpeg","webp","gif"))),
            "videos": len(_list_media(OUT_VID, ("mp4","gif")))}
@router.get("/outputs/images")
def outputs_images():
    return {"ok": True, "items": _list_media(OUT_IMG, ("png","jpg","jpeg","webp","gif"))}
@router.get("/outputs/videos")
def outputs_videos():
    return {"ok": True, "items": _list_media(OUT_VID, ("mp4","gif"))}

# -----------------------------------------------------------------------------#
# Pipeline-Resolver (dynamisch, ohne Pins)
# -----------------------------------------------------------------------------#
def _get_txt2img_pipe(model_dir: Path):
    key = str(model_dir.resolve())
    if key in _TXT2IMG: return _TXT2IMG[key]
    if not _has_model_index(model_dir):
        raise HTTPException(400, f"Model not loadable (no model_index.json): {model_dir}")
    if _looks_like_sdxl(model_dir):
        pipe = StableDiffusionXLPipeline.from_pretrained(str(model_dir), torch_dtype=DTYPE)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(str(model_dir), torch_dtype=DTYPE)
    if DEVICE == "cuda":
        pipe = pipe.to("cuda")
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass
    try: pipe.enable_vae_tiling()
    except Exception: pass
    _TXT2IMG[key] = pipe
    return pipe

def _get_img2img_pipe(model_dir: Path):
    key = str(model_dir.resolve())
    if key in _IMG2IMG: return _IMG2IMG[key]
    if not _has_model_index(model_dir):
        raise HTTPException(400, f"Model not loadable (no model_index.json): {model_dir}")
    if _looks_like_sdxl(model_dir):
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(str(model_dir), torch_dtype=DTYPE)
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(str(model_dir), torch_dtype=DTYPE)
    if DEVICE == "cuda":
        pipe = pipe.to("cuda")
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass
    try: pipe.enable_vae_tiling()
    except Exception: pass
    _IMG2IMG[key] = pipe
    return pipe

def _get_img2vid_pipe(model_dir: Path):
    key = str(model_dir.resolve())
    if key in _IMG2VID: return _IMG2VID[key]
    if not _has_model_index(model_dir):
        raise HTTPException(400, f"Video model not loadable (no model_index.json): {model_dir}")
    pipe = StableVideoDiffusionPipeline.from_pretrained(str(model_dir), torch_dtype=DTYPE)
    if DEVICE == "cuda":
        pipe = pipe.to("cuda")
    _IMG2VID[key] = pipe
    return pipe

# -----------------------------------------------------------------------------#
# GENERIC GENERATE
# -----------------------------------------------------------------------------#
class GenerateRequest(BaseModel):
    task: str = Field(..., pattern="^(txt2img|img2img|img2vid)$")
    model: Optional[str] = Field(None, description="Model-Ordnername oder Pfad; wenn None -> erster loadable Match")
    params: Dict[str, Any] = Field(default_factory=dict)

@router.post("/generate")
async def generate(req: GenerateRequest):
    # Model auswählen
    models = _scan_models()
    domain = "video" if req.task == "img2vid" else "image"

    # bevorzugtes Modell
    if req.model:
        mdir = _model_dir_from_name(req.model, domain)
    else:
        # nimm erstes "loadable" aus passender Domäne
        cand = next((m for m in models[domain] if m["loadable"]), None)
        if not cand:
            raise HTTPException(400, f"No loadable {domain} model found.")
        mdir = ROOT / cand["path"].lstrip("/")

    if req.task == "txt2img":
        pipe = _get_txt2img_pipe(mdir)
        p = req.params
        g = None
        if p.get("seed") is not None:
            g = torch.Generator(device=DEVICE).manual_seed(int(p["seed"]))
        img = pipe(
            prompt=p.get("prompt",""),
            negative_prompt=p.get("negative_prompt"),
            num_inference_steps=int(p.get("steps",20)),
            guidance_scale=float(p.get("guidance",5.0)),
            width=int(p.get("width",1024)),
            height=int(p.get("height",1024)),
            generator=g,
        ).images[0]
        url = _save_image(img)
        return {"ok": True, "output_url": url, "model": mdir.name}

    elif req.task == "img2img":
        # Für generisch müssten Bilddaten kommen; hier nur Hinweis:
        raise HTTPException(400, "Use /api/img2img (multipart/form-data) for image upload.")

    elif req.task == "img2vid":
        raise HTTPException(400, "Use /api/video/img2vid (multipart/form-data) for image upload.")

    raise HTTPException(400, "Unknown task")

# -----------------------------------------------------------------------------#
# TXT2IMG / IMG2IMG (bequeme, UI-freundliche Routen – OHNE Modell-Pins)
# → optionaler 'model' String erlaubt (Ordnername/Pfad), sonst auto-pick
# -----------------------------------------------------------------------------#
@router.post("/txt2img")
def txt2img(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),
    steps: int = Form(20),
    guidance: float = Form(5.0),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: Optional[int] = Form(None),
    model: Optional[str] = Form(None),
):
    # Modell wählen
    models = _scan_models()
    if model:
        mdir = _model_dir_from_name(model, "image")
    else:
        cand = next((m for m in models["image"] if m["loadable"]), None)
        if not cand:
            raise HTTPException(400, "No loadable image model found.")
        mdir = ROOT / cand["path"].lstrip("/")

    pipe = _get_txt2img_pipe(mdir)
    g = torch.Generator(device=DEVICE).manual_seed(int(seed)) if seed is not None else None
    img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=int(width),
        height=int(height),
        generator=g,
    ).images[0]
    url = _save_image(img)
    return {"ok": True, "output_url": url, "model": mdir.name}

@router.post("/img2img")
async def img2img(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: Optional[str] = Form(None),
    strength: float = Form(0.6),
    steps: int = Form(20),
    guidance: float = Form(5.0),
    seed: Optional[int] = Form(None),
    model: Optional[str] = Form(None),
):
    models = _scan_models()
    if model:
        mdir = _model_dir_from_name(model, "image")
    else:
        cand = next((m for m in models["image"] if m["loadable"]), None)
        if not cand:
            raise HTTPException(400, "No loadable image model found.")
        mdir = ROOT / cand["path"].lstrip("/")

    pipe = _get_img2img_pipe(mdir)
    raw = await image.read()
    init = Image.open(io.BytesIO(raw)).convert("RGB")
    g = torch.Generator(device=DEVICE).manual_seed(int(seed)) if seed is not None else None
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=g,
    ).images[0]
    url = _save_image(out)
    return {"ok": True, "output_url": url, "model": mdir.name}

# -----------------------------------------------------------------------------#
# EDITOR – Remove (approx Inpaint, modell-frei; nutzt gewähltes img2img-Modell)
# Erwartet 'mask' = L-Mode Maske (weiß = zu ersetzen)
# -----------------------------------------------------------------------------#
@router.post("/editor/remove")
async def editor_remove(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(""),
    strength: float = Form(0.75),
    steps: int = Form(20),
    guidance: float = Form(4.5),
    model: Optional[str] = Form(None),
):
    models = _scan_models()
    if model:
        mdir = _model_dir_from_name(model, "image")
    else:
        cand = next((m for m in models["image"] if m["loadable"]), None)
        if not cand:
            raise HTTPException(400, "No loadable image model found.")
        mdir = ROOT / cand["path"].lstrip("/")

    pipe = _get_img2img_pipe(mdir)

    raw = await image.read()
    mraw = await mask.read()
    init = Image.open(io.BytesIO(raw)).convert("RGB")
    msk  = Image.open(io.BytesIO(mraw)).convert("L")
    msk = msk.filter(ImageFilter.GaussianBlur(radius=6))

    blurred = init.filter(ImageFilter.MedianFilter(size=7))
    base = Image.composite(blurred, init, msk)

    out = pipe(
        prompt=prompt or "",
        image=base,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
    ).images[0]
    url = _save_image(out)
    return {"ok": True, "output_url": url, "model": mdir.name}

# -----------------------------------------------------------------------------#
# WARDROBE – schnelle Personensegmentierung (modell-frei; MediaPipe optional)
# -----------------------------------------------------------------------------#
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

@router.post("/wardrobe/segment")
async def wardrobe_segment(
    image: UploadFile = File(...),
    mode: str = Form("person")  # "person" | "upper" | "lower"
):
    if not HAS_MEDIAPIPE:
        raise HTTPException(500, "MediaPipe not available. Install mediapipe to use wardrobe segmentation.")

    import numpy as np
    from PIL import ImageDraw

    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    mp_seg = mp.solutions.selfie_segmentation
    with mp_seg.SelfieSegmentation(model_selection=1) as seg:
        arr = np.array(img)
        res = seg.process(arr[:, :, ::-1])  # tolerant bzgl. RGB/BGR
        mask = (res.segmentation_mask * 255).astype("uint8")
        mask_img = Image.fromarray(mask, mode="L")

    # simple upper/lower heuristic
    if mode in ("upper", "lower"):
        cut = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(cut)
        w, h = img.size
        if mode == "upper":
            draw.rectangle([0, 0, w, h//2], fill=255)
        else:
            draw.rectangle([0, h//2, w, h], fill=255)
        mask_img = ImageChops.multiply(mask_img, cut)

    # speichern
    d = _today_dir(OUT_IMG)
    ts = datetime.now().strftime("%H%M%S")
    mpath = d / f"mask_{ts}.png"
    opath = d / f"overlay_{ts}.png"

    overlay = Image.blend(img, Image.new("RGB", img.size, (255,0,0)), alpha=0.35)
    overlay = Image.composite(overlay, img, mask_img)

    mask_img.save(mpath)
    overlay.save(opath)
    return {
        "ok": True,
        "mask_url": "/" + mpath.relative_to(ROOT).as_posix(),
        "overlay_url": "/" + opath.relative_to(ROOT).as_posix()
    }

# -----------------------------------------------------------------------------#
# VIDEO – IMG2VID (modell-frei – wählt erstes ladbares Video-Modell)
# -----------------------------------------------------------------------------#
@router.post("/video/img2vid")
async def video_img2vid(
    image: UploadFile = File(...),
    num_frames: int = Form(14),
    fps: int = Form(8),
    motion_strength: float = Form(0.8),
    model: Optional[str] = Form(None),
):
    models = _scan_models()
    if model:
        mdir = _model_dir_from_name(model, "video")
    else:
        cand = next((m for m in models["video"] if m["loadable"]), None)
        if not cand:
            raise HTTPException(400, "No loadable video model found.")
        mdir = ROOT / cand["path"].lstrip("/")

    pipe = _get_img2vid_pipe(mdir)

    raw = await image.read()
    init = Image.open(io.BytesIO(raw)).convert("RGB")
    init = _letterbox(init, (1024, 576))  # SVD sweet-spot; sicher für die meisten

    with torch.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu"):
        result = pipe(image=init, noise_aug_strength=float(motion_strength), num_frames=int(num_frames))
    frames = result.frames[0]
    url = _save_frames_as_video(frames, fps=int(fps))
    return {"ok": True, "output_url": url, "model": mdir.name}

# -----------------------------------------------------------------------------#
# MOTION – Alias auf video/img2vid
# -----------------------------------------------------------------------------#
@router.post("/motion/animate")
async def motion_animate(
    image: UploadFile = File(...),
    intensity: float = Form(0.8),
    frames: int = Form(14),
    fps: int = Form(8),
    model: Optional[str] = Form(None),
):
    return await video_img2vid(
        image=image, num_frames=frames, fps=fps, motion_strength=intensity, model=model
    )
