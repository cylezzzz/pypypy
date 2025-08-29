# server/api/andio_endpoints.py
# -----------------------------------------------------------------------------
# AndioMediaStudio – Kombinierte API-Router:
# - /api/txt2img            (Text -> Bild; SDXL Base)
# - /api/img2img            (Bild -> Bild; SDXL Base)
# - /api/editor/remove      (Bereich entfernen via einfacher Inpaint-Approx)
# - /api/wardrobe/segment   (Kleidung/Person-Masken via MediaPipe SelfieSeg)
# - /api/video/img2vid      (Bild -> Video; SVD-IMG2VID-XT)
# - /api/motion/animate     (Alias auf SVD: einfache "Motion"-Generierung)
# - /api/models_slim        (gefilterte Modell-Liste für UI)
#
# Lädt Pipelines lazy, nutzt vorhandene Modelle. Keine Fremd-Downloads nötig.
# -----------------------------------------------------------------------------

from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime
import io, os

# Core
import torch
from PIL import Image, ImageFilter, ImageOps

# Diffusers
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableVideoDiffusionPipeline,
)

# Optional: MediaPipe für schnelle Personensegmentierung
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

router = APIRouter(prefix="/api", tags=["andio"])

# -----------------------------------------------------------------------------#
# Pfade / Dtypes / Helpers
# -----------------------------------------------------------------------------#
ROOT = Path(__file__).resolve().parents[2]
IMG_MODELS = ROOT / "models" / "image"
VID_MODELS = ROOT / "models" / "video"
SDXL_BASE = IMG_MODELS / "stable-diffusion-xl-base-1.0"
SVD_XT   = VID_MODELS / "stable-video-diffusion-img2vid-xt"

OUT_IMG = ROOT / "outputs" / "images"
OUT_VID = ROOT / "outputs" / "videos"
for p in (OUT_IMG, OUT_VID):
    p.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if (DEVICE == "cuda") else torch.float32

_txt2img: Optional[StableDiffusionXLPipeline] = None
_img2img: Optional[StableDiffusionXLImg2ImgPipeline] = None
_img2vid: Optional[StableVideoDiffusionPipeline] = None

def _save_image(img: Image.Image, sub="images", suffix="png") -> str:
    base = OUT_IMG if sub == "images" else OUT_VID
    date_dir = base / datetime.now().strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    path = date_dir / f"gen_{ts}.{suffix}"
    img.save(path)
    return "/" + path.relative_to(ROOT).as_posix()

def _save_video(frames: list[Image.Image], fps: int = 8) -> str:
    """
    Speichert Frames als MP4 (ohne ffmpeg-Abhängigkeit fällt Backoff auf GIF).
    """
    date_dir = OUT_VID / datetime.now().strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    mp4_path = date_dir / f"vid_{ts}.mp4"
    gif_path = date_dir / f"vid_{ts}.gif"

    # Versuche MP4 (benötigt imageio[ffmpeg] / cv2). Falls nicht vorhanden -> GIF.
    try:
        import imageio
        imageio.mimsave(mp4_path, [f.convert("RGB") for f in frames], fps=fps)
        return "/" + mp4_path.relative_to(ROOT).as_posix()
    except Exception:
        # Fallback: GIF (breit unterstützt)
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0
        )
        return "/" + gif_path.relative_to(ROOT).as_posix()

def _lazy_txt2img():
    global _txt2img
    if _txt2img is None:
        if not SDXL_BASE.exists():
            raise HTTPException(500, f"SDXL Base not found: {SDXL_BASE}")
        pipe = StableDiffusionXLPipeline.from_pretrained(str(SDXL_BASE), torch_dtype=DTYPE)
        if DEVICE == "cuda":
            pipe = pipe.to("cuda")
            try: pipe.enable_xformers_memory_efficient_attention()
            except Exception: pass
        try: pipe.enable_vae_tiling()
        except Exception: pass
        _txt2img = pipe
    return _txt2img

def _lazy_img2img():
    global _img2img
    if _img2img is None:
        if not SDXL_BASE.exists():
            raise HTTPException(500, f"SDXL Base not found: {SDXL_BASE}")
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(str(SDXL_BASE), torch_dtype=DTYPE)
        if DEVICE == "cuda":
            pipe = pipe.to("cuda")
            try: pipe.enable_xformers_memory_efficient_attention()
            except Exception: pass
        try: pipe.enable_vae_tiling()
        except Exception: pass
        _img2img = pipe
    return _img2img

def _lazy_img2vid():
    global _img2vid
    if _img2vid is None:
        if not SVD_XT.exists():
            raise HTTPException(500, f"SVD-XT not found: {SVD_XT}")
        pipe = StableVideoDiffusionPipeline.from_pretrained(str(SVD_XT), torch_dtype=DTYPE)
        if DEVICE == "cuda":
            pipe = pipe.to("cuda")
        _img2vid = pipe
    return _img2vid

# -----------------------------------------------------------------------------#
# MODELS (gefiltert, ohne "components/checkpoints/loras")
# -----------------------------------------------------------------------------#
@router.get("/models_slim")
def models_slim():
    def usable_img(p: Path) -> bool:
        if not p.is_dir(): return False
        name = p.name.lower()
        ban = {"components", "checkpoints", "loras", ".cache", "motion_modules"}
        if name in ban: return False
        if (p / "model_index.json").exists(): return True
        return name.startswith("stable-diffusion-xl")
    def usable_vid(p: Path) -> bool:
        return p.is_dir() and (p / "model_index.json").exists()

    imgs = [d.name for d in IMG_MODELS.iterdir()] if IMG_MODELS.exists() else []
    vids = [d.name for d in (VID_MODELS.iterdir() if VID_MODELS.exists() else [])]

    return {
        "ok": True,
        "image": [n for n in imgs if usable_img(IMG_MODELS/n)],
        "video": [n for n in vids if usable_vid(VID_MODELS/n)],
    }

# -----------------------------------------------------------------------------#
# TXT2IMG
# -----------------------------------------------------------------------------#
class Txt2ImgReq(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = None
    steps: int = Field(20, ge=1, le=100)
    guidance: float = Field(5.0, ge=0, le=20)
    seed: Optional[int] = None
    width: int = Field(1024, ge=256, le=1536)
    height: int = Field(1024, ge=256, le=1536)

@router.post("/txt2img")
def txt2img(req: Txt2ImgReq):
    pipe = _lazy_txt2img()
    g = torch.Generator(device=DEVICE).manual_seed(int(req.seed)) if req.seed is not None else None
    image = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance,
        generator=g,
        width=req.width, height=req.height
    ).images[0]
    url = _save_image(image, "images", "png")
    return {"ok": True, "output_url": url, "meta": req.dict()}

# -----------------------------------------------------------------------------#
# IMG2IMG
# -----------------------------------------------------------------------------#
@router.post("/img2img")
async def img2img(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),
    strength: float = Form(0.6),
    steps: int = Form(20),
    guidance: float = Form(5.0),
    seed: Optional[int] = Form(None),
    image: UploadFile = File(...),
):
    pipe = _lazy_img2img()
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
    url = _save_image(out, "images", "png")
    return {"ok": True, "output_url": url}

# -----------------------------------------------------------------------------#
# EDITOR – Remove (approx. Inpaint ohne extra Inpaint-Weights)
# Idee: Maske weichzeichnen -> Bereich „neutralisieren“ -> img2img mit hohem strength
# -----------------------------------------------------------------------------#
@router.post("/editor/remove")
async def editor_remove(
    prompt: str = Form(""),
    strength: float = Form(0.75),
    steps: int = Form(20),
    guidance: float = Form(4.5),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    pipe = _lazy_img2img()
    raw = await image.read()
    mraw = await mask.read()
    init = Image.open(io.BytesIO(raw)).convert("RGB")
    msk  = Image.open(io.BytesIO(mraw)).convert("L")

    # Maske sanfter machen
    msk = msk.filter(ImageFilter.GaussianBlur(radius=6))
    # „Neutralisiere“ maskierten Bereich grob (median blur)
    blurred = init.filter(ImageFilter.MedianFilter(size=7))
    base = Image.composite(blurred, init, msk)  # msk=weiß => Bereich ersetzt

    # Re-Guide mit img2img (ohne prompt -> erhält Struktur)
    out = pipe(
        prompt=prompt or "",
        image=base,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
    ).images[0]

    url = _save_image(out, "images", "png")
    return {"ok": True, "output_url": url}

# -----------------------------------------------------------------------------#
# WARDROBE – Schnelle Personen-/Kleidungsmasken via MediaPipe
# Liefert PNG-Maske + Preview-Overlay-URL zurück
# -----------------------------------------------------------------------------#
@router.post("/wardrobe/segment")
async def wardrobe_segment(
    image: UploadFile = File(...),
    mode: str = Form("person"),  # "person" | "upper" | "lower" (heuristisch)
):
    if not HAS_MEDIAPIPE:
        raise HTTPException(500, "MediaPipe nicht verfügbar – bitte installieren.")

    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    import numpy as np
    mp_selfie = mp.solutions.selfie_segmentation
    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        arr = np.array(img)
        res = seg.process(arr[:, :, ::-1])  # RGB->BGR für MP (intern tolerant)
        mask = (res.segmentation_mask * 255).astype("uint8")
        mask_img = Image.fromarray(mask, mode="L")

    # Einfache Heuristik für upper/lower (oben/unten Bildhälfte)
    if mode in ("upper", "lower"):
        cut = Image.new("L", img.size, 0)
        w, h = img.size
        from PIL import ImageDraw
        draw = ImageDraw.Draw(cut)
        if mode == "upper":
            draw.rectangle([0, 0, w, h//2], fill=255)
        else:
            draw.rectangle([0, h//2, w, h], fill=255)
        mask_img = ImageChops.multiply(mask_img, cut)

    # Dateien speichern
    # 1) Maske
    date_dir = OUT_IMG / datetime.now().strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    mask_path = date_dir / f"mask_{ts}.png"
    mask_img.save(mask_path)

    # 2) Overlay-Vorschau
    overlay = Image.blend(img, Image.new("RGB", img.size, (255, 0, 0)), alpha=0.35)
    overlay = Image.composite(overlay, img, mask_img)
    overlay_path = date_dir / f"overlay_{ts}.png"
    overlay.save(overlay_path)

    return {
        "ok": True,
        "mask_url": "/" + mask_path.relative_to(ROOT).as_posix(),
        "overlay_url": "/" + overlay_path.relative_to(ROOT).as_posix(),
    }

# -----------------------------------------------------------------------------#
# VIDEO – IMG2VID (SVD-XT)
# -----------------------------------------------------------------------------#
@router.post("/video/img2vid")
async def video_img2vid(
    image: UploadFile = File(...),
    motion_strength: float = Form(0.8),
    num_frames: int = Form(14),
    fps: int = Form(8),
):
    pipe = _lazy_img2vid()
    raw = await image.read()
    init = Image.open(io.BytesIO(raw)).convert("RGB")

    # SVD akzeptiert 576x1024 / 1024x576 bevorzugt – wir passen an, aber bewahren Aspect mit Letterbox
    init = _letterbox(init, (1024, 576))

    with torch.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu"):
        out = pipe(
            image=init,
            noise_aug_strength=float(motion_strength),
            num_frames=int(num_frames)
        )
    frames = out.frames[0]  # Liste PIL.Image

    url = _save_video(frames, fps=int(fps))
    return {"ok": True, "output_url": url, "meta": {"frames": len(frames), "fps": int(fps)}}

def _letterbox(im: Image.Image, target: Tuple[int,int]) -> Image.Image:
    tw, th = target
    w, h = im.size
    scale = min(tw/w, th/h)
    nw, nh = int(w*scale), int(h*scale)
    im2 = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (tw, th), (0,0,0))
    ox, oy = (tw-nw)//2, (th-nh)//2
    canvas.paste(im2, (ox, oy))
    return canvas

# -----------------------------------------------------------------------------#
# MOTION – alias (nutzt video/img2vid intern)
# -----------------------------------------------------------------------------#
@router.post("/motion/animate")
async def motion_animate(
    image: UploadFile = File(...),
    style: str = Form("basic"),
    intensity: float = Form(0.8),
    frames: int = Form(14),
    fps: int = Form(8),
):
    # Delegiere an SVD
    return await video_img2vid(image=image, motion_strength=intensity, num_frames=frames, fps=fps)
