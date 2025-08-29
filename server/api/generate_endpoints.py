# server/api/generate_endpoints.py
# -----------------------------------------------------------------------------
# AndioMediaStudio – Minimal-pragmatische Generate-Endpoints (TXT2IMG / IMG2IMG)
# - Fügt sich in die bestehende Router-Autodiscovery ein (Namensschema *_endpoints.py)
# - Nutzt lokal vorhandenes SDXL-Base-Modell
# - Speichert Outputs unter outputs/images/YYYYMMDD/...
# - Liefert schlanke JSON-Antworten für die Web-UI
# -----------------------------------------------------------------------------

from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
from datetime import datetime
import io

# Diffusers / Torch
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image

router = APIRouter(prefix="/api", tags=["generate"])

# -----------------------------------------------------------------------------#
# Pfade & Lazy-Ladestrategie
# -----------------------------------------------------------------------------#
ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models" / "image"
SDXL_BASE = MODELS / "stable-diffusion-xl-base-1.0"

OUT_ROOT = ROOT / "outputs" / "images"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if (_device == "cuda") else torch.float32

_txt2img_pipe: Optional[StableDiffusionXLPipeline] = None
_img2img_pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None


def _ensure_pipelines():
    global _txt2img_pipe, _img2img_pipe
    if not SDXL_BASE.exists():
        raise HTTPException(status_code=500, detail=f"SDXL Base not found: {SDXL_BASE}")

    if _txt2img_pipe is None:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(SDXL_BASE), torch_dtype=_dtype
        )
        if _device == "cuda":
            pipe = pipe.to("cuda")
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        _txt2img_pipe = pipe

    if _img2img_pipe is None:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            str(SDXL_BASE), torch_dtype=_dtype
        )
        if _device == "cuda":
            pipe = pipe.to("cuda")
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        _img2img_pipe = pipe


def _save_image(img: Image.Image, suffix: str = "png") -> str:
    date_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    out_path = date_dir / f"gen_{ts}.{suffix}"
    img.save(out_path)
    # Rückgabe als Web-Pfad relativ zum Web-Root
    rel = out_path.relative_to(ROOT).as_posix()
    return "/" + rel


# -----------------------------------------------------------------------------#
# Schemas
# -----------------------------------------------------------------------------#
class Txt2ImgRequest(BaseModel):
    model: Optional[str] = Field(default="stable-diffusion-xl-base-1.0")
    prompt: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = None
    steps: int = Field(default=20, ge=1, le=100)
    guidance: float = Field(default=5.0, ge=0.0, le=20.0)
    seed: Optional[int] = None
    width: int = Field(default=1024, ge=256, le=1536)
    height: int = Field(default=1024, ge=256, le=1536)


@router.post("/txt2img")
def txt2img(req: Txt2ImgRequest):
    _ensure_pipelines()
    g = None
    if req.seed is not None:
        g = torch.Generator(device=_device).manual_seed(int(req.seed))

    image = _txt2img_pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance,
        generator=g,
        width=req.width,
        height=req.height,
    ).images[0]

    url = _save_image(image, "png")
    return {
        "ok": True,
        "model": req.model,
        "output_url": url,
        "meta": {
            "steps": req.steps,
            "guidance": req.guidance,
            "seed": req.seed,
            "size": [req.width, req.height],
        },
    }


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
    _ensure_pipelines()

    # Eingangsbild lesen
    raw = await image.read()
    init = Image.open(io.BytesIO(raw)).convert("RGB")

    g = None
    if seed is not None:
        g = torch.Generator(device=_device).manual_seed(int(seed))

    out = _img2img_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=g,
    ).images[0]

    url = _save_image(out, "png")
    return {
        "ok": True,
        "output_url": url,
        "meta": {
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "strength": strength,
        },
    }


# -----------------------------------------------------------------------------#
# Schlanke Modelle-Liste (nur „benutzbare“ Einträge)
#   – optionaler Ersatz falls /api/models fehlt oder „falsche“ Ordner anzeigt
# -----------------------------------------------------------------------------#
@router.get("/models_slim")
def models_slim():
    if not MODELS.exists():
        return {"ok": True, "image": [], "video": []}

    def usable(d: Path) -> bool:
        if not d.is_dir():
            return False
        name = d.name.lower()
        # Ausfiltern: Komponenten/Cache/LoRAs/Checkpoints usw.
        banned = {"components", "checkpoints", "loras", "motion_modules", ".cache"}
        if name in banned:
            return False
        # diffusers-Style? – model_index.json als Kriterium
        if (d / "model_index.json").exists():
            return True
        # einige bekannte Ordnernamen (SDXL Base z. B.) haben model_index.json, sonst „unknown“
        return name.startswith("stable-diffusion-xl")

    image_models = [p.name for p in MODELS.iterdir() if usable(p)]

    # Video-Ordner (best-effort)
    video_root = ROOT / "models" / "video"
    video_models = []
    if video_root.exists():
        for p in video_root.iterdir():
            if p.is_dir() and (p / "model_index.json").exists():
                video_models.append(p.name)

    return {"ok": True, "image": image_models, "video": video_models}
