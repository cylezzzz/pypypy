
# server/api/generation_endpoints.py
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

# --- Base paths ---
BASE_DIR = Path(__file__).resolve().parents[2]
OUT_IMAGES = BASE_DIR / "outputs" / "images"
OUT_VIDEOS = BASE_DIR / "outputs" / "videos"
OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_VIDEOS.mkdir(parents=True, exist_ok=True)

# --- Utils ---
def _b64_to_image(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64.split(",")[-1])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64 image: {e}")

def _path_to_url(path: Path) -> str:
    # Map filesystem path under outputs/ to public URL mounted at /outputs/*
    try:
        p = path.resolve()
        if OUT_IMAGES in p.parents or p == OUT_IMAGES:
            rel = p.relative_to(OUT_IMAGES)
            return f"/outputs/images/{rel.as_posix()}"
        if OUT_VIDEOS in p.parents or p == OUT_VIDEOS:
            rel = p.relative_to(OUT_VIDEOS)
            return f"/outputs/videos/{rel.as_posix()}"
    except Exception:
        pass
    # fallback: best-effort
    return f"/outputs/{path.name}"

def _ts_name(prefix: str, ext: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{int(time.time()*1000)}.{ext.lstrip('.')}"

# --- Imports of pipelines (local-only) ---
# Text2Img (enhanced)
try:
    from server.pipelines.image_txt2img import EnhancedTxt2ImgPipeline  # type: ignore
except Exception:
    EnhancedTxt2ImgPipeline = None  # type: ignore

# Image2Image
try:
    from server.pipelines.image_img2img import Img2ImgPipeline  # type: ignore
except Exception:
    Img2ImgPipeline = None  # type: ignore

# Inpainting
try:
    from server.pipelines.image_inpaint import SDInpaintEngine, InpaintRequest as InpReq  # type: ignore
except Exception:
    SDInpaintEngine = None  # type: ignore
    InpReq = None  # type: ignore

# Pose / Control
try:
    from server.pipelines.control_pose import extract_pose, draw_pose  # type: ignore
except Exception:
    extract_pose = None  # type: ignore
    draw_pose = None  # type: ignore

# Video (Stable Video Diffusion)
try:
    from server.pipelines.video_svd import SVDPipeline  # type: ignore
except Exception:
    SVDPipeline = None  # type: ignore

# Lip-Sync (Wav2Lip - requires local repo)
try:
    from server.pipelines.lipsync_wav2lip import run_wav2lip  # type: ignore
except Exception:
    run_wav2lip = None  # type: ignore

# --- Router ---
router = APIRouter(prefix="/api", tags=["generation"])

# -------------------- Schemas --------------------
class Txt2ImgBody(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 896
    height: int = 1152
    steps: int = Field(28, ge=1, le=150)
    guidance: float = Field(6.0, ge=0.0, le=50.0)
    num_images: int = Field(1, ge=1, le=8, description="How many images to generate")
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    model: Optional[str] = None
    lora_paths: Optional[List[str]] = None
    lora_weights: Optional[List[float]] = None

class Img2ImgBody(BaseModel):
    init_image_b64: str
    prompt: str
    strength: float = Field(0.65, ge=0.0, le=1.0)
    quality: str = "BALANCED"
    negative: Optional[str] = None
    seed: Optional[int] = None
    model: Optional[str] = None

class InpaintBody(BaseModel):
    image_b64: str
    mask_b64: Optional[str] = None
    prompt: str = ""
    negative_prompt: str = "blurry, artifacts, text, watermark"
    steps: int = 30
    guidance: float = 7.5
    strength: float = 0.85
    seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    model: Optional[str] = None

class PoseOverlayBody(BaseModel):
    image_b64: str
    draw: bool = True

class VideoBody(BaseModel):
    prompt: str
    num_frames: int = 25
    fps: int = 16
    width: int = 576
    height: int = 320
    model: Optional[str] = None

class LipSyncBody(BaseModel):
    face_video_path: str
    audio_path: str
    out_name: Optional[str] = None

# -------------------- Endpoints --------------------

@router.post("/generate/image", summary="Generate image(s) (txt2img)")
def generate_image_txt2img(body: Txt2ImgBody) -> Dict[str, Any]:
    if EnhancedTxt2ImgPipeline is None:
        raise HTTPException(status_code=500, detail="Text2Image pipeline not available")
    pipe = EnhancedTxt2ImgPipeline(BASE_DIR, body.model)
    try:
        images, meta = pipe.generate(
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            width=body.width,
            height=body.height,
            num_inference_steps=body.steps,
            guidance_scale=body.guidance,
            num_images_per_prompt=body.num_images,
            seed=body.seed,
            scheduler=body.scheduler,
            lora_paths=body.lora_paths,
            lora_weights=body.lora_weights,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"txt2img failed: {e}")
    urls: List[str] = []
    for img in images:
        name = _ts_name("txt2img", "png")
        out = OUT_IMAGES / name
        img.save(out)
        urls.append(_path_to_url(out))
    return {"ok": True, "files": urls, "meta": meta}

@router.post("/generate/img2img", summary="Image-to-Image")
def generate_image_img2img(body: Img2ImgBody) -> Dict[str, Any]:
    if Img2ImgPipeline is None:
        raise HTTPException(status_code=500, detail="Img2Img pipeline not available")
    init = _b64_to_image(body.init_image_b64)
    pipe = Img2ImgPipeline(BASE_DIR, body.model)
    try:
        result, info = pipe.run(init_image=init, prompt=body.prompt, strength=body.strength, quality=body.quality, negative=body.negative, seed=body.seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"img2img failed: {e}")
    name = _ts_name("img2img", "png")
    out = OUT_IMAGES / name
    result.save(out)
    return {"ok": True, "file": _path_to_url(out), "meta": info}

@router.post("/generate/inpaint", summary="Inpainting (mask optional if pipeline supports)")
def generate_image_inpaint(body: InpaintBody) -> Dict[str, Any]:
    if SDInpaintEngine is None or InpReq is None:
        raise HTTPException(status_code=500, detail="Inpaint engine not available")
    req = InpReq(
        image_path=None,
        mask_path=None,
        mask_b64=body.mask_b64,
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        steps=body.steps,
        guidance=body.guidance,
        strength=body.strength,
        seed=body.seed,
        width=body.width,
        height=body.height,
        model=body.model,
    )
    # SDInpaintEngine lädt die Bilddatei aus path – wir geben sie via temp
    # deshalb schreiben wir das b64-Bild kurz in eine temp-Datei
    tmp = OUT_IMAGES / _ts_name("inpaint_src", "png")
    img = _b64_to_image(body.image_b64)
    img.save(tmp)
    req.image_path = str(tmp)
    try:
        files = SDInpaintEngine().run(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inpaint failed: {e}")
    urls = [ _path_to_url(Path(f)) for f in files ]
    return {"ok": True, "files": urls}

@router.post("/pose/extract", summary="Extract pose landmarks and optionally draw overlay")
def pose_extract(body: PoseOverlayBody) -> Dict[str, Any]:
    if extract_pose is None:
        raise HTTPException(status_code=500, detail="Pose module not available")
    img = _b64_to_image(body.image_b64)
    try:
        lm = extract_pose(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pose extract failed: {e}")
    resp: Dict[str, Any] = {"ok": True, "landmarks": bool(lm)}
    if body.draw:
        if draw_pose is None:
            raise HTTPException(status_code=500, detail="Draw pose module not available")
        im = draw_pose(img, lm)
        name = _ts_name("pose", "png")
        out = OUT_IMAGES / name
        im.save(out)
        resp["overlay"] = _path_to_url(out)
    return resp

@router.post("/generate/video", summary="Text to Video (Stable Video Diffusion)")
def generate_video(body: VideoBody) -> Dict[str, Any]:
    if SVDPipeline is None:
        raise HTTPException(status_code=500, detail="Video pipeline not available")
    pipe = SVDPipeline(BASE_DIR, body.model)
    try:
        frames = pipe.run_txt2video(prompt=body.prompt, num_frames=body.num_frames, height=body.height, width=body.width)
        name = _ts_name("svd", "mp4")
        out = OUT_VIDEOS / name
        pipe.save_mp4(frames, out, fps=body.fps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"video generation failed: {e}")
    return {"ok": True, "file": _path_to_url(out)}

@router.post("/lipsync", summary="Lip Sync (Wav2Lip)")
def lipsync(body: LipSyncBody) -> Dict[str, Any]:
    if run_wav2lip is None:
        raise HTTPException(status_code=500, detail="Wav2Lip integration not configured. Please add local repo and checkpoint.")
    out = OUT_VIDEOS / (body.out_name or _ts_name("lipsync", "mp4"))
    try:
        path = run_wav2lip(BASE_DIR, Path(body.face_video_path), Path(body.audio_path), out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"lipsync failed: {e}")
    return {"ok": True, "file": _path_to_url(Path(path))}

# -------------------- Outputs listing --------------------

@router.get("/outputs", summary="List generated outputs (images & videos)")
def list_outputs() -> Dict[str, Any]:
    imgs = []
    vids = []
    if OUT_IMAGES.exists():
        for p in sorted(OUT_IMAGES.glob("*.*"), key=lambda x: x.stat().st_mtime, reverse=True):
            imgs.append(_path_to_url(p))
    if OUT_VIDEOS.exists():
        for p in sorted(OUT_VIDEOS.glob("*.*"), key=lambda x: x.stat().st_mtime, reverse=True):
            vids.append(_path_to_url(p))
    return {"images": imgs, "videos": vids}

@router.get("/outputs/images", summary="List generated images")
def list_images() -> List[str]:
    res = []
    if OUT_IMAGES.exists():
        for p in sorted(OUT_IMAGES.glob("*.*"), key=lambda x: x.stat().st_mtime, reverse=True):
            res.append(_path_to_url(p))
    return res

@router.get("/outputs/videos", summary="List generated videos")
def list_videos() -> List[str]:
    res = []
    if OUT_VIDEOS.exists():
        for p in sorted(OUT_VIDEOS.glob("*.*"), key=lambda x: x.stat().st_mtime, reverse=True):
            res.append(_path_to_url(p))
    return res
