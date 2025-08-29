# server/pipelines/image_inpaint.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
import logging
import base64
import io

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# diffusers optional laden
try:
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline, DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except Exception as e:
    DIFFUSERS_AVAILABLE = False
    logger.warning(f"Diffusers not fully available for inpaint: {e}")

# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parents[2]
MODELS_DIR_FLAT = BASE / "models"
MODELS_DIR_LEGACY = BASE / "models" / "image"
CHECKPOINT_EXT = {".safetensors", ".ckpt", ".bin", ".pt"}

def _models_root() -> Path:
    # zuerst flach, dann legacy akzeptieren
    if MODELS_DIR_FLAT.exists():
        return MODELS_DIR_FLAT
    if MODELS_DIR_LEGACY.exists():
        return MODELS_DIR_LEGACY
    raise FileNotFoundError(f"Models directory not found: {MODELS_DIR_FLAT}")

def _find_model_path(name: Optional[str]) -> Path:
    root = _models_root()
    if name:
        candidates = [
            root / name,
            root / f"{name}.safetensors",
            root / f"{name}.ckpt",
            root / f"{name}.bin",
            root / f"{name}.pt",
        ]
        for c in candidates:
            if c.exists():
                return c
        # fuzzy
        for p in root.iterdir():
            if name.lower() in p.name.lower():
                return p

    # fallback: diffusers-ordner
    for p in root.iterdir():
        if p.is_dir() and (p / "model_index.json").exists():
            logger.info(f"[inpaint] fallback diffusers folder: {p.name}")
            return p

    # fallback: irgendein checkpoint (nicht ideal für inpaint)
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in CHECKPOINT_EXT:
            logger.info(f"[inpaint] fallback checkpoint: {p.name}")
            return p

    raise FileNotFoundError(f"No inpaint-capable model found in {root}")

def _load_image(path_or_b64: Optional[str], b64: Optional[str] = None) -> Image.Image:
    if b64:
        raw = base64.b64decode(b64.split(",")[-1])
        return Image.open(io.BytesIO(raw)).convert("RGBA")
    if path_or_b64:
        p = Path(path_or_b64)
        if not p.exists():
            raise FileNotFoundError(f"image not found: {p}")
        return Image.open(p).convert("RGBA")
    raise ValueError("no image provided")

def _load_mask(path_or_b64: Optional[str], b64: Optional[str] = None, size: Optional[tuple[int,int]] = None) -> Optional[Image.Image]:
    if not path_or_b64 and not b64:
        return None
    if b64:
        raw = base64.b64decode(b64.split(",")[-1])
        m = Image.open(io.BytesIO(raw)).convert("L")
    else:
        p = Path(path_or_b64)
        if not p.exists():
            raise FileNotFoundError(f"mask not found: {p}")
        m = Image.open(p).convert("L")
    if size and m.size != size:
        m = m.resize(size, Image.NEAREST)
    return m

# ---------------------------------------------------------------------------

class InpaintRequest(torch.nn.Module):
    def __init__(
        self,
        image_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        mask_b64: Optional[str] = None,
        prompt: str = "",
        negative_prompt: str = "blurry, artifacts, text, watermark",
        steps: int = 30,
        guidance: float = 7.5,
        strength: float = 0.85,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        model: Optional[str] = None,  # <— neu: explizites Modell möglich
    ):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.mask_b64 = mask_b64
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.steps = steps
        self.guidance = guidance
        self.strength = strength
        self.seed = seed
        self.width = width
        self.height = height
        self.model = model

class SDInpaintEngine:
    """
    Lädt **lazy** ein Inpainting-Modell ausschließlich lokal aus `models/` (flat/legacy).
    Unterstützt:
      • Diffusers-Ordner mit model_index.json (empfohlen: SD-Inpaint)
      • Single-File Checkpoints (.safetensors/.ckpt/.bin/.pt) -> fallback mit normaler SD-Pipeline (Maske wird ignoriert!)
    """
    def __init__(self, prefer_gpu: bool = True) -> None:
        self.device = "cuda" if (prefer_gpu and torch.cuda.is_available()) else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe: Optional[DiffusionPipeline] = None
        self.loaded_from: Optional[Path] = None

    def _create_pipeline(self, model_path: Path) -> DiffusionPipeline:
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusers not installed")

        common = dict(
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,  # **nie** ins Netz gehen
        )

        if model_path.is_file():
            # Single-File: Inpainting-Pipeline aus Single-File gibt es nur, wenn das Model kompatibel ist.
            # Viele .safetensors sind "normale" SD-Weights -> wir laden dann normale SD-Pipeline (Maske wird ignoriert).
            if hasattr(StableDiffusionInpaintPipeline, "from_single_file"):
                try:
                    pipe = StableDiffusionInpaintPipeline.from_single_file(str(model_path), **common)
                    logger.info(f"[inpaint] loaded InpaintPipeline from single file: {model_path.name}")
                    return pipe.to(self.device)
                except Exception as e:
                    logger.warning(f"[inpaint] single-file inpaint not supported, falling back to SD pipeline: {e}")

            pipe = StableDiffusionPipeline.from_single_file(str(model_path), **common)
            logger.info(f"[inpaint] loaded **normal** SD pipeline from single file (mask will be ignored): {model_path.name}")
            return pipe.to(self.device)

        # Diffusers-Ordner
        # Versuch Inpaint-Pipeline
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(str(model_path), **common)
            logger.info(f"[inpaint] loaded InpaintPipeline from folder: {model_path.name}")
            return pipe.to(self.device)
        except Exception as e:
            logger.warning(f"[inpaint] folder not inpaint-capable, trying normal SD pipeline: {e}")
            pipe = StableDiffusionPipeline.from_pretrained(str(model_path), **common)
            logger.info(f"[inpaint] loaded **normal** SD pipeline from folder (mask will be ignored): {model_path.name}")
            return pipe.to(self.device)

    def ensure_loaded(self, model_name: Optional[str]) -> None:
        if self.pipe is not None:
            return
        model_path = _find_model_path(model_name)
        self.pipe = self._create_pipeline(model_path)
        self.loaded_from = model_path

        # leichte Optimierungen
        if self.device == "cuda":
            try:
                self.pipe.enable_attention_slicing(1)
            except Exception:
                pass
            try:
                self.pipe.enable_vae_slicing()
            except Exception:
                pass

    def run(self, req: InpaintRequest, progress_cb: Optional[Callable[[int,int,str], None]] = None) -> List[str]:
        self.ensure_loaded(req.model)

        img = _load_image(req.image_path)
        msk = _load_mask(req.mask_path, req.mask_b64, size=img.size)

        # Dimensionen ggf. anpassen
        if req.width and req.height:
            # SD verlangt vielfache von 8
            w = (int(req.width) // 8) * 8
            h = (int(req.height) // 8) * 8
            if (w, h) != img.size:
                img = img.resize((w, h), Image.LANCZOS)
                if msk is not None:
                    msk = msk.resize((w, h), Image.NEAREST)

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(req.seed))

        # Prüfen, ob wir eine echte Inpaint-Pipeline haben
        is_inpaint = isinstance(self.pipe, StableDiffusionInpaintPipeline)

        if msk is None and is_inpaint:
            logger.warning("[inpaint] no mask provided — result will ~equal original. Provide a white=change / black=keep mask.")
        if msk is None and not is_inpaint:
            logger.warning("[inpaint] mask ignored because pipeline is **not** inpaint-capable (normal SD).")

        # Aufruf-Argumente zusammenstellen
        call = dict(
            prompt=req.prompt or "",
            negative_prompt=req.negative_prompt or "",
            num_inference_steps=int(req.steps),
            guidance_scale=float(req.guidance),
            generator=generator,
        )

        if is_inpaint:
            # echte Inpainting-Argumente
            call.update(dict(
                image=img.convert("RGB"),
                mask_image=msk if msk is not None else Image.new("L", img.size, color=0),
                strength=float(req.strength),
            ))
        else:
            # normale SD-Pipeline (Maske kann nicht wirken)
            call.update(dict(
                width=img.size[0],
                height=img.size[1],
            ))

        if progress_cb:
            # diffusers-callback ist bei vielen Pipelines verfügbar; wir nutzen step/total heuristisch
            def _cb(step, t, kwargs):
                try:
                    progress_cb(step, int(req.steps), f"step {step}/{req.steps}")
                except Exception:
                    pass
            call["callback"] = _cb
            call["callback_steps"] = 1

        out = self.pipe(**call)
        imgs: List[Image.Image] = out.images

        out_dir = BASE / "outputs" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        results: List[str] = []
        for i, im in enumerate(imgs):
            p = out_dir / f"inpaint_{(self.loaded_from.name if self.loaded_from else 'model')}_{i}.png"
            im.save(p)
            results.append(str(p))
        return results
