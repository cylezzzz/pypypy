# server/pipelines/image_inpaint.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, List
import io
import base64
import uuid

import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline


class InpaintRequest:
    def __init__(self, **data):
        self.image_path: str = data.get("image_path")
        self.mask_path: Optional[str] = data.get("mask_path")
        self.mask_b64: Optional[str] = data.get("mask_b64")
        self.prompt: str = data.get("prompt", "")
        self.negative_prompt: str = data.get("negative_prompt", "")
        self.steps: int = int(data.get("steps", 30))
        self.guidance: float = float(data.get("guidance", 7.5))
        self.strength: float = float(data.get("strength", 0.85))
        self.seed: Optional[int] = data.get("seed")
        self.width: Optional[int] = data.get("width")
        self.height: Optional[int] = data.get("height")


class SDInpaintEngine:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-inpainting"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load pipeline once
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,  # you control usage locally; keep enabled if public
        )
        self.pipe = self.pipe.to(self.device)
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            # Only available when xformers present
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    def _decode_mask_b64(self, data_uri: str) -> Image.Image:
        """Decode a data URL (PNG) to Pillow image (L)."""
        if "," in data_uri:
            data_uri = data_uri.split(",", 1)[1]
        raw = base64.b64decode(data_uri)
        return Image.open(io.BytesIO(raw)).convert("L")

    def _load_images(self, req: InpaintRequest) -> tuple[Image.Image, Optional[Image.Image]]:
        img = Image.open(req.image_path).convert("RGB")
        if req.width and req.height:
            img = img.resize((int(req.width), int(req.height)), Image.LANCZOS)

        mask = None
        if req.mask_path and Path(req.mask_path).exists():
            mask = Image.open(req.mask_path).convert("L")
        elif req.mask_b64:
            mask = self._decode_mask_b64(req.mask_b64)

        if mask is not None and mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)

        # SD expects white = to-change, black = keep. Ensure correct polarity (no-op placeholder).
        if mask is not None:
            mask = ImageOps.invert(ImageOps.invert(mask))

        return img, mask

    def run(self, req: InpaintRequest, progress: Optional[Callable[[int,int,float], None]] = None) -> List[str]:
        img, mask = self._load_images(req)

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(req.seed))

        images = self.pipe(
            prompt=req.prompt,
            negative_prompt=(req.negative_prompt or None),
            image=img,
            mask_image=mask,
            guidance_scale=float(req.guidance),
            num_inference_steps=int(req.steps),
            strength=float(req.strength),
            generator=generator,
        ).images

        out_dir = Path(__file__).resolve().parents[2] / "outputs" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        for i, im in enumerate(images):
            out = out_dir / f"inpaint_{uuid.uuid4().hex[:8]}_{i:02d}.png"
            im.save(out)
            saved.append(str(out).replace("\\", "/"))
            if progress:
                progress(i + 1, len(images), 0.0)
        return saved
