# server/pipelines/enhanced_txt2img.py
from __future__ import annotations
import os
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
from server.config.model_config import ModelConfig
from server.registry.model_cache import ModelCache
from server.registry.model_loader import load_txt2img_pipeline
try:
    import torch
except Exception:
    torch = None  # type: ignore
Callback = Optional[Callable[[int, int, float], None]]
@dataclass
class Txt2ImgRequest:
    prompt: str
    negative_prompt: str = "worst quality, low quality, jpeg artifacts, watermark, signature, text"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    num_images: int = 1
    batch_size: int = 1
    seed: Optional[int] = None
    safety_checker: bool = True
class EnhancedTxt2Img:
    def __init__(self, cfg: Optional[ModelConfig] = None, cache: Optional[ModelCache] = None):
        self.cfg = cfg or ModelConfig()
        self.cache = cache or ModelCache(self.cfg.cache_capacity)
        if self.cfg.outputs_dir:
            base = Path(self.cfg.outputs_dir)
        else:
            base = Path(__file__).resolve().parents[2] / "outputs"
        self.out_images = base / "images"
        self.out_images.mkdir(parents=True, exist_ok=True)
    def _pipeline_key(self) -> str:
        parts = [f"model={self.cfg.model_id}", f"dev={self.cfg.resolve_device()}"]
        return "|".join(parts)
    def _get_or_load_pipe(self):
        key = self._pipeline_key()
        pipe = self.cache.get(key)
        if pipe is not None:
            return pipe
        pipe = load_txt2img_pipeline(self.cfg)
        self.cache.put(key, pipe)
        return pipe
    @staticmethod
    def _round_to_eight(x: int) -> int:
        return max(64, int(math.floor(x / 8) * 8))
    def _make_generator(self, seed: Optional[int]):
        if seed is None or torch is None:
            return None
        g = torch.Generator(device=self.cfg.resolve_device() if torch.cuda.is_available() else "cpu")
        return g.manual_seed(seed)
    def generate(self, req: Txt2ImgRequest, progress_cb: Callback = None) -> List[Dict]:
        if not req.prompt or not req.prompt.strip():
            raise ValueError("prompt darf nicht leer sein.")
        w = self._round_to_eight(req.width)
        h = self._round_to_eight(req.height)
        req.width, req.height = w, h
        pipe = self._get_or_load_pipe()
        if hasattr(pipe, "safety_checker"):
            if not (req.safety_checker and self.cfg.enable_safety_checker):
                setattr(pipe, "safety_checker", None)
        total = req.num_images
        bs = max(1, int(req.batch_size))
        remaining = total
        made = 0
        results: List[Dict] = []
        started = time.time()
        while remaining > 0:
            now_bs = min(bs, remaining)
            generator = self._make_generator(req.seed + made if req.seed is not None else None)
            images = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                num_images_per_prompt=now_bs,
                generator=generator,
            ).images
            for img in images:
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = f"txt2img_{ts}_{int(time.time()*1000)}.png"
                fpath = self.out_images / fname
                img.save(fpath)
                results.append({"file": str(fpath), "prompt": req.prompt})
                made += 1
                remaining -= 1
                if remaining <= 0:
                    break
        return results
