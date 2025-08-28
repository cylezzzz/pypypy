from __future__ import annotations
from pathlib import Path
import torch, os
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image
import imageio.v3 as iio

class SVDPipeline:
    def __init__(self, base_dir: Path, model_name: str|None=None):
        self.base = base_dir; self.pipe=None; self.model_name=model_name

    def _device_dtype(self):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev=="cuda" else torch.float32
        return dev, dtype

    def _resolve_model_path(self):
        root = self.base/"models"/"video"
        if self.model_name:
            d = root/self.model_name
            if d.exists(): return str(d)
            for p in root.rglob("*"):
                if p.is_file() and p.stem==self.model_name: return str(p.parent)
        # fallback any diffusers folder with model_index.json
        for p in root.iterdir():
            if p.is_dir() and (p/"model_index.json").exists():
                return str(p)
        raise FileNotFoundError("Kein SVD-Modell in models/video gefunden.")

    def ensure(self):
        if self.pipe is not None: return
        path = self._resolve_model_path()
        dev, dtype = self._device_dtype()
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(path, torch_dtype=dtype)
        if dev=="cuda": self.pipe = self.pipe.to("cuda")

    @torch.inference_mode()
    def run_txt2video(self, prompt: str, num_frames: int=25, height: int=320, width: int=576):
        self.ensure()
        # SVD requires an initial image; we can synthesize with a lightweight txt2img if available, else pure noise init
        frames = self.pipe(prompt=prompt, num_frames=num_frames, height=height, width=width).frames[0]
        return frames

    def save_mp4(self, frames, out_path: Path, fps: int=16):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(out_path, frames, fps=fps)
        return str(out_path)
