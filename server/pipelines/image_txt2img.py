from __future__ import annotations
from pathlib import Path
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import yaml

def load_settings(cfg_path: Path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class Txt2ImgPipeline:
    def __init__(self, base_dir: Path, model_name: str|None=None):
        self.base = base_dir
        self.cfg = load_settings(base_dir/"server"/"config"/"presets.yaml")
        self.settings = load_settings(base_dir/"server"/"config"/"settings.yaml")
        self.pipe = None
        self.model_name = model_name

    def _device_dtype(self):
        dev = "cuda" if torch.cuda.is_available() and self.settings.get("prefer_gpu",True) else "cpu"
        dtype = torch.float16 if dev=="cuda" else torch.float32
        return dev, dtype

    def _resolve_model_path(self):
        # Expect diffusers folder or a checkpoint path under models/image
        root = self.base/"models"/"image"
        # model_name is a stem found by scanner; we try to map to folder name
        if self.model_name:
            cand_dir = root/self.model_name
            if cand_dir.exists():
                return str(cand_dir)
            # else try parent of file with same stem
            for p in root.rglob("*"):
                if p.is_file() and p.stem==self.model_name:
                    return str(p.parent)
        # fallback: try any diffusers-like folder
        for p in root.iterdir():
            if p.is_dir() and (p/"model_index.json").exists():
                return str(p)
        raise FileNotFoundError("Kein diffusers Text2Image Modell im models/image gefunden.")

    def ensure(self):
        if self.pipe is not None: return
        path = self._resolve_model_path()
        dev, dtype = self._device_dtype()
        self.pipe = AutoPipelineForText2Image.from_pretrained(path, torch_dtype=dtype)
        if dev=="cuda": self.pipe = self.pipe.to("cuda")
        self.pipe.safety_checker = None

    @torch.inference_mode()
    def run(self, prompt: str, quality: str="BALANCED", negative: str|None=None, seed: int|None=None):
        self.ensure()
        preset = self.cfg["image"].get(quality, self.cfg["image"]["BALANCED"])
        gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        if seed is None: seed = int(torch.randint(0, 2**31-1, (1,)).item())
        gen = gen.manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative or "",
            guidance_scale=float(preset["guidance"]),
            num_inference_steps=int(preset["steps"]),
            width=int(preset["width"]), height=int(preset["height"]),
            generator=gen
        ).images[0]
        return result, {"seed": seed}
