from __future__ import annotations
from pathlib import Path
import torch
from diffusers import AutoPipelineForInpainting, StableDiffusionInpaintPipelineLegacy
from PIL import Image
import yaml

def load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f: 
        import yaml; return yaml.safe_load(f)

class InpaintPipeline:
    def __init__(self, base_dir: Path, model_name: str|None=None):
        self.base = base_dir
        self.cfg = load_yaml(base_dir/"server"/"config"/"presets.yaml")
        self.settings = load_yaml(base_dir/"server"/"config"/"settings.yaml")
        self.pipe = None; self.model_name = model_name

    def _device_dtype(self):
        dev = "cuda" if torch.cuda.is_available() and self.settings.get("prefer_gpu",True) else "cpu"
        dtype = torch.float16 if dev=="cuda" else torch.float32
        return dev, dtype

    def _resolve_model_path(self):
        root = self.base/"models"/"image"
        if self.model_name:
            d = root/self.model_name
            if d.exists(): return str(d)
            for p in root.rglob("*"):
                if p.is_file() and p.stem==self.model_name: return str(p.parent)
        for p in root.iterdir():
            if p.is_dir() and (p/"model_index.json").exists():
                return str(p)
        raise FileNotFoundError("Kein Inpaint-Modell im models/image gefunden.")

    def ensure(self):
        if self.pipe is not None: return
        path = self._resolve_model_path()
        dev, dtype = self._device_dtype()
        try:
            self.pipe = AutoPipelineForInpainting.from_pretrained(path, torch_dtype=dtype)
        except Exception:
            self.pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(path, torch_dtype=dtype)
        if dev=="cuda": self.pipe = self.pipe.to("cuda")
        self.pipe.safety_checker = None

    @torch.inference_mode()
    def run(self, init_image: Image.Image, mask_image: Image.Image, prompt: str, strength: float=1.0, quality: str="BALANCED", negative: str|None=None, seed: int|None=None):
        self.ensure()
        preset = self.cfg["image"].get(quality, self.cfg["image"]["BALANCED"])
        gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        if seed is None: seed = int(torch.randint(0, 2**31-1, (1,)).item())
        gen = gen.manual_seed(seed)
        result = self.pipe(
            image=init_image, mask_image=mask_image, prompt=prompt, negative_prompt=negative or "",
            strength=float(strength),
            guidance_scale=float(preset["guidance"]),
            num_inference_steps=int(preset["steps"]),
            generator=gen
        ).images[0]
        return result, {"seed": seed}
