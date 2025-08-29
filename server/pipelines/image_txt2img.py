"""
Enhanced Text-to-Image Pipeline for AndioMediaStudio
Supports multiple model architectures, advanced sampling, and no content restrictions
"""

from __future__ import annotations
from pathlib import Path
import torch
import logging
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import yaml
import json
import os

# Import all possible pipelines for maximum compatibility
try:
    from diffusers import (
        AutoPipelineForText2Image,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        DiffusionPipeline,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        DDIMScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        KDPM2DiscreteScheduler,
        KDPM2AncestralDiscreteScheduler,
    )
    # legacy conversion fallback (nur wenn from_single_file fehlt)
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        download_from_original_stable_diffusion_ckpt,
    )
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Diffusers not fully available: {e}")
    DIFFUSERS_AVAILABLE = False

try:
    import xformers  # noqa
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file with error handling"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}


class EnhancedTxt2ImgPipeline:
    """
    Enhanced Text-to-Image pipeline with support for:
    - Multiple model architectures (SD1.5, SDXL, Custom)
    - Advanced samplers and schedulers
    - LoRA support
    - Memory optimization
    - No content filtering (creative freedom)
    """

    def __init__(self, base_dir: Path, model_name: str | None = None):
        self.base_dir = Path(base_dir)
        self.model_name = model_name
        self.pipeline: Optional[DiffusionPipeline] = None
        self.model_info = None
        self.device = "cpu"
        self.dtype = torch.float32
        self.scheduler_map = self._init_scheduler_map()

        self.config = load_config(self.base_dir / "server" / "config" / "presets.yaml")
        self.settings = load_config(self.base_dir / "server" / "config" / "settings.yaml")

        self._setup_device()

    def _init_scheduler_map(self) -> Dict[str, Any]:
        if not DIFFUSERS_AVAILABLE:
            return {}
        return {
            "DPMSolverMultistep": DPMSolverMultistepScheduler,
            "DPM++2M": DPMSolverMultistepScheduler,
            "EulerAncestral": EulerAncestralDiscreteScheduler,
            "Euler": EulerDiscreteScheduler,
            "DDIM": DDIMScheduler,
            "LMS": LMSDiscreteScheduler,
            "PNDM": PNDMScheduler,
            "Heun": HeunDiscreteScheduler,
            "KDPM2": KDPM2DiscreteScheduler,
            "KDPM2Ancestral": KDPM2AncestralDiscreteScheduler,
        }

    def _setup_device(self) -> None:
        if torch.cuda.is_available() and self.settings.get("prefer_gpu", True):
            self.device = "cuda"
            self.dtype = torch.float16
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            logger.info("Using CPU")

    # ----------------------------- FLAT MODELS DIR -----------------------------

    def _models_root(self) -> Path:
        """
        Nutze **flaches** models/ Verzeichnis.
        Unterstützt zusätzlich Legacy-Unterordner (models/image).
        """
        flat = self.base_dir / "models"
        legacy = self.base_dir / "models" / "image"
        if flat.exists():
            return flat
        if legacy.exists():
            return legacy
        raise FileNotFoundError(f"Models directory not found: {flat}")

    def _find_model_path(self) -> Path:
        """
        Finde Modell in models/ (flat) **oder** legacy models/image/.
        Unterstützt Ordner (diffusers) und Single-File-Checkpoints.
        """
        root = self._models_root()

        # direkte Kandidaten
        def candidates(name: str) -> List[Path]:
            return [
                root / name,                         # Ordner
                root / f"{name}.safetensors",
                root / f"{name}.ckpt",
                root / f"{name}.bin",
                root / f"{name}.pt",
            ]

        if self.model_name:
            for c in candidates(self.model_name):
                if c.exists():
                    return c
            # fuzzy
            for p in root.iterdir():
                if self.model_name.lower() in p.name.lower():
                    return p

        # Fallback: diffusers-Ordner
        for p in root.iterdir():
            if p.is_dir() and (p / "model_index.json").exists():
                logger.info(f"Using fallback model folder: {p.name}")
                return p

        # Fallback: Single-File
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in {".safetensors", ".ckpt", ".bin", ".pt"}:
                logger.info(f"Using checkpoint model: {p.name}")
                return p

        raise FileNotFoundError(f"No suitable text2img model found in {root}")

    # --------------------------------------------------------------------------

    def _detect_model_architecture(self, model_path: Path) -> str:
        path_str = str(model_path).lower()
        if model_path.is_file():
            return "sd15"
        if any(kw in path_str for kw in ["sdxl", "xl", "stable-diffusion-xl", "refiner"]):
            return "sdxl"
        mi = model_path / "model_index.json"
        if mi.exists():
            try:
                cfg = json.loads(mi.read_text(encoding="utf-8"))
                if "StableDiffusionXLPipeline" in json.dumps(cfg):
                    return "sdxl"
            except Exception:
                pass
        return "sd15"

    def _create_pipeline_from_single_file(self, ckpt_path: Path) -> DiffusionPipeline:
        if hasattr(StableDiffusionPipeline, "from_single_file"):
            pipe = StableDiffusionPipeline.from_single_file(
                str(ckpt_path),
                torch_dtype=self.dtype,
                local_files_only=True,
                safety_checker=None,
                requires_safety_checker=False,
            )
        else:
            logger.info("Converting original SD checkpoint to diffusers format (legacy path)")
            dump_dir = ckpt_path.with_suffix("")
            dump_dir.mkdir(parents=True, exist_ok=True)
            download_from_original_stable_diffusion_ckpt(
                ckpt_path=str(ckpt_path),
                from_safetensors=ckpt_path.suffix.lower() == ".safetensors",
                extract_ema=True,
                device=self.device,
                dump_path=str(dump_dir),
            )
            pipe = StableDiffusionPipeline.from_pretrained(
                str(dump_dir),
                torch_dtype=self.dtype,
                local_files_only=True,
                safety_checker=None,
                requires_safety_checker=False,
            )
        logger.info(f"Loaded SD pipeline from single file: {ckpt_path.name}")
        return pipe

    def _create_pipeline(self, model_path: Path) -> DiffusionPipeline:
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusers library not available")

        if model_path.is_file():
            return self._create_pipeline_from_single_file(model_path)

        arch = self._detect_model_architecture(model_path)
        logger.info(f"Detected architecture: {arch}")

        common = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
            "safety_checker": None,
            "requires_safety_checker": False,
            "local_files_only": True,
        }

        try:
            pipe = AutoPipelineForText2Image.from_pretrained(str(model_path), **common)
            logger.info("Loaded with AutoPipelineForText2Image")
            return pipe
        except Exception as e:
            logger.warning(f"AutoPipeline failed, trying specific: {e}")

        if arch == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(str(model_path), **common)
            logger.info("Loaded with StableDiffusionXLPipeline")
            return pipe
        else:
            pipe = StableDiffusionPipeline.from_pretrained(str(model_path), **common)
            logger.info("Loaded with StableDiffusionPipeline")
            return pipe

    def _optimize_pipeline(self, pipe: DiffusionPipeline) -> DiffusionPipeline:
        pipe = pipe.to(self.device)
        if self.device == "cuda":
            if XFORMERS_AVAILABLE:
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xFormers memory efficient attention")
                except Exception as e:
                    logger.warning(f"xFormers enable failed: {e}")
            try:
                pipe.enable_model_cpu_offload()
                logger.info("Enabled CPU offloading")
            except Exception as e:
                logger.warning(f"CPU offloading not available: {e}")
            try:
                pipe.enable_vae_slicing()
                logger.info("Enabled VAE slicing")
            except Exception:
                pass
            try:
                pipe.enable_attention_slicing(1)
                logger.info("Enabled attention slicing")
            except Exception:
                pass

        if hasattr(torch, "compile") and self.settings.get("compile_unet", False):
            try:
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
                logger.info("Compiled UNet")
            except Exception as e:
                logger.warning(f"UNet compile failed: {e}")
        return pipe

    def ensure_loaded(self) -> None:
        if self.pipeline is not None:
            return
        logger.info(f"Loading Text2Image pipeline: {self.model_name or 'auto'}")
        model_path = self._find_model_path()
        logger.info(f"Using model: {model_path}")
        self.pipeline = self._optimize_pipeline(self._create_pipeline(model_path))
        self.model_info = {
            "name": model_path.name,
            "path": str(model_path),
            "architecture": self._detect_model_architecture(model_path),
            "device": self.device,
            "dtype": str(self.dtype),
        }
        logger.info(f"Pipeline loaded: {self.model_info}")

    def set_scheduler(self, name: str) -> None:
        if not name or name not in self.scheduler_map:
            return
        self.ensure_loaded()
        try:
            cls = self.scheduler_map[name]
            cfg = self.pipeline.scheduler.config
            self.pipeline.scheduler = cls.from_config(cfg)
            logger.info(f"Scheduler set: {name}")
        except Exception as e:
            logger.error(f"Set scheduler failed: {e}")

    def load_lora_weights(self, lora_paths: List[str], weights: List[float] = None) -> None:
        if not lora_paths:
            return
        self.ensure_loaded()
        if not hasattr(self.pipeline, "load_lora_weights"):
            logger.warning("LoRA not supported")
            return
        try:
            for i, lp in enumerate(lora_paths):
                w = weights[i] if weights and i < len(weights) else 1.0
                full = self.base_dir / lp
                if full.exists():
                    self.pipeline.load_lora_weights(str(full))
                    logger.info(f"Loaded LoRA: {lp} (w={w})")
                else:
                    logger.warning(f"LoRA not found: {full}")
        except Exception as e:
            logger.error(f"Load LoRA failed: {e}")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 896,
        height: int = 1152,
        num_inference_steps: int = 28,
        guidance_scale: float = 6.0,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        scheduler: Optional[str] = None,
        lora_paths: Optional[List[str]] = None,
        lora_weights: Optional[List[float]] = None,
        callback: Optional[callable] = None,
        callback_steps: int = 1,
        **kwargs,
    ) -> Tuple[List[Image.Image], Dict[str, Any]]:
        self.ensure_loaded()

        if scheduler:
            self.set_scheduler(scheduler)
        if lora_paths:
            self.load_lora_weights(lora_paths, lora_weights)

        gen = torch.Generator(device=self.device)
        if seed is None:
            seed = torch.randint(0, 2**31 - 1, (1,)).item()
        gen.manual_seed(seed)

        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Width and height must be multiples of 8")

        call_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": gen,
            "callback": callback,
            "callback_steps": callback_steps,
        }
        supported = self.pipeline.__class__.__call__.__code__.co_varnames
        call_args = {k: v for k, v in call_args.items() if (v is not None and k in supported)}

        logger.info(f"Generating {num_images_per_prompt} image(s) seed={seed}")
        out = self.pipeline(**call_args)
        images = out.images

        meta = {
            "seed": seed,
            "model": self.model_info["name"] if self.model_info else "unknown",
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "scheduler": scheduler or "default",
            "architecture": self.model_info.get("architecture", "unknown") if self.model_info else "unknown",
            "device": self.device,
            "num_images": len(images),
        }
        logger.info(f"Generation completed: {len(images)} images")
        return images, meta

    def generate_with_quality_preset(self, prompt: str, quality: str = "BALANCED", **kwargs) -> Tuple[List[Image.Image], Dict[str, Any]]:
        presets = self.config.get("image", {})
        p = presets.get(quality, presets.get("BALANCED", {}))
        args = {
            "width": p.get("width", 896),
            "height": p.get("height", 1152),
            "num_inference_steps": p.get("steps", 28),
            "guidance_scale": p.get("guidance", 6.0),
            **kwargs,
        }
        return self.generate(prompt, **args)

    def get_model_info(self) -> Dict[str, Any]:
        self.ensure_loaded()
        return self.model_info or {}

    def get_available_schedulers(self) -> List[str]:
        return list(self.scheduler_map.keys())

    def clear_cache(self) -> None:
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Cleared CUDA cache")

    def unload(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.model_info = None
            self.clear_cache()
            logger.info("Pipeline unloaded")


class Txt2ImgManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.pipelines: Dict[str, EnhancedTxt2ImgPipeline] = {}
        self.current_model = None

    def get_pipeline(self, model_name: str = None) -> EnhancedTxt2ImgPipeline:
        key = model_name or "default"
        if key not in self.pipelines:
            self.pipelines[key] = EnhancedTxt2ImgPipeline(self.base_dir, model_name)
        self.current_model = key
        return self.pipelines[key]

    def switch_model(self, model_name: str) -> EnhancedTxt2ImgPipeline:
        if self.current_model and self.current_model in self.pipelines:
            self.pipelines[self.current_model].unload()
        return self.get_pipeline(model_name)

    def list_available_models(self) -> List[str]:
        models_dir_flat = self.base_dir / "models"
        models_dir_legacy = self.base_dir / "models" / "image"
        root = models_dir_flat if models_dir_flat.exists() else models_dir_legacy
        if not root.exists():
            return []
        out: List[str] = []
        for p in root.iterdir():
            if p.is_dir() and (p / "model_index.json").exists():
                out.append(p.name)
            elif p.is_file() and p.suffix.lower() in {".safetensors", ".ckpt", ".bin", ".pt"}:
                out.append(p.name)
        return sorted(set(out))

    def cleanup_all(self) -> None:
        for p in self.pipelines.values():
            p.unload()
        self.pipelines.clear()
        self.current_model = None
        logger.info("All pipelines unloaded")


def create_txt2img_pipeline(base_dir: Path, model_name: str = None) -> EnhancedTxt2ImgPipeline:
    return EnhancedTxt2ImgPipeline(base_dir, model_name)


def generate_image(base_dir: Path, prompt: str, model_name: str = None, **kwargs) -> Tuple[List[Image.Image], Dict[str, Any]]:
    pipeline = create_txt2img_pipeline(base_dir, model_name)
    return pipeline.generate(prompt, **kwargs)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    pipeline = create_txt2img_pipeline(base_dir)
    imgs, meta = pipeline.generate(
        prompt="a beautiful sunset over mountains, photorealistic, 8k",
        negative_prompt="low quality, blurry",
        width=896,
        height=1152,
        num_inference_steps=28,
        guidance_scale=6.0,
        num_images_per_prompt=1,
    )
    if imgs:
        out = base_dir / "outputs" / "images" / "test_txt2img.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        imgs[0].save(out)
        print("Saved:", out)
