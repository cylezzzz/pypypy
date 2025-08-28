"""
Enhanced Text-to-Image Pipeline for AndioMediaStudio
Supports multiple model architectures, advanced sampling, and no content restrictions
"""

from __future__ import annotations
from pathlib import Path
import torch
import logging
from typing import Optional, Dict, Any, List, Union, Tuple
from PIL import Image
import yaml
import json

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
        KDPM2AncestralDiscreteScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Diffusers not fully available: {e}")
    DIFFUSERS_AVAILABLE = False

try:
    import xformers
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
        self.pipeline = None
        self.model_info = None
        self.device = "cpu"
        self.dtype = torch.float32
        self.scheduler_map = self._init_scheduler_map()
        
        # Load configurations
        self.config = load_config(self.base_dir / "server" / "config" / "presets.yaml")
        self.settings = load_config(self.base_dir / "server" / "config" / "settings.yaml")
        
        # Initialize device settings
        self._setup_device()
        
    def _init_scheduler_map(self) -> Dict[str, Any]:
        """Initialize available schedulers"""
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
        """Setup computing device and data type"""
        if torch.cuda.is_available() and self.settings.get("prefer_gpu", True):
            self.device = "cuda"
            self.dtype = torch.float16
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            logger.info("Using CPU")
    
    def _find_model_path(self) -> Path:
        """
        Find the best matching model path
        Supports both diffusers format and single checkpoint files
        """
        models_root = self.base_dir / "models" / "image"
        
        if not models_root.exists():
            raise FileNotFoundError(f"Models directory not found: {models_root}")
        
        # If specific model requested, try to find it
        if self.model_name:
            # Try direct folder match
            candidate_path = models_root / self.model_name
            if candidate_path.exists() and candidate_path.is_dir():
                if (candidate_path / "model_index.json").exists():
                    return candidate_path
            
            # Try finding by stem name
            for path in models_root.rglob("*"):
                if path.is_dir() and path.name == self.model_name:
                    if (path / "model_index.json").exists():
                        return path
                elif path.is_file() and path.stem == self.model_name:
                    return path
        
        # Find any suitable model as fallback
        for path in models_root.iterdir():
            if path.is_dir() and (path / "model_index.json").exists():
                logger.info(f"Using fallback model: {path.name}")
                return path
        
        # Check for checkpoint files
        checkpoint_extensions = {".safetensors", ".ckpt", ".bin", ".pt"}
        for path in models_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in checkpoint_extensions:
                logger.info(f"Using checkpoint model: {path.name}")
                return path
        
        raise FileNotFoundError(f"No suitable text2img model found in {models_root}")
    
    def _detect_model_architecture(self, model_path: Path) -> str:
        """Detect model architecture from path and config"""
        path_str = str(model_path).lower()
        
        # Check for SDXL models
        if any(keyword in path_str for keyword in ["sdxl", "xl", "stable-diffusion-xl"]):
            return "sdxl"
        
        # Check model_index.json for architecture hints
        if model_path.is_dir():
            model_index = model_path / "model_index.json"
            if model_index.exists():
                try:
                    with open(model_index, 'r') as f:
                        config = json.load(f)
                        # Detect by UNet config
                        if config.get("unet", [None])[1] == "UNet2DConditionModel":
                            # Check cross attention dim for SDXL (2048) vs SD1.5 (768)
                            return "sdxl" if "xl" in path_str else "sd15"
                except:
                    pass
        
        # Default to SD1.5 compatible
        return "sd15"
    
    def _create_pipeline(self, model_path: Path) -> DiffusionPipeline:
        """Create appropriate pipeline based on model architecture"""
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusers library not available")
        
        architecture = self._detect_model_architecture(model_path)
        logger.info(f"Detected architecture: {architecture}")
        
        # Common arguments
        common_args = {
            "torch_dtype": self.dtype,
            "use_safetensors": True if model_path.suffix == ".safetensors" else None,
            "safety_checker": None,  # Remove content filtering
            "requires_safety_checker": False,
            "local_files_only": True,
        }
        
        try:
            # Try AutoPipeline first (most compatible)
            pipeline = AutoPipelineForText2Image.from_pretrained(
                str(model_path),
                **common_args
            )
            logger.info("Loaded with AutoPipelineForText2Image")
            
        except Exception as e:
            logger.warning(f"AutoPipeline failed, trying specific pipelines: {e}")
            
            # Fallback to specific pipelines
            if architecture == "sdxl":
                try:
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        str(model_path),
                        **common_args
                    )
                    logger.info("Loaded with StableDiffusionXLPipeline")
                except Exception as e2:
                    logger.error(f"SDXL pipeline failed: {e2}")
                    raise e2
            else:
                try:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        str(model_path),
                        **common_args
                    )
                    logger.info("Loaded with StableDiffusionPipeline")
                except Exception as e2:
                    logger.error(f"SD pipeline failed: {e2}")
                    raise e2
        
        return pipeline
    
    def _optimize_pipeline(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """Apply memory and performance optimizations"""
        # Move to device
        pipeline = pipeline.to(self.device)
        
        # Memory optimizations
        if self.device == "cuda":
            # Enable memory efficient attention
            if XFORMERS_AVAILABLE:
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xFormers memory efficient attention")
                except Exception as e:
                    logger.warning(f"Failed to enable xFormers: {e}")
            
            # Enable CPU offloading for large models
            try:
                pipeline.enable_model_cpu_offload()
                logger.info("Enabled CPU offloading")
            except Exception as e:
                logger.warning(f"CPU offloading not available: {e}")
            
            # Enable VAE slicing for memory efficiency
            try:
                pipeline.enable_vae_slicing()
                logger.info("Enabled VAE slicing")
            except:
                pass
                
            # Enable attention slicing
            try:
                pipeline.enable_attention_slicing(1)
                logger.info("Enabled attention slicing")
            except:
                pass
        
        # Compile UNet for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.settings.get('compile_unet', False):
            try:
                pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
                logger.info("Compiled UNet for faster inference")
            except Exception as e:
                logger.warning(f"UNet compilation failed: {e}")
        
        return pipeline
    
    def ensure_loaded(self) -> None:
        """Ensure pipeline is loaded and ready"""
        if self.pipeline is not None:
            return
        
        logger.info(f"Loading Text2Image pipeline: {self.model_name or 'auto'}")
        
        # Find model
        model_path = self._find_model_path()
        logger.info(f"Using model: {model_path}")
        
        # Create and optimize pipeline
        self.pipeline = self._create_pipeline(model_path)
        self.pipeline = self._optimize_pipeline(self.pipeline)
        
        # Store model info
        self.model_info = {
            "name": model_path.name,
            "path": str(model_path),
            "architecture": self._detect_model_architecture(model_path),
            "device": self.device,
            "dtype": str(self.dtype)
        }
        
        logger.info(f"Pipeline loaded successfully: {self.model_info}")
    
    def set_scheduler(self, scheduler_name: str) -> None:
        """Change the scheduler/sampler"""
        if not scheduler_name or scheduler_name not in self.scheduler_map:
            return
        
        self.ensure_loaded()
        
        try:
            scheduler_class = self.scheduler_map[scheduler_name]
            # Copy config from existing scheduler
            scheduler_config = self.pipeline.scheduler.config
            new_scheduler = scheduler_class.from_config(scheduler_config)
            self.pipeline.scheduler = new_scheduler
            logger.info(f"Changed scheduler to: {scheduler_name}")
        except Exception as e:
            logger.error(f"Failed to set scheduler {scheduler_name}: {e}")
    
    def load_lora_weights(self, lora_paths: List[str], weights: List[float] = None) -> None:
        """Load LoRA weights for model customization"""
        if not lora_paths:
            return
        
        self.ensure_loaded()
        
        if not hasattr(self.pipeline, 'load_lora_weights'):
            logger.warning("LoRA not supported by this pipeline")
            return
        
        try:
            for i, lora_path in enumerate(lora_paths):
                weight = weights[i] if weights and i < len(weights) else 1.0
                full_path = self.base_dir / lora_path
                
                if full_path.exists():
                    self.pipeline.load_lora_weights(str(full_path))
                    logger.info(f"Loaded LoRA: {lora_path} (weight: {weight})")
                else:
                    logger.warning(f"LoRA file not found: {full_path}")
        except Exception as e:
            logger.error(f"Failed to load LoRA weights: {e}")
    
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
        **kwargs
    ) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """
        Generate images from text prompts with full creative freedom
        
        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in the image
            width: Image width (must be multiple of 8)
            height: Image height (must be multiple of 8)
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow prompt
            num_images_per_prompt: Number of images to generate
            seed: Random seed for reproducibility
            scheduler: Scheduler/sampler to use
            lora_paths: List of LoRA weight paths to load
            lora_weights: Weights for each LoRA
            callback: Progress callback function
            callback_steps: Steps between callback calls
            **kwargs: Additional pipeline arguments
        
        Returns:
            Tuple of (generated_images, metadata)
        """
        
        self.ensure_loaded()
        
        # Set scheduler if requested
        if scheduler:
            self.set_scheduler(scheduler)
        
        # Load LoRA weights if provided
        if lora_paths:
            self.load_lora_weights(lora_paths, lora_weights)
        
        # Setup random seed
        generator = torch.Generator(device=self.device)
        if seed is None:
            seed = torch.randint(0, 2**31-1, (1,)).item()
        generator.manual_seed(seed)
        
        # Validate dimensions
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Width and height must be multiples of 8")
        
        # Generation parameters
        generation_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator,
            "callback": callback,
            "callback_steps": callback_steps,
            **kwargs
        }
        
        # Remove None values and unsupported args
        generation_args = {k: v for k, v in generation_args.items() 
                          if v is not None and k in self.pipeline.__class__.__call__.__code__.co_varnames}
        
        logger.info(f"Generating {num_images_per_prompt} image(s) with seed {seed}")
        
        try:
            # Generate images
            result = self.pipeline(**generation_args)
            images = result.images
            
            # Metadata
            metadata = {
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
                "num_images": len(images)
            }
            
            logger.info(f"Generation completed: {len(images)} images")
            return images, metadata
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise e
    
    def generate_with_quality_preset(
        self,
        prompt: str,
        quality: str = "BALANCED",
        **kwargs
    ) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """Generate using quality presets"""
        
        # Load quality settings
        quality_presets = self.config.get("image", {})
        preset = quality_presets.get(quality, quality_presets.get("BALANCED", {}))
        
        # Apply preset settings
        generation_kwargs = {
            "width": preset.get("width", 896),
            "height": preset.get("height", 1152),
            "num_inference_steps": preset.get("steps", 28),
            "guidance_scale": preset.get("guidance", 6.0),
            **kwargs  # User overrides
        }
        
        return self.generate(prompt, **generation_kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        self.ensure_loaded()
        return self.model_info or {}
    
    def get_available_schedulers(self) -> List[str]:
        """Get list of available schedulers"""
        return list(self.scheduler_map.keys())
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache"""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Cleared CUDA cache")
    
    def unload(self) -> None:
        """Unload pipeline to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.model_info = None
            self.clear_cache()
            logger.info("Pipeline unloaded")


class Txt2ImgManager:
    """
    Manager for multiple Text2Image pipelines
    Handles model switching and resource management
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.pipelines: Dict[str, EnhancedTxt2ImgPipeline] = {}
        self.current_model = None
    
    def get_pipeline(self, model_name: str = None) -> EnhancedTxt2ImgPipeline:
        """Get or create pipeline for specified model"""
        model_key = model_name or "default"
        
        if model_key not in self.pipelines:
            self.pipelines[model_key] = EnhancedTxt2ImgPipeline(
                self.base_dir, model_name
            )
        
        self.current_model = model_key
        return self.pipelines[model_key]
    
    def switch_model(self, model_name: str) -> EnhancedTxt2ImgPipeline:
        """Switch to different model"""
        # Unload current pipeline to free memory
        if self.current_model and self.current_model in self.pipelines:
            self.pipelines[self.current_model].unload()
        
        return self.get_pipeline(model_name)
    
    def list_available_models(self) -> List[str]:
        """List all available models"""
        models_dir = self.base_dir / "models" / "image"
        if not models_dir.exists():
            return []
        
        models = []
        
        # Scan for diffusers models
        for path in models_dir.iterdir():
            if path.is_dir() and (path / "model_index.json").exists():
                models.append(path.name)
        
        # Scan for checkpoint files
        checkpoint_extensions = {".safetensors", ".ckpt", ".bin", ".pt"}
        for path in models_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in checkpoint_extensions:
                models.append(path.stem)
        
        return sorted(list(set(models)))
    
    def cleanup_all(self) -> None:
        """Unload all pipelines"""
        for pipeline in self.pipelines.values():
            pipeline.unload()
        self.pipelines.clear()
        self.current_model = None
        logger.info("All pipelines unloaded")


# Convenience functions for backward compatibility
def create_txt2img_pipeline(base_dir: Path, model_name: str = None) -> EnhancedTxt2ImgPipeline:
    """Create a text-to-image pipeline"""
    return EnhancedTxt2ImgPipeline(base_dir, model_name)

def generate_image(
    base_dir: Path,
    prompt: str,
    model_name: str = None,
    **kwargs
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """Simple function to generate an image"""
    pipeline = create_txt2img_pipeline(base_dir, model_name)
    return pipeline.generate(prompt, **kwargs)

# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Basic test
    base_dir = Path(__file__).resolve().parents[2]  # AndioMediaStudio root
    
    try:
        pipeline = create_txt2img_pipeline(base_dir)
        images, metadata = pipeline.generate(
            prompt="a beautiful sunset over mountains, photorealistic, 8k",
            negative_prompt="low quality, blurry",
            width=896,
            height=1152,
            num_inference_steps=28,
            guidance_scale=6.0
        )
        
        print(f"Generated {len(images)} images")
        print(f"Metadata: {metadata}")
        
        # Save first image
        if images:
            output_path = base_dir / "outputs" / "images" / "test_txt2img.png"
            images[0].save(output_path)
            print(f"Saved to: {output_path}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)