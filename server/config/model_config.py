# server/config/model_config.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
try:
    import torch
except Exception:
    torch = None  # type: ignore
@dataclass
class ModelConfig:
    model_id: str = os.environ.get("AMS_TXT2IMG_MODEL", "runwayml/stable-diffusion-v1-5")
    lora_path: Optional[str] = os.environ.get("AMS_LORA_PATH") or None
    hf_token: Optional[str] = os.environ.get("HF_TOKEN") or None
    models_dir: Optional[str] = os.environ.get("AMS_MODELS_DIR") or None
    device: str = os.environ.get("AMS_DEVICE", "auto")
    dtype: str = os.environ.get("AMS_DTYPE", "auto")
    enable_safety_checker: bool = os.environ.get("AMS_SAFETY", "1") == "1"
    enable_attention_slicing: bool = os.environ.get("AMS_ATT_SLICING", "1") == "1"
    enable_vae_slicing: bool = os.environ.get("AMS_VAE_SLICING", "1") == "1"
    enable_xformers: bool = os.environ.get("AMS_XFORMERS", "1") == "1"
    enable_offload: bool = os.environ.get("AMS_OFFLOAD", "1") == "1"
    cache_capacity: int = int(os.environ.get("AMS_MODEL_CACHE_CAP", "2"))
    outputs_dir: str = os.environ.get("AMS_OUTPUTS_DIR", "")
    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            return "mps"
        return "cpu"
    def resolve_dtype(self):
        if torch is None:
            return None
        m = (self.dtype or "auto").lower()
        if m == "float16":
            return torch.float16
        if m == "bfloat16":
            return torch.bfloat16
        if m == "float32":
            return torch.float32
        dev = self.resolve_device()
        if dev == "cuda":
            return torch.float16
        return torch.float32
