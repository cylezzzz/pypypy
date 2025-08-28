# server/registry/model_loader.py
from __future__ import annotations
import os
from typing import Optional
from server.config.model_config import ModelConfig
try:
    import torch
    from diffusers import StableDiffusionPipeline
except Exception as e:
    raise RuntimeError("Diffusers/torch nicht verfügbar.") from e
def _maybe_enable_xformers(pipe, enable: bool):
    if not enable:
        return
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
def _maybe_cpu_offload(pipe, enable: bool):
    if not enable:
        return
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            pass
def _apply_lora_if_any(pipe: StableDiffusionPipeline, lora_path: Optional[str]):
    if not lora_path:
        return pipe
    try:
        pipe.load_lora_weights(lora_path)
        return pipe
    except Exception:
        pass
    try:
        pipe.unet.load_attn_procs(lora_path)
    except Exception:
        pass
    return pipe
def load_txt2img_pipeline(cfg: ModelConfig) -> StableDiffusionPipeline:
    device = cfg.resolve_device()
    dtype = cfg.resolve_dtype()
    model_source = cfg.model_id
    if cfg.models_dir and not os.path.isabs(model_source):
        local_candidate = os.path.join(cfg.models_dir, model_source)
        if os.path.isdir(local_candidate):
            model_source = local_candidate
    auth_token = cfg.hf_token if cfg.hf_token else True
    pipe = StableDiffusionPipeline.from_pretrained(
        model_source,
        torch_dtype=dtype,
        use_safetensors=True,
        safety_checker=None if not cfg.enable_safety_checker else None,
        token=auth_token if isinstance(auth_token, str) else None,
    )
    if cfg.enable_vae_slicing:
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    if cfg.enable_attention_slicing:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    _maybe_enable_xformers(pipe, cfg.enable_xformers)
    _apply_lora_if_any(pipe, cfg.lora_path)
    if device == "cuda":
        pipe.to("cuda")
    elif device == "mps":
        pipe.to("mps")
    else:
        pipe.to("cpu")
    _maybe_cpu_offload(pipe, cfg.enable_offload)
    return pipe
