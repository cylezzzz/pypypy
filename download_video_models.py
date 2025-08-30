import os
import requests
from pathlib import Path

def download_file(url, path):
    print(f"Downloading {url}")
    response = requests.get(url, stream=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Saved to {path}")

# SVD Files
base_url = "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main"
model_dir = Path("X:/pypygennew/models/video/stable-video-diffusion-img2vid-xt")

files = [
    "model_index.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json", 
    "vae/diffusion_pytorch_model.safetensors",
    "image_encoder/config.json",
    "image_encoder/pytorch_model.bin",
    "scheduler/scheduler_config.json",
    "feature_extractor/preprocessor_config.json"
]

for file in files:
    url = f"{base_url}/{file}"
    path = model_dir / file
    download_file(url, path)

print("🎉 Video model download complete!")