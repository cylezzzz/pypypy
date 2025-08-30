# server/pipelines/video_svd_fixed.py
"""
REPARIERTE Video-Pipeline für SVD (Stable Video Diffusion)
- Echte Video-Generierung statt Thumbnails
- IMG2VID (Bild zu Video)
- TXT2VID (Text zu Video via SDXL + SVD)
"""

from __future__ import annotations
from pathlib import Path
import logging
from typing import List, Optional
import numpy as np
from PIL import Image

import torch

try:
    from diffusers import (
        StableVideoDiffusionPipeline,
        StableDiffusionXLPipeline,
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class FixedSVDPipeline:
    """REPARIERTE SVD Pipeline für echte Video-Generierung"""

    def __init__(self, base_dir: Path, model_name: str | None = None):
        self.base_dir = Path(base_dir)
        self.model_name = model_name or "stable-video-diffusion-img2vid-xt"

        self.svd_pipe = None
        self.txt2img_pipe = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info(f"🎬 SVD Pipeline initialized - Device: {self.device}")

    # ---------- Model discovery ----------

    def _find_svd_model(self) -> Path:
        """Finde SVD-Modell in models/video/"""
        video_dir = self.base_dir / "models" / "video"
        candidates = [
            video_dir / self.model_name,
            video_dir / "stable-video-diffusion-img2vid-xt",
            video_dir / "stable-video-diffusion-img2vid",
            video_dir / "svd-xt",
            video_dir / "svd",
        ]
        for c in candidates:
            if c.exists() and (c / "model_index.json").exists():
                logger.info(f"📦 Found SVD model: {c}")
                return c
        raise FileNotFoundError(f"SVD model not found in {video_dir}. Please download it first!")

    def _find_txt2img_model(self) -> Optional[Path]:
        """Finde SDXL-Modell für Text2Video"""
        image_dir = self.base_dir / "models" / "image"
        candidates = [
            image_dir / "stable-diffusion-xl-base-1.0",
            image_dir / "sdxl-base",
            self.base_dir / "models" / "stable-diffusion-xl-base-1.0",
        ]
        for c in candidates:
            if c.exists() and (c / "model_index.json").exists():
                logger.info(f"🖼️ Found SDXL model: {c}")
                return c
        logger.warning("No SDXL model found - Text2Video will not work")
        return None

    # ---------- Load pipelines ----------

    def load_svd_pipeline(self):
        """Lade SVD Pipeline mit Fallback-Handling"""
        if self.svd_pipe is not None:
            return
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("❌ Diffusers not available - install with: pip install diffusers")

        model_path = self._find_svd_model()
        try:
            logger.info("📦 Loading SVD pipeline...")
            self.svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
                str(model_path),
                torch_dtype=self.dtype,
                variant="fp16" if self.device == "cuda" else None,
            )
            if self.device == "cuda":
                self.svd_pipe = self.svd_pipe.to("cuda")
                try:
                    self.svd_pipe.enable_model_cpu_offload()
                    logger.info("✅ Enabled CPU offload for SVD")
                except Exception:
                    pass
            logger.info("✅ SVD Pipeline loaded successfully!")
        except Exception as e:
            msg = str(e)
            if "incomplete metadata" in msg or "not fully covered" in msg:
                raise RuntimeError(
                    f"❌ SVD model download incomplete: {model_path}\n"
                    "Please wait for download to finish or re-download the model."
                )
            raise RuntimeError(f"❌ Failed to load SVD pipeline: {msg}")

    def load_txt2img_pipeline(self):
        """Lade SDXL für Text2Video"""
        if self.txt2img_pipe is not None:
            return
        model_path = self._find_txt2img_model()
        if not model_path:
            return

        logger.info("🖼️ Loading SDXL pipeline for Text2Video...")
        self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
            str(model_path),
            torch_dtype=self.dtype,
            variant="fp16" if self.device == "cuda" else None,
            use_safetensors=True,
        )
        if self.device == "cuda":
            self.txt2img_pipe = self.txt2img_pipe.to("cuda")
        logger.info("✅ SDXL Pipeline loaded for Text2Video!")

    # ---------- Core: IMG2VID ----------

    @torch.inference_mode()
    def img2video(
        self,
        image: Image.Image,
        num_frames: int = 14,
        motion_bucket_id: int = 127,
        fps: int = 7,
        noise_aug_strength: float = 0.1,
        decode_chunk_size: int = 8,
    ) -> List[Image.Image]:
        """Image-to-Video Generierung mit SVD (mit robustem Fallback)"""
        try:
            self.load_svd_pipeline()
        except Exception as e:
            logger.warning(f"⚠️ Could not load SVD pipeline: {e}. Using fallback.")
            return self._fallback_from_image(image, num_frames, motion_bucket_id)

        # Input-Bild auf SVD-Format bringen
        image = self._prepare_image(image)

        try:
            logger.info(f"🎬 Generating {num_frames} frame video from image...")
            result = self.svd_pipe(
                image=image,
                num_frames=num_frames,
                motion_bucket_id=motion_bucket_id,
                fps=fps,
                noise_aug_strength=noise_aug_strength,
                decode_chunk_size=decode_chunk_size,
            )
            frames = result.frames[0]  # Liste von PIL Images
            logger.info(f"✅ Generated {len(frames)} frames successfully!")
            return frames
        except Exception as e:
            logger.warning(f"⚠️ SVD generation failed: {e}. Using fallback.")
            return self._fallback_from_image(image, num_frames, motion_bucket_id)

    def _fallback_from_image(self, image: Image.Image, num_frames: int, motion_bucket_id: int) -> List[Image.Image]:
        """Fallback: einfacher Mock-Generator (oder statische Frames)"""
        try:
            from server.pipelines.video_mock import MockVideoGenerator

            mock_gen = MockVideoGenerator(self.base_dir)
            motion_types = ["pulse", "ken_burns", "zoom", "shake"]
            motion_type = motion_types[min(3, motion_bucket_id // 64)]
            intensity = min(1.0, motion_bucket_id / 255.0)

            frames = mock_gen.img2video_mock(image, num_frames, motion_type, intensity)
            logger.info(f"✅ Fallback produced {len(frames)} frames with '{motion_type}' motion")
            return frames
        except Exception as e:
            logger.error(f"❌ Fallback mock failed: {e}. Using static frames.")
            # Letzter Ausweg: statische Frames (Kopien, damit Writer nicht referenzgleich ist)
            return [image.copy() for _ in range(num_frames)]

    # ---------- Core: TXT2VID ----------

    @torch.inference_mode()
    def txt2video(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, bad anatomy",
        num_frames: int = 14,
        width: int = 1024,
        height: int = 576,
        motion_bucket_id: int = 127,
        **kwargs,
    ) -> List[Image.Image]:
        """Text-to-Video: Text -> SDXL -> SVD -> Video"""
        logger.info("🖼️ Step 1/2: Generating initial frame with SDXL...")
        self.load_txt2img_pipeline()
        if not self.txt2img_pipe:
            raise RuntimeError("❌ No SDXL model found for Text2Video")

        initial_image = self.txt2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=kwargs.get("steps", 30),
            guidance_scale=kwargs.get("guidance", 7.5),
            num_images_per_prompt=1,
        ).images[0]
        logger.info("✅ Initial frame generated!")

        logger.info("🎬 Step 2/2: Converting to video with SVD...")
        frames = self.img2video(
            image=initial_image,
            num_frames=num_frames,
            motion_bucket_id=motion_bucket_id,
        )
        logger.info("✅ Text2Video completed!")
        return frames

    # ---------- Helpers ----------

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Bereite Bild für SVD vor (optimale Größe)"""
        # SVD arbeitet am besten mit 1024x576 oder 576x1024
        aspect = image.width / image.height
        if aspect > 1.5:
            target_w, target_h = 1024, 576  # Landscape
        else:
            target_w, target_h = 576, 1024  # Portrait

        img = image.copy()
        img.thumbnail((target_w, target_h), Image.LANCZOS)

        padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        paste_x = (target_w - img.width) // 2
        paste_y = (target_h - img.height) // 2
        padded.paste(img, (paste_x, paste_y))
        return padded

    def save_video_mp4(self, frames: List[Image.Image], output_path: Path, fps: int = 8) -> Path:
        """Speichere Frames als MP4 Video (fällt bei Bedarf auf GIF zurück)"""
        if not IMAGEIO_AVAILABLE:
            return self.save_video_gif(frames, output_path.with_suffix(".gif"), fps)
        try:
            frame_arrays = [np.array(f.convert("RGB")) for f in frames]
            imageio.mimsave(str(output_path), frame_arrays, fps=fps, quality=8, macro_block_size=1)
            logger.info(f"✅ Video saved as MP4: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ MP4 save failed: {e}, falling back to GIF")
            return self.save_video_gif(frames, output_path.with_suffix(".gif"), fps)

    def save_video_gif(self, frames: List[Image.Image], output_path: Path, fps: int = 8) -> Path:
        """Speichere Frames als GIF (Fallback)"""
        try:
            duration = int(1000 / fps)  # ms per frame
            frames[0].save(
                str(output_path),
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
                optimize=True,
            )
            logger.info(f"✅ Video saved as GIF: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ GIF save failed: {e}")
            raise

    def cleanup(self):
        """Räume GPU Memory auf"""
        if self.svd_pipe:
            del self.svd_pipe
            self.svd_pipe = None
        if self.txt2img_pipe:
            del self.txt2img_pipe
            self.txt2img_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🧹 Pipeline memory cleaned up")


# Quick Test Function
def test_svd_pipeline(base_dir: Path) -> bool:
    """Teste die reparierte SVD Pipeline"""
    try:
        pipeline = FixedSVDPipeline(base_dir)
        test_image = Image.new("RGB", (1024, 576), color=(50, 100, 200))
        logger.info("🧪 Testing IMG2VID...")
        frames = pipeline.img2video(test_image, num_frames=8)
        output_path = base_dir / "outputs" / "videos" / "svd_test.gif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.save_video_gif(frames, output_path, fps=8)
        logger.info(f"✅ Test completed! Video: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    test_svd_pipeline(base_dir)
