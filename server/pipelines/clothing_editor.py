# server/pipelines/clothing_editor.py
"""
Advanced Clothing Editor Pipeline
Klamotten entfernen, ändern, anpassen mit KI.

Hinweise:
- Benötigte optionale Abhängigkeiten: diffusers, segment-anything, mediapipe, torch, opencv-python, pillow, numpy
- Modelle werden unter base_dir/models/... erwartet (siehe Pfade unten).
- Wenn ein Baustein fehlt, wird sauber geloggt und ein Fallback genutzt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import logging
import time

import numpy as np

# --- optionale Libs (robust importieren) ---
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

# Segment Anything
try:
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore
    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False
    SamPredictor = None  # type: ignore
    sam_model_registry = {}  # type: ignore
    logging.warning("SAM nicht verfügbar – 'segment-anything' installieren, um Masken zu verfeinern.")

# MediaPipe
try:
    import mediapipe as mp  # type: ignore
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False
    mp = None  # type: ignore
    logging.warning("MediaPipe nicht verfügbar – Pose-gestützte Segmentierung eingeschränkt.")

# Diffusers
try:
    from diffusers import (  # type: ignore
        AutoPipelineForInpainting,
        ControlNetModel,
        StableDiffusionControlNetInpaintPipeline,
    )
    DIFFUSERS_AVAILABLE = True
except Exception:
    AutoPipelineForInpainting = None  # type: ignore
    ControlNetModel = None  # type: ignore
    StableDiffusionControlNetInpaintPipeline = None  # type: ignore
    DIFFUSERS_AVAILABLE = False
    logging.warning("Diffusers nicht verfügbar – KI-Inpainting/ControlNet deaktiviert.")


# ======================================================================
#                         Segmentierung
# ======================================================================

class ClothingSegmentator:
    """Intelligente (pose-gestützte) Kleidungssegmentierung + SAM-Verfeinerung."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.sam_predictor: Optional[SamPredictor] = None
        self.mediapipe_pose = None
        self._setup_models()

    def _setup_models(self):
        # SAM
        if SAM_AVAILABLE and torch is not None:
            sam_path = self.base_dir / "models" / "segmentation" / "sam_vit_h.pth"
            if sam_path.exists():
                try:
                    sam = sam_model_registry["vit_h"](checkpoint=str(sam_path))  # type: ignore
                    if torch.cuda.is_available():
                        sam.to("cuda")
                    self.sam_predictor = SamPredictor(sam)  # type: ignore
                    logging.info("SAM geladen.")
                except Exception as e:
                    logging.warning(f"SAM konnte nicht initialisiert werden: {e}")

        # MediaPipe Pose
        if MEDIAPIPE_AVAILABLE and mp is not None:
            try:
                mp_pose = mp.solutions.pose
                self.mediapipe_pose = mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True
                )
                logging.info("MediaPipe Pose geladen.")
            except Exception as e:
                logging.warning(f"MediaPipe Pose Fehler: {e}")
                self.mediapipe_pose = None

    # ------------------------- Public API -------------------------

    def detect_clothing_regions(self, image: np.ndarray, clothing_type: str = "all") -> Dict[str, np.ndarray]:
        """
        Erzeuge grobe Binärmasken für Kleidungsbereiche (uint8, 0/255).
        image: HxWxC, dtype uint8, RGB oder BGR (wir erkennen das robust).
        """
        if cv2 is None:
            logging.error("OpenCV (cv2) nicht verfügbar – Segmentierung nicht möglich.")
            return {}

        img_rgb = self._ensure_rgb(image)
        pose_landmarks = None
        if self.mediapipe_pose is not None:
            try:
                # MediaPipe erwartet RGB
                results = self.mediapipe_pose.process(img_rgb)
                if getattr(results, "pose_landmarks", None):
                    pose_landmarks = results.pose_landmarks.landmark
            except Exception as e:
                logging.debug(f"MediaPipe Verarbeitung fehlgeschlagen: {e}")

        regions: Dict[str, np.ndarray] = {}
        registry = {
            "shirt": self._get_torso_region,
            "pants": self._get_legs_region,
            "dress": self._get_dress_region,
            "skirt": self._get_skirt_region,
            "jacket": self._get_jacket_region,
            "shoes": self._get_feet_region,
            "hat": self._get_head_region,
            "all": self._get_all_clothing_regions,
        }

        if clothing_type in registry:
            regions[clothing_type] = registry[clothing_type](img_rgb, pose_landmarks)
        else:
            for k, fn in registry.items():
                if k == "all":
                    continue
                regions[k] = fn(img_rgb, pose_landmarks)

        # Stelle sicher: dtype=uint8, Werte {0,255}
        for k, m in list(regions.items()):
            regions[k] = self._sanitize_mask(m)
        return regions

    def refine_mask_with_sam(self, image: np.ndarray, rough_mask: np.ndarray) -> np.ndarray:
        """Verfeinere Maske mit SAM (falls verfügbar), sonst gib rough_mask zurück."""
        if self.sam_predictor is None or SamPredictor is None:
            return self._sanitize_mask(rough_mask)

        try:
            img_rgb = self._ensure_rgb(image)
            self.sam_predictor.set_image(img_rgb)
            mask_points = np.argwhere(rough_mask > 128)
            if len(mask_points) == 0:
                return self._sanitize_mask(rough_mask)

            # Max 50 Punkte sampeln
            k = min(50, len(mask_points))
            idx = np.random.choice(len(mask_points), k, replace=False)
            pts = mask_points[idx]
            input_points = pts[:, [1, 0]]  # (y,x) -> (x,y)
            input_labels = np.ones(len(input_points), dtype=np.int32)

            masks, scores, _ = self.sam_predictor.predict(  # type: ignore
                point_coords=input_points,
                point_labels=input_labels,
                mask_input=None,
                multimask_output=False
            )
            if masks is not None and len(masks) > 0:
                return self._sanitize_mask((masks[0] * 255).astype(np.uint8))
        except Exception as e:
            logging.debug(f"SAM-Verfeinerung fehlgeschlagen: {e}")

        return self._sanitize_mask(rough_mask)

    # --------------------- Region Builder (intern) ---------------------

    def _get_torso_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if pose_landmarks:
            ids = [11, 12, 13, 14, 23, 24]  # Schultern, Ellbogen, Hüften
            pts = []
            for i in ids:
                lm = pose_landmarks[i]
                if lm.visibility > 0.5:
                    pts.append((int(lm.x * w), int(lm.y * h)))
            if len(pts) >= 4 and cv2 is not None:
                cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)

        if mask.sum() == 0:
            # Fallback: obere Körperhälfte Mitte
            mask[h // 4:h // 2, w // 4:3 * w // 4] = 255
        return mask

    def _get_legs_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if pose_landmarks:
            ids = [23, 24, 25, 26, 27, 28]  # Hüften, Knie, Knöchel
            pts = []
            for i in ids:
                lm = pose_landmarks[i]
                if lm.visibility > 0.5:
                    pts.append((int(lm.x * w), int(lm.y * h)))
            if len(pts) >= 4 and cv2 is not None:
                cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)

        if mask.sum() == 0:
            mask[h // 2:3 * h // 4, w // 4:3 * w // 4] = 255
        return mask

    def _get_dress_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        return cv2.bitwise_or(
            self._get_torso_region(image, pose_landmarks),
            self._get_legs_region(image, pose_landmarks),
        ) if cv2 is not None else self._get_torso_region(image, pose_landmarks)

    def _get_skirt_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if pose_landmarks:
            lhip, rhip, lknee, rknee = pose_landmarks[23], pose_landmarks[24], pose_landmarks[25], pose_landmarks[26]
            if all(getattr(lm, "visibility", 0) > 0.5 for lm in [lhip, rhip, lknee, rknee]) and cv2 is not None:
                pts = [
                    (int(lhip.x * w), int(lhip.y * h)),
                    (int(rhip.x * w), int(rhip.y * h)),
                    (int(rknee.x * w), int(rknee.y * h)),
                    (int(lknee.x * w), int(lknee.y * h)),
                ]
                cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
        return mask

    def _get_jacket_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        torso = self._get_torso_region(image, pose_landmarks)
        if cv2 is None:
            return torso
        kernel = np.ones((20, 20), np.uint8)
        return cv2.dilate(torso, kernel, iterations=1)

    def _get_feet_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if pose_landmarks and cv2 is not None:
            for idx in (27, 28):  # Knöchel
                lm = pose_landmarks[idx]
                if lm.visibility > 0.5:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.rectangle(mask, (x - 40, y), (x + 40, min(h, y + 60)), 255, -1)
        return mask

    def _get_head_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if pose_landmarks and cv2 is not None:
            nose = pose_landmarks[0]
            if nose.visibility > 0.5:
                x, y = int(nose.x * w), int(nose.y * h)
                cv2.rectangle(mask, (x - 80, max(0, y - 120)), (x + 80, y - 20), 255, -1)
        return mask

    def _get_all_clothing_regions(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        if cv2 is None:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        all_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for fn in (self._get_torso_region, self._get_legs_region, self._get_feet_region):
            m = fn(image, pose_landmarks)
            all_mask = cv2.bitwise_or(all_mask, m)
        return all_mask

    # --------------------- Utils ---------------------

    @staticmethod
    def _ensure_rgb(image: np.ndarray) -> np.ndarray:
        """Sorge für RGB (MediaPipe erwartet RGB)."""
        if image.ndim == 3 and image.shape[2] == 3 and cv2 is not None:
            # Heuristik: viele Bilder kommen als BGR von cv2; wir konvertieren sicherheitshalber
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def _sanitize_mask(mask: np.ndarray) -> np.ndarray:
        m = mask.astype(np.uint8, copy=False)
        # Binärisieren
        m = np.where(m > 128, 255, 0).astype(np.uint8)
        return m


# ======================================================================
#                         Editor / Pipelines
# ======================================================================

class ClothingEditor:
    """Hauptklasse für Kleidungsbearbeitung (Entfernen/Ändern)."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.segmentator = ClothingSegmentator(base_dir)
        self.inpainting_pipeline = None
        self.controlnet_pipeline = None
        self._setup_pipelines()

    def _setup_pipelines(self):
        if not DIFFUSERS_AVAILABLE or AutoPipelineForInpainting is None:
            logging.error("Diffusers nicht verfügbar – AI-Funktionen deaktiviert.")
            return

        # Device & dtype
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        dtype = torch.float16 if (torch is not None and device == "cuda") else (torch.float32 if torch else None)

        # Inpainting (SDXL-Inpaint Ordner erwartet)
        model_path = self.base_dir / "models" / "image" / "stable-diffusion-xl-inpaint"
        if model_path.exists():
            try:
                pipe = AutoPipelineForInpainting.from_pretrained(str(model_path), torch_dtype=dtype)
                pipe.to(device)
                # Creative freedom: Safety Checker deaktivieren, falls vorhanden
                if hasattr(pipe, "safety_checker"):
                    pipe.safety_checker = None
                self.inpainting_pipeline = pipe
                logging.info(f"Inpainting-Pipeline geladen: {model_path}")
            except Exception as e:
                logging.error(f"Inpainting-Pipeline konnte nicht geladen werden: {e}")

        # ControlNet (OpenPose) optional
        controlnet_path = self.base_dir / "models" / "controlnet" / "openpose"
        if controlnet_path.exists() and ControlNetModel is not None and StableDiffusionControlNetInpaintPipeline is not None:
            if model_path.exists():
                try:
                    cn = ControlNetModel.from_pretrained(str(controlnet_path), torch_dtype=dtype)
                    cpipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                        str(model_path), controlnet=cn, torch_dtype=dtype
                    )
                    cpipe.to(device)
                    if hasattr(cpipe, "safety_checker"):
                        cpipe.safety_checker = None
                    self.controlnet_pipeline = cpipe
                    logging.info(f"ControlNet-Inpaint-Pipeline geladen: {controlnet_path}")
                except Exception as e:
                    logging.warning(f"ControlNet konnte nicht initialisiert werden: {e}")

    # ------------------------- Public Ops -------------------------

    def remove_clothing(self, image_path: str, clothing_type: str = "all",
                        preserve_anatomy: bool = True) -> Dict[str, Any]:
        """
        Intelligente Kleidungsentfernung.
        Rückgabe: { success, results: {ctype: {mask_path, output_path, confidence}}, metadata, errors? }
        """
        if Image is None:
            return {"success": False, "error": "Pillow (PIL) nicht verfügbar."}
        try:
            src = (self.base_dir / image_path).resolve()
            image = Image.open(src).convert("RGB")
            image_array = np.array(image)

            clothing_masks = self.segmentator.detect_clothing_regions(image_array, clothing_type)
            if not clothing_masks:
                return {"success": False, "error": "Keine Kleidung erkannt (oder Segmentierung nicht möglich)."}

            results: Dict[str, Any] = {}
            errors: Dict[str, str] = {}

            out_dir = (self.base_dir / "outputs")
            out_dir.mkdir(parents=True, exist_ok=True)

            for ctype, mask in clothing_masks.items():
                if mask.sum() == 0:
                    continue
                refined_mask = self.segmentator.refine_mask_with_sam(image_array, mask)
                mask_img = Image.fromarray(refined_mask).convert("L")

                if self.inpainting_pipeline is None:
                    # Fallback: kein Inpainting möglich – nur Maske speichern
                    mask_path = f"outputs/mask_{ctype}_{int(time.time())}.png"
                    mask_img.save(self.base_dir / mask_path)
                    results[ctype] = {
                        "mask_path": mask_path,
                        "output_path": None,
                        "confidence": float(self._calculate_confidence(refined_mask)),
                        "note": "Inpainting deaktiviert (Diffusers/Modelle fehlen)."
                    }
                    continue

                # Anatomie-Prompt oder Standard
                result_img = (
                    self._anatomical_inpainting(image, mask_img, ctype)
                    if preserve_anatomy else
                    self._standard_inpainting(image, mask_img)
                )

                # Speichern
                ts = int(time.time())
                out_img_path = f"outputs/clothing_removed_{ctype}_{ts}.png"
                mask_path = f"outputs/mask_{ctype}_{ts}.png"
                (self.base_dir / out_img_path).parent.mkdir(parents=True, exist_ok=True)
                result_img.save(self.base_dir / out_img_path)
                mask_img.save(self.base_dir / mask_path)

                results[ctype] = {
                    "mask_path": mask_path,
                    "output_path": out_img_path,
                    "confidence": float(self._calculate_confidence(refined_mask)),
                }

            return {
                "success": True,
                "results": results,
                "errors": errors or None,
                "metadata": {
                    "clothing_type": clothing_type,
                    "preserve_anatomy": bool(preserve_anatomy),
                    "original_size": image.size,
                    "source": str(src)
                }
            }
        except Exception as e:
            logging.exception("Clothing removal failed")
            return {"success": False, "error": str(e)}

    def change_clothing(self, image_path: str, new_clothing_prompt: str,
                        clothing_type: str, style: str = "realistic") -> Dict[str, Any]:
        """
        Kleidung austauschen/ändern.
        Rückgabe: { success, output_path, mask_path, metadata }
        """
        if Image is None:
            return {"success": False, "error": "Pillow (PIL) nicht verfügbar."}
        try:
            src = (self.base_dir / image_path).resolve()
            image = Image.open(src).convert("RGB")
            image_array = np.array(image)

            masks = self.segmentator.detect_clothing_regions(image_array, clothing_type)
            if clothing_type not in masks or masks[clothing_type].sum() == 0:
                return {"success": False, "error": f"Keine {clothing_type}-Region erkannt."}

            refined_mask = self.segmentator.refine_mask_with_sam(image_array, masks[clothing_type])
            mask_img = Image.fromarray(refined_mask).convert("L")

            enhanced_prompt = self._enhance_clothing_prompt(new_clothing_prompt, clothing_type, style)
            negative_prompt = self._get_clothing_negative_prompt(style)

            if self.controlnet_pipeline is not None:
                result_img = self._controlnet_clothing_generation(image, mask_img, enhanced_prompt, negative_prompt)
            elif self.inpainting_pipeline is not None:
                result_img = self._generate_new_clothing(image, mask_img, enhanced_prompt, negative_prompt)
            else:
                return {"success": False, "error": "Keine Inpainting/ControlNet-Pipeline verfügbar."}

            ts = int(time.time())
            out_img_path = f"outputs/clothing_changed_{clothing_type}_{ts}.png"
            mask_path = f"outputs/mask_{clothing_type}_{ts}.png"
            (self.base_dir / out_img_path).parent.mkdir(parents=True, exist_ok=True)
            result_img.save(self.base_dir / out_img_path)
            mask_img.save(self.base_dir / mask_path)

            return {
                "success": True,
                "output_path": out_img_path,
                "mask_path": mask_path,
                "metadata": {
                    "prompt": enhanced_prompt,
                    "negative_prompt": negative_prompt,
                    "clothing_type": clothing_type,
                    "style": style,
                    "source": str(src)
                }
            }
        except Exception as e:
            logging.exception("Clothing change failed")
            return {"success": False, "error": str(e)}

    def change_clothing_material(self, image_path: str, clothing_type: str,
                                 material: str = "leather") -> Dict[str, Any]:
        material_prompts = {
            "leather": "high quality black leather texture, realistic leather material, glossy",
            "silk": "smooth silk fabric, elegant silk texture, flowing silk material",
            "denim": "blue denim texture, jeans material, cotton denim fabric",
            "cotton": "soft cotton fabric, natural cotton texture, comfortable material",
            "lace": "delicate lace pattern, intricate lace texture, elegant lace fabric",
            "velvet": "luxurious velvet texture, soft velvet material, rich velvet fabric",
            "satin": "glossy satin fabric, smooth satin texture, elegant satin material",
        }
        mat = material if material in material_prompts else "cotton"
        prompt = f"same clothing item but made of {material_prompts[mat]}, keep the same cut and style"
        return self.change_clothing(image_path, prompt, clothing_type, "realistic")

    def adjust_clothing_fit(self, image_path: str, clothing_type: str,
                            fit_adjustment: str = "tighter") -> Dict[str, Any]:
        fit_prompts = {
            "tighter": "form-fitting, tight-fitting, body-hugging",
            "looser": "loose-fitting, baggy, oversized, relaxed fit",
            "fitted": "well-fitted, tailored, perfect fit",
            "baggy": "very loose, oversized, baggy style",
        }
        descr = fit_prompts.get(fit_adjustment, "well-fitted")
        prompt = f"same {clothing_type} but {descr}, keep same color and material"
        return self.change_clothing(image_path, prompt, clothing_type, "realistic")

    # --------------------- Intern: Inpainting/Gen ---------------------

    def _anatomical_inpainting(self, image: Image.Image, mask: Image.Image, clothing_type: str) -> Image.Image:
        if self.inpainting_pipeline is None:
            return image
        anatomy_prompts = {
            "shirt": "natural skin texture, realistic human torso anatomy, proper chest and shoulder definition",
            "pants": "realistic leg anatomy, natural skin tone, proper hip and leg proportions",
            "dress": "natural body silhouette, realistic skin texture, proper torso and leg anatomy",
            "jacket": "realistic upper body anatomy, natural skin tone, proper shoulder definition",
            "skirt": "realistic leg anatomy, natural hip area, proper thigh definition",
        }
        prompt = anatomy_prompts.get(clothing_type, "realistic human anatomy, natural skin texture")
        negative_prompt = "clothing, fabric, artificial, plastic, unrealistic anatomy, deformed"
        result = self.inpainting_pipeline(
            image=image,
            mask_image=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            strength=1.0,
        )
        return result.images[0]

    def _standard_inpainting(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        if self.inpainting_pipeline is None:
            return image
        result = self.inpainting_pipeline(
            image=image,
            mask_image=mask,
            prompt="natural background, seamless inpainting, realistic",
            negative_prompt="artifacts, blurry, distorted",
            num_inference_steps=30,
            guidance_scale=6.0,
        )
        return result.images[0]

    def _generate_new_clothing(self, image: Image.Image, mask: Image.Image,
                               prompt: str, negative_prompt: str) -> Image.Image:
        if self.inpainting_pipeline is None:
            return image
        result = self.inpainting_pipeline(
            image=image,
            mask_image=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=40,
            guidance_scale=8.0,
            strength=1.0,
        )
        return result.images[0]

    def _controlnet_clothing_generation(self, image: Image.Image, mask: Image.Image,
                                        prompt: str, negative_prompt: str) -> Image.Image:
        if self.controlnet_pipeline is None:
            return self._generate_new_clothing(image, mask, prompt, negative_prompt)
        pose_image = self._extract_pose_guide(np.array(image))
        result = self.controlnet_pipeline(
            image=image,
            mask_image=mask,
            control_image=pose_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=40,
            guidance_scale=8.0,
            controlnet_conditioning_scale=1.0,
        )
        return result.images[0]

    def _extract_pose_guide(self, image: np.ndarray) -> Image.Image:
        if cv2 is None or Image is None:
            return Image.fromarray(image)
        if self.segmentator.mediapipe_pose is not None:
            try:
                img_rgb = self.segmentator._ensure_rgb(image)
                results = self.segmentator.mediapipe_pose.process(img_rgb)
                if getattr(results, "pose_landmarks", None):
                    pose_img = np.zeros(image.shape, dtype=np.uint8)
                    # einfache Verbindungen
                    connections = [
                        (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
                        (11, 23), (12, 24), (23, 24),
                        (23, 25), (24, 26), (25, 27), (26, 28)
                    ]
                    h, w = image.shape[:2]
                    lm = results.pose_landmarks.landmark
                    for a, b in connections:
                        p1, p2 = lm[a], lm[b]
                        if p1.visibility > 0.5 and p2.visibility > 0.5:
                            x1, y1 = int(p1.x * w), int(p1.y * h)
                            x2, y2 = int(p2.x * w), int(p2.y * h)
                            cv2.line(pose_img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                    return Image.fromarray(pose_img)
            except Exception as e:
                logging.debug(f"Pose-Guide Fallback (Fehler): {e}")
        return Image.fromarray(image)

    # --------------------- Prompt Helpers ---------------------

    @staticmethod
    def _enhance_clothing_prompt(prompt: str, clothing_type: str, style: str) -> str:
        style_add = {
            "realistic": "photorealistic, high quality, detailed fabric texture",
            "artistic": "artistic, stylized, creative design",
            "casual": "casual wear, everyday clothing, comfortable",
            "formal": "formal attire, elegant, sophisticated",
            "vintage": "vintage style, retro fashion, classic design",
        }
        type_add = {
            "shirt": "well-fitted shirt, proper collar and sleeves",
            "pants": "well-tailored pants, proper leg fit",
            "dress": "elegant dress, flowing fabric",
            "jacket": "stylish jacket, proper shoulders",
            "skirt": "fashionable skirt, appropriate length",
        }
        parts = [prompt.strip()]
        if clothing_type in type_add:
            parts.append(type_add[clothing_type])
        parts.append(style_add.get(style, "high quality"))
        return ", ".join([p for p in parts if p])

    @staticmethod
    def _get_clothing_negative_prompt(style: str) -> str:
        base_neg = "blurry, low quality, distorted, deformed, bad anatomy, wrong proportions"
        style_neg = {
            "realistic": "cartoon, anime, artistic, stylized, unrealistic",
            "artistic": "photorealistic, boring, plain",
            "casual": "formal, elegant, fancy",
            "formal": "casual, sloppy, messy",
            "vintage": "modern, contemporary, futuristic",
        }
        return f"{base_neg}, {style_neg.get(style, '')}"

    @staticmethod
    def _calculate_confidence(mask: np.ndarray) -> float:
        # einfache Heuristik: Flächenanteil + Kreisförmigkeit
        if cv2 is None or mask is None or mask.size == 0:
            return 0.1
        total = mask.shape[0] * mask.shape[1]
        on = float((mask > 128).sum())
        if on == 0:
            return 0.0
        ratio = on / total
        if ratio < 0.01 or ratio > 0.8:
            base = 0.3
        else:
            base = 0.6

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # OpenCV 3 vs 4: Rückgabeformate unterscheiden sich
        cnts = contours[0] if len(contours) == 2 else contours[1]
        if not cnts:
            return 0.3
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        per = cv2.arcLength(c, True)
        if per <= 0:
            return 0.3
        circularity = 4.0 * np.pi * area / (per * per)
        conf = min(1.0, (base + circularity) / 1.6)
        return float(max(0.1, conf))


# ======================================================================
#                          API Wrapper
# ======================================================================

class ClothingEditorAPI:
    """Dünner Wrapper für FastAPI-Handler."""

    def __init__(self, base_dir: Path):
        self.editor = ClothingEditor(base_dir)

    async def remove_clothing_api(self, image_path: str, clothing_type: str = "all",
                                  preserve_anatomy: bool = True) -> Dict[str, Any]:
        return self.editor.remove_clothing(image_path, clothing_type, preserve_anatomy)

    async def change_clothing_api(self, image_path: str, new_clothing_prompt: str,
                                  clothing_type: str, style: str = "realistic") -> Dict[str, Any]:
        return self.editor.change_clothing(image_path, new_clothing_prompt, clothing_type, style)


# ======================================================================
#                      Quick helpers (sync)
# ======================================================================

def quick_remove_clothing(base_dir: Path, image_path: str, clothing_type: str = "shirt") -> str:
    editor = ClothingEditor(base_dir)
    res = editor.remove_clothing(image_path, clothing_type, preserve_anatomy=True)
    if res.get("success"):
        # Nimm bevorzugt die gewünschte clothing_type-Ausgabe
        if "results" in res and clothing_type in res["results"]:
            out = res["results"][clothing_type].get("output_path")
            if out:
                return out
        # oder irgendein Ergebnis
        for data in res.get("results", {}).values():
            if data.get("output_path"):
                return data["output_path"]
    raise RuntimeError(f"Failed to remove {clothing_type}: {res.get('error')}")

def quick_change_clothing(base_dir: Path, image_path: str, new_prompt: str, clothing_type: str = "shirt") -> str:
    editor = ClothingEditor(base_dir)
    res = editor.change_clothing(image_path, new_prompt, clothing_type, "realistic")
    if res.get("success") and res.get("output_path"):
        return res["output_path"]  # type: ignore
    raise RuntimeError(f"Failed to change {clothing_type}: {res.get('error', 'Unknown error')}")
