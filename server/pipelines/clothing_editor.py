# server/pipelines/clothing_editor.py
"""
Advanced Clothing Editor Pipeline
Klamotten entfernen, ändern, anpassen mit KI
"""

from __future__ import annotations
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from typing import Optional, Dict, List, Tuple
import logging

try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logging.warning("SAM not available - install segment-anything")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available")

try:
    from diffusers import (
        AutoPipelineForInpainting,
        ControlNetModel,
        StableDiffusionControlNetInpaintPipeline
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

class ClothingEditor:
    """Hauptklasse für Kleidungsbearbeitung"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.segmentator = ClothingSegmentator(base_dir)
        self.inpainting_pipeline = None
        self.controlnet_pipeline = None
        self.setup_pipelines()
    
    def setup_pipelines(self):
        """Initialisiere AI-Pipelines"""
        if not DIFFUSERS_AVAILABLE:
            logging.error("Diffusers not available - AI features disabled")
            return
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Inpainting Pipeline für Kleidungsentfernung
        model_path = self.base_dir / "models" / "image" / "stable-diffusion-xl-inpaint"
        if model_path.exists():
            self.inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
                str(model_path), torch_dtype=dtype
            ).to(device)
            # Keine Safety Checker für kreative Freiheit
            self.inpainting_pipeline.safety_checker = None
        
        # ControlNet für präzise Kleidungsplatzierung
        controlnet_path = self.base_dir / "models" / "controlnet" / "openpose"
        if controlnet_path.exists():
            controlnet = ControlNetModel.from_pretrained(str(controlnet_path), torch_dtype=dtype)
            self.controlnet_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                str(model_path), controlnet=controlnet, torch_dtype=dtype
            ).to(device)
    
    def remove_clothing(self, image_path: str, clothing_type: str = "all", 
                       preserve_anatomy: bool = True) -> Dict[str, any]:
        """
        Intelligente Kleidungsentfernung
        
        Args:
            image_path: Pfad zum Eingangsbild
            clothing_type: Art der Kleidung ("shirt", "pants", "dress", etc.)
            preserve_anatomy: Ob anatomische Details erhalten werden sollen
        
        Returns:
            Dict mit Ergebnis und Metadaten
        """
        try:
            # Bild laden
            image = Image.open(self.base_dir / image_path).convert("RGB")
            image_array = np.array(image)
            
            # Kleidung segmentieren
            clothing_masks = self.segmentator.detect_clothing_regions(image_array, clothing_type)
            
            results = {}
            
            for cloth_type, mask in clothing_masks.items():
                if mask.sum() == 0:  # Keine Kleidung gefunden
                    continue
                
                # Maske verfeinern
                refined_mask = self.segmentator.refine_mask_with_sam(image_array, mask)
                mask_image = Image.fromarray(refined_mask).convert("L")
                
                if preserve_anatomy:
                    # Anatomisch korrekte Inpainting
                    result_image = self._anatomical_inpainting(image, mask_image, cloth_type)
                else:
                    # Standard Inpainting
                    result_image = self._standard_inpainting(image, mask_image)
                
                results[cloth_type] = {
                    "image": result_image,
                    "mask": mask_image,
                    "confidence": self._calculate_confidence(refined_mask)
                }
            
            return {
                "success": True,
                "results": results,
                "metadata": {
                    "clothing_type": clothing_type,
                    "preserve_anatomy": preserve_anatomy,
                    "original_size": image.size
                }
            }
            
        except Exception as e:
            logging.error(f"Clothing removal failed: {e}")
            return {"success": False, "error": str(e)}
    
    def change_clothing(self, image_path: str, new_clothing_prompt: str, 
                       clothing_type: str, style: str = "realistic") -> Dict[str, any]:
        """
        Kleidung austauschen/ändern
        
        Args:
            image_path: Pfad zum Eingangsbild
            new_clothing_prompt: Beschreibung der neuen Kleidung
            clothing_type: Art der Kleidung die geändert werden soll
            style: Stil der neuen Kleidung ("realistic", "artistic", etc.)
        
        Returns:
            Dict mit Ergebnis
        """
        try:
            # Bild laden
            image = Image.open(self.base_dir / image_path).convert("RGB")
            image_array = np.array(image)
            
            # Existierende Kleidung segmentieren
            clothing_masks = self.segmentator.detect_clothing_regions(image_array, clothing_type)
            
            if not clothing_masks or clothing_type not in clothing_masks:
                return {"success": False, "error": f"No {clothing_type} detected in image"}
            
            mask = clothing_masks[clothing_type]
            refined_mask = self.segmentator.refine_mask_with_sam(image_array, mask)
            mask_image = Image.fromarray(refined_mask).convert("L")
            
            # Neue Kleidung generieren
            enhanced_prompt = self._enhance_clothing_prompt(new_clothing_prompt, clothing_type, style)
            negative_prompt = self._get_clothing_negative_prompt(style)
            
            if self.controlnet_pipeline:
                # Mit ControlNet für bessere Körperanpassung
                result_image = self._controlnet_clothing_generation(
                    image, mask_image, enhanced_prompt, negative_prompt
                )
            else:
                # Standard Inpainting
                result_image = self._generate_new_clothing(
                    image, mask_image, enhanced_prompt, negative_prompt
                )
            
            return {
                "success": True,
                "result_image": result_image,
                "mask": mask_image,
                "metadata": {
                    "prompt": enhanced_prompt,
                    "clothing_type": clothing_type,
                    "style": style
                }
            }
            
        except Exception as e:
            logging.error(f"Clothing change failed: {e}")
            return {"success": False, "error": str(e)}
    
    def change_clothing_material(self, image_path: str, clothing_type: str, 
                               material: str = "leather") -> Dict[str, any]:
        """
        Nur das Material der Kleidung ändern (Farbe/Textur)
        
        Args:
            image_path: Pfad zum Eingangsbild
            clothing_type: Art der Kleidung
            material: Neues Material ("leather", "silk", "denim", "cotton", etc.)
        
        Returns:
            Dict mit Ergebnis
        """
        material_prompts = {
            "leather": "high quality black leather texture, realistic leather material, glossy",
            "silk": "smooth silk fabric, elegant silk texture, flowing silk material", 
            "denim": "blue denim texture, jeans material, cotton denim fabric",
            "cotton": "soft cotton fabric, natural cotton texture, comfortable material",
            "lace": "delicate lace pattern, intricate lace texture, elegant lace fabric",
            "velvet": "luxurious velvet texture, soft velvet material, rich velvet fabric",
            "satin": "glossy satin fabric, smooth satin texture, elegant satin material"
        }
        
        if material not in material_prompts:
            material = "cotton"
        
        material_prompt = f"same clothing item but made of {material_prompts[material]}, keep the same cut and style"
        
        return self.change_clothing(image_path, material_prompt, clothing_type, "realistic")
    
    def adjust_clothing_fit(self, image_path: str, clothing_type: str, 
                           fit_adjustment: str = "tighter") -> Dict[str, any]:
        """
        Passform der Kleidung anpassen
        
        Args:
            image_path: Pfad zum Eingangsbild
            clothing_type: Art der Kleidung
            fit_adjustment: Art der Anpassung ("tighter", "looser", "fitted", "baggy")
        
        Returns:
            Dict mit Ergebnis
        """
        fit_prompts = {
            "tighter": "form-fitting, tight-fitting, body-hugging",
            "looser": "loose-fitting, baggy, oversized, relaxed fit",
            "fitted": "well-fitted, tailored, perfect fit",
            "baggy": "very loose, oversized, baggy style"
        }
        
        fit_prompt = f"same {clothing_type} but {fit_prompts.get(fit_adjustment, 'well-fitted')}, keep same color and material"
        
        return self.change_clothing(image_path, fit_prompt, clothing_type, "realistic")
    
    def _anatomical_inpainting(self, image: Image.Image, mask: Image.Image, 
                              clothing_type: str) -> Image.Image:
        """Anatomisch korrektes Inpainting"""
        if not self.inpainting_pipeline:
            return image
        
        # Anatomie-spezifische Prompts
        anatomy_prompts = {
            "shirt": "natural skin texture, realistic human torso anatomy, proper chest and shoulder definition",
            "pants": "realistic leg anatomy, natural skin tone, proper hip and leg proportions",
            "dress": "natural body silhouette, realistic skin texture, proper torso and leg anatomy",
            "jacket": "realistic upper body anatomy, natural skin tone, proper shoulder definition",
            "skirt": "realistic leg anatomy, natural hip area, proper thigh definition"
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
            strength=1.0
        )
        
        return result.images[0]
    
    def _standard_inpainting(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Standard Inpainting ohne spezielle Anatomie"""
        if not self.inpainting_pipeline:
            return image
        
        prompt = "natural background, seamless inpainting, realistic"
        negative_prompt = "artifacts, blurry, distorted"
        
        result = self.inpainting_pipeline(
            image=image,
            mask_image=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=6.0
        )
        
        return result.images[0]
    
    def _generate_new_clothing(self, image: Image.Image, mask: Image.Image, 
                              prompt: str, negative_prompt: str) -> Image.Image:
        """Neue Kleidung generieren"""
        if not self.inpainting_pipeline:
            return image
        
        result = self.inpainting_pipeline(
            image=image,
            mask_image=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=40,
            guidance_scale=8.0,
            strength=1.0
        )
        
        return result.images[0]
    
    def _controlnet_clothing_generation(self, image: Image.Image, mask: Image.Image,
                                      prompt: str, negative_prompt: str) -> Image.Image:
        """Kleidung mit ControlNet für bessere Körperanpassung"""
        if not self.controlnet_pipeline:
            return self._generate_new_clothing(image, mask, prompt, negative_prompt)
        
        # Pose-Guide aus Original-Bild extrahieren
        pose_image = self._extract_pose_guide(np.array(image))
        
        result = self.controlnet_pipeline(
            image=image,
            mask_image=mask,
            control_image=pose_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=40,
            guidance_scale=8.0,
            controlnet_conditioning_scale=1.0
        )
        
        return result.images[0]
    
    def _extract_pose_guide(self, image: np.ndarray) -> Image.Image:
        """Pose-Guide für ControlNet extrahieren"""
        # OpenPose-Style Skelett extrahieren
        if self.segmentator.mediapipe_pose:
            results = self.segmentator.mediapipe_pose.process(image)
            if results.pose_landmarks:
                pose_image = np.zeros(image.shape, dtype=np.uint8)
                
                # Verbindungen zwischen Körperteilen zeichnen
                connections = [
                    # Torso
                    (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
                    (11, 23), (12, 24), (23, 24),
                    # Beine
                    (23, 25), (24, 26), (25, 27), (26, 28)
                ]
                
                h, w = image.shape[:2]
                landmarks = results.pose_landmarks.landmark
                
                for connection in connections:
                    pt1 = landmarks[connection[0]]
                    pt2 = landmarks[connection[1]]
                    
                    if pt1.visibility > 0.5 and pt2.visibility > 0.5:
                        x1, y1 = int(pt1.x * w), int(pt1.y * h)
                        x2, y2 = int(pt2.x * w), int(pt2.y * h)
                        cv2.line(pose_image, (x1, y1), (x2, y2), (255, 255, 255), 3)
                
                return Image.fromarray(pose_image)
        
        # Fallback: Original-Bild als Pose-Guide verwenden
        return Image.fromarray(image)
    
    def _enhance_clothing_prompt(self, prompt: str, clothing_type: str, style: str) -> str:
        """Prompt für Kleidungsgenerierung optimieren"""
        style_additions = {
            "realistic": "photorealistic, high quality, detailed fabric texture",
            "artistic": "artistic, stylized, creative design",
            "casual": "casual wear, everyday clothing, comfortable",
            "formal": "formal attire, elegant, sophisticated",
            "vintage": "vintage style, retro fashion, classic design"
        }
        
        type_additions = {
            "shirt": "well-fitted shirt, proper collar and sleeves",
            "pants": "well-tailored pants, proper leg fit",
            "dress": "elegant dress, flowing fabric",
            "jacket": "stylish jacket, proper shoulders",
            "skirt": "fashionable skirt, appropriate length"
        }
        
        enhanced = f"{prompt}, {type_additions.get(clothing_type, '')}, {style_additions.get(style, 'high quality')}"
        return enhanced.strip(", ")
    
    def _get_clothing_negative_prompt(self, style: str) -> str:
        """Negative Prompts für bessere Kleidungsgenerierung"""
        base_negative = "blurry, low quality, distorted, deformed, bad anatomy, wrong proportions"
        
        style_negatives = {
            "realistic": "cartoon, anime, artistic, stylized, unrealistic",
            "artistic": "photorealistic, boring, plain",
            "casual": "formal, elegant, fancy",
            "formal": "casual, sloppy, messy",
            "vintage": "modern, contemporary, futuristic"
        }
        
        return f"{base_negative}, {style_negatives.get(style, '')}"
    
    def _calculate_confidence(self, mask: np.ndarray) -> float:
        """Berechne Vertrauen in die Segmentierung"""
        # Einfache Metrik basierend auf Maskengröße und -form
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_pixels = np.sum(mask > 128)
        
        if mask_pixels == 0:
            return 0.0
        
        # Verhältnis der Maskenpixel
        ratio = mask_pixels / total_pixels
        
        # Penalize zu kleine oder zu große Masken
        if ratio < 0.01 or ratio > 0.8:
            return 0.3
        
        # Berechne Kompaktheit der Maske
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.3
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.3
        
        # Kreisförmigkeit als Qualitätsmaß
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Kombiniere Faktoren für finales Vertrauen
        confidence = min(1.0, (ratio * 2 + circularity) / 2)
        
        return max(0.1, confidence)


# API Integration für FastAPI
class ClothingEditorAPI:
    """API-Wrapper für den Clothing Editor"""
    
    def __init__(self, base_dir: Path):
        self.editor = ClothingEditor(base_dir)
    
    async def remove_clothing_api(self, image_path: str, clothing_type: str = "all",
                                 preserve_anatomy: bool = True) -> Dict[str, any]:
        """API-Endpoint für Kleidungsentfernung"""
        try:
            result = self.editor.remove_clothing(image_path, clothing_type, preserve_anatomy)
            
            if result["success"]:
                # Ergebnisse speichern
                output_paths = {}
                for cloth_type, data in result["results"].items():
                    output_path = f"outputs/clothing_removed_{cloth_type}_{int(time.time())}.png"
                    full_output_path = self.editor.base_dir / output_path
                    data["image"].save(full_output_path)
                    output_paths[cloth_type] = output_path
                
                result["output_paths"] = output_paths
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def change_clothing_api(self, image_path: str, new_clothing_prompt: str,
                                 clothing_type: str, style: str = "realistic") -> Dict[str, any]:
        """API-Endpoint für Kleidungsänderung"""
        try:
            result = self.editor.change_clothing(image_path, new_clothing_prompt, clothing_type, style)
            
            if result["success"]:
                # Ergebnis speichern
                output_path = f"outputs/clothing_changed_{clothing_type}_{int(time.time())}.png"
                full_output_path = self.editor.base_dir / output_path
                result["result_image"].save(full_output_path)
                result["output_path"] = output_path
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Hilfsfunktionen für schnelle Integration
def quick_remove_clothing(base_dir: Path, image_path: str, clothing_type: str = "shirt") -> str:
    """Schnelle Kleidungsentfernung - gibt Output-Pfad zurück"""
    editor = ClothingEditor(base_dir)
    result = editor.remove_clothing(image_path, clothing_type, preserve_anatomy=True)
    
    if result["success"] and clothing_type in result["results"]:
        output_path = f"outputs/quick_remove_{int(time.time())}.png"
        full_path = base_dir / output_path
        result["results"][clothing_type]["image"].save(full_path)
        return output_path
    
    raise Exception(f"Failed to remove {clothing_type}")


def quick_change_clothing(base_dir: Path, image_path: str, new_prompt: str, clothing_type: str = "shirt") -> str:
    """Schneller Kleidungswechsel - gibt Output-Pfad zurück"""
    editor = ClothingEditor(base_dir)
    result = editor.change_clothing(image_path, new_prompt, clothing_type, "realistic")
    
    if result["success"]:
        output_path = f"outputs/quick_change_{int(time.time())}.png"
        full_path = base_dir / output_path
        result["result_image"].save(full_path)
        return output_path
    
    raise Exception(f"Failed to change {clothing_type}: {result.get('error', 'Unknown error')}")


import timeothingSegmentator:
    """Intelligente Kleidungssegmentierung"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.sam_predictor = None
        self.mediapipe_pose = None
        self.setup_models()
    
    def setup_models(self):
        """Initialisiere Segmentierungsmodelle"""
        # SAM für präzise Segmentierung
        if SAM_AVAILABLE:
            sam_path = self.base_dir / "models" / "segmentation" / "sam_vit_h.pth"
            if sam_path.exists():
                sam = sam_model_registry["vit_h"](checkpoint=str(sam_path))
                if torch.cuda.is_available():
                    sam.to("cuda")
                self.sam_predictor = SamPredictor(sam)
        
        # MediaPipe für Körperpose
        if MEDIAPIPE_AVAILABLE:
            mp_pose = mp.solutions.pose
            self.mediapipe_pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True
            )
    
    def detect_clothing_regions(self, image: np.ndarray, clothing_type: str = "all") -> Dict[str, np.ndarray]:
        """Erkenne verschiedene Kleidungsbereiche"""
        masks = {}
        h, w = image.shape[:2]
        
        # Körperpose erkennen
        pose_landmarks = None
        if self.mediapipe_pose:
            results = self.mediapipe_pose.process(image)
            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks.landmark
        
        # Definiere Kleidungsbereiche basierend auf Körperpose
        clothing_regions = {
            "shirt": self._get_torso_region,
            "pants": self._get_legs_region, 
            "dress": self._get_dress_region,
            "skirt": self._get_skirt_region,
            "jacket": self._get_jacket_region,
            "shoes": self._get_feet_region,
            "hat": self._get_head_region,
            "all": self._get_all_clothing_regions
        }
        
        if clothing_type in clothing_regions:
            region_func = clothing_regions[clothing_type]
            masks[clothing_type] = region_func(image, pose_landmarks)
        else:
            # Alle Kleidungstypen erkennen
            for ctype, region_func in clothing_regions.items():
                if ctype != "all":
                    masks[ctype] = region_func(image, pose_landmarks)
        
        return masks
    
    def _get_torso_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        """Torso/Shirt Bereich segmentieren"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if pose_landmarks:
            # Schultern, Ellbogen, Hüften für Shirt-Region
            landmarks = {
                'left_shoulder': pose_landmarks[11],
                'right_shoulder': pose_landmarks[12], 
                'left_elbow': pose_landmarks[13],
                'right_elbow': pose_landmarks[14],
                'left_hip': pose_landmarks[23],
                'right_hip': pose_landmarks[24]
            }
            
            # Polygon für Shirt-Region erstellen
            points = []
            for landmark in landmarks.values():
                if landmark.visibility > 0.5:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append((x, y))
            
            if len(points) >= 4:
                points = np.array(points, np.int32)
                cv2.fillPoly(mask, [points], 255)
        
        # Fallback: obere Körperhälfte
        if mask.sum() == 0:
            mask[h//4:h//2, w//4:3*w//4] = 255
        
        return mask
    
    def _get_legs_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        """Beine/Hosen Bereich"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if pose_landmarks:
            # Hüften, Knie, Knöchel für Hosen-Region
            landmarks = {
                'left_hip': pose_landmarks[23],
                'right_hip': pose_landmarks[24],
                'left_knee': pose_landmarks[25], 
                'right_knee': pose_landmarks[26],
                'left_ankle': pose_landmarks[27],
                'right_ankle': pose_landmarks[28]
            }
            
            points = []
            for landmark in landmarks.values():
                if landmark.visibility > 0.5:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append((x, y))
            
            if len(points) >= 4:
                points = np.array(points, np.int32)
                cv2.fillPoly(mask, [points], 255)
        
        # Fallback: untere Körperhälfte
        if mask.sum() == 0:
            mask[h//2:3*h//4, w//4:3*w//4] = 255
        
        return mask
    
    def _get_dress_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        """Kleid-Region (Kombination aus Torso und Beine)"""
        torso_mask = self._get_torso_region(image, pose_landmarks)
        legs_mask = self._get_legs_region(image, pose_landmarks)
        return cv2.bitwise_or(torso_mask, legs_mask)
    
    def _get_skirt_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        """Rock-Region (oberer Teil der Beine)"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if pose_landmarks:
            # Hüften bis Knie
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
            left_knee = pose_landmarks[25]
            right_knee = pose_landmarks[26]
            
            if all(lm.visibility > 0.5 for lm in [left_hip, right_hip, left_knee, right_knee]):
                points = [
                    (int(left_hip.x * w), int(left_hip.y * h)),
                    (int(right_hip.x * w), int(right_hip.y * h)),
                    (int(right_knee.x * w), int(right_knee.y * h)),
                    (int(left_knee.x * w), int(left_knee.y * h))
                ]
                cv2.fillPoly(mask, [np.array(points)], 255)
        
        return mask
    
    def _get_jacket_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        """Jacken-Region (erweiterte Torso-Region)"""
        torso_mask = self._get_torso_region(image, pose_landmarks)
        # Jacke ist typischerweise größer als normales Shirt
        kernel = np.ones((20, 20), np.uint8)
        return cv2.dilate(torso_mask, kernel, iterations=1)
    
    def _get_feet_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        """Schuh-Region"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if pose_landmarks:
            left_ankle = pose_landmarks[27]
            right_ankle = pose_landmarks[28]
            
            for ankle in [left_ankle, right_ankle]:
                if ankle.visibility > 0.5:
                    x = int(ankle.x * w)
                    y = int(ankle.y * h)
                    # Schuhbereich unter dem Knöchel
                    cv2.rectangle(mask, (x-40, y), (x+40, min(h, y+60)), 255, -1)
        
        return mask
    
    def _get_head_region(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        """Kopf/Hut-Region"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if pose_landmarks:
            nose = pose_landmarks[0]
            if nose.visibility > 0.5:
                x = int(nose.x * w)
                y = int(nose.y * h)
                # Kopfbereich oberhalb der Nase
                cv2.rectangle(mask, (x-80, max(0, y-120)), (x+80, y-20), 255, -1)
        
        return mask
    
    def _get_all_clothing_regions(self, image: np.ndarray, pose_landmarks) -> np.ndarray:
        """Alle Kleidungsbereiche kombiniert"""
        all_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        regions = [
            self._get_torso_region,
            self._get_legs_region,
            self._get_feet_region
        ]
        
        for region_func in regions:
            mask = region_func(image, pose_landmarks)
            all_mask = cv2.bitwise_or(all_mask, mask)
        
        return all_mask
    
    def refine_mask_with_sam(self, image: np.ndarray, rough_mask: np.ndarray) -> np.ndarray:
        """Verfeinere Maske mit SAM für pixelgenaue Segmentierung"""
        if not self.sam_predictor:
            return rough_mask
        
        # SAM mit grober Maske als Eingabe
        self.sam_predictor.set_image(image)
        
        # Positive Punkte aus der groben Maske extrahieren
        mask_points = np.argwhere(rough_mask > 128)
        if len(mask_points) > 50:
            # Sample 50 Punkte für SAM
            indices = np.random.choice(len(mask_points), 50, replace=False)
            sample_points = mask_points[indices]
            # Format: (x, y) für SAM
            input_points = sample_points[:, [1, 0]]  # y,x -> x,y
            input_labels = np.ones(len(input_points))
            
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                mask_input=None,
                multimask_output=False
            )
            
            if len(masks) > 0:
                return (masks[0] * 255).astype(np.uint8)
        
        return rough_mask


class Cl