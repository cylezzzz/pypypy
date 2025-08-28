# server/orchestrator/intelligent_model_selector.py
"""
Intelligenter Model-Orchestrator für AndioMediaStudio
Wählt automatisch die besten Modelle für jede Aufgabe
"""

from __future__ import annotations
from pathlib import Path
import torch
import json
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Verschiedene Task-Typen"""
    CLOTHING_REMOVAL = "clothing_removal"
    CLOTHING_CHANGE = "clothing_change"
    OBJECT_INSERTION = "object_insertion"
    BACKGROUND_REPLACEMENT = "background_replacement"
    STYLE_TRANSFER = "style_transfer"
    UPSCALING = "upscaling"
    FACE_RESTORATION = "face_restoration"
    INPAINTING = "inpainting"
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"

class ContentCategory(Enum):
    """Content-Kategorien für NSFW-Handling"""
    SAFE = "safe"
    SUGGESTIVE = "suggestive" 
    NSFW = "nsfw"
    EXPLICIT = "explicit"

@dataclass
class ModelCapabilities:
    """Modell-Fähigkeiten definieren"""
    name: str
    path: str
    tasks: List[TaskType]
    quality_score: float  # 0-100
    speed_score: float   # 0-100
    vram_usage: int      # MB
    content_restrictions: bool  # True = has safety checker
    specializations: List[str]  # ["photorealistic", "anime", "artistic", etc.]
    architecture: str    # "sdxl", "sd15", "flux", etc.

@dataclass
class TaskContext:
    """Kontext für Task-Ausführung"""
    task_type: TaskType
    prompt: str
    negative_prompt: str = ""
    content_category: ContentCategory = ContentCategory.SAFE
    quality_priority: str = "balanced"  # "speed", "balanced", "quality"
    style_preference: str = "realistic"  # "realistic", "artistic", "anime"
    custom_requirements: Dict[str, Any] = None

class IntelligentModelOrchestrator:
    """Hauptklasse für intelligente Modellauswahl"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.models_dir = base_dir / "models"
        self.available_models: Dict[str, ModelCapabilities] = {}
        self.model_instances: Dict[str, Any] = {}
        self.usage_stats: Dict[str, Dict[str, float]] = {}
        self.performance_cache: Dict[str, Dict[str, float]] = {}
        
        self.scan_available_models()
        self.load_performance_data()
    
    def scan_available_models(self):
        """Scanne verfügbare Modelle und analysiere ihre Fähigkeiten"""
        logger.info("🔍 Scanning available AI models...")
        
        # Definiere bekannte Modell-Patterns und ihre Fähigkeiten
        model_patterns = {
            r"stable.*diffusion.*xl.*inpaint": {
                "tasks": [TaskType.INPAINTING, TaskType.CLOTHING_CHANGE, TaskType.OBJECT_INSERTION],
                "quality_score": 95,
                "speed_score": 70,
                "vram_usage": 8000,
                "specializations": ["photorealistic", "high_resolution"],
                "architecture": "sdxl"
            },
            r"realistic.*vision": {
                "tasks": [TaskType.TXT2IMG, TaskType.CLOTHING_CHANGE, TaskType.OBJECT_INSERTION],
                "quality_score": 92,
                "speed_score": 80,
                "vram_usage": 4000,
                "specializations": ["photorealistic", "portraits"],
                "architecture": "sd15"
            },
            r"flux.*inpaint": {
                "tasks": [TaskType.INPAINTING, TaskType.OBJECT_INSERTION],
                "quality_score": 90,
                "speed_score": 85,
                "vram_usage": 6000,
                "specializations": ["high_quality", "creative"],
                "architecture": "flux"
            },
            r"real.*esrgan": {
                "tasks": [TaskType.UPSCALING],
                "quality_score": 98,
                "speed_score": 60,
                "vram_usage": 2000,
                "specializations": ["upscaling", "enhancement"],
                "architecture": "esrgan"
            },
            r"gfpgan": {
                "tasks": [TaskType.FACE_RESTORATION],
                "quality_score": 94,
                "speed_score": 85,
                "vram_usage": 1500,
                "specializations": ["face_restoration"],
                "architecture": "gfpgan"
            },
            r"wav2lip": {
                "tasks": [],  # Special audio-video model
                "quality_score": 88,
                "speed_score": 70,
                "vram_usage": 3000,
                "specializations": ["lip_sync"],
                "architecture": "wav2lip"
            }
        }
        
        # Scanne Modell-Verzeichnisse
        for model_type in ["image", "video", "audio"]:
            type_dir = self.models_dir / model_type
            if not type_dir.exists():
                continue
                
            for model_path in type_dir.iterdir():
                if model_path.is_dir() or model_path.suffix in ['.safetensors', '.ckpt', '.pth']:
                    model_name = model_path.stem.lower()
                    
                    # Pattern matching für Modell-Typ
                    capabilities = None
                    for pattern, caps in model_patterns.items():
                        if re.search(pattern, model_name):
                            capabilities = ModelCapabilities(
                                name=model_path.name,
                                path=str(model_path),
                                tasks=caps["tasks"],
                                quality_score=caps["quality_score"],
                                speed_score=caps["speed_score"],
                                vram_usage=caps["vram_usage"],
                                content_restrictions=False,  # AndioMediaStudio = No Limits!
                                specializations=caps["specializations"],
                                architecture=caps["architecture"]
                            )
                            break
                    
                    # Fallback für unbekannte Modelle
                    if not capabilities:
                        capabilities = self._analyze_unknown_model(model_path)
                    
                    self.available_models[model_path.name] = capabilities
                    logger.info(f"📦 Found model: {model_path.name} ({capabilities.architecture})")
        
        logger.info(f"✅ Scanned {len(self.available_models)} AI models")
    
    def _analyze_unknown_model(self, model_path: Path) -> ModelCapabilities:
        """Analysiere unbekannte Modelle"""
        name = model_path.name.lower()
        
        # Basis-Klassifikation basierend auf Namen/Verzeichnis
        if "inpaint" in name:
            tasks = [TaskType.INPAINTING, TaskType.CLOTHING_CHANGE]
        elif "xl" in name or "sdxl" in name:
            tasks = [TaskType.TXT2IMG, TaskType.IMG2IMG, TaskType.OBJECT_INSERTION]
        else:
            tasks = [TaskType.TXT2IMG]
        
        return ModelCapabilities(
            name=model_path.name,
            path=str(model_path),
            tasks=tasks,
            quality_score=75,  # Standard-Wert
            speed_score=75,
            vram_usage=4000,
            content_restrictions=False,
            specializations=["general"],
            architecture="unknown"
        )
    
    def select_best_model(self, context: TaskContext) -> Tuple[str, str]:
        """
        Wähle das beste Modell für einen Task
        
        Returns:
            Tuple[model_name, reason]: Name des besten Modells und Begründung
        """
        candidates = []
        
        # Filtere Modelle die den Task unterstützen
        for model_name, capabilities in self.available_models.items():
            if context.task_type in capabilities.tasks:
                score = self._calculate_model_score(capabilities, context)
                candidates.append((model_name, capabilities, score))
        
        if not candidates:
            # Fallback: Nehme bestes Allround-Modell
            fallback = self._get_fallback_model(context)
            return fallback, f"Fallback model (no specialized model found for {context.task_type.value})"
        
        # Sortiere nach Score
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_model, best_capabilities, best_score = candidates[0]
        
        # Generiere Begründung
        reason = self._generate_selection_reason(best_capabilities, context, best_score)
        
        logger.info(f"🎯 Selected model: {best_model} (score: {best_score:.1f}) - {reason}")
        return best_model, reason
    
    def _calculate_model_score(self, capabilities: ModelCapabilities, context: TaskContext) -> float:
        """Berechne Score für Modell basierend auf Kontext"""
        score = 0.0
        
        # Basis-Scores
        quality_weight = {"speed": 0.2, "balanced": 0.5, "quality": 0.8}[context.quality_priority]
        speed_weight = 1.0 - quality_weight
        
        score += capabilities.quality_score * quality_weight
        score += capabilities.speed_score * speed_weight
        
        # Task-spezifische Boni
        if context.task_type in capabilities.tasks:
            score += 15  # Task wird direkt unterstützt
        
        # Style-Matching
        if context.style_preference in capabilities.specializations:
            score += 10
        
        # Content-Handling (AndioMediaStudio = No Restrictions!)
        if not capabilities.content_restrictions and context.content_category != ContentCategory.SAFE:
            score += 20  # Bonus für NSFW-Capability
        
        # VRAM-Optimierung
        available_vram = self._get_available_vram()
        if capabilities.vram_usage <= available_vram:
            score += 5
        else:
            score -= 15  # Penalty wenn nicht genug VRAM
        
        # Performance History
        if capabilities.name in self.performance_cache:
            perf_data = self.performance_cache[capabilities.name]
            if context.task_type.value in perf_data:
                # Bonus für bewährte Performance
                score += perf_data[context.task_type.value] * 0.1
        
        return min(100.0, max(0.0, score))
    
    def _generate_selection_reason(self, capabilities: ModelCapabilities, 
                                 context: TaskContext, score: float) -> str:
        """Generiere menschenlesbare Begründung für Modellauswahl"""
        reasons = []
        
        # Primärer Grund
        if context.task_type in capabilities.tasks:
            task_names = {
                TaskType.CLOTHING_REMOVAL: "clothing removal",
                TaskType.CLOTHING_CHANGE: "clothing transformation", 
                TaskType.OBJECT_INSERTION: "object insertion",
                TaskType.UPSCALING: "image upscaling",
                TaskType.FACE_RESTORATION: "face restoration"
            }
            reasons.append(f"specialized for {task_names.get(context.task_type, 'this task')}")
        
        # Quality/Speed Fokus
        if context.quality_priority == "quality" and capabilities.quality_score > 90:
            reasons.append("highest quality output")
        elif context.quality_priority == "speed" and capabilities.speed_score > 85:
            reasons.append("fastest processing")
        
        # Style Matching
        if context.style_preference in capabilities.specializations:
            reasons.append(f"optimized for {context.style_preference} style")
        
        # NSFW Handling
        if not capabilities.content_restrictions and context.content_category != ContentCategory.SAFE:
            reasons.append("unrestricted content generation")
        
        # Architecture Benefits
        if capabilities.architecture == "sdxl":
            reasons.append("SDXL architecture for superior quality")
        elif capabilities.architecture == "flux":
            reasons.append("FLUX for creative generation")
        
        return ", ".join(reasons[:3])  # Maximal 3 Gründe
    
    def _get_fallback_model(self, context: TaskContext) -> str:
        """Fallback-Modell wenn kein spezialisiertes gefunden wird"""
        # Suche bestes Allround-Modell
        best_score = 0
        best_model = None
        
        for model_name, capabilities in self.available_models.items():
            # Bevorzuge SDXL für Allround
            base_score = capabilities.quality_score * 0.7 + capabilities.speed_score * 0.3
            if capabilities.architecture == "sdxl":
                base_score += 15
            elif capabilities.architecture == "flux":
                base_score += 10
            
            if base_score > best_score:
                best_score = base_score
                best_model = model_name
        
        return best_model or "stable-diffusion-xl-base-1.0"
    
    def _get_available_vram(self) -> int:
        """Ermittle verfügbaren VRAM"""
        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                available = (total_memory - allocated_memory) // (1024 * 1024)  # MB
                return max(2000, available - 1000)  # Reserve 1GB
            except:
                return 4000  # Fallback
        return 0  # CPU only
    
    def load_performance_data(self):
        """Lade Performance-Daten aus vorherigen Durchläufen"""
        perf_file = self.base_dir / "server" / "cache" / "model_performance.json"
        if perf_file.exists():
            try:
                with open(perf_file, 'r') as f:
                    self.performance_cache = json.load(f)
                logger.info(f"📊 Loaded performance data for {len(self.performance_cache)} models")
            except Exception as e:
                logger.warning(f"Failed to load performance data: {e}")
    
    def save_performance_data(self):
        """Speichere Performance-Daten"""
        perf_file = self.base_dir / "server" / "cache" / "model_performance.json"
        perf_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(perf_file, 'w') as f:
                json.dump(self.performance_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save performance data: {e}")
    
    def record_performance(self, model_name: str, task_type: TaskType, 
                          execution_time: float, quality_rating: float):
        """Zeichne Performance-Daten auf"""
        if model_name not in self.performance_cache:
            self.performance_cache[model_name] = {}
        
        task_key = task_type.value
        if task_key not in self.performance_cache[model_name]:
            self.performance_cache[model_name][task_key] = {"times": [], "ratings": []}
        
        # Sliding window für Performance-Daten
        data = self.performance_cache[model_name][task_key]
        data["times"].append(execution_time)
        data["ratings"].append(quality_rating)
        
        # Nur letzten 20 Einträge behalten
        if len(data["times"]) > 20:
            data["times"] = data["times"][-20:]
            data["ratings"] = data["ratings"][-20:]
        
        # Auto-save alle 5 Minuten
        current_time = time.time()
        if not hasattr(self, '_last_save') or current_time - self._last_save > 300:
            self.save_performance_data()
            self._last_save = current_time
    
    def get_model_recommendations(self, context: TaskContext) -> List[Tuple[str, str, float]]:
        """Erhalte Top 3 Modell-Empfehlungen mit Scores"""
        recommendations = []
        
        for model_name, capabilities in self.available_models.items():
            if context.task_type in capabilities.tasks or len(capabilities.tasks) == 0:
                score = self._calculate_model_score(capabilities, context)
                reason = self._generate_selection_reason(capabilities, context, score)
                recommendations.append((model_name, reason, score))
        
        # Sortiere nach Score und gib Top 3 zurück
        recommendations.sort(key=lambda x: x[2], reverse=True)
        return recommendations[:3]
    
    async def execute_with_best_model(self, context: TaskContext, 
                                    inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Führe Task mit automatisch gewähltem bestem Modell aus"""
        start_time = time.time()
        
        # Wähle bestes Modell
        model_name, reason = self.select_best_model(context)
        logger.info(f"🚀 Executing {context.task_type.value} with {model_name}")
        
        try:
            # Lade Modell wenn nötig
            model_instance = await self._load_model_instance(model_name, context.task_type)
            
            # Führe Task aus
            result = await self._execute_task(model_instance, context, inputs)
            
            execution_time = time.time() - start_time
            
            # Performance aufzeichnen (Mock quality rating für jetzt)
            quality_rating = 85.0  # In echter Implementierung: automatische Qualitätsbewertung
            self.record_performance(model_name, context.task_type, execution_time, quality_rating)
            
            result.update({
                "model_used": model_name,
                "model_reason": reason,
                "execution_time": execution_time,
                "success": True
            })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Model execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_used": model_name,
                "execution_time": time.time() - start_time
            }
    
    async def _load_model_instance(self, model_name: str, task_type: TaskType):
        """Lade Modell-Instanz (mit Caching)"""
        cache_key = f"{model_name}_{task_type.value}"
        
        if cache_key in self.model_instances:
            return self.model_instances[cache_key]
        
        # Hier würde die echte Modell-Ladung stattfinden
        # Für Demo-Zwecke simulieren wir das
        logger.info(f"📦 Loading {model_name} for {task_type.value}...")
        
        # Simuliere Ladezeit
        await asyncio.sleep(1)
        
        # Mock-Modell-Instanz
        model_instance = {
            "name": model_name,
            "type": task_type,
            "loaded_at": time.time()
        }
        
        self.model_instances[cache_key] = model_instance
        return model_instance
    
    async def _execute_task(self, model_instance: Dict, context: TaskContext, 
                           inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Führe spezifischen Task mit Modell aus"""
        task_handlers = {
            TaskType.CLOTHING_REMOVAL: self._handle_clothing_removal,
            TaskType.CLOTHING_CHANGE: self._handle_clothing_change,
            TaskType.OBJECT_INSERTION: self._handle_object_insertion,
            TaskType.BACKGROUND_REPLACEMENT: self._handle_background_replacement,
            TaskType.STYLE_TRANSFER: self._handle_style_transfer,
            TaskType.UPSCALING: self._handle_upscaling,
            TaskType.FACE_RESTORATION: self._handle_face_restoration
        }
        
        handler = task_handlers.get(context.task_type)
        if not handler:
            raise ValueError(f"No handler for task type: {context.task_type}")
        
        return await handler(model_instance, context, inputs)
    
    async def _handle_clothing_removal(self, model: Dict, context: TaskContext, 
                                     inputs: Dict) -> Dict[str, Any]:
        """Handle Clothing Removal"""
        logger.info("👕 Removing clothing with AI...")
        
        # Simuliere Verarbeitung
        await asyncio.sleep(2)
        
        return {
            "result_type": "clothing_removal",
            "clothing_type": inputs.get("clothing_type", "shirt"),
            "preserve_anatomy": inputs.get("preserve_anatomy", True),
            "confidence": 0.95,
            "processing_details": {
                "segmentation_method": "SAM + MediaPipe",
                "inpainting_model": model["name"],
                "anatomical_correction": True
            }
        }
    
    async def _handle_clothing_change(self, model: Dict, context: TaskContext,
                                    inputs: Dict) -> Dict[str, Any]:
        """Handle Clothing Change"""
        logger.info("🔄 Changing clothing with AI...")
        
        await asyncio.sleep(3)
        
        return {
            "result_type": "clothing_change",
            "new_clothing": inputs.get("new_clothing_prompt"),
            "clothing_type": inputs.get("clothing_type"),
            "material": inputs.get("material", "fabric"),
            "style_match": 0.92,
            "processing_details": {
                "segmentation_accuracy": 0.96,
                "generation_model": model["name"],
                "style_consistency": True
            }
        }
    
    async def _handle_object_insertion(self, model: Dict, context: TaskContext,
                                     inputs: Dict) -> Dict[str, Any]:
        """Handle Object Insertion"""
        logger.info("⭐ Inserting object with AI...")
        
        await asyncio.sleep(2.5)
        
        return {
            "result_type": "object_insertion",
            "object_description": inputs.get("object_prompt"),
            "position": inputs.get("position", {"x": 0.5, "y": 0.5}),
            "insertion_strength": inputs.get("strength", 0.8),
            "integration_quality": 0.89,
            "processing_details": {
                "depth_estimation": True,
                "lighting_match": True,
                "shadow_generation": True,
                "model_used": model["name"]
            }
        }
    
    async def _handle_background_replacement(self, model: Dict, context: TaskContext,
                                           inputs: Dict) -> Dict[str, Any]:
        """Handle Background Replacement"""
        logger.info("🖼️ Replacing background with AI...")
        
        await asyncio.sleep(2)
        
        return {
            "result_type": "background_replacement", 
            "new_background": inputs.get("background_prompt"),
            "subject_preservation": 0.94,
            "edge_blending": 0.91
        }
    
    async def _handle_style_transfer(self, model: Dict, context: TaskContext,
                                   inputs: Dict) -> Dict[str, Any]:
        """Handle Style Transfer"""
        logger.info("🎨 Applying style transfer...")
        
        await asyncio.sleep(1.5)
        
        return {
            "result_type": "style_transfer",
            "target_style": inputs.get("style_prompt"),
            "style_strength": inputs.get("style_strength", 0.8),
            "content_preservation": 0.88
        }
    
    async def _handle_upscaling(self, model: Dict, context: TaskContext,
                              inputs: Dict) -> Dict[str, Any]:
        """Handle Image Upscaling"""
        logger.info("🔍 Upscaling image...")
        
        await asyncio.sleep(4)  # Upscaling dauert länger
        
        return {
            "result_type": "upscaling",
            "scale_factor": inputs.get("scale_factor", 4),
            "target_resolution": "4K",
            "detail_enhancement": 0.93
        }
    
    async def _handle_face_restoration(self, model: Dict, context: TaskContext,
                                     inputs: Dict) -> Dict[str, Any]:
        """Handle Face Restoration"""
        logger.info("😊 Restoring face details...")
        
        await asyncio.sleep(1)
        
        return {
            "result_type": "face_restoration",
            "faces_detected": inputs.get("face_count", 1),
            "restoration_quality": 0.91,
            "detail_recovery": 0.88
        }


class ContentAnalyzer:
    """Analysiert Content für automatische Kategorisierung"""
    
    def __init__(self):
        self.nsfw_keywords = {
            "explicit": ["nude", "naked", "sex", "porn", "explicit", "xxx"],
            "suggestive": ["bikini", "lingerie", "underwear", "sexy", "seductive", "sensual"],
            "clothing": ["dress", "shirt", "pants", "jacket", "skirt", "boots", "shoes"],
            "safe": ["landscape", "nature", "food", "animals", "architecture", "art"]
        }
    
    def analyze_prompt(self, prompt: str) -> Tuple[ContentCategory, Dict[str, Any]]:
        """Analysiere Prompt und bestimme Content-Kategorie"""
        prompt_lower = prompt.lower()
        
        # Explicit Content Check
        if any(keyword in prompt_lower for keyword in self.nsfw_keywords["explicit"]):
            return ContentCategory.EXPLICIT, {"confidence": 0.95, "keywords_found": "explicit"}
        
        # Suggestive Content Check  
        if any(keyword in prompt_lower for keyword in self.nsfw_keywords["suggestive"]):
            return ContentCategory.SUGGESTIVE, {"confidence": 0.85, "keywords_found": "suggestive"}
        
        # NSFW but not explicit
        nsfw_indicators = ["remove clothes", "no clothes", "undress", "revealing"]
        if any(indicator in prompt_lower for indicator in nsfw_indicators):
            return ContentCategory.NSFW, {"confidence": 0.8, "keywords_found": "nsfw_action"}
        
        # Safe Content
        return ContentCategory.SAFE, {"confidence": 0.9, "keywords_found": "safe"}
    
    def get_style_preference(self, prompt: str) -> str:
        """Bestimme Style-Präferenz aus Prompt"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["photorealistic", "photo", "realistic", "portrait"]):
            return "photorealistic"
        elif any(word in prompt_lower for word in ["anime", "manga", "cartoon"]):
            return "anime" 
        elif any(word in prompt_lower for word in ["artistic", "painting", "art", "creative"]):
            return "artistic"
        elif any(word in prompt_lower for word in ["cinematic", "movie", "film"]):
            return "cinematic"
        
        return "realistic"  # Default


# FastAPI Integration
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

class TaskRequest(BaseModel):
    task_type: str
    prompt: str
    negative_prompt: str = ""
    inputs: Dict[str, Any] = {}
    quality_priority: str = "balanced"
    
class ModelRecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    selected_model: str
    selection_reason: str

router = APIRouter()

# Global orchestrator instance
orchestrator = None

@router.on_event("startup")
async def setup_orchestrator():
    global orchestrator
    from pathlib import Path
    base_dir = Path(__file__).resolve().parents[2]  # AndioMediaStudio root
    orchestrator = IntelligentModelOrchestrator(base_dir)

@router.post("/api/intelligent/execute")
async def execute_intelligent_task(request: TaskRequest):
    """Führe Task mit automatisch gewähltem bestem Modell aus"""
    global orchestrator
    if not orchestrator:
        raise HTTPException(500, "Model orchestrator not initialized")
    
    try:
        # Content-Analyse
        analyzer = ContentAnalyzer()
        content_category, analysis = analyzer.analyze_prompt(request.prompt)
        style_preference = analyzer.get_style_preference(request.prompt)
        
        # Task-Kontext erstellen
        context = TaskContext(
            task_type=TaskType(request.task_type),
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            content_category=content_category,
            quality_priority=request.quality_priority,
            style_preference=style_preference,
            custom_requirements=request.inputs
        )
        
        # Ausführung mit bestem Modell
        result = await orchestrator.execute_with_best_model(context, request.inputs)
        
        return {
            "success": result["success"],
            "result": result,
            "content_analysis": {
                "category": content_category.value,
                "style": style_preference,
                "analysis_details": analysis
            },
            "model_info": {
                "selected": result.get("model_used"),
                "reason": result.get("model_reason"),
                "execution_time": result.get("execution_time")
            }
        }
        
    except Exception as e:
        logger.error(f"Intelligent execution failed: {e}")
        raise HTTPException(500, f"Execution failed: {str(e)}")

@router.get("/api/intelligent/recommendations/{task_type}")
async def get_model_recommendations(task_type: str, prompt: str = "", quality_priority: str = "balanced"):
    """Erhalte Modell-Empfehlungen für Task"""
    global orchestrator
    if not orchestrator:
        raise HTTPException(500, "Model orchestrator not initialized")
    
    try:
        # Content-Analyse
        analyzer = ContentAnalyzer()
        content_category, _ = analyzer.analyze_prompt(prompt)
        style_preference = analyzer.get_style_preference(prompt)
        
        # Kontext für Empfehlungen
        context = TaskContext(
            task_type=TaskType(task_type),
            prompt=prompt,
            content_category=content_category,
            quality_priority=quality_priority,
            style_preference=style_preference
        )
        
        # Empfehlungen holen
        recommendations = orchestrator.get_model_recommendations(context)
        selected_model, reason = orchestrator.select_best_model(context)
        
        return {
            "recommendations": [
                {
                    "model_name": name,
                    "reason": reason,
                    "score": score,
                    "architecture": orchestrator.available_models[name].architecture
                }
                for name, reason, score in recommendations
            ],
            "selected_model": selected_model,
            "selection_reason": reason,
            "content_analysis": {
                "category": content_category.value,
                "style": style_preference
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(500, f"Recommendation failed: {str(e)}")

@router.get("/api/intelligent/models")
async def list_available_models():
    """Liste alle verfügbaren Modelle mit Capabilities"""
    global orchestrator
    if not orchestrator:
        raise HTTPException(500, "Model orchestrator not initialized")
    
    models_info = []
    for name, capabilities in orchestrator.available_models.items():
        models_info.append({
            "name": name,
            "path": capabilities.path,
            "tasks": [task.value for task in capabilities.tasks],
            "quality_score": capabilities.quality_score,
            "speed_score": capabilities.speed_score,
            "vram_usage": capabilities.vram_usage,
            "content_restrictions": capabilities.content_restrictions,
            "specializations": capabilities.specializations,
            "architecture": capabilities.architecture
        })
    
    return {
        "models": models_info,
        "total_count": len(models_info),
        "unrestricted_models": len([m for m in orchestrator.available_models.values() 
                                   if not m.content_restrictions])
    }


# Utility Functions für schnelle Integration
def quick_select_model(base_dir: Path, task_type: str, prompt: str = "") -> Tuple[str, str]:
    """Schnelle Modellauswahl ohne vollständige Orchestrierung"""
    orchestrator = IntelligentModelOrchestrator(base_dir)
    analyzer = ContentAnalyzer()
    
    content_category, _ = analyzer.analyze_prompt(prompt)
    style_preference = analyzer.get_style_preference(prompt)
    
    context = TaskContext(
        task_type=TaskType(task_type),
        prompt=prompt,
        content_category=content_category,
        style_preference=style_preference
    )
    
    return orchestrator.select_best_model(context)

def is_nsfw_content(prompt: str) -> bool:
    """Schnelle NSFW-Erkennung"""
    analyzer = ContentAnalyzer()
    category, _ = analyzer.analyze_prompt(prompt)
    return category in [ContentCategory.NSFW, ContentCategory.EXPLICIT, ContentCategory.SUGGESTIVE]