# server/models_registry.py
"""
Dynamisches Model Registry System für AndioMediaStudio
- Lädt models/manifest.json
- Unterstützt flaches models/ Layout + Legacy-Unterordner
- Smart Model Picking basierend auf Genre/NSFW
- NSFW-Consent Handling
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    id: str
    name: str
    type: str  # image, video, voice
    genres: List[str] = None
    tags: List[str] = None
    nsfw: bool = False
    hint: str = ""
    path: str = ""
    size_mb: int = 0
    installed: bool = False
    architecture: str = "unknown"  # sd15, sdxl, flux, svd, etc.

class ModelsRegistry:
    """Zentrale Model-Verwaltung für AndioMediaStudio"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.manifest_path = self.models_dir / "manifest.json"
        self.models: Dict[str, ModelInfo] = {}
        self.load_manifest()
        self.scan_installed_models()
    
    def load_manifest(self) -> None:
        """Lade manifest.json mit allen verfügbaren Modellen"""
        try:
            if self.manifest_path.exists():
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for model_data in data.get('models', []):
                    model = ModelInfo(
                        id=model_data['id'],
                        name=model_data['name'],
                        type=model_data.get('type', 'image'),
                        genres=model_data.get('genres', []),
                        tags=model_data.get('tags', []),
                        nsfw=model_data.get('nsfw', False),
                        hint=model_data.get('hint', ''),
                        path=model_data.get('path', ''),
                        size_mb=model_data.get('size_mb', 0),
                        architecture=model_data.get('architecture', 'unknown')
                    )
                    self.models[model.id] = model
                    
                logger.info(f"✅ Loaded {len(self.models)} models from manifest.json")
            else:
                logger.warning("📝 No manifest.json found, creating default")
                self._create_default_manifest()
                
        except Exception as e:
            logger.error(f"❌ Failed to load manifest: {e}")
            self._create_default_manifest()
    
    def _create_default_manifest(self) -> None:
        """Erstelle Standard-Manifest mit Beispiel-Modellen"""
        default_models = [
            {
                "id": "sd15-realistic", "name": "Stable Diffusion 1.5 Realistic",
                "type": "image", "genres": ["portrait", "landscape", "art"],
                "tags": ["txt2img", "img2img", "inpaint"], "nsfw": False,
                "hint": "Klassisches SD 1.5 - gut für realistische Bilder",
                "architecture": "sd15"
            },
            {
                "id": "sdxl-base", "name": "SDXL Base Model",
                "type": "image", "genres": ["portrait", "art", "photography"],
                "tags": ["txt2img", "img2img"], "nsfw": False,
                "hint": "SDXL für hochauflösende, detaillierte Bilder",
                "architecture": "sdxl"
            },
            {
                "id": "flux-dev", "name": "FLUX.1 Dev",
                "type": "image", "genres": ["creative", "art", "abstract"],
                "tags": ["txt2img"], "nsfw": False,
                "hint": "FLUX für kreative und künstlerische Inhalte",
                "architecture": "flux"
            },
            {
                "id": "svd-xt", "name": "Stable Video Diffusion XT",
                "type": "video", "genres": ["cinematic", "nature", "abstract"],
                "tags": ["img2video", "txt2video"], "nsfw": False,
                "hint": "Video-Generation aus Bildern oder Text",
                "architecture": "svd"
            },
            {
                "id": "realistic-nsfw", "name": "Realistic Vision NSFW",
                "type": "image", "genres": ["portrait", "artistic"],
                "tags": ["txt2img", "img2img", "inpaint"], "nsfw": True,
                "hint": "Realistische Bilder ohne Content-Filter",
                "architecture": "sd15"
            }
        ]
        
        manifest = {"models": default_models}
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        # Models in Registry laden
        for model_data in default_models:
            model = ModelInfo(
                id=model_data['id'],
                name=model_data['name'],
                type=model_data.get('type', 'image'),
                genres=model_data.get('genres', []),
                tags=model_data.get('tags', []),
                nsfw=model_data.get('nsfw', False),
                hint=model_data.get('hint', ''),
                path=model_data.get('path', ''),
                architecture=model_data.get('architecture', 'unknown')
            )
            self.models[model.id] = model
        
        logger.info("📝 Created default manifest.json with example models")
    
    def scan_installed_models(self) -> None:
        """Scanne physisch vorhandene Modelle (flach + legacy)"""
        scan_paths = [
            self.models_dir,  # Flaches Layout
            self.models_dir / "image",  # Legacy
            self.models_dir / "video",  # Legacy
        ]
        
        installed_models = set()
        
        for scan_path in scan_paths:
            if not scan_path.exists():
                continue
                
            # Diffusers-Ordner (mit model_index.json)
            for path in scan_path.iterdir():
                if path.is_dir() and (path / "model_index.json").exists():
                    model_id = self._path_to_model_id(path)
                    installed_models.add(model_id)
                    logger.debug(f"📦 Found diffusers model: {path.name}")
            
            # Single-File Checkpoints
            for path in scan_path.glob("*.safetensors"):
                model_id = self._path_to_model_id(path)
                installed_models.add(model_id)
                logger.debug(f"📦 Found checkpoint: {path.name}")
            
            for path in scan_path.glob("*.ckpt"):
                model_id = self._path_to_model_id(path)
                installed_models.add(model_id)
                logger.debug(f"📦 Found checkpoint: {path.name}")
        
        # Update installation status
        for model_id, model in self.models.items():
            model.installed = model_id in installed_models
            if model.installed and not model.path:
                # Try to find actual path
                model.path = self._find_model_path(model_id) or ""
        
        logger.info(f"🔍 Scanned models: {len(installed_models)} installed, {len(self.models)} total")
    
    def _path_to_model_id(self, path: Path) -> str:
        """Konvertiere Pfad zu Model-ID"""
        name = path.stem if path.is_file() else path.name
        # Normalisiere Namen zu IDs
        model_id = name.lower().replace(" ", "-").replace("_", "-")
        return model_id
    
    def _find_model_path(self, model_id: str) -> Optional[str]:
        """Finde physischen Pfad für Model-ID"""
        search_patterns = [
            model_id,
            model_id.replace("-", "_"),
            model_id.replace("-", " "),
        ]
        
        scan_paths = [self.models_dir, self.models_dir / "image", self.models_dir / "video"]
        
        for scan_path in scan_paths:
            if not scan_path.exists():
                continue
                
            for pattern in search_patterns:
                # Diffusers-Ordner
                candidate = scan_path / pattern
                if candidate.is_dir() and (candidate / "model_index.json").exists():
                    return str(candidate)
                
                # Checkpoints
                for ext in [".safetensors", ".ckpt", ".bin", ".pt"]:
                    candidate = scan_path / f"{pattern}{ext}"
                    if candidate.exists():
                        return str(candidate)
        
        return None
    
    def get_models(self, 
                   type: Optional[str] = None,
                   nsfw: bool = False,
                   genre: Optional[str] = None,
                   installed_only: bool = False) -> List[Dict[str, Any]]:
        """
        Hole gefilterte Modell-Liste
        
        Args:
            type: 'image', 'video', 'voice' oder None für alle
            nsfw: True für NSFW-Modelle, False für SFW
            genre: Spezifisches Genre filtern
            installed_only: Nur installierte Modelle
        """
        results = []
        
        for model in self.models.values():
            # Filter: Typ
            if type and model.type != type:
                continue
            
            # Filter: NSFW
            if not nsfw and model.nsfw:
                continue
            
            # Filter: Genre
            if genre and genre not in model.genres:
                continue
            
            # Filter: Installiert
            if installed_only and not model.installed:
                continue
            
            results.append({
                "id": model.id,
                "name": model.name,
                "type": model.type,
                "genres": model.genres,
                "tags": model.tags,
                "nsfw": model.nsfw,
                "hint": model.hint,
                "path": model.path,
                "installed": model.installed,
                "architecture": model.architecture,
                "size_mb": model.size_mb
            })
        
        # Sortiere: Installierte zuerst, dann nach Name
        results.sort(key=lambda x: (not x["installed"], x["name"]))
        return results
    
    def smart_pick_model(self, 
                        type: str = "image",
                        nsfw: bool = False,
                        genre: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Intelligente Modell-Auswahl basierend auf Parametern
        
        Returns:
            Bestes Modell für die Anforderungen oder None
        """
        candidates = self.get_models(type=type, nsfw=nsfw, installed_only=True)
        
        if not candidates:
            # Fallback: Alle installierten Modelle des Typs
            candidates = self.get_models(type=type, installed_only=True)
        
        if not candidates:
            return None
        
        # Scoring-System
        def score_model(model: Dict[str, Any]) -> float:
            score = 0.0
            
            # Basis-Score für Installation
            if model["installed"]:
                score += 100
            
            # Genre-Match
            if genre and genre in model.get("genres", []):
                score += 50
            
            # NSFW-Match
            if nsfw and model.get("nsfw", False):
                score += 30
            elif not nsfw and not model.get("nsfw", False):
                score += 20
            
            # Architektur-Präferenz (modernere Modelle bevorzugen)
            arch_scores = {
                "flux": 25, "sdxl": 20, "sd15": 15, "svd": 30, "unknown": 5
            }
            score += arch_scores.get(model.get("architecture", "unknown"), 5)
            
            # Kleine Präferenz für Modelle mit mehr Tags (vielseitiger)
            score += len(model.get("tags", [])) * 2
            
            return score
        
        # Beste Kandidaten finden
        scored_candidates = [(score_model(m), m) for m in candidates]
        scored_candidates.sort(reverse=True)  # Höchste Scores zuerst
        
        best_model = scored_candidates[0][1]
        logger.info(f"🎯 Smart picked model: {best_model['name']} (score: {scored_candidates[0][0]:.1f})")
        
        return best_model
    
    def get_model_by_id(self, model_id: str) -> Optional[ModelInfo]:
        """Hole spezifisches Modell nach ID"""
        return self.models.get(model_id)
    
    def add_model(self, model_data: Dict[str, Any]) -> bool:
        """Füge neues Modell zur Registry hinzu"""
        try:
            model = ModelInfo(
                id=model_data['id'],
                name=model_data['name'],
                type=model_data.get('type', 'image'),
                genres=model_data.get('genres', []),
                tags=model_data.get('tags', []),
                nsfw=model_data.get('nsfw', False),
                hint=model_data.get('hint', ''),
                path=model_data.get('path', ''),
                size_mb=model_data.get('size_mb', 0),
                architecture=model_data.get('architecture', 'unknown')
            )
            
            self.models[model.id] = model
            self.save_manifest()
            logger.info(f"➕ Added model: {model.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add model: {e}")
            return False
    
    def save_manifest(self) -> None:
        """Speichere aktuelle Registry in manifest.json"""
        try:
            manifest_data = {
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "type": model.type,
                        "genres": model.genres or [],
                        "tags": model.tags or [],
                        "nsfw": model.nsfw,
                        "hint": model.hint,
                        "path": model.path,
                        "size_mb": model.size_mb,
                        "architecture": model.architecture
                    }
                    for model in self.models.values()
                ]
            }
            
            with open(self.manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            
            logger.info("💾 Saved manifest.json")
            
        except Exception as e:
            logger.error(f"❌ Failed to save manifest: {e}")
    
    def install_model(self, model_id: str, source_path: str) -> bool:
        """Installiere Modell von Quelle"""
        # TODO: Implementiere Download/Copy-Logik
        # Für jetzt nur Status aktualisieren
        if model_id in self.models:
            self.models[model_id].installed = True
            self.models[model_id].path = source_path
            logger.info(f"📦 Marked model as installed: {model_id}")
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Hole Registry-Statistiken"""
        total = len(self.models)
        installed = sum(1 for m in self.models.values() if m.installed)
        by_type = {}
        by_arch = {}
        nsfw_count = sum(1 for m in self.models.values() if m.nsfw)
        
        for model in self.models.values():
            # Nach Typ zählen
            by_type[model.type] = by_type.get(model.type, 0) + 1
            # Nach Architektur zählen
            by_arch[model.architecture] = by_arch.get(model.architecture, 0) + 1
        
        return {
            "total_models": total,
            "installed_models": installed,
            "nsfw_models": nsfw_count,
            "by_type": by_type,
            "by_architecture": by_arch,
            "manifest_path": str(self.manifest_path),
            "models_dir": str(self.models_dir)
        }

# Globale Registry-Instanz
_registry: Optional[ModelsRegistry] = None

def get_registry(base_dir: Path = None) -> ModelsRegistry:
    """Hole oder erstelle globale Registry-Instanz"""
    global _registry
    if _registry is None:
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[1]  # AndioMediaStudio root
        _registry = ModelsRegistry(base_dir)
    return _registry

def smart_pick_model(type: str = "image", nsfw: bool = False, genre: str = None) -> Optional[Dict[str, Any]]:
    """Convenience-Funktion für Smart Model Picking"""
    registry = get_registry()
    return registry.smart_pick_model(type=type, nsfw=nsfw, genre=genre)