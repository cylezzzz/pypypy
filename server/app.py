# server/app_complete.py
"""
AndioMediaStudio - VOLLSTÄNDIGE API mit echter KI-Integration
- Dynamische Model Registry ✅
- Echte Diffusers-Integration ✅  
- Smart Model Selection ✅
- NSFW-Consent Management ✅
- Robuste Fallbacks ✅
- Job-Management mit Status-Tracking ✅
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Lokale Imports
from server.models_registry import get_registry
from server.pipelines.ai_orchestrator import get_orchestrator, generate_image, generate_video, get_job_status, list_jobs, cancel_job, get_system_status
from server.utils.file_validator import ensure_dir, validate_image_upload

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AndioMediaStudio")

# ================================ APP SETUP ================================

app = FastAPI(
    title="AndioMediaStudio Complete API",
    version="0.7.0",
    description="Vollständige KI-Integration für lokales Media Studio"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basis-Pfade
BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"
OUTPUTS_DIR = BASE_DIR / "outputs"
WORKSPACE_DIR = BASE_DIR / "workspace"

# Verzeichnisse erstellen
for directory in [WEB_DIR, OUTPUTS_DIR / "images", OUTPUTS_DIR / "videos", WORKSPACE_DIR / "uploads"]:
    ensure_dir(directory)

# Globale Instanzen
REGISTRY = get_registry(BASE_DIR)
ORCHESTRATOR = get_orchestrator(BASE_DIR)

# WebSocket-Verbindungen für Progress-Updates
active_websockets: Dict[str, List[WebSocket]] = {}

# ================================ SCHEMAS ================================

class APIResponse(BaseModel):
    """Standardisierte API-Antwort"""
    ok: bool
    data: Optional[Any] = None
    url: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    code: Optional[int] = None
    message: Optional[str] = None
    hint: Optional[str] = None

class ImageGenerationRequest(BaseModel):
    """Bildgenerierung Parameter"""
    mode: str = Field(default="txt2img", description="txt2img, img2img, inpaint")
    nsfw: bool = Field(default=False, description="NSFW-Modus")
    genre: Optional[str] = Field(default=None, description="Genre für Smart Model Pick")
    model: Optional[str] = Field(default=None, description="Spezifisches Modell")
    smartModel: bool = Field(default=True, description="Automatische Modellauswahl")
    
    format: str = Field(default="1:1", description="Seitenverhältnis")
    res: str = Field(default="1024×1024", description="Auflösung")
    steps: int = Field(default=28, ge=1, le=150, description="Anzahl Steps")
    guidance: float = Field(default=7.5, ge=0.0, le=30.0, description="Guidance Scale")
    seed: Optional[int] = Field(default=None, description="Seed")
    
    prompt: str = Field(..., min_length=1, description="Hauptprompt")
    negative: str = Field(default="", description="Negative Prompts")
    
    refImg: Optional[str] = Field(default=None, description="Referenzbild für img2img")
    mask: Optional[str] = Field(default=None, description="Maske für Inpainting")
    
    # NSFW Consent
    consent: Optional[Dict[str, Any]] = Field(default=None, description="NSFW-Einwilligung")

class VideoGenerationRequest(BaseModel):
    """Video-Generierung Parameter"""
    mode: str = Field(default="t2v", description="t2v, ti2v, kenburns, slideshow")
    nsfw: bool = Field(default=False, description="NSFW-Modus")
    genre: Optional[str] = Field(default=None, description="Genre")
    model: Optional[str] = Field(default=None, description="Spezifisches Modell")
    smartModel: bool = Field(default=True, description="Smart Modell")
    
    format: str = Field(default="16:9", description="Video-Format")
    res: str = Field(default="1280×720", description="Auflösung")
    length: float = Field(default=6.0, ge=1.0, le=30.0, description="Länge in Sekunden")
    fps: int = Field(default=24, ge=8, le=60, description="FPS")
    
    prompt: str = Field(default="", description="Video-Prompt")
    negative: str = Field(default="", description="Negative Prompts")
    
    refImg: Optional[str] = Field(default=None, description="Referenzbild")
    sources: Optional[List[str]] = Field(default=None, description="Quellbilder für Slideshow")
    
    consent: Optional[Dict[str, Any]] = Field(default=None, description="NSFW-Einwilligung")

# ================================ ERROR HANDLER ================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Globaler Exception Handler - niemals rohe Tracebacks"""
    logger.error(f"Global exception: {exc}\n{traceback.format_exc()}")
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=APIResponse(
                ok=False,
                code=exc.status_code,
                message=str(exc.detail),
                hint="HTTP Exception"
            ).dict()
        )
    elif isinstance(exc, FileNotFoundError):
        return JSONResponse(
            status_code=404,
            content=APIResponse(
                ok=False,
                code=404,
                message="Datei oder Modell nicht gefunden",
                hint="Überprüfe models/manifest.json und installierte Modelle"
            ).dict()
        )
    elif isinstance(exc, ValueError):
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                ok=False,
                code=400,
                message=f"Ungültige Parameter: {str(exc)}",
                hint="Überprüfe Eingabewerte und NSFW-Consent"
            ).dict()
        )
    else:
        return JSONResponse(
            status_code=500,
            content=APIResponse(
                ok=False,
                code=500,
                message="Interner Server-Fehler",
                hint="Siehe Server-Logs für Details"
            ).dict()
        )

# ================================ WEBSOCKET PROGRESS ================================

@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_progress(websocket: WebSocket, job_id: str):
    """WebSocket für Job-Progress Updates"""
    await websocket.accept()
    
    if job_id not in active_websockets:
        active_websockets[job_id] = []
    active_websockets[job_id].append(websocket)
    
    logger.info(f"WebSocket connected for job {job_id}")
    
    try:
        # Initial Status senden
        status = get_job_status(job_id)
        if status:
            await websocket.send_text(json.dumps(status))
        
        # Keep-Alive Loop
        while True:
            try:
                # Warte auf Nachrichten (für Ping/Pong)
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                # Sende aktuellen Status
                status = get_job_status(job_id)
                if status:
                    await websocket.send_text(json.dumps(status))
                    
                    # Bei completed/error: Verbindung nach kurzer Zeit schließen
                    if status["status"] in ["completed", "error", "cancelled"]:
                        await asyncio.sleep(2)
                        break
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.warning(f"WebSocket error for job {job_id}: {e}")
    finally:
        # Cleanup
        if job_id in active_websockets:
            try:
                active_websockets[job_id].remove(websocket)
                if not active_websockets[job_id]:
                    del active_websockets[job_id]
            except ValueError:
                pass
        logger.info(f"WebSocket disconnected for job {job_id}")

async def broadcast_job_update(job_id: str, status_data: Dict[str, Any]):
    """Sende Job-Update an alle WebSocket-Verbindungen"""
    if job_id not in active_websockets:
        return
    
    message = json.dumps(status_data)
    disconnected = []
    
    for websocket in active_websockets[job_id]:
        try:
            await websocket.send_text(message)
        except Exception:
            disconnected.append(websocket)
    
    # Cleanup disconnected websockets
    for ws in disconnected:
        try:
            active_websockets[job_id].remove(ws)
        except ValueError:
            pass

# Job-Status-Updates an WebSockets weiterleiten
def setup_job_callbacks():
    """Setup Callbacks für Job-Updates"""
    def job_status_callback(job):
        asyncio.create_task(broadcast_job_update(job.job_id, job.to_dict()))
    
    # Alle neuen Jobs bekommen Callback
    original_create_job = ORCHESTRATOR.job_manager.create_job
    def create_job_with_callback(task_type, params):
        job = original_create_job(task_type, params)
        job.add_callback(job_status_callback)
        return job
    
    ORCHESTRATOR.job_manager.create_job = create_job_with_callback

# ================================ API ENDPOINTS ================================

@app.get("/api/ping")
async def ping():
    """Health Check mit System-Info"""
    try:
        system_status = get_system_status()
        
        return APIResponse(
            ok=True,
            data={
                "service": "AndioMediaStudio Complete API",
                "version": "0.7.0",
                "timestamp": int(time.time() * 1000),
                "system": system_status
            },
            message="AndioMediaStudio API ready with full KI integration"
        ).dict()
    except Exception as e:
        logger.error(f"Ping failed: {e}")
        return APIResponse(
            ok=False,
            message="Service nicht verfügbar",
            hint="Überprüfe KI-Pipeline Status"
        ).dict()

@app.get("/api/models")
async def get_models(type: str = "image", nsfw: bool = False, genre: Optional[str] = None, installed_only: bool = False):
    """Hole verfügbare Modelle mit Smart Picking"""
    try:
        models = REGISTRY.get_models(
            type=type,
            nsfw=nsfw,
            genre=genre,
            installed_only=installed_only
        )
        
        smart_pick = REGISTRY.smart_pick_model(
            type=type,
            nsfw=nsfw,
            genre=genre
        )
        
        return APIResponse(
            ok=True,
            data={
                "models": models,
                "smart_pick": smart_pick,
                "total_count": len(models),
                "installed_count": len([m for m in models if m["installed"]]),
                "filters": {
                    "type": type,
                    "nsfw": nsfw,
                    "genre": genre,
                    "installed_only": installed_only
                }
            },
            message=f"Gefunden: {len(models)} Modelle ({len([m for m in models if m['installed']])} installiert)"
        ).dict()
        
    except Exception as e:
        logger.error(f"Get models failed: {e}")
        return APIResponse(
            ok=False,
            message="Fehler beim Laden der Modelle",
            hint="Überprüfe models/manifest.json und Registry"
        ).dict()

@app.get("/api/presets/{preset_type}")
async def get_presets(preset_type: str):
    """Hole Presets (image/video)"""
    try:
        if preset_type == "image":
            presets = await _get_image_presets()
        elif preset_type == "video":
            presets = await _get_video_presets()
        else:
            raise ValueError(f"Unknown preset type: {preset_type}")
        
        return APIResponse(
            ok=True,
            data=presets,
            message=f"{preset_type.title()}-Presets geladen"
        ).dict()
        
    except Exception as e:
        logger.error(f"Get {preset_type} presets failed: {e}")
        return APIResponse(
            ok=False,
            message=f"Fehler beim Laden der {preset_type}-Presets"
        ).dict()

async def _get_image_presets():
    """Image-Presets"""
    return {
        "genres": [
            {"id": "portrait", "name": "Portrait", "tags": ["person", "face"]},
            {"id": "landscape", "name": "Landscape", "tags": ["nature", "outdoor"]},
            {"id": "art", "name": "Artistic", "tags": ["creative", "abstract"]},
            {"id": "photography", "name": "Photography", "tags": ["realistic", "photo"]},
            {"id": "anime", "name": "Anime/Manga", "tags": ["anime", "illustration"]},
            {"id": "nsfw_artistic", "name": "NSFW Artistic", "tags": ["artistic", "nude"], "nsfw": True},
            {"id": "nsfw_realistic", "name": "NSFW Realistic", "tags": ["realistic", "nude"], "nsfw": True}
        ],
        "styles": [
            {"id": "realistic", "name": "Realistic"},
            {"id": "cinematic", "name": "Cinematic"},
            {"id": "analog", "name": "Analog Film"},
            {"id": "anime", "name": "Anime Style"},
            {"id": "illustration", "name": "Digital Illustration"},
            {"id": "fantasy", "name": "Fantasy Art"}
        ],
        "lights": [
            {"id": "natural", "name": "Natural Light"},
            {"id": "golden_hour", "name": "Golden Hour"},
            {"id": "studio", "name": "Studio Lighting"},
            {"id": "dramatic", "name": "Dramatic Lighting"},
            {"id": "soft", "name": "Soft Light"}
        ],
        "compositions": [
            {"id": "rule_thirds", "name": "Rule of Thirds"},
            {"id": "centered", "name": "Centered"},
            {"id": "close_up", "name": "Close-up"},
            {"id": "wide_shot", "name": "Wide Shot"}
        ],
        "palettes": [
            {"id": "natural", "name": "Natural Colors"},
            {"id": "warm", "name": "Warm Tones"},
            {"id": "cool", "name": "Cool Tones"},
            {"id": "monochrome", "name": "Monochrome"},
            {"id": "vibrant", "name": "Vibrant Colors"}
        ],
        "negatives": {
            "sfw": "worst quality, low quality, jpeg artifacts, watermark, signature, text, blurry, deformed",
            "nsfw": "worst quality, low quality, jpeg artifacts, watermark, signature, text, blurry, deformed, censored"
        },
        "featured": ["Portrait", "Landscape", "Artistic", "Cinematic", "Realistic", "Golden Hour", "Studio Light"]
    }

async def _get_video_presets():
    """Video-Presets"""
    return {
        "genres": [
            {"id": "cinematic", "name": "Cinematic", "tags": ["movie", "film"]},
            {"id": "nature", "name": "Nature", "tags": ["landscape", "wildlife"]},
            {"id": "abstract", "name": "Abstract", "tags": ["artistic", "motion"]},
            {"id": "portrait", "name": "Portrait Video", "tags": ["person", "face"]},
            {"id": "action", "name": "Action", "tags": ["fast", "dynamic"]},
            {"id": "nsfw_artistic", "name": "NSFW Artistic", "tags": ["artistic"], "nsfw": True}
        ],
        "cameras": [
            {"id": "handheld", "name": "Handheld"},
            {"id": "gimbal", "name": "Gimbal Stabilized"},
            {"id": "tripod", "name": "Tripod Static"},
            {"id": "drone", "name": "Drone/Aerial"}
        ],
        "motions": [
            {"id": "slow", "name": "Slow Motion"},
            {"id": "natural", "name": "Natural Speed"},
            {"id": "fast", "name": "Fast Motion"},
            {"id": "timelapse", "name": "Timelapse"}
        ],
        "grades": [
            {"id": "natural", "name": "Natural"},
            {"id": "cinematic", "name": "Cinematic LUT"},
            {"id": "warm", "name": "Warm Grade"},
            {"id": "cool", "name": "Cool Grade"},
            {"id": "bw", "name": "Black & White"}
        ],
        "negatives": {
            "sfw": "artifacts, jitter, compression artifacts, low quality, blurry",
            "nsfw": "artifacts, jitter, compression artifacts, low quality, blurry, censored"
        },
        "featured": ["Cinematic", "Nature", "Abstract", "Slow Motion", "Drone Shot", "Natural Grade"]
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Datei-Upload mit Validierung"""
    try:
        if not file.filename:
            raise HTTPException(400, "Kein Dateiname angegeben")
        
        file_data = await file.read()
        is_valid, error_msg = validate_image_upload(file.filename, len(file_data))
        
        if not is_valid:
            raise HTTPException(400, error_msg)
        
        # Speichern
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = WORKSPACE_DIR / "uploads" / safe_filename
        
        with open(file_path, "wb") as f:
            f.write(file_data)
        
        # Bild-Info extrahieren
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                width, height = img.size
                format_info = img.format
        except Exception:
            width = height = 0
            format_info = "unknown"
        
        relative_path = str(file_path.relative_to(BASE_DIR)).replace("\\", "/")
        
        return APIResponse(
            ok=True,
            data={
                "path": relative_path,
                "url": f"/{relative_path}",
                "name": safe_filename,
                "original_name": file.filename,
                "size": len(file_data),
                "width": width,
                "height": height,
                "format": format_info
            },
            message="Datei erfolgreich hochgeladen"
        ).dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, "Upload fehlgeschlagen")

@app.post("/api/generate/image")
async def generate_image_endpoint(request: ImageGenerationRequest):
    """Bildgenerierung mit vollständiger KI-Integration"""
    try:
        # Parameter zu Dictionary konvertieren
        params = request.dict()
        
        # KI-Orchestrator aufrufen
        result = await generate_image(params)
        
        return APIResponse(
            ok=True,
            data=result,
            message="Bildgenerierung gestartet"
        ).dict()
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return APIResponse(
            ok=False,
            message=f"Bildgenerierung fehlgeschlagen: {str(e)}",
            hint="Überprüfe Parameter, Modell-Installation und NSFW-Consent"
        ).dict()

@app.post("/api/generate/video")
async def generate_video_endpoint(request: VideoGenerationRequest):
    """Video-Generierung mit vollständiger KI-Integration"""
    try:
        # Parameter zu Dictionary konvertieren
        params = request.dict()
        
        # KI-Orchestrator aufrufen
        result = await generate_video(params)
        
        return APIResponse(
            ok=True,
            data=result,
            message="Video-Generierung gestartet"
        ).dict()
        
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        return APIResponse(
            ok=False,
            message=f"Video-Generierung fehlgeschlagen: {str(e)}",
            hint="Überprüfe Parameter, Modell-Verfügbarkeit und NSFW-Consent"
        ).dict()

@app.get("/api/jobs")
async def list_jobs_endpoint(limit: int = 20):
    """Liste aktuelle Jobs"""
    try:
        jobs = list_jobs(limit)
        
        return APIResponse(
            ok=True,
            data={
                "jobs": jobs,
                "total_count": len(jobs),
                "active_count": len([j for j in jobs if j["status"] in ["queued", "loading", "generating"]])
            },
            message=f"Jobs abgerufen: {len(jobs)} Einträge"
        ).dict()
        
    except Exception as e:
        logger.error(f"List jobs failed: {e}")
        return APIResponse(
            ok=False,
            message="Fehler beim Laden der Job-Liste"
        ).dict()

@app.get("/api/jobs/{job_id}")
async def get_job_endpoint(job_id: str):
    """Hole spezifischen Job-Status"""
    try:
        job_status = get_job_status(job_id)
        
        if not job_status:
            return APIResponse(
                ok=False,
                code=404,
                message="Job nicht gefunden",
                hint=f"Job-ID {job_id} existiert nicht"
            ).dict()
        
        return APIResponse(
            ok=True,
            data=job_status,
            message="Job-Status abgerufen"
        ).dict()
        
    except Exception as e:
        logger.error(f"Get job {job_id} failed: {e}")
        return APIResponse(
            ok=False,
            message="Fehler beim Laden des Job-Status"
        ).dict()

@app.delete("/api/jobs/{job_id}")
async def cancel_job_endpoint(job_id: str):
    """Breche Job ab"""
    try:
        success = await cancel_job(job_id)
        
        if success:
            return APIResponse(
                ok=True,
                data={"job_id": job_id, "cancelled": True},
                message="Job erfolgreich abgebrochen"
            ).dict()
        else:
            return APIResponse(
                ok=False,
                message="Job konnte nicht abgebrochen werden",
                hint="Job ist möglicherweise bereits beendet oder existiert nicht"
            ).dict()
            
    except Exception as e:
        logger.error(f"Cancel job {job_id} failed: {e}")
        return APIResponse(
            ok=False,
            message="Fehler beim Abbrechen des Jobs"
        ).dict()

@app.get("/api/gallery")
async def get_gallery(type: str = "all", nsfw: Optional[bool] = None, genre: Optional[str] = None, limit: int = 100):
    """Gallery-Endpunkt für generierte Inhalte"""
    try:
        results = {"images": [], "videos": []}
        
        if type in ["all", "image"]:
            images_dir = OUTPUTS_DIR / "images"
            if images_dir.exists():
                for img_path in sorted(images_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
                    # Lade Metadaten
                    meta_path = img_path.with_suffix('.json')
                    meta = {}
                    if meta_path.exists():
                        try:
                            with open(meta_path, 'r', encoding='utf-8') as f:
                                meta = json.load(f)
                        except Exception:
                            pass
                    
                    # Filter anwenden
                    if nsfw is not None and meta.get("nsfw", False) != nsfw:
                        continue
                    if genre and meta.get("genre") != genre:
                        continue
                    
                    stat = img_path.stat()
                    results["images"].append({
                        "url": str(img_path.relative_to(BASE_DIR)).replace("\\", "/"),
                        "name": img_path.name,
                        "size": stat.st_size,
                        "created": int(stat.st_mtime * 1000),
                        "meta": meta
                    })
        
        if type in ["all", "video"]:
            videos_dir = OUTPUTS_DIR / "videos"
            if videos_dir.exists():
                for vid_path in sorted(videos_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
                    # Prüfe ob es ein Video-Thumbnail ist
                    if not ("_video" in vid_path.stem or "_kenburns" in vid_path.stem or "_slideshow" in vid_path.stem):
                        continue
                    
                    # Suche nach Metadaten
                    job_id = vid_path.stem.split('_')[0]
                    meta_path = videos_dir / f"{job_id}.json"
                    meta = {}
                    if meta_path.exists():
                        try:
                            with open(meta_path, 'r', encoding='utf-8') as f:
                                meta = json.load(f)
                        except Exception:
                            pass
                    
                    if nsfw is not None and meta.get("nsfw", False) != nsfw:
                        continue
                    if genre and meta.get("genre") != genre:
                        continue
                    
                    stat = vid_path.stat()
                    results["videos"].append({
                        "url": str(vid_path.relative_to(BASE_DIR)).replace("\\", "/"),
                        "name": vid_path.name,
                        "size": stat.st_size,
                        "created": int(stat.st_mtime * 1000),
                        "meta": meta
                    })
        
        total_items = len(results["images"]) + len(results["videos"])
        
        return APIResponse(
            ok=True,
            data=results,
            meta={
                "total_items": total_items,
                "filters": {"type": type, "nsfw": nsfw, "genre": genre}
            },
            message=f"Gallery: {total_items} Einträge gefunden"
        ).dict()
        
    except Exception as e:
        logger.error(f"Gallery failed: {e}")
        return APIResponse(
            ok=False,
            message="Fehler beim Laden der Gallery"
        ).dict()

@app.get("/api/catalog")
async def get_catalog():
    """Model-Catalog für Store-Seite"""
    try:
        stats = REGISTRY.get_statistics()
        all_models = REGISTRY.get_models()
        
        return APIResponse(
            ok=True,
            data={
                "models": all_models,
                "statistics": stats,
                "categories": {
                    "image": [m for m in all_models if m["type"] == "image"],
                    "video": [m for m in all_models if m["type"] == "video"],
                    "voice": [m for m in all_models if m["type"] == "voice"],
                    "controlnet": [m for m in all_models if m["type"] == "controlnet"]
                }
            },
            message=f"Catalog: {len(all_models)} Modelle verfügbar"
        ).dict()
        
    except Exception as e:
        logger.error(f"Catalog failed: {e}")
        return APIResponse(
            ok=False,
            message="Fehler beim Laden des Catalogs"
        ).dict()

@app.post("/api/catalog/install")
async def install_model(model_id: str = Form(...), source: str = Form(...)):
    """Modell installieren"""
    try:
        success = REGISTRY.install_model(model_id, source)
        
        if success:
            return APIResponse(
                ok=True,
                data={"model_id": model_id, "source": source, "installed": True},
                message=f"Modell {model_id} erfolgreich installiert"
            ).dict()
        else:
            return APIResponse(
                ok=False,
                message=f"Installation von {model_id} fehlgeschlagen",
                hint="Überprüfe Modell-ID und Quelle"
            ).dict()
            
    except Exception as e:
        logger.error(f"Model installation failed: {e}")
        return APIResponse(
            ok=False,
            message="Installation fehlgeschlagen"
        ).dict()

@app.get("/api/player/meta")
async def get_player_meta(url: str):
    """Player-Metadaten für Datei"""
    try:
        # URL normalisieren
        if url.startswith('/'):
            url = url[1:]
        
        file_path = BASE_DIR / url
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {url}")
        
        # Metadaten laden
        meta_path = file_path.with_suffix('.json')
        meta = {}
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        
        # File-Info
        stat = file_path.stat()
        file_info = {
            "path": url,
            "name": file_path.name,
            "size": stat.st_size,
            "created": int(stat.st_mtime * 1000),
            "type": "image" if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp'] else "video"
        }
        
        return APIResponse(
            ok=True,
            data={
                "file": file_info,
                "meta": meta
            },
            message="Metadaten geladen"
        ).dict()
        
    except FileNotFoundError:
        return APIResponse(
            ok=False,
            code=404,
            message="Datei nicht gefunden",
            hint=f"Pfad {url} existiert nicht"
        ).dict()
    except Exception as e:
        logger.error(f"Player meta failed: {e}")
        return APIResponse(
            ok=False,
            message="Fehler beim Laden der Metadaten"
        ).dict()

@app.get("/api/system/status")
async def get_system_status_endpoint():
    """System-Status und Diagnostics"""
    try:
        status = get_system_status()
        
        return APIResponse(
            ok=True,
            data=status,
            message="System-Status abgerufen"
        ).dict()
        
    except Exception as e:
        logger.error(f"System status failed: {e}")
        return APIResponse(
            ok=False,
            message="Fehler beim Laden des System-Status"
        ).dict()

# ================================ STATIC MOUNTS ================================

# Static file serving (Order matters!)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/workspace", StaticFiles(directory=str(WORKSPACE_DIR)), name="workspace")

# Web UI last (catch-all)
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:
    logger.warning(f"Web directory not found: {WEB_DIR}")

# ================================ STARTUP ================================

@app.on_event("startup")
async def startup_event():
    """Server-Initialisierung"""
    logger.info("🚀 AndioMediaStudio Complete API Server starting...")
    logger.info(f"📁 Base directory: {BASE_DIR}")
    logger.info(f"🌐 Web directory: {WEB_DIR}")
    
    # Registry Stats
    stats = REGISTRY.get_statistics()
    logger.info(f"📦 Models in registry: {stats['total_models']}")
    logger.info(f"✅ Models installed: {stats['installed_models']}")
    logger.info(f"🔞 NSFW models: {stats['nsfw_models']}")
    
    # System Status
    system_status = get_system_status()
    logger.info(f"💾 CUDA available: {system_status['cuda_available']}")
    logger.info(f"🔧 Models cached: {system_status['models_cached']}")
    
    # WebSocket-Callbacks einrichten
    setup_job_callbacks()
    
    logger.info("✅ AndioMediaStudio Complete API ready with full KI integration!")
    logger.info("📋 Available endpoints:")
    logger.info("   • /api/generate/image - Vollständige Bildgenerierung")
    logger.info("   • /api/generate/video - Vollständige Video-Generierung")
    logger.info("   • /api/models - Dynamische Modell-Registry")
    logger.info("   • /api/jobs/{job_id} - Job-Status mit WebSocket")
    logger.info("   • /ws/jobs/{job_id} - WebSocket Progress-Updates")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)