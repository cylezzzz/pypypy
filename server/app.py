# FIXED: AndioMediaStudio - Minimaler funktionierender Kern
# Ersetzt das gesamte Chaos mit EINER funktionierenden Implementierung
from server.api import gallery_endpoints
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import torch
from PIL import Image
import io
import uuid
import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
import json

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# NEU: Router für Auto-Dateinamen Uploads (Wardrobe, Editor, Motion)
# Stelle sicher, dass die Dateien vorhanden sind:
#   server/utils/file_namer.py
#   server/api/wardrobe_endpoints.py
#   server/api/editor_endpoints.py
#   server/api/motion_endpoints.py
from server.api import wardrobe_endpoints, editor_endpoints, motion_endpoints
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Diffusers - DIE Basis für alles
try:
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableVideoDiffusionPipeline
    )
    DIFFUSERS_OK = True
except ImportError:
    DIFFUSERS_OK = False
    logging.error("❌ DIFFUSERS FEHLT - pip install diffusers accelerate")

# =============================================================================
# FIXED: Einheitliche Pfad-Behandlung
# =============================================================================

class ModelPaths:
    """EINE Quelle der Wahrheit für alle Pfade"""

    def __init__(self, base_dir: Path):
        self.base = Path(base_dir)
        self.models = self.base / "models"
        self.outputs = self.base / "outputs"
        self.web = self.base / "web"

        # Outputs sicherstellen
        (self.outputs / "images").mkdir(parents=True, exist_ok=True)
        (self.outputs / "videos").mkdir(parents=True, exist_ok=True)

    def find_sdxl_model(self) -> Path:
        """Finde SDXL-Modell - KLAR und SIMPEL"""
        candidates = [
            self.models / "stable-diffusion-xl-base-1.0",
            self.models / "image" / "stable-diffusion-xl-base-1.0",
            self.models / "sdxl-base",
            self.models / "image" / "sdxl-base"
        ]

        for path in candidates:
            if path.exists() and (path / "model_index.json").exists():
                return path

        raise FileNotFoundError(
            f"❌ KEIN SDXL-MODELL GEFUNDEN!\n"
            f"Erwartete Pfade: {[str(p) for p in candidates[:2]]}\n"
            f"Lade SDXL Base von HuggingFace herunter!"
        )

    def find_svd_model(self) -> Optional[Path]:
        """Finde SVD für Video - optional"""
        candidates = [
            self.models / "stable-video-diffusion-img2vid-xt",
            self.models / "video" / "stable-video-diffusion-img2vid-xt",
            self.models / "svd-xt"
        ]

        for path in candidates:
            if path.exists() and (path / "model_index.json").exists():
                return path
        return None

# =============================================================================
# FIXED: EIN Pipeline-Manager - macht was er soll
# =============================================================================

class FixedPipelineManager:
    """EIN Pipeline-Manager - KEINE Mocks, KEINE Komplexität"""

    def __init__(self, paths: ModelPaths):
        self.paths = paths
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Lazy-loaded Pipelines
        self._txt2img: Optional[StableDiffusionXLPipeline] = None
        self._img2img: Optional[StableDiffusionXLImg2ImgPipeline] = None
        self._img2vid: Optional[StableVideoDiffusionPipeline] = None

        logging.info(f"🔧 Pipeline Manager - Device: {self.device}")

    def get_txt2img(self) -> StableDiffusionXLPipeline:
        """Lade SDXL Text2Img - einmal, richtig"""
        if self._txt2img is None:
            if not DIFFUSERS_OK:
                raise RuntimeError("❌ Diffusers nicht installiert!")

            model_path = self.paths.find_sdxl_model()
            logging.info(f"📦 Lade SDXL: {model_path}")

            self._txt2img = StableDiffusionXLPipeline.from_pretrained(
                str(model_path),
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None
            )

            if self.device == "cuda":
                self._txt2img = self._txt2img.to("cuda")
                # Nur bewährte Optimierungen
                try:
                    self._txt2img.enable_vae_slicing()
                    self._txt2img.enable_attention_slicing()
                except Exception:
                    pass

            # ❌ KEIN Safety Checker - Kreative Freiheit!
            self._txt2img.safety_checker = None

            logging.info("✅ SDXL Text2Img geladen!")

        return self._txt2img

    def get_img2img(self) -> StableDiffusionXLImg2ImgPipeline:
        """Lade SDXL Img2Img"""
        if self._img2img is None:
            model_path = self.paths.find_sdxl_model()
            logging.info(f"📦 Lade SDXL Img2Img: {model_path}")

            self._img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                str(model_path),
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None
            )

            if self.device == "cuda":
                self._img2img = self._img2img.to("cuda")
                try:
                    self._img2img.enable_vae_slicing()
                    self._img2img.enable_attention_slicing()
                except Exception:
                    pass

            self._img2img.safety_checker = None
            logging.info("✅ SDXL Img2Img geladen!")

        return self._img2img

    def get_img2vid(self) -> Optional[StableVideoDiffusionPipeline]:
        """Lade SVD für Video - falls vorhanden"""
        if self._img2vid is None:
            svd_path = self.paths.find_svd_model()
            if not svd_path:
                logging.warning("⚠️ Kein SVD-Modell gefunden - Video-Features deaktiviert")
                return None

            logging.info(f"📦 Lade SVD: {svd_path}")

            self._img2vid = StableVideoDiffusionPipeline.from_pretrained(
                str(svd_path),
                torch_dtype=self.dtype,
                variant="fp16" if self.device == "cuda" else None
            )

            if self.device == "cuda":
                self._img2vid = self._img2vid.to("cuda")

            logging.info("✅ SVD Video-Pipeline geladen!")

        return self._img2vid

# =============================================================================
# FIXED: EIN Job-Manager - ECHT, nicht simuliert
# =============================================================================

class RealJob:
    """Echter Job - KEIN Mock"""

    def __init__(self, job_id: str, task: str):
        self.id = job_id
        self.task = task
        self.status = "queued"
        self.progress = 0.0
        self.message = ""
        self.result = None
        self.error = None
        self.created = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.id,
            "task": self.task,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "elapsed": round(time.time() - self.created, 1)
        }

class RealJobManager:
    """Echter Job-Manager - führt echte AI aus"""

    def __init__(self, pipeline_manager: FixedPipelineManager):
        self.pipes = pipeline_manager
        self.jobs: Dict[str, RealJob] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}

    def create_job(self, task: str) -> RealJob:
        job_id = f"{task}_{uuid.uuid4().hex[:8]}"
        job = RealJob(job_id, task)
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[RealJob]:
        return self.jobs.get(job_id)

    async def run_txt2img(self, job: RealJob, prompt: str, **params) -> None:
        """ECHTE Text2Img Generierung - KEIN Mock!"""
        try:
            job.status = "loading"
            job.message = "Pipeline wird geladen..."

            # Echte Pipeline laden
            pipe = self.pipes.get_txt2img()

            job.status = "generating"
            job.message = "Bild wird generiert..."
            job.progress = 0.3

            # ECHTE AI-Generierung
            generator = torch.Generator(device=self.pipes.device)
            if params.get('seed'):
                generator.manual_seed(int(params['seed']))

            with torch.autocast(device_type=self.pipes.device):
                result = pipe(
                    prompt=prompt,
                    negative_prompt=params.get('negative_prompt', ''),
                    width=params.get('width', 1024),
                    height=params.get('height', 1024),
                    num_inference_steps=params.get('steps', 25),
                    guidance_scale=params.get('guidance', 7.0),
                    generator=generator
                )

            job.progress = 0.9
            job.message = "Speichere Ergebnis..."

            # Echtes Bild speichern
            image = result.images[0]
            output_path = self.pipes.paths.outputs / "images" / f"{job.id}.png"
            image.save(output_path)

            # Erfolg!
            job.status = "completed"
            job.progress = 1.0
            job.message = "Fertig!"
            job.result = {
                "image_url": f"/outputs/images/{output_path.name}",
                "prompt": prompt,
                **params
            }

        except Exception as e:
            logging.error(f"❌ Job {job.id} failed: {e}")
            job.status = "error"
            job.error = str(e)
            job.message = f"Fehler: {str(e)}"

    async def start_job(self, task: str, **params) -> RealJob:
        """Starte echten Job"""
        job = self.create_job(task)

        if task == "txt2img":
            task_coro = self.run_txt2img(job, **params)
        else:
            raise ValueError(f"Task '{task}' noch nicht implementiert")

        # Starte als Background-Task
        self.active_tasks[job.id] = asyncio.create_task(task_coro)

        return job

# =============================================================================
# FIXED: EIN klarer API-Router
# =============================================================================

# Setup
ROOT = Path(__file__).resolve().parents[1]
paths = ModelPaths(ROOT)
pipeline_manager = FixedPipelineManager(paths)
job_manager = RealJobManager(pipeline_manager)

# FastAPI App
app = FastAPI(title="AndioMediaStudio - FIXED", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---- Router registrieren (Wardrobe/Editor/Motion Upload mit Auto-Dateinamen)
app.include_router(wardrobe_endpoints.router)
app.include_router(editor_endpoints.router)
app.include_router(motion_endpoints.router)
app.include_router(gallery_endpoints.router)

# =============================================================================
# API Endpoints - KLAR und FUNKTIONAL
# =============================================================================

@app.get("/")
async def root():
    return {"status": "✅ AndioMediaStudio FIXED läuft!", "version": "0.1.0"}

@app.get("/api/health")
async def health():
    """System-Gesundheit prüfen"""
    try:
        sdxl_path = paths.find_sdxl_model()
        sdxl_ok = True
    except FileNotFoundError:
        sdxl_path = None
        sdxl_ok = False

    svd_path = paths.find_svd_model()

    return {
        "status": "healthy" if sdxl_ok else "degraded",
        "gpu_available": torch.cuda.is_available(),
        "diffusers_ok": DIFFUSERS_OK,
        "models": {
            "sdxl": {"available": sdxl_ok, "path": str(sdxl_path) if sdxl_path else None},
            "svd": {"available": bool(svd_path), "path": str(svd_path) if svd_path else None}
        }
    }

@app.post("/api/txt2img")
async def txt2img(
    prompt: str = Form(...),
    negative_prompt: str = Form("low quality, blurry"),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(25),
    guidance: float = Form(7.0),
    seed: Optional[int] = Form(None)
):
    """ECHTE Text-zu-Bild Generierung"""
    try:
        job = await job_manager.start_job(
            "txt2img",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed
        )

        return {
            "success": True,
            "job_id": job.id,
            "message": "Bildgenerierung gestartet",
            "estimated_time": f"{steps * 0.3:.1f}s"
        }

    except Exception as e:
        logging.error(f"❌ txt2img failed: {e}")
        raise HTTPException(500, f"Generierung fehlgeschlagen: {str(e)}")

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Job-Status abfragen"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job nicht gefunden")

    return job.to_dict()

@app.get("/api/jobs")
async def list_jobs():
    """Alle Jobs auflisten"""
    return {
        "jobs": [job.to_dict() for job in job_manager.jobs.values()],
        "total": len(job_manager.jobs)
    }

# =============================================================================
# Static File Serving
# =============================================================================

# Outputs verfügbar machen
app.mount("/outputs", StaticFiles(directory=str(paths.outputs)), name="outputs")

# Web-Interface (falls vorhanden)
if paths.web.exists():
    app.mount("/", StaticFiles(directory=str(paths.web), html=True), name="web")

# =============================================================================
# Startup-Validierung
# =============================================================================

@app.on_event("startup")
async def startup():
    """Beim Start: Kritische Validierung"""
    logging.basicConfig(level=logging.INFO)

    logging.info("🚀 AndioMediaStudio FIXED startet...")

    # Kritische Prüfungen
    if not DIFFUSERS_OK:
        logging.error("❌ DIFFUSERS FEHLT! pip install diffusers accelerate")
        return

    try:
        sdxl_path = paths.find_sdxl_model()
        logging.info(f"✅ SDXL gefunden: {sdxl_path}")
    except FileNotFoundError as e:
        logging.error(str(e))
        logging.error("💡 Download: huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0")

    svd_path = paths.find_svd_model()
    if svd_path:
        logging.info(f"✅ SVD gefunden: {svd_path}")
    else:
        logging.warning("⚠️ Kein SVD - Video-Features deaktiviert")

    logging.info("✅ AndioMediaStudio FIXED bereit!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000, log_level="info")
