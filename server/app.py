# server/app.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

# ---------------------------------
# Setup
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"
ASSETS_DIR = WEB_DIR / "assets"
OUTPUTS_DIR = BASE_DIR / "outputs"

logger = logging.getLogger("AndioMediaStudio")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

app = FastAPI(title="AndioMediaStudio API", version="0.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------
# ⚠️ Wichtig: Keine Root-Mounts!
# StaticFiles NICHT auf "/" mounten, sonst verschluckt es /api/* → 404.
# ---------------------------------
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
else:
    logger.warning(f"⚠️  ASSETS_DIR fehlt: {ASSETS_DIR}")

# Optional: Ausgaben lesbar machen
if (OUTPUTS_DIR / "images").exists():
    app.mount("/outputs/images", StaticFiles(directory=str(OUTPUTS_DIR / "images")), name="outputs-images")
if (OUTPUTS_DIR / "videos").exists():
    app.mount("/outputs/videos", StaticFiles(directory=str(OUTPUTS_DIR / "videos")), name="outputs-videos")

# ---------------------------------
# Orchestrator Import (Fallback)
# ---------------------------------
try:
    from server.ai_orchestrator import (
        get_orchestrator, generate_image, generate_video, get_job_status,
        list_jobs, cancel_job, get_system_status
    )  # type: ignore
    logger.info("✅ Orchestrator: server.ai_orchestrator aktiv")
except Exception as e:
    logger.warning(f"⚠️  server.ai_orchestrator nicht gefunden/nutzbar: {e}")
    from server.pipelines.ai_orchestrator import (
        get_orchestrator, generate_image, generate_video, get_job_status,
        list_jobs, cancel_job, get_system_status
    )
    logger.info("✅ Orchestrator: server.pipelines.ai_orchestrator aktiv")

# ---------------------------------
# Optional: Enhanced Endpoints
# ---------------------------------
try:
    from server.api.enhanced_endpoints import router as enhanced_router
    app.include_router(enhanced_router)
    logger.info("✅ Enhanced endpoints eingebunden")
except Exception as e:
    logger.warning(f"Enhanced endpoints nicht geladen: {e}")

# Intelligent Model Selector (falls vorhanden)
try:
    from server.orchestrator.intelligent_model_selector import router as ims_router
    app.include_router(ims_router)
    logger.info("✅ Intelligent model selector endpoints eingebunden")
except Exception as e:
    logger.warning(f"Intelligent model selector nicht geladen: {e}")

# ---------------------------------
# Core endpoints
# ---------------------------------
from pydantic import BaseModel

class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: int = 28
    guidance: float = 7.5
    width: int = 768
    height: int = 768
    model: Optional[str] = None
    nsfw: bool = False
    seed: Optional[int] = None

class VideoGenRequest(BaseModel):
    prompt: str
    num_frames: int = 25
    fps: int = 10
    width: int = 512
    height: int = 512
    model: Optional[str] = None
    seed: Optional[int] = None

@app.get("/api/ping")
def ping():
    return {"ok": True}

@app.post("/api/generate/image")
async def api_generate_image(req: Txt2ImgRequest):
    job = await generate_image(req.model, req.dict())
    return job

@app.post("/api/generate/video")
async def api_generate_video(req: VideoGenRequest):
    job = await generate_video(req.model, req.dict())
    return job

@app.get("/api/jobs/{job_id}")
async def api_job_status(job_id: str):
    return await get_job_status(job_id)

@app.get("/api/jobs")
async def api_list_jobs():
    return await list_jobs()

@app.delete("/api/jobs/{job_id}")
async def api_cancel_job(job_id: str):
    return await cancel_job(job_id)

@app.get("/api/system")
async def api_system():
    return await get_system_status()

# Optional: einfache /api/catalog, wenn Registry verfügbar ist
try:
    from server.models_registry import get_registry  # type: ignore
    @app.get("/api/catalog")
    def api_catalog():
        reg = get_registry()
        return {"models": reg.list_models() if hasattr(reg, "list_models") else []}
except Exception as e:
    logger.warning(f"/api/catalog nicht aktiviert: {e}")

# ---------------------------------
# Frontend-Seiten sicher ausliefern
# ---------------------------------
def file_or_404(rel: str):
    p = WEB_DIR / rel
    if p.exists():
        return FileResponse(str(p))
    return JSONResponse({"error": "file not found"}, status_code=404)

@app.get("/")
def index():
    # Liefert index.html ohne Root-Static-Mount
    return file_or_404("index.html")

# Convenience-Routen für deine bestehenden Seiten-Namen
for page in ["images.html", "video-gen.html", "wandrobe.html", "motion.html", "gallery.html", "editor.html", "catalog.html", "store.html"]:
    route_path = f"/{page}"
    async def serve_page(page=page):
        return file_or_404(page)
    app.add_api_route(route_path, serve_page, methods=["GET"]) 
