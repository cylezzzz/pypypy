# server/app.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# ---------------------------------
# Setup
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"
ASSETS_DIR = WEB_DIR / "assets"

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
# Static mounts
# ---------------------------------
# / -> serve web/
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web-root")
else:
    logger.warning(f"⚠️  WEB_DIR fehlt: {WEB_DIR}")

# /assets -> serve web/assets
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
else:
    logger.warning(f"⚠️  ASSETS_DIR fehlt: {ASSETS_DIR}")

# ---------------------------------
# Orchestrator Import (robuster Fallback)
# ---------------------------------
orchestrator_module = None
try:
    from server.ai_orchestrator import (
        get_orchestrator, generate_image, generate_video, get_job_status,
        list_jobs, cancel_job, get_system_status
    )  # type: ignore
    logger.info("✅ Orchestrator: server.ai_orchestrator aktiv")
except Exception as e:
    logger.warning(f"⚠️  server.ai_orchestrator nicht gefunden/nutzbar: {e}")
    from server.pipelines.ai_orchestrator import (  # Fallback
        get_orchestrator, generate_image, generate_video, get_job_status,
        list_jobs, cancel_job, get_system_status
    )
    logger.info("✅ Orchestrator: server.pipelines.ai_orchestrator aktiv")

# ---------------------------------
# Optional: Enhanced Endpoints
# ---------------------------------
try:
    from server.api.enhanced_endpoints import router as enhanced_router  # nutzt sniff_image
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

# Core endpoints
from fastapi import Body
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

# Root: index.html ausliefern
@app.get("/")
def root():
    # StaticFiles mount liefert index.html automatisch; hier nur als Fallback JSON
    return JSONResponse({"message": "AndioMediaStudio API running"})
