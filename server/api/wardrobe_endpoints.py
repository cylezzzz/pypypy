# server/api/wardrobe_endpoints.py
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

# Inpainting-Engine (flat models support)
from server.pipelines.image_inpaint import SDInpaintEngine, InpaintRequest
# Wardrobe / Clothing-Editor
from server.pipelines.clothing_editor import ClothingEditorAPI

# -----------------------------------------------------------------------------
# Router & Globals
# -----------------------------------------------------------------------------

# Wichtig: Prefix passt zum Frontend (/api/wardrobe/...)
router = APIRouter(prefix="/api/wardrobe", tags=["wardrobe"])

# Projektwurzel (…/pypygennew)
BASE_DIR = Path(__file__).resolve().parents[2]

# Engines (Lazy-Konstruktion hier explizit angelegt)
ENGINE = SDInpaintEngine(prefer_gpu=True)
CLOTHING = ClothingEditorAPI(BASE_DIR)

# -----------------------------------------------------------------------------
# Job-Struktur (für lange Läufe / Polling)
# -----------------------------------------------------------------------------

class Job:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.status: str = "queued"     # queued | running | done | error
        self.progress: float = 0.0      # 0..1
        self.message: str = ""
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.t0: float = time.time()

JOBS: Dict[str, Job] = {}

# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------

class InpaintIn(BaseModel):
    image_path: str
    mask_path: Optional[str] = None
    mask_b64: Optional[str] = None
    prompt: str = ""
    negative_prompt: str = "blurry, artifacts, text, watermark"
    steps: int = Field(default=30, ge=1, le=150)
    guidance: float = Field(default=7.5, ge=0, le=30)
    strength: float = Field(default=0.85, ge=0.05, le=1.0)
    seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    model: Optional[str] = None  # Name/Datei im flachen models/-Ordner

    @validator("image_path")
    def _validate_image_exists(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"image not found: {v}")
        return v

class JobOut(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    elapsed: float
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# --- Wardrobe: Remove / Change ------------------------------------------------

class RemoveIn(BaseModel):
    image: str                          # relativer Pfad unter BASE_DIR
    clothing_type: str = "all"          # "shirt" | "pants" | "dress" | "all" ...
    preserve_anatomy: bool = True

    @validator("image")
    def _validate_image_exists(cls, v: str) -> str:
        p = (BASE_DIR / v)
        if not p.exists():
            raise ValueError(f"image not found (relative to BASE_DIR): {p}")
        return v

class ChangeIn(BaseModel):
    image: str
    clothing_type: str
    prompt: str
    style: str = "realistic"

    @validator("image")
    def _validate_image_exists(cls, v: str) -> str:
        p = (BASE_DIR / v)
        if not p.exists():
            raise ValueError(f"image not found (relative to BASE_DIR): {p}")
        return v

# -----------------------------------------------------------------------------
# Inpainting (bestehend) – Jobbasiert
# -----------------------------------------------------------------------------

@router.post("/image/inpaint", response_model=JobOut)
async def start_inpaint(payload: InpaintIn):
    """
    Startet einen Inpaint-Job.
    - Lädt das Modell lazy und ausschließlich lokal (flat oder legacy).
    - Wenn ein Single-File-Checkpoint (z. B. sd15.safetensors) kein echtes Inpaint kann,
      wird die Maske ignoriert (geloggt) und es verhält sich wie normales Img2Img.
    """
    job = Job()
    JOBS[job.id] = job

    async def run():
        job.status = "running"
        job.message = "initializing"

        # progress-callback -> aktualisiert Job
        def _cb(step: int, total: int, msg: str):
            try:
                job.progress = min(1.0, step / max(1, total))
                job.message = msg or "running"
            except Exception:
                pass

        try:
            job.message = "loading model / preparing"
            # Blockierende diffusers-Calls in Thread ausführen
            results: List[str] = await asyncio.to_thread(
                ENGINE.run,
                InpaintRequest(
                    image_path=payload.image_path,
                    mask_path=payload.mask_path,
                    mask_b64=payload.mask_b64,
                    prompt=payload.prompt,
                    negative_prompt=payload.negative_prompt,
                    steps=payload.steps,
                    guidance=payload.guidance,
                    strength=payload.strength,
                    seed=payload.seed,
                    width=payload.width,
                    height=payload.height,
                    model=payload.model,  # flat models/ support
                ),
                _cb,
            )

            job.status = "done"
            job.progress = 1.0
            job.message = "done"
            job.result = {"images": results}

        except FileNotFoundError as e:
            job.status = "error"
            job.error = f"{e}"
            job.message = "model not found"
        except Exception as e:
            job.status = "error"
            job.error = str(e)
            job.message = "error"

    asyncio.create_task(run())
    return JobOut(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        elapsed=0.0,
    )

@router.get("/image/inpaint/{job_id}", response_model=JobOut)
async def get_inpaint(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JobOut(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        elapsed=time.time() - job.t0,
        results=job.result,
        error=job.error,
    )

# -----------------------------------------------------------------------------
# Wardrobe: Remove / Change – SYNCHRON (direkte Antwort)
# -----------------------------------------------------------------------------

@router.post("/remove")
async def remove_sync(payload: RemoveIn):
    """
    Synchronous Wardrobe Remove:
    - Erwartet { image, clothing_type, preserve_anatomy }
    - Liefert direkt die Pfade (ohne Job-Polling).
    """
    result = await CLOTHING.remove_clothing_api(
        image_path=payload.image,
        clothing_type=payload.clothing_type,
        preserve_anatomy=payload.preserve_anatomy,
    )
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "remove failed"))
    return result

@router.post("/change")
async def change_sync(payload: ChangeIn):
    """
    Synchronous Wardrobe Change:
    - Erwartet { image, clothing_type, prompt, style }
    - Liefert direkt den Output-Pfad (ohne Job-Polling).
    """
    result = await CLOTHING.change_clothing_api(
        image_path=payload.image,
        new_clothing_prompt=payload.prompt,
        clothing_type=payload.clothing_type,
        style=payload.style,
    )
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "change failed"))
    return result

# -----------------------------------------------------------------------------
# Wardrobe: Remove / Change – JOB-Varianten (optional)
# -----------------------------------------------------------------------------

@router.post("/remove/job", response_model=JobOut)
async def remove_job(payload: RemoveIn):
    job = Job()
    JOBS[job.id] = job

    async def run():
        job.status = "running"
        job.message = "wardrobe/remove"
        try:
            result = await CLOTHING.remove_clothing_api(
                image_path=payload.image,
                clothing_type=payload.clothing_type,
                preserve_anatomy=payload.preserve_anatomy,
            )
            if not result.get("success"):
                raise RuntimeError(result.get("error", "remove failed"))

            job.status = "done"
            job.progress = 1.0
            job.message = "done"
            job.result = result  # enthält output_paths/masken usw.

        except Exception as e:
            job.status = "error"
            job.error = str(e)
            job.message = "error"

    asyncio.create_task(run())
    return JobOut(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        elapsed=0.0,
    )

@router.post("/change/job", response_model=JobOut)
async def change_job(payload: ChangeIn):
    job = Job()
    JOBS[job.id] = job

    async def run():
        job.status = "running"
        job.message = "wardrobe/change"
        try:
            result = await CLOTHING.change_clothing_api(
                image_path=payload.image,
                new_clothing_prompt=payload.prompt,
                clothing_type=payload.clothing_type,
                style=payload.style,
            )
            if not result.get("success"):
                raise RuntimeError(result.get("error", "change failed"))

            job.status = "done"
            job.progress = 1.0
            job.message = "done"
            job.result = result

        except Exception as e:
            job.status = "error"
            job.error = str(e)
            job.message = "error"

    asyncio.create_task(run())
    return JobOut(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        elapsed=0.0,
    )

@router.get("/job/{job_id}", response_model=JobOut)
async def get_wardrobe_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JobOut(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        elapsed=time.time() - job.t0,
        results=job.result,
        error=job.error,
    )
