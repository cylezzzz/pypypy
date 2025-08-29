# server/api/wardrobe_endpoints.py
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

# Unsere Engine mit Flat-Models-Support
from server.pipelines.image_inpaint import SDInpaintEngine, InpaintRequest

router = APIRouter(prefix="/api", tags=["wardrobe"])

# ----------------------------- Job-Struktur ----------------------------------

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

# Engine: lazy load, nur lokal
ENGINE = SDInpaintEngine(prefer_gpu=True)

# --------------------------- Request / Response ------------------------------

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
    model: Optional[str] = None  # <- NEU: Name/Datei im flachen models/ Ordner

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

# ------------------------------- Endpoints -----------------------------------

@router.post("/image/inpaint", response_model=JobOut)
async def start_inpaint(payload: InpaintIn):
    """
    Startet einen Inpaint-Job.
    - Lädt das Modell lazy und ausschließlich aus X:/pypygennew/models (flat oder legacy).
    - Wenn ein Single-File-Checkpoint (z. B. sd15.safetensors) kein echtes Inpaint kann,
      wird die Maske ignoriert (wird geloggt) und es verhält sich wie normales Img2Img.
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
            # Ausführung im Thread (blockierende diffusers-Calls)
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
                    model=payload.model,  # <- wichtig: flat models/ support
                ),
                _cb,
            )

            job.status = "done"
            job.progress = 1.0
            job.message = "done"
            job.result = {"images": results}

        except FileNotFoundError as e:
            # z. B. wenn kein inpaint-fähiges Modell lokal gefunden wird
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
