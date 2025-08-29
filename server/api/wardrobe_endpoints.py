# server/api/wardrobe_endpoints.py
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from server.pipelines.image_inpaint import SDInpaintEngine, InpaintRequest

router = APIRouter(prefix="/api", tags=["wardrobe"])

class Job:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.status = "queued"
        self.progress = 0.0
        self.message = ""
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.t0 = time.time()

JOBS: Dict[str, Job] = {}
ENGINE = SDInpaintEngine()  # load once

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

class JobOut(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    elapsed: float
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@router.post("/image/inpaint", response_model=JobOut)
async def start_inpaint(payload: InpaintIn):
    if not Path(payload.image_path).exists():
        raise HTTPException(status_code=404, detail="image not found")
    job = Job()
    JOBS[job.id] = job

    async def run():
        job.status = "running"
        job.message = "inpainting"
        try:
            def cb(step, total, _):
                job.progress = min(1.0, step / max(1, total))
            results = await asyncio.to_thread(
                ENGINE.run,
                InpaintRequest(**payload.dict()),
                cb,
            )
            job.status = "done"
            job.progress = 1.0
            job.message = "done"
            job.result = {"images": results}
        except Exception as e:
            job.status = "error"
            job.error = str(e)
            job.message = "error"

    asyncio.create_task(run())
    return JobOut(job_id=job.id, status=job.status, progress=job.progress, message=job.message, elapsed=0.0)

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
