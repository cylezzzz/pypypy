# server/api/enhanced_endpoints.py
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field, field_validator

from server.pipelines.enhanced_txt2img import EnhancedTxt2Img, Txt2ImgRequest
from server.config.model_config import ModelConfig
from server.utils.file_validator import ensure_dir, validate_image_upload, sniff_image
from server.utils.video_processor import ken_burns_from_image

router = APIRouter(prefix="/api", tags=["andio-media-studio"])

# ----------------------------- Job Manager --------------------------------------

class JobStatus(str):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class Job:
    def __init__(self, job_id: str) -> None:
        self.id = job_id
        self.status: str = JobStatus.QUEUED
        self.progress: float = 0.0
        self.message: str = ""
        self.results: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.t0: float = time.time()
        self.t1: Optional[float] = None
        self._listeners: set[WebSocket] = set()

    async def broadcast(self):
        payload = json.dumps({
            "job_id": self.id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "elapsed": round(time.time() - self.t0, 3),
        })
        dead = []
        for ws in list(self._listeners):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._listeners.discard(ws)


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._engine = EnhancedTxt2Img()
        self._cfg = ModelConfig()

    # --- jobs ---
    def create(self) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id)
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return job

    def list_jobs(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for j in self._jobs.values():
            out.append({
                "job_id": j.id,
                "status": j.status,
                "progress": j.progress,
                "message": j.message,
                "elapsed": round((j.t1 or time.time()) - j.t0, 3),
                "has_result": j.results is not None,
                "error": j.error,
            })
        out.sort(key=lambda x: x["elapsed"], reverse=True)
        return out

    async def run_txt2img(self, job: Job, req: Txt2ImgRequest):
        job.status = JobStatus.RUNNING
        job.message = "starting"
        await job.broadcast()

        def cb(step: int, total: int, _elapsed: float):
            job.progress = min(1.0, float(step) / max(1, total))
            job.message = f"{step}/{total} images"
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(job.broadcast())
            except RuntimeError:
                pass

        try:
            results = await asyncio.to_thread(self._engine.generate, req, cb)
            job.status = JobStatus.DONE
            job.message = "done"
            job.results = {"results": results}
            job.progress = 1.0
            job.t1 = time.time()
            await job.broadcast()
        except Exception as e:
            job.status = JobStatus.ERROR
            job.error = str(e)
            job.message = "error"
            job.t1 = time.time()
            await job.broadcast()

    async def run_kenburns(self, job: Job, image_path: Path, w: int, h: int, duration: float, fps: int):
        job.status = JobStatus.RUNNING
        job.message = "processing"
        await job.broadcast()
        try:
            vid_dir = Path(__file__).resolve().parents[2] / "outputs" / "videos"
            ensure_dir(vid_dir)
            out_path = vid_dir / f"kenburns_{uuid.uuid4().hex[:8]}.mp4"
            # process in thread
            def _work():
                ken_burns_from_image(image_path, out_path, duration_sec=duration, fps=fps, out_w=w, out_h=h)
                return str(out_path)
            result_path = await asyncio.to_thread(_work)
            job.status = JobStatus.DONE
            job.progress = 1.0
            job.message = "done"
            job.t1 = time.time()
            job.results = {"video": result_path.replace("\\", "/")}
            await job.broadcast()
        except Exception as e:
            job.status = JobStatus.ERROR
            job.error = str(e)
            job.message = "error"
            job.t1 = time.time()
            await job.broadcast()

    # --- models ---
    def _scan_models_dir(self, base: Path) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not base.exists():
            return items
        for modality in sorted([d for d in base.iterdir() if d.is_dir()]):
            for model_dir in sorted([d for d in modality.iterdir() if d.is_dir()]):
                size_bytes = 0
                try:
                    for p in model_dir.rglob("*"):
                        if p.is_file():
                            size_bytes += p.stat().st_size
                except Exception:
                    pass
                items.append({
                    "modality": modality.name,
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "size_bytes": size_bytes,
                })
        return items

    def list_models(self) -> Dict[str, Any]:
        cfg = self._cfg
        candidates: List[Path] = []
        if cfg.models_dir:
            candidates.append(Path(cfg.models_dir))
        candidates.append(Path(__file__).resolve().parents[2] / "models")
        if os.name == "nt":
            candidates.append(Path("X:/pypygennew/models"))
        discovered: List[Dict[str, Any]] = []
        for c in candidates:
            try:
                discovered.extend(self._scan_models_dir(c))
            except Exception:
                continue
        current_txt2img = cfg.model_id
        return {
            "txt2img_current": current_txt2img,
            "discovered": discovered,
            "device": cfg.resolve_device(),
            "dtype": str(cfg.dtype),
        }


JOBS = JobManager()

# ----------------------------- Schemas ------------------------------------------

class Txt2ImgIn(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str = Field(default="worst quality, low quality, jpeg artifacts, watermark, signature, text")
    width: int = 512
    height: int = 512
    steps: int = Field(default=30, ge=1, le=150)
    guidance: float = Field(default=7.5, ge=0, le=30)
    num_images: int = Field(default=1, ge=1, le=16)
    batch_size: int = Field(default=1, ge=1, le=8)
    seed: Optional[int] = None
    safety_checker: bool = True

    @field_validator("width", "height")
    @classmethod
    def _snap_to_valid(cls, v: int) -> int:
        v = max(64, min(2048, v))
        return (v // 8) * 8


class JobOut(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    elapsed: float
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ----------------------------- Endpoints ----------------------------------------

@router.get("/ping")
async def ping():
    return {"ok": True, "time": time.time()}

@router.get("/health")
async def health():
    return {"ok": True, "service": "andio-media-studio", "time": time.time()}

@router.get("/models")
async def list_models():
    return JOBS.list_models()

@router.get("/jobs")
async def list_jobs():
    return {"jobs": JOBS.list_jobs()}

@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: str):
    try:
        job = JOBS.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    return JobOut(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        elapsed=round(time.time() - job.t0, 3),
        results=job.results,
        error=job.error,
    )

# ---- Upload (images only) ------------------------------------------------------
@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    data = await file.read()
    ok, msg = validate_image_upload(file.filename, len(data))
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    ws = Path(__file__).resolve().parents[2] / "workspace" / "uploads"
    ensure_dir(ws)
    dst = ws / f"{uuid.uuid4().hex[:10]}_{file.filename}"
    with open(dst, "wb") as f:
        f.write(data)

    if not sniff_image(dst):
        try:
            dst.unlink(missing_ok=True)  # type: ignore
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Upload is not a valid image")
    return {"ok": True, "path": str(dst).replace("\\", "/"), "name": file.filename, "size": len(data)}

# ---- Txt2Img -------------------------------------------------------------------
@router.post("/txt2img", response_model=JobOut)
async def start_txt2img(payload: Txt2ImgIn):
    job = JOBS.create()
    req = Txt2ImgRequest(
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        width=payload.width,
        height=payload.height,
        num_inference_steps=payload.steps,
        guidance_scale=payload.guidance,
        num_images=payload.num_images,
        batch_size=payload.batch_size,
        seed=payload.seed,
        safety_checker=payload.safety_checker,
    )
    asyncio.create_task(JOBS.run_txt2img(job, req))
    return JobOut(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        elapsed=0.0,
        results=None,
        error=None,
    )

# ---- Video: Ken Burns from image -----------------------------------------------
class KenBurnsIn(BaseModel):
    image_path: str
    width: int = 1280
    height: int = 720
    duration: float = Field(default=5.0, gt=0, le=30)
    fps: int = Field(default=30, ge=10, le=60)

@router.post("/video/kenburns", response_model=JobOut)
async def start_kenburns(payload: KenBurnsIn):
    p = Path(payload.image_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="image not found")
    job = JOBS.create()
    asyncio.create_task(JOBS.run_kenburns(job, p, payload.width, payload.height, payload.duration, payload.fps))
    return JobOut(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        elapsed=0.0,
        results=None,
        error=None,
    )

@router.websocket("/ws/progress/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        job = JOBS.get(job_id)
    except KeyError:
        await websocket.send_text(json.dumps({"error": "job not found", "job_id": job_id}))
        await websocket.close()
        return
    job._listeners.add(websocket)
    await job.broadcast()
    try:
        while True:
            if job.status in (JobStatus.DONE, JobStatus.ERROR):
                await job.broadcast()
                await asyncio.sleep(0.1)
                break
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                await asyncio.sleep(0.25)
    finally:
        job._listeners.discard(websocket)
        try:
            await websocket.close()
        except Exception:
            pass
