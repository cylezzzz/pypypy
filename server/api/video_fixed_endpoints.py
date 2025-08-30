# server/api/video_fixed_endpoints.py
"""
REPARIERTE Video-API für echte Video-Generierung
- IMG2VID mit SVD 
- TXT2VID mit SDXL + SVD
- Echte MP4/GIF Ausgabe statt Thumbnails
"""

from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import io
import logging
import asyncio
import uuid
import time

# Import der reparierten Pipeline
from server.pipelines.video_svd_fixed import FixedSVDPipeline
from PIL import Image

router = APIRouter(prefix="/api/video", tags=["video-fixed"])
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------#
# Pfade & Setup
# -----------------------------------------------------------------------------#
ROOT = Path(__file__).resolve().parents[2]
OUT_VID = ROOT / "outputs" / "videos"
OUT_VID.mkdir(parents=True, exist_ok=True)

# Globale Pipeline-Instanz
_svd_pipeline: Optional[FixedSVDPipeline] = None

def get_pipeline() -> FixedSVDPipeline:
    """Hole oder erstelle SVD Pipeline"""
    global _svd_pipeline
    if _svd_pipeline is None:
        _svd_pipeline = FixedSVDPipeline(ROOT)
    return _svd_pipeline

# -----------------------------------------------------------------------------#
# Job Management für Async Processing
# -----------------------------------------------------------------------------#
class VideoJob:
    def __init__(self, job_id: str, job_type: str):
        self.job_id = job_id
        self.job_type = job_type
        self.status = "queued"  # queued, processing, completed, failed
        self.progress = 0.0
        self.message = ""
        self.result = None
        self.error = None
        self.created_at = time.time()

# In-Memory Job Storage (in Production: Redis/Database)
active_jobs: Dict[str, VideoJob] = {}

# -----------------------------------------------------------------------------#
# API Models
# -----------------------------------------------------------------------------#
class Img2VideoRequest(BaseModel):
    motion_strength: float = Field(0.8, ge=0.1, le=1.0, description="Bewegungsintensität")
    num_frames: int = Field(14, ge=8, le=25, description="Anzahl Frames")
    fps: int = Field(8, ge=6, le=24, description="Frames per Second")
    format: str = Field("mp4", description="Ausgabeformat: mp4 oder gif")

class Txt2VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text-Beschreibung")
    negative_prompt: str = Field("low quality, blurry, bad anatomy", description="Negative Prompts")
    num_frames: int = Field(14, ge=8, le=25, description="Anzahl Frames")
    fps: int = Field(8, ge=6, le=24, description="Frames per Second")
    width: int = Field(1024, ge=512, le=1536, description="Breite (Vielfaches von 64)")
    height: int = Field(576, ge=512, le=1536, description="Höhe (Vielfaches von 64)")
    motion_strength: float = Field(0.8, ge=0.1, le=1.0, description="Bewegungsintensität")
    steps: int = Field(30, ge=10, le=50, description="Diffusion Steps für Initial Frame")
    guidance: float = Field(7.5, ge=1.0, le=20.0, description="Guidance Scale")
    format: str = Field("mp4", description="Ausgabeformat: mp4 oder gif")

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    elapsed_seconds: float

# -----------------------------------------------------------------------------#
# Helper Functions
# -----------------------------------------------------------------------------#
def create_job(job_type: str) -> VideoJob:
    """Erstelle neuen Video-Job"""
    job_id = f"{job_type}_{uuid.uuid4().hex[:8]}"
    job = VideoJob(job_id, job_type)
    active_jobs[job_id] = job
    return job

def save_video_file(frames, format: str = "mp4", fps: int = 8) -> str:
    """Speichere Video und gib relative URL zurück"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format.lower() == "mp4":
        filename = f"video_{timestamp}.mp4"
        output_path = OUT_VID / filename
        
        try:
            pipeline = get_pipeline()
            pipeline.save_video_mp4(frames, output_path, fps)
        except Exception as e:
            logger.warning(f"MP4 save failed, using GIF: {e}")
            filename = f"video_{timestamp}.gif"
            output_path = OUT_VID / filename
            pipeline.save_video_gif(frames, output_path, fps)
    else:
        filename = f"video_{timestamp}.gif"
        output_path = OUT_VID / filename
        pipeline = get_pipeline()
        pipeline.save_video_gif(frames, output_path, fps)
    
    # Gib relative URL zurück
    return f"/outputs/videos/{filename}"

# -----------------------------------------------------------------------------#
# Background Tasks
# -----------------------------------------------------------------------------#
async def process_img2video(job: VideoJob, image: Image.Image, params: Img2VideoRequest):
    """Background Task für Image-to-Video"""
    try:
        job.status = "processing"
        job.message = "Loading video pipeline..."
        job.progress = 0.1
        
        pipeline = get_pipeline()
        
        job.message = "Generating video frames..."
        job.progress = 0.3
        
        # Konvertiere Motion Strength zu Motion Bucket ID
        motion_bucket_id = int(params.motion_strength * 255)
        
        # Generiere Video
        frames = await asyncio.to_thread(
            pipeline.img2video,
            image=image,
            num_frames=params.num_frames,
            motion_bucket_id=motion_bucket_id,
            fps=params.fps
        )
        
        job.message = "Saving video file..."
        job.progress = 0.9
        
        # Speichere Video
        video_url = save_video_file(frames, params.format, params.fps)
        
        job.status = "completed"
        job.progress = 1.0
        job.message = "Video generation completed!"
        job.result = {
            "video_url": video_url,
            "frames": len(frames),
            "duration_seconds": len(frames) / params.fps,
            "format": params.format
        }
        
    except Exception as e:
        logger.error(f"IMG2VID failed for job {job.job_id}: {e}")
        job.status = "failed"
        job.error = str(e)
        job.message = f"Error: {str(e)}"

async def process_txt2video(job: VideoJob, params: Txt2VideoRequest):
    """Background Task für Text-to-Video"""
    try:
        job.status = "processing"
        job.message = "Loading pipelines..."
        job.progress = 0.1
        
        pipeline = get_pipeline()
        
        job.message = "Generating initial frame with SDXL..."
        job.progress = 0.2
        
        # Konvertiere Motion Strength zu Motion Bucket ID
        motion_bucket_id = int(params.motion_strength * 255)
        
        # Generiere Video
        frames = await asyncio.to_thread(
            pipeline.txt2video,
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            num_frames=params.num_frames,
            width=params.width,
            height=params.height,
            motion_bucket_id=motion_bucket_id,
            steps=params.steps,
            guidance=params.guidance
        )
        
        job.message = "Saving video file..."
        job.progress = 0.9
        
        # Speichere Video
        video_url = save_video_file(frames, params.format, params.fps)
        
        job.status = "completed"
        job.progress = 1.0
        job.message = "Text2Video completed!"
        job.result = {
            "video_url": video_url,
            "frames": len(frames),
            "duration_seconds": len(frames) / params.fps,
            "format": params.format,
            "prompt": params.prompt
        }
        
    except Exception as e:
        logger.error(f"TXT2VID failed for job {job.job_id}: {e}")
        job.status = "failed"
        job.error = str(e)
        job.message = f"Error: {str(e)}"

# -----------------------------------------------------------------------------#
# API Endpoints
# -----------------------------------------------------------------------------#

@router.post("/img2vid")
async def start_img2video(
    image: UploadFile = File(..., description="Input image"),
    motion_strength: float = Form(0.8),
    num_frames: int = Form(14),
    fps: int = Form(8),
    format: str = Form("mp4")
):
    """
    🎬 REPARIERT: Image-to-Video mit SVD
    Generiert ECHTES Video aus einem Bild!
    """
    try:
        # Validierung
        if not image.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        
        # Lade Bild
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Erstelle Job
        job = create_job("img2vid")
        
        # Parameter
        params = Img2VideoRequest(
            motion_strength=motion_strength,
            num_frames=num_frames,
            fps=fps,
            format=format
        )
        
        # Starte Background Processing
        asyncio.create_task(process_img2video(job, pil_image, params))
        
        return {
            "success": True,
            "job_id": job.job_id,
            "message": "Image-to-Video generation started",
            "estimated_time_seconds": num_frames * 2,  # Grobe Schätzung
            "status_url": f"/api/video/status/{job.job_id}"
        }
        
    except Exception as e:
        logger.error(f"IMG2VID start failed: {e}")
        raise HTTPException(500, f"Failed to start image-to-video: {str(e)}")

@router.post("/txt2vid") 
async def start_txt2video(params: Txt2VideoRequest):
    """
    🎬 REPARIERT: Text-to-Video mit SDXL + SVD
    Generiert ECHTES Video aus Text!
    """
    try:
        # Validierung
        if not params.prompt.strip():
            raise HTTPException(400, "Prompt cannot be empty")
        
        # Dimension Validation (muss Vielfaches von 64 sein)
        if params.width % 64 != 0 or params.height % 64 != 0:
            raise HTTPException(400, "Width and height must be multiples of 64")
        
        # Erstelle Job
        job = create_job("txt2vid")
        
        # Starte Background Processing
        asyncio.create_task(process_txt2video(job, params))
        
        return {
            "success": True,
            "job_id": job.job_id,
            "message": "Text-to-Video generation started",
            "estimated_time_seconds": params.num_frames * 3,  # Länger wegen SDXL + SVD
            "status_url": f"/api/video/status/{job.job_id}",
            "prompt": params.prompt[:100] + "..." if len(params.prompt) > 100 else params.prompt
        }
        
    except Exception as e:
        logger.error(f"TXT2VID start failed: {e}")
        raise HTTPException(500, f"Failed to start text-to-video: {str(e)}")

@router.get("/status/{job_id}")
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    📊 Job-Status abfragen
    """
    if job_id not in active_jobs:
        raise HTTPException(404, "Job not found")
    
    job = active_jobs[job_id]
    
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        result=job.result,
        error=job.error,
        elapsed_seconds=time.time() - job.created_at
    )

@router.get("/jobs")
async def list_jobs():
    """📋 Liste aktuelle Video-Jobs"""
    jobs = []
    for job in sorted(active_jobs.values(), key=lambda x: x.created_at, reverse=True):
        jobs.append({
            "job_id": job.job_id,
            "type": job.job_type,
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "has_result": job.result is not None,
            "elapsed_seconds": time.time() - job.created_at
        })
    
    return {"jobs": jobs, "total": len(jobs)}

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """❌ Job abbrechen (nur für Warteschlange)"""
    if job_id not in active_jobs:
        raise HTTPException(404, "Job not found")
    
    job = active_jobs[job_id]
    
    if job.status == "processing":
        return {"success": False, "message": "Cannot cancel job that is already processing"}
    
    if job.status in ["completed", "failed"]:
        return {"success": False, "message": "Job already finished"}
    
    # Remove from queue
    del active_jobs[job_id]
    
    return {"success": True, "message": "Job cancelled"}

@router.get("/test")
async def test_video_pipeline():
    """🧪 Teste Video-Pipeline mit einfachem Bild"""
    try:
        pipeline = get_pipeline()
        
        # Erstelle Test-Bild
        test_image = Image.new('RGB', (1024, 576), color=(100, 150, 200))
        
        logger.info("Testing video pipeline...")
        
        # Test mit wenigen Frames
        frames = await asyncio.to_thread(
            pipeline.img2video,
            image=test_image,
            num_frames=6,
            motion_bucket_id=127
        )
        
        # Speichere Test-Video
        video_url = save_video_file(frames, "gif", 6)
        
        return {
            "success": True,
            "message": "Video pipeline test completed!",
            "test_video": video_url,
            "frames_generated": len(frames)
        }
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Video pipeline test failed - check if SVD model is downloaded"
        }

@router.get("/health")
async def video_health_check():
    """❤️ Health Check für Video-System"""
    try:
        pipeline = get_pipeline()
        
        # Prüfe ob SVD-Modell verfügbar
        svd_available = False
        try:
            pipeline.load_svd_pipeline()
            svd_available = True
        except Exception as e:
            svd_error = str(e)
        
        # Prüfe ob SDXL verfügbar 
        sdxl_available = False
        try:
            pipeline.load_txt2img_pipeline()
            sdxl_available = pipeline.txt2img_pipe is not None
        except Exception:
            pass
        
        return {
            "status": "healthy" if svd_available else "degraded",
            "svd_available": svd_available,
            "sdxl_available": sdxl_available,
            "active_jobs": len(active_jobs),
            "gpu_available": torch.cuda.is_available() if 'torch' in globals() else False,
            "models_path": str(ROOT / "models" / "video"),
            "outputs_path": str(OUT_VID)
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "svd_available": False,
            "sdxl_available": False
        }