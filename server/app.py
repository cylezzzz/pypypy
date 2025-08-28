"""
AndioMediaStudio - Funktionsfähiger FastAPI Server
Mit echter Diffusers-Integration und AI-Generation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
import asyncio
import time
import json
import shutil
import uuid
import logging
import base64
from typing import Optional, Dict, List, Any
from PIL import Image
import io
import torch
import numpy as np

# AI Imports
try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        AutoPipelineForText2Image,
        AutoPipelineForImage2Image,
        AutoPipelineForInpainting,
        DiffusionPipeline
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("⚠️ Diffusers not available - install with: pip install diffusers")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AndioMediaStudio")

# Base paths
BASE = Path(__file__).resolve().parents[1]
WEB_DIR = BASE / "web"
MODELS_DIR = BASE / "models"
OUTPUTS_DIR = BASE / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
WORKSPACE_DIR = BASE / "workspace"

# Ensure directories exist
for directory in [MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR, VIDEOS_DIR, WORKSPACE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# FastAPI app
app = FastAPI(title="AndioMediaStudio", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

# Pydantic models
class JobRequest(BaseModel):
    task: str
    prompt: str
    negative: Optional[str] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    quality: str = "BALANCED"
    model_preference: Optional[str] = None
    seed: Optional[int] = None
    batch_size: int = 1

# Global state
active_jobs: Dict[str, Dict[str, Any]] = {}
jobs_log: List[Dict[str, Any]] = []
progress_channels: Dict[str, set] = {}
pipelines: Dict[str, Any] = {}

# AI Pipeline Manager
class PipelineManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipelines = {}
        self.models_loaded = False
        
    async def load_models(self):
        """Load available AI models"""
        if not DIFFUSERS_AVAILABLE:
            logger.error("Diffusers not available")
            return False
            
        logger.info(f"Loading models on {self.device}")
        
        # Try to load any available model
        model_candidates = [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "CompVis/stable-diffusion-v1-4"
        ]
        
        for model_name in model_candidates:
            try:
                logger.info(f"Attempting to load {model_name}")
                
                # Load txt2img pipeline
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    safety_checker=None,  # No restrictions
                    requires_safety_checker=False
                )
                pipeline = pipeline.to(self.device)
                
                self.pipelines["txt2img"] = pipeline
                self.pipelines["img2img"] = AutoPipelineForImage2Image.from_pipe(pipeline)
                self.pipelines["inpaint"] = AutoPipelineForInpainting.from_pipe(pipeline)
                
                logger.info(f"✅ Successfully loaded {model_name}")
                self.models_loaded = True
                return True
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        # Fallback: Create dummy pipelines for demo
        logger.warning("No models loaded - creating demo mode")
        self.models_loaded = False
        return False
    
    async def generate_txt2img(self, prompt: str, negative: str = "", width: int = 768, 
                              height: int = 768, steps: int = 20, guidance: float = 7.5, 
                              seed: Optional[int] = None) -> Image.Image:
        """Generate image from text"""
        if not self.models_loaded or "txt2img" not in self.pipelines:
            # Demo mode - create colored noise image
            return self._create_demo_image(prompt, width, height)
        
        pipeline = self.pipelines["txt2img"]
        
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        )
        
        return result.images[0]
    
    def _create_demo_image(self, prompt: str, width: int, height: int) -> Image.Image:
        """Create a demo image with text overlay"""
        # Create colorful gradient based on prompt hash
        hash_val = hash(prompt) % 360
        
        # Create image
        img = Image.new('RGB', (width, height))
        pixels = []
        
        for y in range(height):
            for x in range(width):
                r = int(127 + 127 * np.sin((x + hash_val) * 0.01))
                g = int(127 + 127 * np.sin((y + hash_val) * 0.01))
                b = int(127 + 127 * np.sin((x + y + hash_val) * 0.01))
                pixels.append((r, g, b))
        
        img.putdata(pixels)
        
        return img

# Global pipeline manager
pipeline_manager = PipelineManager()

async def send_progress(job_id: str, status: str, step: int = 0, total: int = 100, text: str = ""):
    """Send progress to connected WebSocket clients"""
    connections = progress_channels.get(job_id, set())
    if not connections:
        return
    
    percent = min(100, max(0, (step / total * 100) if total > 0 else 0))
    
    message = {
        "status": status,
        "step": step,
        "total": total,
        "percent": round(percent, 1),
        "text": text,
        "job_id": job_id,
        "timestamp": int(time.time() * 1000)
    }
    
    # Send to all connected clients
    disconnected = set()
    for ws in list(connections):
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            disconnected.add(ws)
    
    # Clean up disconnected clients
    for ws in disconnected:
        connections.discard(ws)

@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for job progress"""
    await websocket.accept()
    progress_channels.setdefault(job_id, set()).add(websocket)
    logger.info(f"WebSocket connected for job {job_id}")
    
    try:
        while True:
            await websocket.receive_text()
    except:
        logger.info(f"WebSocket disconnected for job {job_id}")
    finally:
        progress_channels.get(job_id, set()).discard(websocket)

def _new_job_id() -> str:
    """Generate unique job ID"""
    return f"job_{int(time.time())}_{str(uuid.uuid4())[:8]}"

@app.get("/api/ping")
async def ping():
    """Health check"""
    return {
        "ok": True,
        "app": "AndioMediaStudio",
        "version": "0.5.0",
        "timestamp": int(time.time() * 1000),
        "system": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "models_loaded": pipeline_manager.models_loaded,
            "active_jobs": len(active_jobs)
        }
    }

@app.get("/api/models")
async def get_models():
    """Get available models"""
    models = []
    
    if pipeline_manager.models_loaded:
        models.extend([
            {
                "name": "Stable Diffusion",
                "path": "txt2img_pipeline",
                "type": "diffusers",
                "group": "image",
                "tags": ["txt2img", "img2img", "inpaint"],
                "size_mb": 4000
            }
        ])
    else:
        models.append({
            "name": "Demo Mode",
            "path": "demo",
            "type": "demo",
            "group": "image", 
            "tags": ["txt2img"],
            "size_mb": 0
        })
    
    return {"models": models}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload file endpoint"""
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    # Save file
    timestamp = int(time.time())
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = WORKSPACE_DIR / safe_filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get image info if it's an image
    file_info = {
        "ok": True,
        "path": str(file_path.relative_to(BASE)),
        "name": safe_filename,
        "size": file_path.stat().st_size,
        "url": f"/workspace/{safe_filename}"
    }
    
    if file.content_type and file.content_type.startswith('image/'):
        try:
            with Image.open(file_path) as img:
                file_info.update({
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "type": "image"
                })
        except:
            pass
    
    return file_info

@app.get("/api/outputs")
async def get_outputs():
    """Get generated outputs"""
    results = []
    
    if IMAGES_DIR.exists():
        for file_path in IMAGES_DIR.glob("*.png"):
            stat = file_path.stat()
            results.append({
                "url": str(file_path.relative_to(BASE)).replace("\\", "/"),
                "name": file_path.name,
                "size": stat.st_size,
                "mtime": int(stat.st_mtime * 1000),
                "type": "image"
            })
    
    results.sort(key=lambda x: x["mtime"], reverse=True)
    return results

@app.post("/api/jobs")
async def create_job(request: JobRequest, background_tasks: BackgroundTasks):
    """Create and execute generation job"""
    job_id = _new_job_id()
    
    # Create job record
    job_record = {
        "id": job_id,
        "task": request.task,
        "prompt": request.prompt[:200] + "..." if len(request.prompt) > 200 else request.prompt,
        "time": int(time.time() * 1000),
        "status": "queued",
        "quality": request.quality
    }
    
    jobs_log.append(job_record)
    active_jobs[job_id] = {"status": "queued", "started": time.time()}
    
    # Queue job for execution
    background_tasks.add_task(execute_job, job_id, request)
    
    return {"ok": True, "job": {"id": job_id, "status": "queued"}}

async def execute_job(job_id: str, request: JobRequest):
    """Execute the generation job"""
    try:
        active_jobs[job_id]["status"] = "running"
        await send_progress(job_id, "starting", 1, 100, "Initializing...")
        
        # Load models if not already loaded
        if not pipeline_manager.models_loaded:
            await send_progress(job_id, "loading", 10, 100, "Loading AI models...")
            await pipeline_manager.load_models()
        
        await send_progress(job_id, "generating", 30, 100, "Generating image...")
        
        # Get parameters
        width = request.inputs.get("width", 768)
        height = request.inputs.get("height", 768)
        steps = request.inputs.get("steps", 20)
        guidance = request.inputs.get("guidance", 7.5)
        
        # Generate image
        if request.task == "txt2img":
            image = await pipeline_manager.generate_txt2img(
                prompt=request.prompt,
                negative=request.negative or "",
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=request.seed
            )
        else:
            # For other tasks, fall back to txt2img for now
            image = await pipeline_manager.generate_txt2img(
                prompt=request.prompt,
                negative=request.negative or "",
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=request.seed
            )
        
        await send_progress(job_id, "saving", 90, 100, "Saving image...")
        
        # Save result
        output_path = IMAGES_DIR / f"{job_id}.png"
        image.save(output_path)
        
        # Update job record
        for job in jobs_log:
            if job["id"] == job_id:
                job["artifacts"] = [str(output_path.relative_to(BASE)).replace("\\", "/")]
                job["status"] = "completed"
                break
        
        await send_progress(job_id, "completed", 100, 100, "Generation completed!")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        active_jobs[job_id]["status"] = "failed"
        await send_progress(job_id, "failed", text=f"Failed: {str(e)}")
    finally:
        if job_id in active_jobs:
            del active_jobs[job_id]

@app.get("/api/jobs")
async def get_jobs():
    """Get job history"""
    return sorted(jobs_log, key=lambda x: x["time"], reverse=True)

@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """Download generated files"""
    full_path = BASE / file_path
    if not full_path.exists():
        raise HTTPException(404, "File not found")
    
    return FileResponse(
        path=str(full_path),
        filename=full_path.name,
        media_type='application/octet-stream'
    )

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🎨 AndioMediaStudio starting up...")
    
    # Load models in background
    asyncio.create_task(pipeline_manager.load_models())
    
    logger.info("✅ AndioMediaStudio ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)