"""
AndioMediaStudio FastAPI Application
Enhanced with better error handling, monitoring, and direct AI access
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Form, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from pathlib import Path
import asyncio
import time
import json
import shutil
import uuid
import logging
from typing import Optional, Dict, List, Any, Union
from PIL import Image
import io

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
TEMP_DIR = BASE / "server" / "temp"

# Ensure directories exist
for directory in [MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR, VIDEOS_DIR, WORKSPACE_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# FastAPI app initialization
app = FastAPI(
    title="AndioMediaStudio",
    description="Universal Local AI Media Studio - No Limits, Full Creative Control",
    version="0.5.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware - allow all origins for local development
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
    task: str = Field(..., description="Task type: txt2img, img2img, inpaint, pose_transfer, txt2video, img2video, lipsync, talking")
    prompt: str = Field("", description="Text prompt for generation")
    negative: Optional[str] = Field(None, description="Negative prompt")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Additional input parameters")
    quality: str = Field("BALANCED", description="Quality preset: FAST, BALANCED, ULTRA")
    format: Optional[str] = Field(None, description="Output format preference")
    model_preference: Optional[str] = Field(None, description="Preferred model name")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    batch_size: int = Field(1, description="Number of images to generate", ge=1, le=4)

class DirectAIRequest(BaseModel):
    """Direct AI access for advanced users"""
    pipeline_type: str = Field(..., description="Pipeline type: txt2img, img2img, inpaint, controlnet")
    model_path: str = Field(..., description="Direct model path")
    prompt: str = Field(..., description="Generation prompt")
    negative_prompt: Optional[str] = Field("", description="Negative prompt")
    width: int = Field(896, ge=64, le=2048)
    height: int = Field(1152, ge=64, le=2048)
    num_inference_steps: int = Field(28, ge=1, le=100)
    guidance_scale: float = Field(6.0, ge=0.1, le=30.0)
    strength: float = Field(0.65, ge=0.1, le=1.0)
    seed: Optional[int] = Field(None)
    scheduler: Optional[str] = Field(None, description="Scheduler type")
    lora_weights: Optional[List[str]] = Field(None, description="LoRA weight paths")

class ModelInfo(BaseModel):
    name: str
    path: str
    type: str
    group: str
    tags: List[str]
    license: str = "unknown"
    size_mb: Optional[float] = None
    description: Optional[str] = None

# Global state management
progress_channels: Dict[str, set[WebSocket]] = {}
jobs_log: List[Dict[str, Any]] = []
active_jobs: Dict[str, Dict[str, Any]] = {}
_eta: Dict[str, Dict[str, Any]] = {}

# Progress tracking
async def send_progress(job_id: str, status: str, step: int = 0, total: int = 100, text: str = "", metadata: Optional[Dict] = None):
    """Enhanced progress reporting with metadata support"""
    connections = progress_channels.get(job_id, set())
    if not connections:
        return
    
    percent = 0 if total <= 0 else min(100, max(0, (step / total * 100)))
    
    # Calculate ETA
    eta_ms = None
    if step > 0 and total > step:
        elapsed = time.time() - _eta.get(job_id, {}).get("start", time.time())
        per_step = elapsed / step
        eta_ms = int(max(0, (total - step) * per_step * 1000))
    
    message = {
        "status": status,
        "step": step,
        "total": total,
        "percent": round(percent, 1),
        "eta_ms": eta_ms,
        "text": text,
        "job_id": job_id,
        "timestamp": int(time.time() * 1000),
        "metadata": metadata or {}
    }
    
    # Send to all connected clients
    disconnected = set()
    for ws in list(connections):
        try:
            await ws.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send progress to WebSocket: {e}")
            disconnected.add(ws)
    
    # Clean up disconnected clients
    for ws in disconnected:
        connections.discard(ws)

@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job progress"""
    await websocket.accept()
    progress_channels.setdefault(job_id, set()).add(websocket)
    logger.info(f"WebSocket connected for job {job_id}")
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    finally:
        progress_channels.get(job_id, set()).discard(websocket)

# Utility functions
def _new_job_id() -> str:
    """Generate unique job ID"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}"

def _detect_file_type(file_path: Path) -> str:
    """Detect file type from extension"""
    suffix = file_path.suffix.lower()
    if suffix in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}:
        return 'image'
    elif suffix in {'.mp4', '.webm', '.mov', '.avi', '.mkv'}:
        return 'video'
    elif suffix in {'.wav', '.mp3', '.flac', '.ogg'}:
        return 'audio'
    else:
        return 'unknown'

def _get_model_info(model_path: Path) -> ModelInfo:
    """Get detailed model information"""
    try:
        size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024 * 1024)
    except:
        size_mb = None
    
    # Determine group from path
    group = "unknown"
    if "image" in str(model_path):
        group = "image"
    elif "video" in str(model_path):
        group = "video"
    elif "audio" in str(model_path):
        group = "audio"
    
    # Determine tags based on model name and structure
    tags = []
    name_lower = model_path.name.lower()
    
    if any(keyword in name_lower for keyword in ['sd', 'stable-diffusion', 'flux', 'sdxl']):
        tags.extend(['txt2img', 'img2img', 'inpaint'])
    if 'control' in name_lower or 'pose' in name_lower:
        tags.append('controlnet')
    if 'svd' in name_lower or 'video' in name_lower:
        tags.extend(['txt2video', 'img2video'])
    if 'wav2lip' in name_lower:
        tags.append('lipsync')
    if 'sadtalker' in name_lower:
        tags.append('talking')
    
    return ModelInfo(
        name=model_path.name,
        path=str(model_path.relative_to(MODELS_DIR)),
        type="diffusers" if (model_path / "model_index.json").exists() else model_path.suffix.lstrip('.'),
        group=group,
        tags=tags,
        size_mb=size_mb
    )

# API Endpoints
@app.get("/api/ping")
async def ping():
    """Health check and system info"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name() if cuda_available else None
    except ImportError:
        cuda_available = False
        cuda_device = None
    
    return {
        "ok": True,
        "app": "AndioMediaStudio",
        "version": "0.5.0",
        "nsfw": True,
        "creative_mode": True,
        "timestamp": int(time.time() * 1000),
        "system": {
            "cuda_available": cuda_available,
            "cuda_device": cuda_device,
            "active_jobs": len(active_jobs),
            "models_path": str(MODELS_DIR),
        }
    }

@app.get("/api/models", response_model=Dict[str, List[ModelInfo]])
async def get_models():
    """Get all available models with detailed information"""
    MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".bin", ".pt", ".onnx", ".gguf", ".json", ".pth"}
    models = []
    
    # Scan models directory
    for group_dir in ["image", "video", "audio"]:
        group_path = MODELS_DIR / group_dir
        if not group_path.exists():
            continue
        
        # Scan for diffusers models (directories with model_index.json)
        for path in group_path.iterdir():
            if path.is_dir() and (path / "model_index.json").exists():
                models.append(_get_model_info(path))
        
        # Scan for single model files
        for path in group_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS:
                models.append(_get_model_info(path))
    
    return {"models": models}

@app.get("/api/outputs")
async def get_outputs(kind: str = "all", limit: int = 200):
    """Get generated outputs"""
    def _scan_media(directory: Path, extensions: set) -> List[Dict]:
        items = []
        if not directory.exists():
            return items
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                stat = file_path.stat()
                items.append({
                    "url": str(file_path.relative_to(BASE)).replace("\\", "/"),
                    "name": file_path.name,
                    "size": stat.st_size,
                    "mtime": int(stat.st_mtime * 1000),
                    "type": _detect_file_type(file_path)
                })
        return items
    
    results = []
    
    if kind in ("all", "images"):
        results.extend(_scan_media(IMAGES_DIR, {".png", ".jpg", ".jpeg", ".webp", ".bmp"}))
    
    if kind in ("all", "videos"):
        results.extend(_scan_media(VIDEOS_DIR, {".mp4", ".webm", ".mov", ".avi", ".mkv"}))
    
    # Sort by modification time (newest first)
    results.sort(key=lambda x: x["mtime"], reverse=True)
    
    return results[:limit]

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), target: str = Form("workspace")):
    """Upload files with enhanced validation and processing"""
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    # Sanitize filename
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "-_.() []").strip()
    if not safe_filename:
        safe_filename = f"file_{int(time.time())}.bin"
    
    # Determine target directory
    target_dirs = {
        "workspace": WORKSPACE_DIR,
        "images": IMAGES_DIR,
        "videos": VIDEOS_DIR,
        "temp": TEMP_DIR
    }
    
    target_dir = target_dirs.get(target, WORKSPACE_DIR)
    file_path = target_dir / safe_filename
    
    # Handle duplicate filenames
    counter = 1
    original_path = file_path
    while file_path.exists():
        stem = original_path.stem
        suffix = original_path.suffix
        file_path = target_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Additional processing for images
        file_info = {
            "ok": True,
            "path": str(file_path.relative_to(BASE)).replace("\\", "/"),
            "name": file_path.name,
            "size": file_path.stat().st_size,
            "type": _detect_file_type(file_path)
        }
        
        # If it's an image, get dimensions
        if file_info["type"] == "image":
            try:
                with Image.open(file_path) as img:
                    file_info["width"] = img.width
                    file_info["height"] = img.height
                    file_info["format"] = img.format
            except Exception as e:
                logger.warning(f"Could not read image metadata: {e}")
        
        return file_info
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/api/auto_mask")
async def auto_mask(image_path: str = Form(...), method: str = Form("mediapipe")):
    """Enhanced automatic mask generation with multiple methods"""
    src_path = BASE / image_path
    if not src_path.exists():
        raise HTTPException(404, "Image not found")
    
    try:
        image = Image.open(src_path).convert("RGB")
        
        if method == "mediapipe":
            # Use MediaPipe Selfie Segmentation
            import mediapipe as mp
            mp_selfie = mp.solutions.selfie_segmentation
            
            with mp_selfie.SelfieSegmentation(model_selection=1) as selfie:
                import numpy as np
                results = selfie.process(np.array(image))
                mask_array = (results.segmentation_mask > 0.5) * 255
                mask = Image.fromarray(mask_array.astype('uint8'), mode='L')
        
        elif method == "sam":
            # Placeholder for SAM integration
            # In production, you would load SAM model here
            raise HTTPException(501, "SAM method not implemented yet - install segment-anything-py")
        
        else:
            # Simple threshold-based mask as fallback
            gray = image.convert('L')
            mask = gray.point(lambda p: 255 if p > 10 else 0)
        
        # Save mask
        mask_path = WORKSPACE_DIR / (src_path.stem + "_mask.png")
        mask.save(mask_path)
        
        return {
            "ok": True,
            "mask_path": str(mask_path.relative_to(BASE)).replace("\\", "/"),
            "method": method,
            "size": {"width": mask.width, "height": mask.height}
        }
        
    except Exception as e:
        logger.error(f"Auto mask failed: {e}")
        raise HTTPException(500, f"Mask generation failed: {str(e)}")

@app.get("/api/jobs")
async def get_jobs():
    """Get job history"""
    return sorted(jobs_log, key=lambda x: x["time"], reverse=True)

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get specific job details"""
    job = next((j for j in jobs_log if j["id"] == job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")
    
    # Add current status if job is active
    if job_id in active_jobs:
        job["active"] = True
        job["current_status"] = active_jobs[job_id]
    
    return job

@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel running job"""
    if job_id not in active_jobs:
        raise HTTPException(404, "Job not found or not active")
    
    # Mark job as cancelled
    active_jobs[job_id]["cancelled"] = True
    await send_progress(job_id, "cancelled", text="Job cancelled by user")
    
    return {"ok": True, "message": "Job cancelled"}

@app.post("/api/jobs")
async def create_job(request: JobRequest, background_tasks: BackgroundTasks):
    """Create and execute AI generation job"""
    job_id = _new_job_id()
    
    # Create job record
    job_record = {
        "id": job_id,
        "task": request.task,
        "prompt": request.prompt[:200] + "..." if len(request.prompt) > 200 else request.prompt,
        "time": int(time.time() * 1000),
        "status": "queued",
        "quality": request.quality,
        "model": request.model_preference or "AUTO"
    }
    
    jobs_log.append(job_record)
    active_jobs[job_id] = {"status": "queued", "started": time.time()}
    _eta[job_id] = {"start": time.time()}
    
    # Queue job for execution
    background_tasks.add_task(execute_job, job_id, request)
    
    # Send initial progress
    await send_progress(job_id, "queued", 0, 100, "Job queued for execution")
    
    return {
        "ok": True,
        "job": {
            "id": job_id,
            "status": "queued",
            "message": "Job created successfully"
        }
    }

@app.post("/api/direct")
async def direct_ai_access(request: DirectAIRequest, background_tasks: BackgroundTasks):
    """Direct AI access for advanced users - bypass all filters and use custom parameters"""
    job_id = _new_job_id()
    
    job_record = {
        "id": job_id,
        "task": "direct_ai",
        "prompt": request.prompt[:200] + "..." if len(request.prompt) > 200 else request.prompt,
        "time": int(time.time() * 1000),
        "status": "queued",
        "mode": "direct",
        "pipeline": request.pipeline_type
    }
    
    jobs_log.append(job_record)
    active_jobs[job_id] = {"status": "queued", "started": time.time()}
    
    background_tasks.add_task(execute_direct_ai_job, job_id, request)
    
    await send_progress(job_id, "queued", 0, 100, "Direct AI job queued")
    
    return {
        "ok": True,
        "job": {
            "id": job_id,
            "status": "queued",
            "message": "Direct AI job created - no filters applied"
        }
    }

async def execute_job(job_id: str, request: JobRequest):
    """Execute AI generation job with enhanced error handling"""
    try:
        active_jobs[job_id]["status"] = "running"
        await send_progress(job_id, "starting", 1, 100, "Initializing AI pipeline...")
        
        if request.task == "txt2img":
            await _execute_txt2img(job_id, request)
        elif request.task == "img2img":
            await _execute_img2img(job_id, request)
        elif request.task == "inpaint":
            await _execute_inpaint(job_id, request)
        elif request.task in ("txt2video", "img2video"):
            await _execute_video_generation(job_id, request)
        else:
            raise ValueError(f"Unsupported task: {request.task}")
        
        # Mark job as completed
        active_jobs[job_id]["status"] = "completed"
        await send_progress(job_id, "completed", 100, 100, "Job completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)
        await send_progress(job_id, "failed", text=f"Job failed: {str(e)}")
    finally:
        # Clean up
        if job_id in active_jobs:
            del active_jobs[job_id]
        if job_id in _eta:
            del _eta[job_id]

async def execute_direct_ai_job(job_id: str, request: DirectAIRequest):
    """Execute direct AI job with full parameter control"""
    try:
        await send_progress(job_id, "loading", 5, 100, "Loading AI model...")
        
        # Import required libraries
        import torch
        from diffusers import (
            AutoPipelineForText2Image,
            AutoPipelineForImage2Image, 
            AutoPipelineForInpainting,
            StableDiffusionControlNetPipeline
        )
        
        # Load model from specified path
        model_path = MODELS_DIR / request.model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {request.model_path}")
        
        await send_progress(job_id, "loading", 20, 100, "Initializing pipeline...")
        
        # Select pipeline based on type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        if request.pipeline_type == "txt2img":
            pipeline = AutoPipelineForText2Image.from_pretrained(
                str(model_path), torch_dtype=dtype
            )
        elif request.pipeline_type == "img2img":
            pipeline = AutoPipelineForImage2Image.from_pretrained(
                str(model_path), torch_dtype=dtype
            )
        elif request.pipeline_type == "inpaint":
            pipeline = AutoPipelineForInpainting.from_pretrained(
                str(model_path), torch_dtype=dtype
            )
        else:
            raise ValueError(f"Unsupported pipeline type: {request.pipeline_type}")
        
        if device == "cuda":
            pipeline = pipeline.to("cuda")
        
        # Disable safety checker for creative freedom
        pipeline.safety_checker = None
        pipeline.requires_safety_checker = False
        
        await send_progress(job_id, "generating", 40, 100, "Generating with AI...")
        
        # Set up generation parameters
        generator = torch.Generator(device=device)
        if request.seed is not None:
            generator.manual_seed(request.seed)
        else:
            seed = torch.randint(0, 2**31-1, (1,)).item()
            generator.manual_seed(seed)
        
        # Custom callback for progress updates
        def progress_callback(step, timestep, latents):
            progress = 40 + (step / request.num_inference_steps) * 50
            asyncio.create_task(send_progress(
                job_id, "generating", int(progress), 100, 
                f"AI Step {step}/{request.num_inference_steps}"
            ))
        
        # Generate
        kwargs = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "generator": generator,
            "callback": progress_callback,
            "callback_steps": 1
        }
        
        result = pipeline(**kwargs)
        
        await send_progress(job_id, "saving", 95, 100, "Saving result...")
        
        # Save result
        output_path = IMAGES_DIR / f"{job_id}.png"
        result.images[0].save(output_path)
        
        # Update job record with results
        for job in jobs_log:
            if job["id"] == job_id:
                job["artifacts"] = [str(output_path.relative_to(BASE)).replace("\\", "/")]
                job["metadata"] = {
                    "seed": generator.initial_seed(),
                    "model_path": request.model_path,
                    "parameters": request.dict()
                }
                break
        
        await send_progress(job_id, "completed", 100, 100, "Direct AI generation completed")
        
    except Exception as e:
        logger.error(f"Direct AI job {job_id} failed: {e}")
        await send_progress(job_id, "failed", text=f"Direct AI job failed: {str(e)}")

async def _execute_txt2img(job_id: str, request: JobRequest):
    """Execute text-to-image generation"""
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        
        await send_progress(job_id, "loading", 10, 100, "Loading Stable Diffusion model...")
        
        # Find suitable model
        model_path = None
        for group_dir in ["image"]:
            group_path = MODELS_DIR / group_dir
            if group_path.exists():
                for path in group_path.iterdir():
                    if path.is_dir() and (path / "model_index.json").exists():
                        model_path = path
                        break
                if model_path:
                    break
        
        if not model_path:
            raise FileNotFoundError("No suitable text2img model found")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        pipeline = AutoPipelineForText2Image.from_pretrained(
            str(model_path), torch_dtype=dtype
        )
        
        if device == "cuda":
            pipeline = pipeline.to("cuda")
        
        pipeline.safety_checker = None  # Remove content filters
        
        await send_progress(job_id, "generating", 30, 100, "Generating image...")
        
        # Quality presets
        quality_settings = {
            "FAST": {"steps": 15, "guidance": 3.5, "width": 768, "height": 1024},
            "BALANCED": {"steps": 28, "guidance": 6.0, "width": 896, "height": 1152},
            "ULTRA": {"steps": 40, "guidance": 7.5, "width": 1024, "height": 1344}
        }
        
        settings = quality_settings.get(request.quality, quality_settings["BALANCED"])
        
        # Override with user inputs
        width = request.inputs.get("width", settings["width"])
        height = request.inputs.get("height", settings["height"])
        steps = request.inputs.get("steps", settings["steps"])
        guidance = request.inputs.get("guidance", settings["guidance"])
        
        # Progress callback
        def callback(step, timestep, latents):
            progress = 30 + (step / steps) * 60
            asyncio.create_task(send_progress(
                job_id, "generating", int(progress), 100, 
                f"Step {step}/{steps}"
            ))
        
        # Generate
        generator = torch.Generator(device=device)
        seed = request.seed or torch.randint(0, 2**31-1, (1,)).item()
        generator.manual_seed(seed)
        
        result = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative or "",
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            callback=callback,
            callback_steps=1
        )
        
        await send_progress(job_id, "saving", 95, 100, "Saving image...")
        
        # Save results
        artifacts = []
        for i, image in enumerate(result.images):
            output_path = IMAGES_DIR / f"{job_id}_{i}.png"
            image.save(output_path)
            artifacts.append(str(output_path.relative_to(BASE)).replace("\\", "/"))
        
        # Update job record
        for job in jobs_log:
            if job["id"] == job_id:
                job["artifacts"] = artifacts
                job["metadata"] = {"seed": seed, "model": str(model_path.name)}
                break
                
    except Exception as e:
        raise e

async def _execute_img2img(job_id: str, request: JobRequest):
    """Execute image-to-image generation"""
    image_path = request.inputs.get("image_path")
    if not image_path:
        raise ValueError("image_path required for img2img")
    
    try:
        import torch
        from diffusers import AutoPipelineForImage2Image
        
        # Load input image
        input_image = Image.open(BASE / image_path).convert("RGB")
        
        await send_progress(job_id, "loading", 15, 100, "Loading img2img model...")
        
        # Similar model loading logic as txt2img
        model_path = None
        for group_dir in ["image"]:
            group_path = MODELS_DIR / group_dir
            if group_path.exists():
                for path in group_path.iterdir():
                    if path.is_dir() and (path / "model_index.json").exists():
                        model_path = path
                        break
        
        if not model_path:
            raise FileNotFoundError("No suitable img2img model found")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            str(model_path), torch_dtype=dtype
        )
        
        if device == "cuda":
            pipeline = pipeline.to("cuda")
        
        pipeline.safety_checker = None
        
        await send_progress(job_id, "generating", 35, 100, "Transforming image...")
        
        # Parameters
        steps = request.inputs.get("steps", 28)
        guidance = request.inputs.get("guidance", 6.0)
        strength = request.inputs.get("strength", 0.65)
        
        def callback(step, timestep, latents):
            progress = 35 + (step / steps) * 55
            asyncio.create_task(send_progress(
                job_id, "generating", int(progress), 100,
                f"Step {step}/{steps}"
            ))
        
        generator = torch.Generator(device=device)
        seed = request.seed or torch.randint(0, 2**31-1, (1,)).item()
        generator.manual_seed(seed)
        
        result = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative or "",
            image=input_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            callback=callback,
            callback_steps=1
        )
        
        await send_progress(job_id, "saving", 95, 100, "Saving transformed image...")
        
        output_path = IMAGES_DIR / f"{job_id}.png"
        result.images[0].save(output_path)
        
        # Update job record
        for job in jobs_log:
            if job["id"] == job_id:
                job["artifacts"] = [str(output_path.relative_to(BASE)).replace("\\", "/")]
                job["metadata"] = {"seed": seed, "strength": strength}
                break
                
    except Exception as e:
        raise e

async def _execute_inpaint(job_id: str, request: JobRequest):
    """Execute inpainting generation"""
    image_path = request.inputs.get("image_path")
    mask_path = request.inputs.get("mask_path")
    
    if not image_path or not mask_path:
        raise ValueError("Both image_path and mask_path required for inpainting")
    
    try:
        import torch
        from diffusers import AutoPipelineForInpainting
        
        # Load images
        input_image = Image.open(BASE / image_path).convert("RGB")
        mask_image = Image.open(BASE / mask_path).convert("L")
        
        await send_progress(job_id, "loading", 20, 100, "Loading inpainting model...")
        
        # Model loading
        model_path = None
        for group_dir in ["image"]:
            group_path = MODELS_DIR / group_dir
            if group_path.exists():
                for path in group_path.iterdir():
                    if path.is_dir() and (path / "model_index.json").exists():
                        model_path = path
                        break
        
        if not model_path:
            raise FileNotFoundError("No suitable inpainting model found")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        pipeline = AutoPipelineForInpainting.from_pretrained(
            str(model_path), torch_dtype=dtype
        )
        
        if device == "cuda":
            pipeline = pipeline.to("cuda")
        
        pipeline.safety_checker = None
        
        await send_progress(job_id, "generating", 40, 100, "Inpainting...")
        
        steps = request.inputs.get("steps", 28)
        guidance = request.inputs.get("guidance", 6.0)
        strength = request.inputs.get("strength", 1.0)
        
        def callback(step, timestep, latents):
            progress = 40 + (step / steps) * 50
            asyncio.create_task(send_progress(
                job_id, "generating", int(progress), 100,
                f"Step {step}/{steps}"
            ))
        
        generator = torch.Generator(device=device)
        seed = request.seed or torch.randint(0, 2**31-1, (1,)).item()
        generator.manual_seed(seed)
        
        result = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative or "",
            image=input_image,
            mask_image=mask_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            callback=callback,
            callback_steps=1
        )
        
        await send_progress(job_id, "saving", 95, 100, "Saving inpainted image...")
        
        output_path = IMAGES_DIR / f"{job_id}.png"
        result.images[0].save(output_path)
        
        # Update job record
        for job in jobs_log:
            if job["id"] == job_id:
                job["artifacts"] = [str(output_path.relative_to(BASE)).replace("\\", "/")]
                job["metadata"] = {"seed": seed, "strength": strength}
                break
                
    except Exception as e:
        raise e

async def _execute_video_generation(job_id: str, request: JobRequest):
    """Execute video generation (placeholder implementation)"""
    await send_progress(job_id, "generating", 10, 100, "Setting up video generation...")
    
    # This is a placeholder - in production you would integrate with
    # Stable Video Diffusion, AnimateDiff, or similar video models
    frames = request.inputs.get("frames", 25)
    fps = request.inputs.get("fps", 16)
    
    for i in range(frames):
        if active_jobs.get(job_id, {}).get("cancelled"):
            break
        await asyncio.sleep(0.1)  # Simulate processing time
        progress = 10 + (i / frames) * 80
        await send_progress(job_id, "rendering", int(progress), 100, f"Frame {i+1}/{frames}")
    
    await send_progress(job_id, "encoding", 95, 100, "Encoding video...")
    
    # Create placeholder video file
    output_path = VIDEOS_DIR / f"{job_id}.mp4"
    with open(output_path, "wb") as f:
        f.write(b"")  # Placeholder
    
    # Update job record
    for job in jobs_log:
        if job["id"] == job_id:
            job["artifacts"] = [str(output_path.relative_to(BASE)).replace("\\", "/")]
            job["metadata"] = {"frames": frames, "fps": fps}
            break

# File serving endpoints
@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """Download generated files"""
    full_path = BASE / file_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(404, "File not found")
    
    # Security check - ensure file is within allowed directories
    allowed_dirs = [OUTPUTS_DIR, WORKSPACE_DIR]
    if not any(full_path.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs):
        raise HTTPException(403, "Access denied")
    
    return FileResponse(
        path=str(full_path),
        filename=full_path.name,
        media_type='application/octet-stream'
    )

@app.get("/stream/{file_path:path}")
async def stream_file(file_path: str):
    """Stream media files"""
    full_path = BASE / file_path
    if not full_path.exists():
        raise HTTPException(404, "File not found")
    
    def file_streamer():
        with open(full_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk
    
    media_type = "video/mp4" if full_path.suffix.lower() == ".mp4" else "application/octet-stream"
    
    return StreamingResponse(
        file_streamer(),
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename={full_path.name}"}
    )

# System management endpoints
@app.post("/api/system/cleanup")
async def cleanup_temp_files():
    """Clean up temporary files"""
    cleaned = 0
    for temp_file in TEMP_DIR.glob("*"):
        if temp_file.is_file() and (time.time() - temp_file.stat().st_mtime) > 3600:  # 1 hour old
            temp_file.unlink()
            cleaned += 1
    
    return {"ok": True, "cleaned_files": cleaned}

@app.get("/api/system/status")
async def system_status():
    """Get detailed system status"""
    try:
        import torch
        import psutil
    except ImportError as e:
        return {"error": f"Required packages not installed: {e}"}
    
    # GPU info
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_cached": torch.cuda.memory_reserved()
        }
    
    # System info
    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory": psutil.virtual_memory()._asdict(),
        "disk": psutil.disk_usage(str(BASE))._asdict(),
        "active_jobs": len(active_jobs),
        "total_jobs": len(jobs_log)
    }
    
    return {
        "ok": True,
        "gpu": gpu_info,
        "system": system_info,
        "directories": {
            "models": str(MODELS_DIR),
            "outputs": str(OUTPUTS_DIR),
            "workspace": str(WORKSPACE_DIR)
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🎨 AndioMediaStudio starting up...")
    logger.info(f"📁 Models directory: {MODELS_DIR}")
    logger.info(f"📁 Outputs directory: {OUTPUTS_DIR}")
    logger.info(f"📁 Workspace directory: {WORKSPACE_DIR}")
    
    # Check for models
    model_count = 0
    for ext in [".safetensors", ".ckpt", ".bin"]:
        model_count += len(list(MODELS_DIR.rglob(f"*{ext}")))
    
    logger.info(f"🤖 Found {model_count} model files")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"🚀 GPU available: {torch.cuda.get_device_name()}")
        else:
            logger.info("💻 Running on CPU")
    except ImportError:
        logger.warning("⚠ PyTorch not installed - limited functionality")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 AndioMediaStudio shutting down...")
    
    # Cancel active jobs
    for job_id in list(active_jobs.keys()):
        active_jobs[job_id]["cancelled"] = True
        await send_progress(job_id, "cancelled", text="Server shutting down")
    
    # Close WebSocket connections
    for connections in progress_channels.values():
        for ws in list(connections):
            try:
                await ws.close()
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)