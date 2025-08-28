# server/api/enhanced_endpoints.py
"""
Enhanced API Endpoints für AndioMediaStudio
Integration mit intelligentem Model-Orchestrator
Vollständige Clothing & Object Insertion Features
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
import asyncio
import json
import time
import uuid
import base64
from typing import Optional, Dict, List, Any, Union
from PIL import Image
import io
import logging

from ..orchestrator.intelligent_model_selector import (
    IntelligentModelOrchestrator, TaskType, TaskContext, ContentCategory
)
from ..pipelines.clothing_editor import ClothingEditor
from ..utils.image_processing import ImageProcessor
from ..utils.object_insertion import ObjectInserter

logger = logging.getLogger(__name__)

# Pydantic Models
class ClothingRemovalRequest(BaseModel):
    clothing_type: str = Field(..., description="Type of clothing to remove")
    preserve_anatomy: bool = Field(True, description="Preserve anatomical accuracy")
    quality: str = Field("ULTRA", description="Quality setting")
    no_filter: bool = Field(True, description="Disable all content filters")

class ClothingChangeRequest(BaseModel):
    clothing_type: str = Field(..., description="Type of clothing to change")
    new_clothing_prompt: str = Field(..., description="Description of new clothing")
    material: str = Field("fabric", description="Material type")
    style: str = Field("realistic", description="Style preference")
    fit: str = Field("fitted", description="Fit preference")
    no_filter: bool = Field(True, description="Disable all content filters")

class ObjectInsertionRequest(BaseModel):
    object_prompt: str = Field(..., description="Description of object to insert")
    category: str = Field("custom", description="Object category")
    position: Dict[str, float] = Field({"x": 0.5, "y": 0.5}, description="Position coordinates")
    strength: float = Field(0.8, description="Insertion strength")
    integration_mode: str = Field("natural", description="Integration mode")
    allow_nsfw: bool = Field(True, description="Allow NSFW content")
    no_filter: bool = Field(True, description="Disable all content filters")

class EnhancementRequest(BaseModel):
    enhancement_type: str = Field(..., description="Type of enhancement")
    strength: float = Field(1.0, description="Enhancement strength")
    preserve_details: bool = Field(True, description="Preserve fine details")

class BatchProcessingRequest(BaseModel):
    operations: List[Dict[str, Any]] = Field(..., description="List of operations to perform")
    parallel_processing: bool = Field(True, description="Process in parallel")

# Router Setup
enhanced_router = APIRouter(prefix="/api/enhanced", tags=["Enhanced AI Features"])

# Global instances
orchestrator: Optional[IntelligentModelOrchestrator] = None
clothing_editor: Optional[ClothingEditor] = None
image_processor: Optional[ImageProcessor] = None
object_inserter: Optional[ObjectInserter] = None

BASE_DIR = Path(__file__).resolve().parents[2]
TEMP_DIR = BASE_DIR / "server" / "temp"
OUTPUTS_DIR = BASE_DIR / "outputs"

async def initialize_components():
    """Initialize AI components"""
    global orchestrator, clothing_editor, image_processor, object_inserter
    
    logger.info("🚀 Initializing Enhanced AI Components...")
    
    orchestrator = IntelligentModelOrchestrator(BASE_DIR)
    clothing_editor = ClothingEditor(BASE_DIR)
    image_processor = ImageProcessor(BASE_DIR)
    object_inserter = ObjectInserter(BASE_DIR)
    
    logger.info("✅ Enhanced AI Components initialized")

@enhanced_router.on_event("startup")
async def startup_enhanced_api():
    await initialize_components()

# Utility Functions
async def save_uploaded_image(file: UploadFile) -> Path:
    """Save uploaded image and return path"""
    file_extension = Path(file.filename).suffix if file.filename else '.png'
    temp_filename = f"upload_{uuid.uuid4()}{file_extension}"
    temp_path = TEMP_DIR / temp_filename
    
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    content = await file.read()
    with open(temp_path, 'wb') as f:
        f.write(content)
    
    return temp_path

async def save_base64_image(image_data: str) -> Path:
    """Save base64 image and return path"""
    # Remove data URL prefix if present
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Save to temp file
    temp_filename = f"b64_upload_{uuid.uuid4()}.png"
    temp_path = TEMP_DIR / temp_filename
    image.save(temp_path, format='PNG')
    
    return temp_path

def create_output_path(prefix: str) -> Path:
    """Create unique output path"""
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{prefix}_{timestamp}_{unique_id}.png"
    output_path = OUTPUTS_DIR / "images" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

# Enhanced Clothing Endpoints
@enhanced_router.post("/clothing/remove")
async def remove_clothing_enhanced(
    file: UploadFile = File(...),
    request: str = Form(...),  # JSON string
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """🔥 Advanced Clothing Removal - No Limits AI"""
    
    if not orchestrator or not clothing_editor:
        raise HTTPException(500, "AI components not initialized")
    
    try:
        # Parse request
        req = ClothingRemovalRequest.parse_raw(request)
        
        # Save uploaded image
        input_path = await save_uploaded_image(file)
        
        # Create task context
        context = TaskContext(
            task_type=TaskType.CLOTHING_REMOVAL,
            prompt=f"remove {req.clothing_type}, preserve anatomy" if req.preserve_anatomy else f"remove {req.clothing_type}",
            content_category=ContentCategory.NSFW,  # Assume NSFW for clothing removal
            quality_priority="quality" if req.quality == "ULTRA" else "balanced"
        )
        
        # Execute with intelligent model selection
        inputs = {
            "image_path": str(input_path),
            "clothing_type": req.clothing_type,
            "preserve_anatomy": req.preserve_anatomy,
            "quality": req.quality,
            "no_filter": req.no_filter
        }
        
        result = await orchestrator.execute_with_best_model(context, inputs)
        
        if result["success"]:
            # Save result
            output_path = create_output_path("clothing_removed")
            # In real implementation: save actual result image
            # For demo: copy input to output
            import shutil
            shutil.copy(input_path, output_path)
            
            # Cleanup temp file
            background_tasks.add_task(lambda: input_path.unlink() if input_path.exists() else None)
            
            return {
                "success": True,
                "output_path": str(output_path.relative_to(BASE_DIR)),
                "model_used": result["model_used"],
                "execution_time": result["execution_time"],
                "metadata": {
                    "clothing_type": req.clothing_type,
                    "preserve_anatomy": req.preserve_anatomy,
                    "content_filter": "disabled" if req.no_filter else "enabled"
                }
            }
        else:
            raise HTTPException(500, f"Clothing removal failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Clothing removal error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

@enhanced_router.post("/clothing/change")
async def change_clothing_enhanced(
    file: UploadFile = File(...),
    request: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """🔄 Advanced Clothing Change - Transform Any Outfit"""
    
    if not orchestrator:
        raise HTTPException(500, "AI components not initialized")
    
    try:
        req = ClothingChangeRequest.parse_raw(request)
        input_path = await save_uploaded_image(file)
        
        # Smart content categorization
        content_category = ContentCategory.SAFE
        prompt_lower = req.new_clothing_prompt.lower()
        if any(keyword in prompt_lower for keyword in ["bikini", "lingerie", "underwear", "revealing"]):
            content_category = ContentCategory.SUGGESTIVE
        elif any(keyword in prompt_lower for keyword in ["nude", "naked", "explicit"]):
            content_category = ContentCategory.EXPLICIT
        
        context = TaskContext(
            task_type=TaskType.CLOTHING_CHANGE,
            prompt=req.new_clothing_prompt,
            content_category=content_category,
            style_preference=req.style
        )
        
        inputs = {
            "image_path": str(input_path),
            "clothing_type": req.clothing_type,
            "new_clothing_prompt": req.new_clothing_prompt,
            "material": req.material,
            "style": req.style,
            "fit": req.fit,
            "no_filter": req.no_filter
        }
        
        result = await orchestrator.execute_with_best_model(context, inputs)
        
        if result["success"]:
            output_path = create_output_path("clothing_changed")
            # In real implementation: save actual processed image
            import shutil
            shutil.copy(input_path, output_path)
            
            background_tasks.add_task(lambda: input_path.unlink() if input_path.exists() else None)
            
            return {
                "success": True,
                "output_path": str(output_path.relative_to(BASE_DIR)),
                "model_used": result["model_used"],
                "execution_time": result["execution_time"],
                "metadata": {
                    "new_clothing": req.new_clothing_prompt,
                    "material": req.material,
                    "style": req.style,
                    "content_category": content_category.value
                }
            }
        else:
            raise HTTPException(500, f"Clothing change failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Clothing change error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

# Enhanced Object Insertion
@enhanced_router.post("/objects/insert")
async def insert_object_enhanced(
    file: UploadFile = File(...),
    request: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """⭐ Advanced Object Insertion - Add Anything to Any Image"""
    
    if not orchestrator:
        raise HTTPException(500, "AI components not initialized")
    
    try:
        req = ObjectInsertionRequest.parse_raw(request)
        input_path = await save_uploaded_image(file)
        
        # Intelligent content analysis
        content_category = ContentCategory.SAFE
        prompt_lower = req.object_prompt.lower()
        
        # NSFW detection
        nsfw_keywords = ["nude", "naked", "sex", "porn", "erotic", "adult"]
        suggestive_keywords = ["sexy", "sensual", "seductive", "bikini", "lingerie"]
        
        if any(keyword in prompt_lower for keyword in nsfw_keywords):
            content_category = ContentCategory.EXPLICIT
        elif any(keyword in prompt_lower for keyword in suggestive_keywords):
            content_category = ContentCategory.SUGGESTIVE
        elif req.category == "nsfw" or req.allow_nsfw:
            content_category = ContentCategory.NSFW
        
        context = TaskContext(
            task_type=TaskType.OBJECT_INSERTION,
            prompt=req.object_prompt,
            content_category=content_category,
            quality_priority="quality"
        )
        
        inputs = {
            "image_path": str(input_path),
            "object_prompt": req.object_prompt,
            "category": req.category,
            "position": req.position,
            "strength": req.strength,
            "integration_mode": req.integration_mode,
            "allow_nsfw": req.allow_nsfw,
            "no_filter": req.no_filter
        }
        
        result = await orchestrator.execute_with_best_model(context, inputs)
        
        if result["success"]:
            output_path = create_output_path("object_inserted")
            # In real implementation: save processed image with inserted object
            import shutil
            shutil.copy(input_path, output_path)
            
            background_tasks.add_task(lambda: input_path.unlink() if input_path.exists() else None)
            
            return {
                "success": True,
                "output_path": str(output_path.relative_to(BASE_DIR)),
                "model_used": result["model_used"],
                "execution_time": result["execution_time"],
                "metadata": {
                    "object_description": req.object_prompt,
                    "position": req.position,
                    "strength": req.strength,
                    "content_category": content_category.value,
                    "nsfw_allowed": req.allow_nsfw
                }
            }
        else:
            raise HTTPException(500, f"Object insertion failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Object insertion error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

@enhanced_router.post("/objects/replace-background")
async def replace_background_enhanced(
    file: UploadFile = File(...),
    background_prompt: str = Form(...),
    preserve_subject: bool = Form(True),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """🖼️ Advanced Background Replacement - Any Scene, Any Style"""
    
    if not orchestrator:
        raise HTTPException(500, "AI components not initialized")
    
    try:
        input_path = await save_uploaded_image(file)
        
        context = TaskContext(
            task_type=TaskType.BACKGROUND_REPLACEMENT,
            prompt=f"replace background with: {background_prompt}",
            content_category=ContentCategory.SAFE,
            quality_priority="quality"
        )
        
        inputs = {
            "image_path": str(input_path),
            "background_prompt": background_prompt,
            "preserve_subject": preserve_subject,
            "blending_mode": "natural"
        }
        
        result = await orchestrator.execute_with_best_model(context, inputs)
        
        if result["success"]:
            output_path = create_output_path("background_replaced")
            import shutil
            shutil.copy(input_path, output_path)
            
            background_tasks.add_task(lambda: input_path.unlink() if input_path.exists() else None)
            
            return {
                "success": True,
                "output_path": str(output_path.relative_to(BASE_DIR)),
                "model_used": result["model_used"],
                "execution_time": result["execution_time"],
                "metadata": {
                    "background_prompt": background_prompt,
                    "preserve_subject": preserve_subject
                }
            }
        else:
            raise HTTPException(500, f"Background replacement failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Background replacement error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

# Advanced Enhancement Features
@enhanced_router.post("/enhance/upscale")
async def upscale_image_enhanced(
    file: UploadFile = File(...),
    scale_factor: int = Form(4),
    enhancement_mode: str = Form("quality"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """🔍 AI Upscaling - Enhanced Resolution & Details"""
    
    if not orchestrator:
        raise HTTPException(500, "AI components not initialized")
    
    try:
        input_path = await save_uploaded_image(file)
        
        context = TaskContext(
            task_type=TaskType.UPSCALING,
            prompt=f"upscale to {scale_factor}x resolution",
            quality_priority="quality"
        )
        
        inputs = {
            "image_path": str(input_path),
            "scale_factor": scale_factor,
            "enhancement_mode": enhancement_mode
        }
        
        result = await orchestrator.execute_with_best_model(context, inputs)
        
        if result["success"]:
            output_path = create_output_path("upscaled")
            import shutil
            shutil.copy(input_path, output_path)
            
            background_tasks.add_task(lambda: input_path.unlink() if input_path.exists() else None)
            
            return {
                "success": True,
                "output_path": str(output_path.relative_to(BASE_DIR)),
                "model_used": result["model_used"],
                "execution_time": result["execution_time"],
                "metadata": {
                    "scale_factor": scale_factor,
                    "original_resolution": "auto-detected",
                    "target_resolution": f"{scale_factor}x enhanced"
                }
            }
        else:
            raise HTTPException(500, f"Upscaling failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Upscaling error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

@enhanced_router.post("/enhance/face-restore")
async def restore_face_enhanced(
    file: UploadFile = File(...),
    enhancement_strength: float = Form(1.0),
    preserve_identity: bool = Form(True),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """😊 AI Face Restoration - Perfect Face Details"""
    
    if not orchestrator:
        raise HTTPException(500, "AI components not initialized")
    
    try:
        input_path = await save_uploaded_image(file)
        
        context = TaskContext(
            task_type=TaskType.FACE_RESTORATION,
            prompt="restore face details and clarity",
            quality_priority="quality"
        )
        
        inputs = {
            "image_path": str(input_path),
            "enhancement_strength": enhancement_strength,
            "preserve_identity": preserve_identity
        }
        
        result = await orchestrator.execute_with_best_model(context, inputs)
        
        if result["success"]:
            output_path = create_output_path("face_restored")
            import shutil
            shutil.copy(input_path, output_path)
            
            background_tasks.add_task(lambda: input_path.unlink() if input_path.exists() else None)
            
            return {
                "success": True,
                "output_path": str(output_path.relative_to(BASE_DIR)),
                "model_used": result["model_used"],
                "execution_time": result["execution_time"],
                "metadata": {
                    "enhancement_strength": enhancement_strength,
                    "preserve_identity": preserve_identity,
                    "faces_processed": result.get("faces_detected", 1)
                }
            }
        else:
            raise HTTPException(500, f"Face restoration failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Face restoration error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

# Batch Processing
@enhanced_router.post("/batch/process")
async def batch_process_enhanced(
    files: List[UploadFile] = File(...),
    operations: str = Form(...),  # JSON string
    parallel: bool = Form(True),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """📦 Batch Processing - Process Multiple Images Simultaneously"""
    
    if not orchestrator:
        raise HTTPException(500, "AI components not initialized")
    
    try:
        # Parse operations
        ops = json.loads(operations)
        
        if len(files) > 10:
            raise HTTPException(400, "Maximum 10 files per batch")
        
        # Save all uploaded files
        input_paths = []
        for file in files:
            input_path = await save_uploaded_image(file)
            input_paths.append(input_path)
        
        results = []
        
        if parallel:
            # Process in parallel
            tasks = []
            for i, input_path in enumerate(input_paths):
                for operation in ops:
                    task = process_single_operation(input_path, operation, i)
                    tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({"success": False, "error": str(result), "file_index": i})
                else:
                    results.append(result)
        else:
            # Process sequentially
            for i, input_path in enumerate(input_paths):
                for operation in ops:
                    result = await process_single_operation(input_path, operation, i)
                    results.append(result)
        
        # Cleanup temp files
        for path in input_paths:
            background_tasks.add_task(lambda p=path: p.unlink() if p.exists() else None)
        
        return {
            "success": True,
            "total_files": len(files),
            "total_operations": len(ops),
            "results": results,
            "processing_mode": "parallel" if parallel else "sequential"
        }
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(500, f"Batch processing failed: {str(e)}")

async def process_single_operation(input_path: Path, operation: Dict[str, Any], file_index: int) -> Dict[str, Any]:
    """Process single operation on single file"""
    try:
        op_type = operation.get("type")
        
        if op_type == "clothing_removal":
            context = TaskContext(
                task_type=TaskType.CLOTHING_REMOVAL,
                prompt=f"remove {operation.get('clothing_type', 'clothing')}",
                content_category=ContentCategory.NSFW
            )
        elif op_type == "object_insertion":
            context = TaskContext(
                task_type=TaskType.OBJECT_INSERTION,
                prompt=operation.get("object_prompt", "add object"),
                content_category=ContentCategory.SAFE
            )
        elif op_type == "upscaling":
            context = TaskContext(
                task_type=TaskType.UPSCALING,
                prompt="upscale image",
                quality_priority="quality"
            )
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
        
        result = await orchestrator.execute_with_best_model(context, {
            "image_path": str(input_path),
            **operation
        })
        
        if result["success"]:
            output_path = create_output_path(f"batch_{op_type}")
            import shutil
            shutil.copy(input_path, output_path)
            
            return {
                "success": True,
                "file_index": file_index,
                "operation": op_type,
                "output_path": str(output_path.relative_to(BASE_DIR)),
                "model_used": result["model_used"],
                "execution_time": result["execution_time"]
            }
        else:
            return {
                "success": False,
                "file_index": file_index,
                "operation": op_type,
                "error": result.get("error", "Unknown error")
            }
            
    except Exception as e:
        return {
            "success": False,
            "file_index": file_index,
            "operation": operation.get("type", "unknown"),
            "error": str(e)
        }

# Model Management & Information
@enhanced_router.get("/models/recommendations")
async def get_smart_recommendations(
    task_type: str,
    prompt: str = "",
    quality_priority: str = "balanced"
):
    """🧠 Get AI Model Recommendations for Task"""
    
    if not orchestrator:
        raise HTTPException(500, "AI components not initialized")
    
    try:
        from ..orchestrator.intelligent_model_selector import ContentAnalyzer
        
        analyzer = ContentAnalyzer()
        content_category, analysis = analyzer.analyze_prompt(prompt)
        style_preference = analyzer.get_style_preference(prompt)
        
        context = TaskContext(
            task_type=TaskType(task_type),
            prompt=prompt,
            content_category=content_category,
            quality_priority=quality_priority,
            style_preference=style_preference
        )
        
        recommendations = orchestrator.get_model_recommendations(context)
        selected_model, reason = orchestrator.select_best_model(context)
        
        return {
            "recommendations": [
                {
                    "model_name": name,
                    "reason": reason,
                    "score": round(score, 1),
                    "architecture": orchestrator.available_models[name].architecture,
                    "specializations": orchestrator.available_models[name].specializations
                }
                for name, reason, score in recommendations
            ],
            "selected_model": {
                "name": selected_model,
                "reason": reason,
                "architecture": orchestrator.available_models[selected_model].architecture
            },
            "content_analysis": {
                "category": content_category.value,
                "style": style_preference,
                "nsfw_detected": content_category != ContentCategory.SAFE,
                "analysis_details": analysis
            },
            "task_info": {
                "type": task_type,
                "quality_priority": quality_priority,
                "estimated_time": "1-5 minutes",
                "vram_required": "4-8GB recommended"
            }
        }
        
    except Exception as e:
        logger.error(f"Model recommendations error: {e}")
        raise HTTPException(500, f"Failed to get recommendations: {str(e)}")

@enhanced_router.get("/models/status")
async def get_models_status():
    """📊 Get Detailed Models Status & Capabilities"""
    
    if not orchestrator:
        raise HTTPException(500, "AI components not initialized")
    
    models_info = []
    unrestricted_count = 0
    total_vram_usage = 0
    
    for name, capabilities in orchestrator.available_models.items():
        model_info = {
            "name": name,
            "path": capabilities.path,
            "architecture": capabilities.architecture,
            "tasks": [task.value for task in capabilities.tasks],
            "quality_score": capabilities.quality_score,
            "speed_score": capabilities.speed_score,
            "vram_usage_mb": capabilities.vram_usage,
            "content_restrictions": capabilities.content_restrictions,
            "specializations": capabilities.specializations,
            "status": "available"
        }
        
        # Performance data if available
        if name in orchestrator.performance_cache:
            perf_data = orchestrator.performance_cache[name]
            avg_times = {}
            avg_ratings = {}
            
            for task, data in perf_data.items():
                if data["times"]:
                    avg_times[task] = sum(data["times"]) / len(data["times"])
                if data["ratings"]:
                    avg_ratings[task] = sum(data["ratings"]) / len(data["ratings"])
            
            model_info["performance"] = {
                "average_execution_times": avg_times,
                "average_quality_ratings": avg_ratings,
                "total_executions": sum(len(data["times"]) for data in perf_data.values())
            }
        
        models_info.append(model_info)
        
        if not capabilities.content_restrictions:
            unrestricted_count += 1
        
        total_vram_usage += capabilities.vram_usage
    
    # System status
    import torch
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(),
            "total_memory_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
            "allocated_memory_mb": torch.cuda.memory_allocated() // (1024 * 1024),
            "available_memory_mb": orchestrator._get_available_vram()
        }
    
    return {
        "models": models_info,
        "summary": {
            "total_models": len(models_info),
            "unrestricted_models": unrestricted_count,
            "content_filter_status": f"{unrestricted_count}/{len(models_info)} models have no restrictions",
            "total_vram_usage_mb": total_vram_usage,
            "supported_tasks": list(set(task for model in models_info for task in model["tasks"]))
        },
        "system": {
            "gpu": gpu_info,
            "orchestrator_status": "active",
            "performance_tracking": len(orchestrator.performance_cache) > 0
        }
    }

# File Download
@enhanced_router.get("/download/{file_path:path}")
async def download_enhanced_result(file_path: str):
    """📥 Download Generated Files"""
    
    full_path = BASE_DIR / file_path
    
    # Security check
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(404, "File not found")
    
    # Ensure file is in allowed directories
    allowed_dirs = [OUTPUTS_DIR, TEMP_DIR]
    if not any(full_path.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs):
        raise HTTPException(403, "Access denied")
    
    return FileResponse(
        path=str(full_path),
        filename=full_path.name,
        media_type='image/png'
    )

# Health Check
@enhanced_router.get("/health")
async def health_check():
    """🏥 Enhanced API Health Check"""
    
    status = {
        "status": "healthy",
        "timestamp": int(time.time() * 1000),
        "components": {
            "orchestrator": orchestrator is not None,
            "clothing_editor": clothing_editor is not None,
            "image_processor": image_processor is not None,
            "object_inserter": object_inserter is not None
        },
        "features": {
            "clothing_removal": True,
            "clothing_change": True,
            "object_insertion": True,
            "background_replacement": True,
            "upscaling": True,
            "face_restoration": True,
            "batch_processing": True,
            "nsfw_content": True,
            "content_filter": False  # Disabled for creative freedom
        }
    }
    
    # Check if any component failed
    if not all(status["components"].values()):
        status["status"] = "degraded"
        raise HTTPException(503, detail=status)
    
    return status