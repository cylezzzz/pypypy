# server/app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict, Any, List
from pathlib import Path
import time, uuid, json, threading
from server.api.video_fixed_endpoints import router as video_fixed_router

app = FastAPI(title="AndioMediaStudio API")
app.include_router(video_fixed_router)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# --- Projektpfade ---
ROOT      = Path(__file__).resolve().parents[1]   # X:\pypygennew
MODELS    = ROOT / "models"
OUTPUTS_I = ROOT / "outputs" / "images"
OUTPUTS_V = ROOT / "outputs" / "videos"
WORK      = ROOT / "workspace"
for p in (MODELS, OUTPUTS_I, OUTPUTS_V, WORK):
    p.mkdir(parents=True, exist_ok=True)

# --- In-Memory Stores ---
JOBS: Dict[str, Dict[str, Any]] = {}
OUTPUTS: List[Dict[str, Any]] = []

# =======================
#        API ROUTES
# =======================

@app.get("/api/system")
def api_system():
    return {
        "uptime_h": 4.2,
        "gpu":  {"util": 37},
        "ram":  {"used_gb": 10.8, "total_gb": 32},
        "vram": {"used_gb": 3.1,  "total_gb": 8},
        "queue_depth": len([j for j in JOBS.values() if j["status"] in ("queued","running")])
    }

@app.get("/api/models")
def api_models():
    return [
        {"name": "Stable Diffusion XL", "type": "image", "status": "loaded", "vram_gb": 5.2},
        {"name": "SD 1.5",              "type": "image", "status": "available"},
        {"name": "FLUX",                "type": "image", "status": "available"},
    ]

@app.get("/api/jobs")
def api_jobs():
    return list(JOBS.values())

@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    return JOBS.get(job_id, {"id": job_id, "status": "unknown", "progress": 0})

@app.get("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str):
    if job_id in JOBS and JOBS[job_id]["status"] in ("queued","running"):
        JOBS[job_id]["status"] = "failed"
    return {"ok": True}

@app.get("/api/jobs/{job_id}/retry")
def api_retry(job_id: str):
    if job_id in JOBS and JOBS[job_id]["status"] == "failed":
        _spawn_job(job_id, JOBS[job_id]["payload"])
    return {"ok": True}

@app.get("/api/outputs")
def api_outputs(limit: int = 12, onlyGenerated: bool = False, id: Optional[str] = None):
    if id:
        m = next((o for o in OUTPUTS if o.get("id") == id or o.get("job_id") == id), None)
        return [m] if m else []
    items = OUTPUTS[-limit:][::-1]
    if onlyGenerated:
        items = [o for o in items if o.get("generated")]
    return items

@app.get("/api/outputs/save")
def api_outputs_save(id: str):
    for o in OUTPUTS:
        if o.get("id") == id or o.get("job_id") == id:
            o["saved"] = True
            return {"ok": True}
    return {"ok": False}

def _job_worker(job_id: str):
    for step in range(1, 21):
        time.sleep(0.25)
        if JOBS[job_id]["status"] == "failed":
            return
        JOBS[job_id]["progress"] = min(step * 5, 99)
        JOBS[job_id]["status"] = "running"

    out_name = f"{job_id}.png"
    out_path = OUTPUTS_I / out_name
    try:
        from PIL import Image
        img = Image.new("RGB", (512, 512), (22, 26, 34))
        img.save(out_path)
    except Exception:
        out_path.write_bytes(b"")

    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["progress"] = 100
    OUTPUTS.append({
        "id": job_id,
        "job_id": job_id,
        "type": "image",
        "path": f"/outputs/images/{out_name}",
        "thumb": f"/outputs/images/{out_name}",
        "generated": True,
        "model": JOBS[job_id]["payload"].get("model") or "SDXL",
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nsfw": (JOBS[job_id]["payload"].get("mode") == "nsfw"),
        "folder_url": "/outputs/images"
    })

def _spawn_job(job_id: str, payload: Dict[str, Any]):
    JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 0, "payload": payload}
    t = threading.Thread(target=_job_worker, args=(job_id,), daemon=True)
    t.start()

@app.post("/api/generate/image")
async def api_generate_image(params: UploadFile = File(...), image: Optional[UploadFile] = File(None)):
    payload = json.loads((await params.read()).decode("utf-8"))
    if image:
        ref_path = WORK / f"reference_{uuid.uuid4().hex}.png"
        ref_path.write_bytes(await image.read())
    job_id = uuid.uuid4().hex[:12]
    _spawn_job(job_id, payload)
    return {"job_id": job_id}

@app.get("/api/health")
def health():
    return {"ok": True}

# =======================
#   STATIC MOUNTS (LAST)
# =======================

# Outputs (für generierte Dateien)
app.mount("/outputs", StaticFiles(directory=str(ROOT / "outputs"), html=False), name="outputs")

# Frontend ausliefern
WEB = ROOT / "web"
WEB.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")

# Optionaler Index-Fallback (wird nur benutzt, wenn index.html fehlt)
@app.get("/")
def serve_index():
    index_file = WEB / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"error": "index.html not found in /web"}
