# app.py
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid, time, threading, os, shutil, json

app = FastAPI(title="Andio Backend")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Static (Platzhalter-Dateien hier hineinlegen) ---
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Simple Job Registry (Demo) ---
JOBS = {}  # job_id -> dict(stage, progress, etaSec, status, payload, result)

def _run_job(job_id: str, stages=("vorbereiten","inferenz","verfeinern","schreiben"), seconds=6):
    t0 = time.time()
    for i, st in enumerate(stages):
        slice_len = seconds / len(stages)
        while True:
            now = time.time()
            elapsed = now - t0
            frac = min(1.0, elapsed / seconds)
            JOBS[job_id]["progress"] = frac
            JOBS[job_id]["etaSec"] = max(0, seconds - elapsed)
            JOBS[job_id]["stage"] = st
            if elapsed >= (i + 1) * slice_len: break
            time.sleep(0.25)
    JOBS[job_id]["progress"] = 1.0
    JOBS[job_id]["etaSec"] = 0
    JOBS[job_id]["stage"] = "fertig"
    JOBS[job_id]["status"] = "done"

def start_job(seconds=6, stages=None, payload=None, result=None):
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    JOBS[job_id] = {
        "stage": "initialisieren", "progress": 0.0, "etaSec": seconds,
        "status": "running", "payload": payload or {}, "result": result or {}
    }
    thread = threading.Thread(target=_run_job, args=(job_id, stages or ("vorbereiten","inferenz","verfeinern","schreiben"), seconds), daemon=True)
    thread.start()
    return job_id

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = JOBS.get(job_id)
    if not j: return JSONResponse({"error":"unknown job"}, status_code=404)
    return {"stage": j["stage"], "progress": j["progress"], "etaSec": j["etaSec"], "status": j["status"]}

# ---------- Helpers: JSON oder Multipart annehmen ----------
async def parse_json_or_form(request: Request):
    """
    Akzeptiert:
      - application/json  -> body als JSON
      - multipart/form-data -> Feld 'json' (JSON-String) + optionale Dateien
    Gibt zurück: (payload:dict, files:dict[str, UploadFile|list[UploadFile]])
    """
    ctype = request.headers.get("content-type", "")
    files = {}
    if "multipart/form-data" in ctype:
        form = await request.form()
        payload = {}
        if "json" in form:
            try:
                payload = json.loads(form["json"])
            except Exception:
                payload = {}
        # bekannte Datei-Felder:
        for key in ["image", "mask", "audio", "vae"]:
            if key in form and isinstance(form[key], UploadFile):
                files[key] = form[key]
        # storyboard (mehrere)
        if "storyboard" in form:
            items = form.getlist("storyboard")
            files["storyboard"] = [it for it in items if isinstance(it, UploadFile)]
        # extras: beliebige
        for k, v in form.items():
            if isinstance(v, UploadFile) and k not in files:
                files[k] = v
        return payload, files
    else:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        return payload, {}

def save_upload(up: UploadFile, subdir="uploads", name=None) -> str:
    os.makedirs(subdir, exist_ok=True)
    filename = name or up.filename or f"file_{uuid.uuid4().hex}"
    dst = os.path.join(subdir, filename)
    with open(dst, "wb") as f:
        shutil.copyfileobj(up.file, f)
    return dst

# ===========================
# MODELS: download / install / set default
# ===========================
class ModelDownload(BaseModel):
    id: str
    url: str | None = None
    vae: str | None = None
    loras: list[str] | None = None
    sha256: str | None = None
    size: str | None = None

@app.post("/api/models/download")
async def models_download(body: ModelDownload):
    job = start_job(seconds=8, stages=("download","verify","extract","finish"), payload=body.dict())
    return {"jobId": job}

class ModelInstall(BaseModel):
    id: str
    dirs: dict | None = None
    files: dict | None = None

@app.post("/api/models/install")
async def models_install(body: ModelInstall):
    job = start_job(seconds=6, stages=("prepare","copy","index","finish"), payload=body.dict())
    return {"jobId": job}

class SetDefault(BaseModel):
    modelId: str

@app.post("/api/models/set-default")
async def models_set_default(body: SetDefault):
    # persistiere Konfiguration nach Bedarf
    return {"ok": True, "default": body.modelId}

# ===========================
# IMAGE
# ===========================
@app.post("/api/image/start")
async def image_start(request: Request):
    payload, files = await parse_json_or_form(request)
    # optional: Dateien speichern
    if "image" in files: save_upload(files["image"], "uploads/images")
    if "mask"  in files: save_upload(files["mask"],  "uploads/masks")
    job = start_job(seconds=6, stages=("encode","denoise","refine","save"), payload=payload)
    JOBS[job]["result"] = {"files":[{"url":"/static/sample_result.jpg","type":"image","name":"image.jpg","generated":True}]}
    return {"jobId": job}

@app.get("/api/image/finish")
async def image_finish(jobId: str | None = None):
    if jobId and jobId in JOBS:
        return JOBS[jobId].get("result", {"files":[]})
    return {"files":[{"url":"/static/sample_result.jpg","type":"image","name":"image.jpg","generated":True}]}

# Optional schnelle Vorschau (low-res)
@app.post("/api/editor/preview")
async def editor_preview(request: Request):
    payload, files = await parse_json_or_form(request)
    # hier könnte eine low-res Inpaint-Vorschau laufen
    return {"url": "/static/sample_result.jpg"}

# ===========================
# VIDEO
# ===========================
@app.post("/api/video/start")
async def video_start(request: Request):
    payload, files = await parse_json_or_form(request)
    if "image" in files: save_upload(files["image"], "uploads/ref")
    if "mask"  in files: save_upload(files["mask"],  "uploads/masks")
    if "audio" in files: save_upload(files["audio"], "uploads/audio")
    if "storyboard" in files:
        for i, f in enumerate(files["storyboard"]):
            save_upload(f, "uploads/storyboard", f"frame_{i:03d}.png")
    job = start_job(seconds=10, stages=("plan","denoise-frames","temporal","encode"), payload=payload)
    JOBS[job]["result"] = {"files":[{"url":"/static/motion_preview.webm","type":"video","name":"video.webm","generated":True}]}
    return {"jobId": job}

@app.get("/api/video/finish")
async def video_finish(jobId: str | None = None):
    if jobId and jobId in JOBS:
        return JOBS[jobId].get("result", {"files":[]})
    return {"files":[{"url":"/static/motion_preview.webm","type":"video","name":"video.webm","generated":True}]}

# ===========================
# EDITOR
# ===========================
@app.post("/api/editor/run/start")
async def editor_run_start(request: Request):
    payload, files = await parse_json_or_form(request)
    if "image" in files: save_upload(files["image"], "uploads/editor")
    if "mask"  in files: save_upload(files["mask"],  "uploads/masks")
    job = start_job(seconds=5, stages=("mask","inpaint","blend","save"), payload=payload)
    JOBS[job]["result"] = {"files":[{"url":"/static/sample_result.jpg","type":"image","name":"editor.jpg","generated":True}]}
    return {"jobId": job}

@app.get("/api/editor/run/finish")
async def editor_run_finish(jobId: str | None = None):
    if jobId and jobId in JOBS:
        return JOBS[jobId].get("result", {"files":[]})
    return {"files":[{"url":"/static/sample_result.jpg","type":"image","name":"editor.jpg","generated":True}]}

# ===========================
# WARDROBE (inkl. Preview)
# ===========================
@app.post("/api/wardrobe/detect/start")
async def wardrobe_detect_start():
    job = start_job(seconds=4, stages=("detect","classify","measure","done"))
    JOBS[job]["result"] = {"detected":{"top":"blouse","bottom":"jeans","shape":"medium"}}
    return {"jobId": job}

@app.get("/api/wardrobe/detect/finish")
async def wardrobe_detect_finish(jobId: str | None = None):
    return {"detected":{"top":"blouse","bottom":"jeans","shape":"medium"}}

@app.post("/api/wardrobe/apply/start")
async def wardrobe_apply_start(request: Request):
    payload, files = await parse_json_or_form(request)
    if "image" in files: save_upload(files["image"], "uploads/wardrobe")
    if "mask"  in files: save_upload(files["mask"],  "uploads/masks")
    job = start_job(seconds=7, stages=("segment","garment-fit","render","blend"))
    JOBS[job]["result"] = {"files":[{"url":"/static/wardrobe_result.jpg","type":"image","name":"wardrobe.jpg","generated":True}]}
    return {"jobId": job}

@app.get("/api/wardrobe/apply/finish")
async def wardrobe_apply_finish(jobId: str | None = None):
    if jobId and jobId in JOBS:
        return JOBS[jobId].get("result", {"files":[]})
    return {"files":[{"url":"/static/wardrobe_result.jpg","type":"image","name":"wardrobe.jpg","generated":True}]}

@app.post("/api/wardrobe/preview")
async def wardrobe_preview(request: Request):
    payload, files = await parse_json_or_form(request)
    # low-res garment preview
    return {"url": "/static/wardrobe_result.jpg"}

# ===========================
# MOTION (inkl. Live-Preview)
# ===========================
class MotionStart(BaseModel):
    mode: str
    preview: str | None = None
    softIk: bool | None = None
    speed: float | None = None
    loop: bool | None = None
    duration: int | None = 6
    keypoints: list | None = None
    regions: list | None = None
    tracks: dict | None = None
    genre: str | None = None
    pose: str | None = None
    action: str | None = None
    globalPrompt: str | None = None
    forPreview: bool | None = None

@app.post("/api/motion/start")
async def motion_start(body: MotionStart):
    job = start_job(seconds=8, stages=("rig","motion","temporal","encode"))
    JOBS[job]["result"] = {"files":[{"url":"/static/motion_preview.webm","type":"video","name":"motion.webm","generated":True}]}
    return {"jobId": job}

@app.post("/api/motion/preview")
async def motion_preview(body: MotionStart):
    return {"url": "/static/motion_preview.webm"}

@app.get("/api/motion/finish")
async def motion_finish(jobId: str | None = None):
    if jobId and jobId in JOBS:
        return JOBS[jobId].get("result", {"files":[]})
    return {"files":[{"url":"/static/motion_preview.webm","type":"video","name":"motion.webm","generated":True}]}
# === Root & Static Fallbacks (safe append) ==============================
from fastapi import Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

# Falls StaticFiles für / schon existiert, NICHT erneut mounten.
# Sonst mounten wir / auf den Web-Ordner (liefert CSS/JS/HTML).
if not any(getattr(r, "path", None) == "/" and isinstance(getattr(r, "app", None), StaticFiles) for r in app.router.routes):
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="webroot")

# Root "/" → index.html (falls vorhanden), sonst Redirect auf /gallery.html
@app.get("/", include_in_schema=False)
async def _root(request: Request):
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    # Fallback: wenn keine index.html, nimm z.B. die Gallery
    return RedirectResponse(url="/gallery.html", status_code=302)
