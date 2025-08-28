from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import asyncio, time, json, shutil, os, yaml
from PIL import Image

from .registry.scanner import scan_models
from .registry.catalog import load_catalog, save_catalog
from .agents.selector import select_plan
from .utils.io_files import ensure_dir, safe_name, list_media

# Pipelines
from .pipelines.image_txt2img import Txt2ImgPipeline
from .pipelines.image_img2img import Img2ImgPipeline
from .pipelines.image_inpaint import InpaintPipeline
from .pipelines.control_pose import extract_pose, draw_pose
from .pipelines.video_svd import SVDPipeline
# Lipsync requires external setup; function will raise if not configured
from .pipelines.lipsync_wav2lip import run_wav2lip

BASE = Path(__file__).resolve().parents[1]
WEB_DIR = BASE / "web"
MODELS_DIR = BASE / "models"
OUTPUTS_DIR = BASE / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
WORKSPACE_DIR = BASE / "workspace"
CONFIG_DIR = BASE / "server" / "config"
SETTINGS_YAML = CONFIG_DIR / "settings.yaml"
CATALOG_JSON = MODELS_DIR / "catalog.json"

for d in [MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR, VIDEOS_DIR, WORKSPACE_DIR]:
    ensure_dir(d)

def load_settings():
    if SETTINGS_YAML.exists():
        with open(SETTINGS_YAML, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}
settings = load_settings()

app = FastAPI(title="AndioMediaStudio", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

class JobRequest(BaseModel):
    task: str
    prompt: str = ""
    negative: str | None = None
    inputs: dict = {}
    quality: str = "BALANCED"
    format: str | None = None
    model_preference: str | None = None

progress_channels: dict[str, set[WebSocket]] = {}

async def send_progress(job_id: str, event: dict):
    conns = progress_channels.get(job_id, set())
    if not conns: return
    msg = json.dumps(event)
    for ws in list(conns):
        try:
            await ws.send_text(msg)
        except:
            try: conns.remove(ws)
            except: pass

@app.on_event("startup")
async def on_startup():
    catalog = scan_models(MODELS_DIR)
    save_catalog(CATALOG_JSON, catalog)
    print(f"📦 Models indexed: {len(catalog.get('models', []))}")

@app.get("/api/ping")
def ping():
    return {"ok": True, "app": "AndioMediaStudio", "mode": "creative"}

@app.get("/api/models")
def api_models():
    return load_catalog(CATALOG_JSON)

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), target: str = Form("workspace")):
    fname = safe_name(file.filename or "upload.bin")
    if target not in {"workspace","images","videos"}:
        raise HTTPException(400, "invalid target")
    folder = {"workspace": WORKSPACE_DIR, "images": IMAGES_DIR, "videos": VIDEOS_DIR}[target]
    ensure_dir(folder)
    path = folder / fname
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "path": str(path.relative_to(BASE)).replace("\","/")}

@app.get("/api/outputs")
def api_outputs(kind: str = "all", limit: int = 200):
    items = []
    if kind in ("all","images"):
        items += list_media(IMAGES_DIR, kind="image")
    if kind in ("all","videos"):
        items += list_media(VIDEOS_DIR, kind="video")
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items[:limit]

@app.websocket("/ws/jobs/{job_id}")
async def ws_jobs(websocket: WebSocket, job_id: str):
    await websocket.accept()
    progress_channels.setdefault(job_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        progress_channels.get(job_id, set()).discard(websocket)

def _new_job_id():
    return time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%100000}"

def _save_image(img: Image.Image, out_dir: Path, stem: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir/f"{stem}.png"
    img.save(out_path)
    return str(out_path.relative_to(BASE)).replace("\","/")

@app.post("/api/jobs")
async def api_jobs(req: JobRequest):
    catalog = load_catalog(CATALOG_JSON)
    plan = select_plan(req.model_preference, req.task, req.prompt, req.inputs, req.quality, catalog)
    job_id = _new_job_id()
    await send_progress(job_id, {"status":"accepted","job_id":job_id,"plan":plan})

    # Execute synchronously here; for long runs, consider background tasks
    if req.task == "txt2img":
        pipe = Txt2ImgPipeline(BASE, plan["model"] if plan["model"]!="AUTO" else None)
        img, meta = pipe.run(prompt=req.prompt, quality=req.quality, negative=req.negative)
        rel = _save_image(img, IMAGES_DIR, job_id)
        await send_progress(job_id, {"status":"completed","job_id":job_id,"artifacts":[rel]})
        return {"ok": True, "job": {"id":job_id, "artifacts":[rel], "meta":meta, "plan":plan}}

    elif req.task == "img2img":
        path = req.inputs.get("image_path")
        if not path: raise HTTPException(400, "inputs.image_path fehlt")
        img = Image.open(BASE/path).convert("RGB")
        pipe = Img2ImgPipeline(BASE, plan["model"] if plan["model"]!="AUTO" else None)
        out, meta = pipe.run(init_image=img, prompt=req.prompt, quality=req.quality, negative=req.negative, strength=float(req.inputs.get("strength",0.65)))
        rel = _save_image(out, IMAGES_DIR, job_id)
        await send_progress(job_id, {"status":"completed","job_id":job_id,"artifacts":[rel]})
        return {"ok": True, "job": {"id":job_id, "artifacts":[rel], "meta":meta, "plan":plan}}

    elif req.task == "inpaint":
        ipath = req.inputs.get("image_path"); mpath = req.inputs.get("mask_path")
        if not ipath or not mpath: raise HTTPException(400, "inputs.image_path & inputs.mask_path erforderlich")
        img = Image.open(BASE/ipath).convert("RGB")
        mask = Image.open(BASE/mpath).convert("L")
        pipe = InpaintPipeline(BASE, plan["model"] if plan["model"]!="AUTO" else None)
        out, meta = pipe.run(init_image=img, mask_image=mask, prompt=req.prompt, quality=req.quality, negative=req.negative)
        rel = _save_image(out, IMAGES_DIR, job_id)
        await send_progress(job_id, {"status":"completed","job_id":job_id,"artifacts":[rel]})
        return {"ok": True, "job": {"id":job_id, "artifacts":[rel], "meta":meta, "plan":plan}}

    elif req.task == "pose_transfer":
        ipath = req.inputs.get("image_path")
        if not ipath: raise HTTPException(400, "inputs.image_path fehlt")
        img = Image.open(BASE/ipath).convert("RGB")
        landmarks = extract_pose(img)
        guide = draw_pose(img, landmarks)
        # For now, return guide image; next you can plug into ControlNet pose
        rel = _save_image(guide, IMAGES_DIR, job_id)
        await send_progress(job_id, {"status":"completed","job_id":job_id,"artifacts":[rel]})
        return {"ok": True, "job": {"id":job_id, "artifacts":[rel], "plan":plan}}

    elif req.task in ("txt2video","img2video"):
        pipe = SVDPipeline(BASE, plan["model"] if plan["model"]!="AUTO" else None)
        frames = pipe.run_txt2video(req.prompt or "cinematic scene", num_frames=int(req.inputs.get("frames",25)))
        out_path = VIDEOS_DIR/f"{job_id}.mp4"
        pipe.save_mp4(frames, out_path, fps=int(req.inputs.get("fps",16)))
        rel = str(out_path.relative_to(BASE)).replace("\","/")
        await send_progress(job_id, {"status":"completed","job_id":job_id,"artifacts":[rel]})
        return {"ok": True, "job": {"id":job_id, "artifacts":[rel], "plan":plan}}

    elif req.task == "lipsync":
        vpath = req.inputs.get("face_path"); apath = req.inputs.get("audio_path")
        if not vpath or not apath: raise HTTPException(400, "inputs.face_path & inputs.audio_path erforderlich")
        out_path = VIDEOS_DIR/f"{job_id}.mp4"
        rel = str(out_path.relative_to(BASE)).replace("\","/")
        # This will raise if not configured
        res = run_wav2lip(BASE, BASE/vpath, BASE/apath, out_path)
        await send_progress(job_id, {"status":"completed","job_id":job_id,"artifacts":[rel]})
        return {"ok": True, "job": {"id":job_id, "artifacts":[rel], "plan":plan}}

    else:
        raise HTTPException(400, f"unbekannte task: {req.task}")
