
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import asyncio, time, json, shutil, os, yaml
from PIL import Image

# ---- Minimal utils
def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def safe_name(name: str) -> str:
    keep = "-_.() []"
    s = "".join([c for c in name or "" if c.isalnum() or c in keep]).strip()
    return s or f"file_{int(time.time())}"

def list_media(root: Path, kind: str):
    items = []
    if kind == "image":
        exts = {".png",".jpg",".jpeg",".webp"}
    else:
        exts = {".mp4",".webm",".mov",".mkv"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            st = p.stat()
            items.append({
                "path": p.as_posix(),
                "url": p.relative_to(BASE).as_posix(),
                "size": st.st_size,
                "mtime": int(st.st_mtime*1000)
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items

# ---- Basic registry: scan models folder
MODEL_EXTS = {".safetensors",".ckpt",".bin",".pt",".onnx",".gguf",".json"}
def tag_by_name(name: str):
    n = name.lower(); tags = []
    if any(k in n for k in ["flux","sdxl","sd3","stable-diffusion","sd15","realistic"]):
        tags += ["txt2img","img2img","inpaint","edit_ok","avatar_ok","nsfw_ok"]
    if "control" in n or "openpose" in n or "pose" in n:
        tags += ["pose"]
    if "svd" in n or "video" in n:
        tags += ["txt2video","img2video"]
    if "animatediff" in n:
        tags += ["img2video"]
    if "wav2lip" in n:
        tags += ["lipsync"]
    if "sadtalker" in n:
        tags += ["talking"]
    if "sam" in n:
        tags += ["mask"]
    return sorted(set(tags))

def scan_models(root: Path):
    models = []
    for sub in ["image","video"]:
        base = root/sub
        if not base.exists(): continue
        for f in base.rglob("*"):
            if f.is_file() and f.suffix.lower() in MODEL_EXTS:
                rel = f.relative_to(root).as_posix()
                models.append({
                    "name": f.stem,
                    "path": rel,
                    "type": f.suffix.lower().lstrip("."),
                    "tags": tag_by_name(f.stem),
                    "group": sub,
                    "license": "unknown"
                })
            if f.is_dir() and (f/"model_index.json").exists():
                rel = f.relative_to(root).as_posix()
                models.append({
                    "name": f.name,
                    "path": rel,
                    "type": "diffusers",
                    "tags": tag_by_name(f.name),
                    "group": sub,
                    "license": "unknown"
                })
    return {"models": models}

# ---- App constants
BASE = Path(__file__).resolve().parents[1]
WEB_DIR = BASE / "web"
MODELS_DIR = BASE / "models"
OUTPUTS_DIR = BASE / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
WORKSPACE_DIR = BASE / "workspace"

for d in [MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR, VIDEOS_DIR, WORKSPACE_DIR]:
    ensure_dir(d)

# ---- FastAPI
app = FastAPI(title="AndioMediaStudio", version="0.3.0")
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
jobs_log: list[dict] = []

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

def new_job_id():
    return time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%100000}"

def save_image(img: Image.Image, out_dir: Path, stem: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir/f"{stem}.png"
    img.save(out_path)
    return out_path.relative_to(BASE).as_posix()

# ---- Endpoints
@app.get("/api/ping")
def ping():
    return {"ok": True, "app": "AndioMediaStudio", "mode": "creative", "nsfw": True}

@app.get("/api/models")
def api_models():
    return scan_models(MODELS_DIR)

@app.get("/api/outputs")
def api_outputs(kind: str = "all", limit: int = 200):
    items = []
    if kind in ("all","images"):
        items += list_media(IMAGES_DIR, kind="image")
    if kind in ("all","videos"):
        items += list_media(VIDEOS_DIR, kind="video")
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items[:limit]

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
    return {"ok": True, "path": path.relative_to(BASE).as_posix()}

@app.websocket("/ws/jobs/{job_id}")
async def ws_jobs(websocket: WebSocket, job_id: str):
    await websocket.accept()
    progress_channels.setdefault(job_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        progress_channels.get(job_id, set()).discard(websocket)

@app.get("/api/jobs")
def api_jobs_list():
    # latest first
    return sorted(jobs_log, key=lambda x: x["time"], reverse=True)

# --- Auto mask endpoint (SAM / mediapipe fallback)
@app.post("/api/auto_mask")
async def api_auto_mask(image_path: str = Form(...)):
    # lightweight fallback: convert to grayscale threshold mask
    src = BASE / image_path
    if not src.exists(): raise HTTPException(404, "image not found")
    img = Image.open(src).convert("L")
    mask = img.point(lambda p: 255 if p>10 else 0)  # very simple fallback
    out = WORKSPACE_DIR / (Path(image_path).stem + "_mask.png")
    mask.save(out)
    rel = out.relative_to(BASE).as_posix()
    return {"ok": True, "mask_path": rel}

# --- Simple pipelines using diffusers if available; otherwise return 400
def have_diffusers():
    try:
        import diffusers  # noqa
        return True
    except Exception:
        return False

@app.post("/api/jobs")
async def api_jobs(req: JobRequest):
    job_id = new_job_id()
    plan = {"task": req.task, "model": req.model_preference or "AUTO"}
    record = {"id": job_id, "task": req.task, "time": int(time.time()*1000), "plan": plan}
    jobs_log.append(record)

    if req.task == "txt2img":
        if not have_diffusers():
            raise HTTPException(400, "diffusers nicht installiert")
        from diffusers import AutoPipelineForText2Image
        import torch
        # resolve model directory
        model_dir = None
        root = MODELS_DIR/"image"
        if req.model_preference:
            cand = root/req.model_preference
            if cand.exists(): model_dir = cand
            else:
                # try find by stem file
                for p in root.rglob("*"):
                    if p.is_file() and p.stem == req.model_preference:
                        model_dir = p.parent; break
        if model_dir is None:
            for p in root.iterdir():
                if p.is_dir() and (p/"model_index.json").exists():
                    model_dir = p; break
        if model_dir is None:
            raise HTTPException(404, "Kein diffusers Text2Image Modell in models/image gefunden.")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev=="cuda" else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(str(model_dir), torch_dtype=dtype)
        if dev=="cuda": pipe = pipe.to("cuda")
        pipe.safety_checker = None
        gen = torch.Generator(device=dev).manual_seed(int(time.time())%2_147_483_647)
        width = int(req.inputs.get("width") or 896)
        height = int(req.inputs.get("height") or 1152)
        steps = int(req.inputs.get("steps") or 28)
        guidance = float(req.inputs.get("guidance") or 6.0)
        image = pipe(prompt=req.prompt, negative_prompt=req.negative or "", width=width, height=height,
                     num_inference_steps=steps, guidance_scale=guidance, generator=gen).images[0]
        rel = save_image(image, IMAGES_DIR, job_id)
        await send_progress(job_id, {"status":"completed","job_id":job_id,"artifacts":[rel]})
        return {"ok": True, "job": {"id":job_id, "artifacts":[rel], "plan":plan}}

    elif req.task == "img2img":
        if not have_diffusers():
            raise HTTPException(400, "diffusers nicht installiert")
        from diffusers import AutoPipelineForImage2Image, StableDiffusionImg2ImgPipeline
        import torch
        ipath = req.inputs.get("image_path")
        if not ipath: raise HTTPException(400, "inputs.image_path fehlt")
        init_image = Image.open(BASE/ipath).convert("RGB")
        model_dir = None
        root = MODELS_DIR/"image"
        if req.model_preference and (root/req.model_preference).exists():
            model_dir = root/req.model_preference
        else:
            for p in root.iterdir():
                if p.is_dir() and (p/"model_index.json").exists():
                    model_dir = p; break
        if model_dir is None: raise HTTPException(404, "Kein Image2Image Modell gefunden.")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev=="cuda" else torch.float32
        try:
            pipe = AutoPipelineForImage2Image.from_pretrained(str(model_dir), torch_dtype=dtype)
        except Exception:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(str(model_dir), torch_dtype=dtype)
        if dev=="cuda": pipe = pipe.to("cuda")
        pipe.safety_checker = None
        gen = torch.Generator(device=dev).manual_seed(int(time.time())%2_147_483_647)
        steps = int(req.inputs.get("steps") or 28)
        guidance = float(req.inputs.get("guidance") or 6.0)
        strength = float(req.inputs.get("strength") or 0.65)
        out = pipe(prompt=req.prompt, negative_prompt=req.negative or "", image=init_image,
                   num_inference_steps=steps, guidance_scale=guidance, strength=strength, generator=gen).images[0]
        rel = save_image(out, IMAGES_DIR, job_id)
        await send_progress(job_id, {"status":"completed","job_id":job_id,"artifacts":[rel]})
        return {"ok": True, "job": {"id":job_id, "artifacts":[rel], "plan":plan}}

    elif req.task == "inpaint":
        if not have_diffusers():
            raise HTTPException(400, "diffusers nicht installiert")
        from diffusers import AutoPipelineForInpainting, StableDiffusionInpaintPipelineLegacy
        import torch
        ipath = req.inputs.get("image_path"); mpath = req.inputs.get("mask_path")
        if not ipath or not mpath: raise HTTPException(400, "inputs.image_path & inputs.mask_path erforderlich")
        image = Image.open(BASE/ipath).convert("RGB")
        mask = Image.open(BASE/mpath).convert("L")
        root = MODELS_DIR/"image"
        model_dir = None
        for p in root.iterdir():
            if p.is_dir() and (p/"model_index.json").exists():
                model_dir = p; break
        if model_dir is None: raise HTTPException(404, "Kein Inpaint Modell gefunden.")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev=="cuda" else torch.float32
        try:
            pipe = AutoPipelineForInpainting.from_pretrained(str(model_dir), torch_dtype=dtype)
        except Exception:
            pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(str(model_dir), torch_dtype=dtype)
        if dev=="cuda": pipe = pipe.to("cuda")
        pipe.safety_checker = None
        gen = torch.Generator(device=dev).manual_seed(int(time.time())%2_147_483_647)
        steps = int(req.inputs.get("steps") or 28)
        guidance = float(req.inputs.get("guidance") or 6.0)
        strength = float(req.inputs.get("strength") or 1.0)
        out = pipe(prompt=req.prompt, negative_prompt=req.negative or "", image=image, mask_image=mask,
                   num_inference_steps=steps, guidance_scale=guidance, strength=strength, generator=gen).images[0]
        rel = save_image(out, IMAGES_DIR, job_id)
        await send_progress(job_id, {"status":"completed","job_id":job_id,"artifacts":[rel]})
        return {"ok": True, "job": {"id":job_id, "artifacts":[rel], "plan":plan}}

    elif req.task in ("txt2video","img2video"):
        # Minimal SVD stub: check if a model folder exists; return 400 if not.
        root = MODELS_DIR/"video"
        has_svd = any((p/"model_index.json").exists() for p in root.iterdir() if p.is_dir())
        if not has_svd:
            raise HTTPException(404, "Kein SVD-Modell in models/video gefunden.")
        # We won't run generation here (heavy); just acknowledge job for now.
        rel = ""  # In einer erweiterten Version hier Frames generieren und speichern
        await send_progress(job_id, {"status":"accepted","job_id":job_id,"artifacts":[]})
        return {"ok": True, "job": {"id":job_id, "artifacts":[], "plan":plan}}

    else:
        raise HTTPException(400, f"unbekannte task: {req.task}")
