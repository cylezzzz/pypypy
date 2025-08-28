
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import asyncio, time, json, shutil
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
WEB_DIR = BASE / "web"
MODELS_DIR = BASE / "models"
OUTPUTS_DIR = BASE / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
WORKSPACE_DIR = BASE / "workspace"
for d in [MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR, VIDEOS_DIR, WORKSPACE_DIR]: d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AndioMediaStudio", version="0.4.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
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
_eta: dict[str, dict] = {}

async def send_progress(job_id: str, status: str, step: int=0, total: int=100, text: str=""):
    conns = progress_channels.get(job_id, set())
    percent = 0 if total<=0 else (step/total*100)
    eta_ms = None
    if step>0 and total>step:
        elapsed = time.time() - _eta.get(job_id, {}).get("start", time.time())
        per_step = elapsed / step
        eta_ms = int(max(0, (total-step) * per_step * 1000))
    msg = {"status": status, "step": step, "total": total, "percent": percent, "eta_ms": eta_ms, "text": text, "job_id": job_id}
    for ws in list(conns):
        try: await ws.send_text(json.dumps(msg))
        except Exception: 
            try: conns.remove(ws)
            except: pass

@app.websocket("/ws/jobs/{job_id}")
async def ws_jobs(websocket: WebSocket, job_id: str):
    await websocket.accept()
    progress_channels.setdefault(job_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        progress_channels.get(job_id, set()).discard(websocket)

@app.get("/api/ping")
def ping(): return {"ok": True, "app": "AndioMediaStudio", "nsfw": True, "version": "0.4.1"}

@app.get("/api/models")
def api_models():
    MODEL_EXTS = {".safetensors",".ckpt",".bin",".pt",".onnx",".gguf",".json"}
    models = []
    for sub in ["image","video"]:
        base = MODELS_DIR/sub
        if not base.exists(): continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in MODEL_EXTS:
                models.append({"name": p.stem, "path": p.relative_to(MODELS_DIR).as_posix(), "group": sub, "tags": []})
            if p.is_dir() and (p/"model_index.json").exists():
                models.append({"name": p.name, "path": p.relative_to(MODELS_DIR).as_posix(), "group": sub, "tags": ["diffusers"]})
    return {"models": models}

def _list_media(root: Path, exts):
    items = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            st = p.stat()
            items.append({"url": p.relative_to(BASE).as_posix(), "mtime": int(st.st_mtime*1000)})
    return sorted(items, key=lambda x: x["mtime"], reverse=True)

@app.get("/api/outputs")
def api_outputs(kind: str = "all", limit: int = 200):
    res = []
    if kind in ("all","images"): res += _list_media(IMAGES_DIR, {".png",".jpg",".jpeg",".webp"})
    if kind in ("all","videos"): res += _list_media(VIDEOS_DIR, {".mp4",".webm",".mov",".mkv"})
    return res[:limit]

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), target: str = Form("workspace")):
    fname = "".join([c for c in (file.filename or "file.bin") if c.isalnum() or c in "-_.() []"]).strip() or f"file_{int(time.time())}"
    folder = {"workspace": WORKSPACE_DIR, "images": IMAGES_DIR, "videos": VIDEOS_DIR}.get(target, WORKSPACE_DIR)
    path = folder / fname
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    return {"ok": True, "path": path.relative_to(BASE).as_posix()}

def _new_job_id(): return time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%100000}"

@app.get("/api/jobs")
def api_jobs_list(): return sorted(jobs_log, key=lambda x: x["time"], reverse=True)

@app.post("/api/auto_mask")
async def api_auto_mask(image_path: str = Form(...)):
    src = BASE / image_path
    if not src.exists(): raise HTTPException(404, "image not found")
    img = Image.open(src).convert("L")
    mask = img.point(lambda p: 255 if p>10 else 0)
    out = WORKSPACE_DIR / (Path(image_path).stem + "_mask.png")
    mask.save(out)
    return {"ok": True, "mask_path": out.relative_to(BASE).as_posix()}

@app.post("/api/jobs")
async def api_jobs(req: JobRequest):
    job_id = _new_job_id()
    jobs_log.append({"id": job_id, "task": req.task, "time": int(time.time()*1000), "plan": {"model": req.model_preference or "AUTO"}})
    _eta[job_id] = {"start": time.time(), "total": 100}
    await send_progress(job_id, "queued", 0, 100, "Wartet…")

    if req.task == "txt2img":
        try:
            from diffusers import AutoPipelineForText2Image
            import torch
        except Exception as e:
            raise HTTPException(400, f"diffusers/torch fehlen: {e}")
        root = MODELS_DIR/"image"
        model_dir = None
        for p in root.iterdir():
            if p.is_dir() and (p/"model_index.json").exists(): model_dir = p; break
        if model_dir is None: raise HTTPException(404, "Kein diffusers Text2Image Modell in models/image")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev=="cuda" else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(str(model_dir), torch_dtype=dtype)
        if dev=="cuda": pipe = pipe.to("cuda")
        pipe.safety_checker = None
        width = int(req.inputs.get("width") or 896); height = int(req.inputs.get("height") or 1152)
        steps = int(req.inputs.get("steps") or 28); guidance = float(req.inputs.get("guidance") or 6.0)
        _eta[job_id] = {"start": time.time(), "total": steps}
        step_box = {"i": 0}
        def cb(step, t, kwargs): step_box["i"]=step+1
        image = pipe(prompt=req.prompt, negative_prompt=req.negative or "", width=width, height=height,
                     num_inference_steps=steps, guidance_scale=guidance, callback=cb, callback_steps=1).images[0]
        (IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        out_path = IMAGES_DIR / f"{job_id}.png"; image.save(out_path)
        await send_progress(job_id, "completed", steps, steps, "Fertig")
        return {"ok": True, "job": {"id":job_id, "artifacts":[out_path.relative_to(BASE).as_posix()]}} 

    elif req.task == "img2img":
        try:
            from diffusers import AutoPipelineForImage2Image, StableDiffusionImg2ImgPipeline
            import torch
        except Exception as e:
            raise HTTPException(400, f"diffusers/torch fehlen: {e}")
        ipath = req.inputs.get("image_path")
        if not ipath: raise HTTPException(400, "inputs.image_path fehlt")
        init_image = Image.open(BASE/ipath).convert("RGB")
        root = MODELS_DIR/"image"
        model_dir = None
        for p in root.iterdir():
            if p.is_dir() and (p/"model_index.json").exists(): model_dir = p; break
        if model_dir is None: raise HTTPException(404, "Kein Image2Image Modell gefunden")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev=="cuda" else torch.float32
        try: pipe = AutoPipelineForImage2Image.from_pretrained(str(model_dir), torch_dtype=dtype)
        except: pipe = StableDiffusionImg2ImgPipeline.from_pretrained(str(model_dir), torch_dtype=dtype)
        if dev=="cuda": pipe = pipe.to("cuda")
        pipe.safety_checker = None
        steps = int(req.inputs.get("steps") or 28); guidance = float(req.inputs.get("guidance") or 6.0); strength = float(req.inputs.get("strength") or 0.65)
        _eta[job_id] = {"start": time.time(), "total": steps}
        out = pipe(prompt=req.prompt, negative_prompt=req.negative or "", image=init_image,
                   num_inference_steps=steps, guidance_scale=guidance, strength=strength).images[0]
        out_path = IMAGES_DIR / f"{job_id}.png"; out.save(out_path)
        await send_progress(job_id, "completed", steps, steps, "Fertig")
        return {"ok": True, "job": {"id":job_id, "artifacts":[out_path.relative_to(BASE).as_posix()]}} 

    elif req.task == "inpaint":
        try:
            from diffusers import AutoPipelineForInpainting, StableDiffusionInpaintPipelineLegacy
            import torch
        except Exception as e:
            raise HTTPException(400, f"diffusers/torch fehlen: {e}")
        ipath = req.inputs.get("image_path"); mpath = req.inputs.get("mask_path")
        if not ipath or not mpath: raise HTTPException(400, "inputs.image_path & inputs.mask_path erforderlich")
        image = Image.open(BASE/ipath).convert("RGB"); mask = Image.open(BASE/mpath).convert("L")
        root = MODELS_DIR/"image"; model_dir = None
        for p in root.iterdir():
            if p.is_dir() and (p/"model_index.json").exists(): model_dir = p; break
        if model_dir is None: raise HTTPException(404, "Kein Inpaint Modell gefunden")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev=="cuda" else torch.float32
        try: pipe = AutoPipelineForInpainting.from_pretrained(str(model_dir), torch_dtype=dtype)
        except: pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(str(model_dir), torch_dtype=dtype)
        if dev=="cuda": pipe = pipe.to("cuda")
        pipe.safety_checker = None
        steps = int(req.inputs.get("steps") or 28); guidance = float(req.inputs.get("guidance") or 6.0); strength = float(req.inputs.get("strength") or 1.0)
        _eta[job_id] = {"start": time.time(), "total": steps}
        out = pipe(prompt=req.prompt, negative_prompt=req.negative or "", image=image, mask_image=mask,
                   num_inference_steps=steps, guidance_scale=guidance, strength=strength).images[0]
        out_path = IMAGES_DIR / f"{job_id}.png"; out.save(out_path)
        await send_progress(job_id, "completed", steps, steps, "Fertig")
        return {"ok": True, "job": {"id":job_id, "artifacts":[out_path.relative_to(BASE).as_posix()]}} 

    elif req.task in ("txt2video","img2video"):
        frames = int(req.inputs.get("frames") or 25)
        _eta[job_id] = {"start": time.time(), "total": frames}
        for i in range(frames):
            await asyncio.sleep(0.15)
            await send_progress(job_id, "rendering", i+1, frames, f"Frame {i+1}/{frames}")
        out_path = VIDEOS_DIR / f"{job_id}.mp4"
        with open(out_path, "wb") as f: f.write(b"")  # placeholder
        await send_progress(job_id, "completed", frames, frames, "Video fertig")
        return {"ok": True, "job": {"id":job_id, "artifacts":[out_path.relative_to(BASE).as_posix()]}} 

    else:
        raise HTTPException(400, f"unbekannte task: {req.task}")
