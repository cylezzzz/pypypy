#!/usr/bin/env python3
"""
Diagnose: GPU, Torch, Diffusers, Model-Registry, Model-Quickscan, Minimal-Inferenz
- Ändert bestehende Dateien NICHT. Nur lesen & kleine Testläufe.
- Ausgabe: Klartext + diagnose_report.json (+ optional ein Mini-Bild aus SDXL)

CLI:
  --skip-smoketest      SDXL-Minilauf überspringen
  --svd-smoketest       Kleiner SVD-Test (nur wenn installiert; standardmäßig AUS)
  --steps N             Schritte für Smoketest (default 6)
  --base-url URL        FastAPI-Basis (z.B. http://127.0.0.1:8000)
  --timeout SEC         Timeout für Webchecks/Jobs (default 5)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

REPORT: Dict[str, Any] = {
    "env": {},
    "torch": {},
    "diffusers": {},
    "deps": {},
    "models": {
        "registry_present": False,
        "registry": None,
        "image_subdirs": [],
        "video_subdirs": [],
        "audio_subdirs": [],
        "loose_components": [],
        "scan": []
    },
    "sdxl_smoketest": {},
    "svd_smoketest": {},
    "web": {},
    "summary": []
}

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
IMAGE_MODELS = MODELS / "image"
VIDEO_MODELS = MODELS / "video"
AUDIO_MODELS = MODELS / "audio"
WEB = ROOT / "web"
CONFIG = ROOT / "config"

# ---------- helpers ----------

def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def ok(msg: str):
    print(f"✅ {msg}")

def warn(msg: str):
    print(f"⚠️  {msg}")

def err(msg: str):
    print(f"❌ {msg}")
    REPORT["summary"].append(msg)

def safe_import(modname: str):
    try:
        m = __import__(modname)
        return m, None
    except Exception as e:
        return None, e

def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def human_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"

def try_run(cmd: List[str], timeout: int = 5) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 1, "", str(e)

# ---------- environment ----------

def check_env():
    section("UMGEBUNG")
    REPORT["env"]["python"] = sys.version
    REPORT["env"]["root"] = str(ROOT)
    print(f"- Python: {sys.version.split()[0]}")
    for k in ["CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "HF_HOME", "HUGGINGFACE_HUB_CACHE"]:
        REPORT["env"][k] = os.environ.get(k, "")
        print(f"- {k} = {REPORT['env'][k]}")
    if MODELS.exists():
        ok(f"Models-Ordner: {MODELS}")
    else:
        warn(f"Models-Ordner fehlt: {MODELS}")

# ---------- torch / gpu ----------

def check_torch():
    section("PYTORCH / CUDA")
    torch, e = safe_import("torch")
    if not torch:
        err(f"Torch Import fehlgeschlagen: {e}")
        return

    REPORT["torch"]["version"] = torch.__version__
    REPORT["torch"]["cuda_available"] = bool(torch.cuda.is_available())
    print(f"- torch = {torch.__version__}")
    print(f"- CUDA available = {torch.cuda.is_available()}")

    # cuDNN, CUDA Version
    try:
        REPORT["torch"]["cudnn"] = {
            "enabled": bool(torch.backends.cudnn.enabled),
            "version": torch.backends.cudnn.version()
        }
        print(f"- cuDNN enabled = {torch.backends.cudnn.enabled}, version = {torch.backends.cudnn.version()}")
    except Exception as e:
        warn(f"cuDNN-Info nicht verfügbar: {e}")

    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            REPORT["torch"]["device_count"] = device_count
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                REPORT["torch"].setdefault("devices", []).append({"index": i, "name": name})
                ok(f"GPU[{i}]: {name}")
            # Speicher
            try:
                free, total = torch.cuda.mem_get_info(0)
                REPORT["torch"]["mem"] = {"free": free, "total": total}
                print(f"- VRAM: free {human_bytes(free)} / total {human_bytes(total)}")
            except Exception as e:
                warn(f"VRAM-Info (torch) fehlgeschlagen: {e}")
        except Exception as e:
            warn(f"GPU-Details nicht lesbar: {e}")
    else:
        err("CUDA nicht verfügbar – GPU-Funktionen werden langsam/instabil sein.")

    # xformers / triton (optional)
    for mod in ["xformers", "triton"]:
        m, ex = safe_import(mod)
        REPORT["torch"][f"{mod}_ok"] = bool(m)
        if m:
            ok(f"{mod} importierbar")
        else:
            warn(f"{mod} fehlt/inkompatibel: {ex}")

    # NVML via nvidia-smi
    code, out, _ = try_run(["nvidia-smi", "-L"], timeout=3)
    if code == 0 and out:
        REPORT["torch"]["nvidia_smi_devices"] = out.splitlines()
        print("- nvidia-smi Geräte:")
        for line in out.splitlines():
            print("  ", line)
    else:
        print("- nvidia-smi nicht verfügbar (ok, optional).")

# ---------- diffusers / transformers ----------

def check_diffusers_and_registry():
    section("DIFFUSERS / TRANSFORMERS / REGISTRY")
    diffusers, de = safe_import("diffusers")
    transformers, te = safe_import("transformers")
    accelerate, ae = safe_import("accelerate")

    REPORT["diffusers"]["ok"] = bool(diffusers)
    REPORT["deps"]["transformers"] = {"ok": bool(transformers), "error": str(te) if te else None}
    REPORT["deps"]["accelerate"] = {"ok": bool(accelerate), "error": str(ae) if ae else None}

    if diffusers:
        ok(f"diffusers importierbar ({diffusers.__version__})")
        REPORT["diffusers"]["version"] = getattr(diffusers, "__version__", "?")
    else:
        err(f"diffusers Import fehlgeschlagen: {de}")

    if transformers:
        print(f"- transformers = {transformers.__version__}")
    else:
        warn("transformers nicht importierbar – einige Pipelines benötigen es.")

    reg = CONFIG / "model_registry.json"
    if reg.exists():
        try:
            data = json.loads(reg.read_text(encoding="utf-8"))
            REPORT["models"]["registry_present"] = True
            REPORT["models"]["registry"] = data
            ok(f"Registry gefunden: {reg}")
            # kleine Validierung
            problems = []
            if isinstance(data, dict):
                iterable = data.get("models") or data.get("entries") or []
            elif isinstance(data, list):
                iterable = data
            else:
                iterable = []
                problems.append("Registry-Format unbekannt (weder Liste noch {models:[...]})")

            for i, m in enumerate(iterable):
                mid = m.get("id") or m.get("name") or f"idx_{i}"
                mtype = m.get("type")
                mpath = m.get("path") or m.get("local_path")
                if not mtype:
                    problems.append(f"{mid}: 'type' fehlt")
                if mpath and not Path(mpath).exists():
                    problems.append(f"{mid}: Pfad fehlt ({mpath})")
            if problems:
                for p in problems:
                    warn("Registry: " + p)
                REPORT["models"]["registry_issues"] = problems
        except Exception as e:
            err(f"Registry defekt: {e}")
    else:
        warn("Keine model_registry.json – ein Scan/Refresh ist empfohlen.")
        REPORT["models"]["registry_present"] = False

    # Ordnerstruktur quick check
    def list_subdirs(base: Path) -> List[str]:
        return [d.name for d in base.iterdir()] if base.exists() else []

    REPORT["models"]["image_subdirs"] = list_subdirs(IMAGE_MODELS)
    REPORT["models"]["video_subdirs"] = list_subdirs(VIDEO_MODELS)
    REPORT["models"]["audio_subdirs"] = list_subdirs(AUDIO_MODELS)
    print(f"- image models: {REPORT['models']['image_subdirs']}")
    print(f"- video models: {REPORT['models']['video_subdirs']}")
    if REPORT["models"]["audio_subdirs"]:
        print(f"- audio models: {REPORT['models']['audio_subdirs']}")

    # Lose Komponenten, die den Scanner verwirren
    trouble = []
    for name in ["scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet"]:
        if (IMAGE_MODELS / name).exists():
            trouble.append(f"image/{name}")
    if trouble:
        warn(f"Lose Komponenten im Models-Verzeichnis: {trouble}")
        REPORT["models"]["loose_components"] = trouble

# ---------- model quick scan ----------

def quickscan_models():
    section("MODEL-QUICKSCAN (Verfügbarkeit)")
    scan: List[Dict[str, Any]] = []

    def scan_dir(kind: str, base: Path):
        if not base.exists():
            return
        for d in base.iterdir():
            if not d.is_dir():
                continue
            entry = {"kind": kind, "name": d.name, "path": str(d), "complete": False, "suspect": []}
            # Diffusers typische Dateien
            model_index = d / "model_index.json"
            config = d / "config.json"
            has_index = model_index.exists()
            has_config = config.exists()
            entry["has_model_index"] = has_index
            entry["has_config"] = has_config

            # Grob-Check: Gewisse Unterordner
            required_any = [
                d / "unet", d / "vae", d / "text_encoder", d / "scheduler"
            ]
            any_present = any(p.exists() for p in required_any)
            entry["any_core_folders"] = any_present

            # Dateien-Größe (Mini-Heuristik gegen kaputte Downloads)
            big_files = [f for f in d.rglob("*.safetensors")]
            entry["safetensors_files"] = len(big_files)
            if big_files:
                total_bytes = sum(f.stat().st_size for f in big_files)
                entry["safetensors_total"] = total_bytes
            else:
                entry["safetensors_total"] = 0

            # Bewertung
            if has_index or has_config or any_present:
                entry["complete"] = True
            if (has_index is False and has_config is False) or (big_files and entry["safetensors_total"] < 1024*1024):
                entry["suspect"].append("Struktur/Files wirken unvollständig")

            scan.append(entry)
            print(f"- {kind}: {d.name}  [{'OK' if entry['complete'] else '??'}]")

    scan_dir("image", IMAGE_MODELS)
    scan_dir("video", VIDEO_MODELS)
    scan_dir("audio", AUDIO_MODELS)

    REPORT["models"]["scan"] = scan
    if not scan:
        warn("Keine Modelle gefunden (models/ leer?)")

# ---------- SDXL mini run ----------

def sdxl_smoketest(steps: int = 6):
    section(f"SDXL SMOKETEST (mini, {steps} steps)")
    sdxl_dir = IMAGE_MODELS / "stable-diffusion-xl-base-1.0"
    if not sdxl_dir.exists():
        warn("SDXL Base fehlt. Smoketest übersprungen.")
        REPORT["sdxl_smoketest"]["skipped"] = True
        return
    try:
        torch, _ = safe_import("torch")
        diffusers, _ = safe_import("diffusers")
        if not torch or not diffusers:
            err("Torch/Diffusers fehlen – Smoketest kann nicht laufen.")
            REPORT["sdxl_smoketest"]["ok"] = False
            return

        from diffusers import StableDiffusionXLPipeline
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        t0 = time.time()
        pipe = StableDiffusionXLPipeline.from_pretrained(str(sdxl_dir), torch_dtype=dtype)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        dt_load = time.time() - t0
        print(f"- Load time: {dt_load:.1f}s")

        prompt = "a small wooden cabin in the snowy mountains, warm light, dusk"
        t1 = time.time()
        img = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=5.0).images[0]
        dt_gen = time.time() - t1

        out = ROOT / "outputs" / "diagnose_sdxl.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        ok(f"Mini-Bild generiert: {out.name} (gen {dt_gen:.1f}s)")
        REPORT["sdxl_smoketest"].update({
            "ok": True,
            "load_seconds": round(dt_load, 1),
            "gen_seconds": round(dt_gen, 1),
            "output": str(out)
        })
    except RuntimeError as e:
        err(f"SDXL Smoketest fehlgeschlagen (Runtime): {e}")
        REPORT["sdxl_smoketest"]["ok"] = False
        REPORT["sdxl_smoketest"]["error"] = str(e)
    except Exception as e:
        err(f"SDXL Smoketest fehlgeschlagen: {e}")
        REPORT["sdxl_smoketest"]["ok"] = False
        REPORT["sdxl_smoketest"]["error"] = str(e)

# ---------- Stable Video Diffusion mini run (optional) ----------

def svd_smoketest(steps: int = 6):
    section(f"SVD SMOKETEST (mini, {steps} steps)")
    video_dir = VIDEO_MODELS / "stable-video-diffusion"
    if not video_dir.exists():
        warn("Stable Video Diffusion fehlt. SVD-Smoke übersprungen.")
        REPORT["svd_smoketest"]["skipped"] = True
        return
    try:
        torch, _ = safe_import("torch")
        diffusers, _ = safe_import("diffusers")
        if not torch or not diffusers:
            err("Torch/Diffusers fehlen – SVD-Test kann nicht laufen.")
            REPORT["svd_smoketest"]["ok"] = False
            return

        from diffusers import StableVideoDiffusionPipeline
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Referenzbild: nimm ein kleines Placeholder, wenn vorhanden
        ref = (ROOT / "web" / "assets" / "placeholders" / "demo.jpg")
        if not ref.exists():
            warn("Kein Referenzbild für SVD gefunden (web/assets/placeholders/demo.jpg).")
            REPORT["svd_smoketest"]["skipped"] = True
            return

        t0 = time.time()
        pipe = StableVideoDiffusionPipeline.from_pretrained(str(video_dir), torch_dtype=dtype)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        dt_load = time.time() - t0
        print(f"- Load time: {dt_load:.1f}s")

        from PIL import Image
        image = Image.open(ref).convert("RGB").resize((576, 320))
        t1 = time.time()
        result = pipe(image, decode_chunk_size=1, motion_bucket_id=127, noise_aug_strength=0.02, num_inference_steps=steps)
        frames = result.frames[0]  # [T, H, W, C]
        outdir = ROOT / "outputs"
        outdir.mkdir(parents=True, exist_ok=True)
        # Save as simple frame set PNGs (kein ffmpeg Zwang)
        for i, fr in enumerate(frames[:4]):
            Image.fromarray(fr).save(outdir / f"diagnose_svd_{i:02d}.png")
        ok(f"SVD Frames gespeichert: {outdir}")
        REPORT["svd_smoketest"].update({
            "ok": True,
            "load_seconds": round(dt_load, 1),
            "frames_saved": min(4, len(frames)),
            "output_dir": str(outdir)
        })
    except Exception as e:
        err(f"SVD Smoketest fehlgeschlagen: {e}")
        REPORT["svd_smoketest"]["ok"] = False
        REPORT["svd_smoketest"]["error"] = str(e)

# ---------- web wiring ----------

def web_wiring(base_url: str, timeout: int):
    section("WEB / WIRING (Basis-Checks)")
    issues = []
    index_html = WEB / "index.html"
    app_js = WEB / "assets" / "app.js"

    if not index_html.exists():
        issues.append("web/index.html fehlt")
    if not app_js.exists():
        issues.append("web/assets/app.js fehlt")
    if issues:
        for m in issues:
            err(m)
        REPORT["web"]["issues"] = issues
    else:
        try:
            js = app_js.read_text(encoding="utf-8", errors="ignore")
            REPORT["web"]["app_js_bytes"] = len(js)
            hints = []
            for key in ["/api/", "fetch(", "XMLHttpRequest"]:
                if key in js:
                    hints.append(key)
            print(f"- app.js Hinweise: {hints}")
            if "http://" in js or "https://" in js:
                warn("Harte API-BaseURL gefunden – kann bei Ports/Hosts brechen.")
        except Exception as e:
            warn(f"app.js nicht lesbar: {e}")

    # Server-Endpunkte testen (nur wenn erreichbar)
    import urllib.request
    REPORT["web"]["endpoints"] = {}
    for path in ["/api/health", "/api/catalog", "/api/models"]:
        url = base_url.rstrip("/") + path
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                code = r.getcode()
                body = r.read()[:2048]
                ok(f"{path} → HTTP {code}")
                REPORT["web"]["endpoints"][path] = {"ok": True, "http": code, "sample": body.decode(errors="ignore")}
        except Exception as e:
            warn(f"{path} nicht erreichbar: {e}")
            REPORT["web"]["endpoints"][path] = {"ok": False, "error": str(e)}

# ---------- report ----------

def write_report():
    out = ROOT / "diagnose_report.json"
    out.write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    section("FAZIT")
    for line in REPORT["summary"]:
        print("• " + line)
    print(f"\nReport gespeichert: {out}")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Andio Diagnose")
    ap.add_argument("--skip-smoketest", action="store_true")
    ap.add_argument("--svd-smoketest", action="store_true")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=int, default=5)
    args = ap.parse_args()

    check_env()
    check_torch()
    check_diffusers_and_registry()
    quickscan_models()

    if not args.skip_smoketest:
        sdxl_smoketest(steps=max(1, args.steps))
    else:
        REPORT["sdxl_smoketest"]["skipped"] = True
        warn("SDXL Smoketest: übersprungen (per Flag).")

    if args.svd_smoketest:
        svd_smoketest(steps=max(1, args.steps))
    else:
        REPORT["svd_smoketest"]["skipped"] = True

    web_wiring(args.base_url, args.timeout)
    write_report()

if __name__ == "__main__":
    main()
