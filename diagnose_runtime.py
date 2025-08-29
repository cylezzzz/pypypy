#!/usr/bin/env python3
"""
Diagnose: GPU, Torch, Diffusers, Model-Registry, Minimal-Inferenz
Ändert bestehende Dateien NICHT. Nur lesen & kleine Testläufe.

Ausgabe: Klartext + diagnose_report.json
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

REPORT = {
    "env": {},
    "torch": {},
    "diffusers": {},
    "models": {},
    "sdxl_smoketest": {},
    "web": {},
    "summary": []
}

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
IMAGE_MODELS = MODELS / "image"
WEB = ROOT / "web"
CONFIG = ROOT / "config"

def section(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def ok(msg): print(f"✅ {msg}")
def warn(msg): print(f"⚠️  {msg}")
def err(msg): print(f"❌ {msg}"); REPORT["summary"].append(msg)

def check_env():
    section("UMGEBUNG")
    REPORT["env"]["python"] = sys.version
    ok(f"Python: {sys.version.split()[0]}")
    for k in ["CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF"]:
        REPORT["env"][k] = os.environ.get(k, "")
        print(f"- {k} = {REPORT['env'][k]}")
    REPORT["env"]["root"] = str(ROOT)

def check_torch():
    section("PYTORCH / CUDA")
    try:
        import torch
        REPORT["torch"]["version"] = torch.__version__
        REPORT["torch"]["cuda_available"] = torch.cuda.is_available()
        print(f"- torch = {torch.__version__}")
        print(f"- CUDA available = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                device = torch.cuda.get_device_name(0)
                REPORT["torch"]["device"] = device
                ok(f"GPU: {device}")
            except Exception as e:
                warn(f"GPU Name konnte nicht gelesen werden: {e}")
        else:
            err("CUDA nicht verfügbar – GPU Features werden sehr langsam/fehlerhaft sein.")
    except Exception as e:
        err(f"Torch Import fehlgeschlagen: {e}")

    # xformers/triton sind nice-to-have
    for mod in ["xformers", "triton"]:
        try:
            __import__(mod)
            ok(f"{mod} importierbar")
            REPORT["torch"][f"{mod}_ok"] = True
        except Exception as e:
            warn(f"{mod} fehlt/inkompatibel: {e}")
            REPORT["torch"][f"{mod}_ok"] = False

def check_diffusers_and_registry():
    section("DIFFUSERS / MODEL-REGISTRY")
    try:
        import diffusers  # noqa
        REPORT["diffusers"]["ok"] = True
        ok("diffusers importierbar")
    except Exception as e:
        REPORT["diffusers"]["ok"] = False
        err(f"diffusers Import fehlgeschlagen: {e}")
        return

    reg = CONFIG / "model_registry.json"
    if reg.exists():
        try:
            data = json.loads(reg.read_text(encoding="utf-8"))
            REPORT["models"]["registry_present"] = True
            REPORT["models"]["registry"] = data
            ok(f"Registry gefunden: {reg}")
        except Exception as e:
            err(f"Registry defekt: {e}")
    else:
        warn("Keine model_registry.json – der Scanner/Refresh wird nötig sein.")
        REPORT["models"]["registry_present"] = False

    # Ordnerstruktur quick check
    expected_dirs = [
        "stable-diffusion-xl-base-1.0",
        "stable-diffusion-xl-refiner-1.0"
    ]
    found = []
    if IMAGE_MODELS.exists():
        for d in IMAGE_MODELS.iterdir():
            if d.is_dir():
                found.append(d.name)
    REPORT["models"]["image_subdirs"] = found
    print(f"- image models: {found}")

    # Warnungen für lose Komponenten
    trouble = []
    for name in ["scheduler","text_encoder","text_encoder_2","tokenizer","tokenizer_2","unet"]:
        if (IMAGE_MODELS / name).exists():
            trouble.append(name)
    if trouble:
        warn(f"Lose Komponenten verwirren Scanner: {trouble}")
        REPORT["models"]["loose_components"] = trouble

def sdxl_smoketest():
    section("SDXL SMOKETEST (mini, 6 steps)")
    # Wir testen NUR das Basis-SDXL, wenn vorhanden
    sdxl_dir = IMAGE_MODELS / "stable-diffusion-xl-base-1.0"
    if not sdxl_dir.exists():
        warn("SDXL Base fehlt. Smoketest übersprungen.")
        REPORT["sdxl_smoketest"]["skipped"] = True
        return
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
        t0 = time.time()
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(sdxl_dir),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        dt_load = time.time() - t0
        print(f"- Load time: {dt_load:.1f}s")
        prompt = "a small wooden cabin in the snowy mountains, warm light, dusk"
        t1 = time.time()
        img = pipe(prompt=prompt, num_inference_steps=6, guidance_scale=5.0).images[0]
        dt_gen = time.time() - t1
        out = ROOT / "outputs" / "diagnose_sdxl.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        ok(f"Mini-Bild generiert: {out.name} (gen {dt_gen:.1f}s)")
        REPORT["sdxl_smoketest"].update({
            "ok": True,
            "load_seconds": round(dt_load,1),
            "gen_seconds": round(dt_gen,1),
            "output": str(out)
        })
    except Exception as e:
        err(f"SDXL Smoketest fehlgeschlagen: {e}")
        REPORT["sdxl_smoketest"]["ok"] = False
        REPORT["sdxl_smoketest"]["error"] = str(e)

def web_wiring():
    section("WEB / WIRING (Basis-Checks)")
    issues = []
    index_html = WEB / "index.html"
    app_js = WEB / "assets" / "app.js"
    if not index_html.exists():
        issues.append("web/index.html fehlt")
    if not app_js.exists():
        issues.append("web/assets/app.js fehlt")
    if issues:
        for m in issues: err(m)
        REPORT["web"]["issues"] = issues
        return

    # naive Prüfung auf API-Basis-Pfad
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

def write_report():
    out = ROOT / "diagnose_report.json"
    out.write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    section("FAZIT")
    for line in REPORT["summary"]:
        print("• " + line)
    print(f"\nReport gespeichert: {out}")

if __name__ == "__main__":
    check_env()
    check_torch()
    check_diffusers_and_registry()
    sdxl_smoketest()
    web_wiring()
    write_report()
