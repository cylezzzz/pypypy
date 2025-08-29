from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AndioMediaStudio", version="0.5")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
OUTPUTS_DIR = ROOT / "outputs"
WORKSPACE_DIR = ROOT / "workspace"

# ensure dirs
WEB_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "videos").mkdir(parents=True, exist_ok=True)
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# 1) API-Router ZUERST
from server.api.enhanced_endpoints import router as andio_router
app.include_router(andio_router)

# (optional) Wardrobe + Outputs-Router
try:
    from server.api.wardrobe_endpoints import router as wardrobe_router
    app.include_router(wardrobe_router)
except Exception:
    pass

try:
    from server.api.enhanced_endpoints_outputs_only import router as outputs_router
    app.include_router(outputs_router)
except Exception:
    pass

# 2) Spezifische Statics danach
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/workspace", StaticFiles(directory=str(WORKSPACE_DIR)), name="workspace")

# 3) Catch-all Web-Root GANZ ZUM SCHLUSS
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
