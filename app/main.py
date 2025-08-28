# app/main.py
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# -----------------------------------------------------------------------------
# Create app
# -----------------------------------------------------------------------------
app = FastAPI(title="AndioMediaStudio", version="0.5")

# CORS (allow LAN/dev access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Static mounts
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
OUTPUTS_DIR = ROOT / "outputs"

WEB_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "videos").mkdir(parents=True, exist_ok=True)

# Serve web UI and outputs (so generated PNG/MP4 are reachable)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

# -----------------------------------------------------------------------------
# API routes
# -----------------------------------------------------------------------------
try:
    from server.api.enhanced_endpoints import router as andio_router
    app.include_router(andio_router)
except Exception as e:
    # Do not crash the app if router missing; UI can still open.
    print(f"[WARN] Failed to include API router: {e}")
