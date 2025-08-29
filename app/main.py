# app/main.py
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("AndioMediaStudio")

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
# Paths & directories
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
OUTPUTS_DIR = ROOT / "outputs"
WORKSPACE_DIR = ROOT / "workspace"

# Ensure required directories exist
WEB_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "videos").mkdir(parents=True, exist_ok=True)
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# API routes (IMPORTANT: include routers BEFORE mounting "/" static)
# -----------------------------------------------------------------------------
try:
    from server.api.enhanced_endpoints import router as andio_router
    app.include_router(andio_router)
    log.info("✓ enhanced_endpoints router mounted")
except Exception as e:
    log.warning(f"[WARN] Failed to include enhanced_endpoints router: {e}")

# Wardrobe/Inpaint endpoints (optional; only if present)
try:
    from server.api.wardrobe_endpoints import router as wardrobe_router
    app.include_router(wardrobe_router)
    log.info("✓ wardrobe_endpoints router mounted")
except Exception as e:
    log.warning(f"[WARN] Wardrobe router not available: {e}")

# Optional outputs listing (gallery)
try:
    from server.api.enhanced_endpoints_outputs_only import router as outputs_router
    app.include_router(outputs_router)
    log.info("✓ outputs_only router mounted")
except Exception as e:
    log.warning(f"[WARN] Outputs-only router not available: {e}")

# -----------------------------------------------------------------------------
# Static mounts (order matters: specific first, root "/" last)
# -----------------------------------------------------------------------------
# Serve generated media so PNG/MP4 are reachable
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
# Serve workspace (for upload previews like /workspace/uploads/...)
app.mount("/workspace", StaticFiles(directory=str(WORKSPACE_DIR)), name="workspace")
# Catch-all web UI LAST so it doesn't swallow /api/*
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
