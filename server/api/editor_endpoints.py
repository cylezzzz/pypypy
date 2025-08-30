from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from server.utils.file_namer import next_filename

router = APIRouter(prefix="/api/editor", tags=["editor"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "workspace" / "uploads"

ALLOWED = {".png", ".jpg", ".jpeg"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported extension '{ext}'. Allowed: {sorted(ALLOWED)}")

    new_name = next_filename("editor", ext, UPLOAD_DIR)
    save_path = UPLOAD_DIR / new_name

    contents = await file.read()
    save_path.write_bytes(contents)

    return {
        "ok": True,
        "filename": new_name,
        "relative_path": str(Path("workspace") / "uploads" / new_name),
    }