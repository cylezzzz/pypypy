from fastapi import APIRouter
from pathlib import Path
from server.utils.gallery_indexer import list_outputs
router = APIRouter(prefix="/api", tags=["outputs"])
@router.get("/outputs")
async def outputs():
    root = Path(__file__).resolve().parents[2] / "outputs"
    return list_outputs(root)
