# server/api/gallery_endpoints.py
from __future__ import annotations
from fastapi import APIRouter
from pathlib import Path
from server.utils.gallery_indexer import list_gallery

router = APIRouter(prefix="/api/gallery", tags=["gallery"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@router.get("/all")
async def gallery_all():
    return list_gallery(PROJECT_ROOT)

@router.get("/uploads")
async def gallery_uploads():
    data = list_gallery(PROJECT_ROOT)
    return {"uploads": data["uploads"], "total": len(data["uploads"])}

@router.get("/images")
async def gallery_images():
    data = list_gallery(PROJECT_ROOT)
    return {"images": data["images"], "total": len(data["images"])}
