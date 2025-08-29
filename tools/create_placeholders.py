# tools/create_placeholders.py
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[1]
WEB = BASE / "web"
ASSETS = WEB / "assets"
DEMO = ASSETS / "demo"
INFOS = ASSETS / "infos"

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def make_placeholder(path: Path, text: str, size=(768, 512), bg=(32, 32, 36)):
    ensure_dir(path.parent)
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    w, h = d.textbbox((0,0), text, font=font)[2:]
    d.text(((size[0]-w)//2, (size[1]-h)//2), text, fill=(220,220,220), font=font)
    img.save(path)

def main():
    demo_files = {
        "portrait_a.jpg": "Demo Portrait A",
        "clip_thumb.jpg": "Demo Clip Thumb",
        "nsfw.jpg": "NSFW Demo (Placeholder)",
        "fabric.jpg": "Fabric Texture",
        "pose.jpg": "Pose Reference",
        "wardrobe.jpg": "Wardrobe Demo",
        "sdxl.jpg": "SDXL Demo",
        "anime.jpg": "Anime Demo",
    }
    for fname, label in demo_files.items():
        make_placeholder(DEMO / fname, label)

    categories = {
        "brust": "👙 Brust",
        "taille": "〰️ Taille",
        "huefte": "🜁 Hüfte",
        "po": "🍑 Po",
    }
    for cat, label in categories.items():
        for i in (1,2,3):
            make_placeholder(INFOS / f"{cat}-{i}.png", f"{label} {i}", size=(256,256), bg=(40,40,44))

    print(f"✅ Placeholders erstellt unter:\n  {DEMO}\n  {INFOS}")

if __name__ == "__main__":
    main()
