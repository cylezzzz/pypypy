# mini_test.py
from pathlib import Path
from PIL import Image
import sys

ROOT = Path("X:/pypygennew")
sys.path.insert(0, str(ROOT))

try:
    print("🔧 Testing basic setup...")
    
    # Test 1: PIL funktioniert
    img = Image.new('RGB', (100, 100), color='red')
    print("✅ PIL works")
    
    # Test 2: Outputs-Ordner
    out_dir = ROOT / "outputs" / "videos" 
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output dir: {out_dir}")
    
    # Test 3: Modelle-Struktur
    models_video = ROOT / "models" / "video"
    print(f"📦 Models dir exists: {models_video.exists()}")
    
    if models_video.exists():
        for item in models_video.iterdir():
            if item.is_dir():
                print(f"   📁 {item.name}")
    
    # Test 4: Einfaches "Video" (GIF aus statischen Frames)
    frames = [
        Image.new('RGB', (200, 200), color=(255, 0, 0)),    # Rot  
        Image.new('RGB', (200, 200), color=(0, 255, 0)),    # Grün
        Image.new('RGB', (200, 200), color=(0, 0, 255)),    # Blau
    ]
    
    gif_path = out_dir / "test_basic.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                   duration=500, loop=0)
    
    print(f"✅ Basic video created: {gif_path}")
    print(f"📊 File size: {gif_path.stat().st_size} bytes")
    
    print("\n🎉 Basic setup working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()