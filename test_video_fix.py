#!/usr/bin/env python3
"""
Schneller Test für reparierte Video-Pipeline
Läuft während das SVD-Modell noch downloadet
"""

import sys
from pathlib import Path
from PIL import Image
import torch
import time

# Projekt-Root finden
ROOT = Path(__file__).resolve().parent
if not (ROOT / "server").exists():
    ROOT = ROOT.parent

print(f"🔍 Testing from: {ROOT}")

def check_environment():
    """Prüfe Python-Environment"""
    print("🔧 Checking environment...")
    
    # Python Version
    print(f"   Python: {sys.version}")
    
    # PyTorch
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        print("   ❌ PyTorch not found")
        return False
    
    # Diffusers
    try:
        import diffusers
        print(f"   Diffusers: {diffusers.__version__}")
    except ImportError:
        print("   ❌ Diffusers not found")
        return False
    
    # PIL
    try:
        from PIL import Image
        print(f"   PIL: available")
    except ImportError:
        print("   ❌ PIL not found")
        return False
    
    return True

def check_models():
    """Prüfe verfügbare Modelle"""
    print("\n📦 Checking models...")
    
    models_dir = ROOT / "models"
    video_dir = models_dir / "video"
    image_dir = models_dir / "image"
    
    print(f"   Models dir: {models_dir}")
    print(f"   Exists: {models_dir.exists()}")
    
    if not models_dir.exists():
        print("   ❌ No models directory found")
        return False
    
    # Video Models
    print(f"\n   Video dir: {video_dir}")
    if video_dir.exists():
        for item in video_dir.iterdir():
            if item.is_dir():
                has_index = (item / "model_index.json").exists()
                print(f"   📁 {item.name} {'✅' if has_index else '❓'}")
    else:
        print("   ❌ No video models directory")
    
    # Image Models  
    print(f"\n   Image dir: {image_dir}")
    if image_dir.exists():
        for item in image_dir.iterdir():
            if item.is_dir():
                has_index = (item / "model_index.json").exists()
                print(f"   📁 {item.name} {'✅' if has_index else '❓'}")
    else:
        print("   ❌ No image models directory")
    
    # Auch flaches models/ Layout prüfen
    print(f"\n   Flat models layout:")
    for item in models_dir.iterdir():
        if item.is_dir() and item.name not in ["image", "video"]:
            has_index = (item / "model_index.json").exists()
            print(f"   📁 {item.name} {'✅' if has_index else '❓'}")
    
    return True

def test_basic_pipeline():
    """Teste ob die neue Pipeline importiert werden kann"""
    print("\n🧪 Testing pipeline import...")
    
    try:
        # Füge server zum Python Path hinzu
        sys.path.insert(0, str(ROOT))
        
        from server.pipelines.video_svd_fixed import FixedSVDPipeline
        print("   ✅ FixedSVDPipeline imported successfully")
        
        # Erstelle Pipeline-Instanz
        pipeline = FixedSVDPipeline(ROOT)
        print("   ✅ Pipeline instance created")
        
        return pipeline
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Pipeline creation failed: {e}")
        return None

def test_model_detection(pipeline):
    """Teste Modell-Erkennung"""
    print("\n🔍 Testing model detection...")
    
    try:
        # SVD Model suchen
        try:
            svd_path = pipeline._find_svd_model()
            print(f"   ✅ SVD model found: {svd_path.name}")
        except FileNotFoundError as e:
            print(f"   ❌ SVD model not found: {e}")
            print("   💡 This is expected if download is still running")
        
        # SDXL Model suchen
        try:
            sdxl_path = pipeline._find_txt2img_model()
            if sdxl_path:
                print(f"   ✅ SDXL model found: {sdxl_path.name}")
            else:
                print("   ❓ SDXL model not found (Text2Video won't work)")
        except Exception as e:
            print(f"   ❌ SDXL detection failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model detection failed: {e}")
        return False

def test_image_processing(pipeline):
    """Teste Bild-Verarbeitung"""
    print("\n🖼️ Testing image processing...")
    
    try:
        # Erstelle Test-Bild
        test_image = Image.new('RGB', (800, 600), color=(100, 150, 200))
        print("   ✅ Test image created (800x600)")
        
        # Teste _prepare_image
        prepared = pipeline._prepare_image(test_image)
        print(f"   ✅ Image prepared: {prepared.size}")
        
        # Erwarte 1024x576 oder 576x1024
        if prepared.size in [(1024, 576), (576, 1024)]:
            print("   ✅ Correct SVD dimensions")
        else:
            print(f"   ❓ Unexpected dimensions: {prepared.size}")
        
        return prepared
        
    except Exception as e:
        print(f"   ❌ Image processing failed: {e}")
        return None

def test_fallback_generation(pipeline):
    """Teste Fallback für fehlende Modelle"""
    print("\n🔄 Testing fallback generation...")
    
    try:
        # Erstelle Test-Bild
        test_image = Image.new('RGB', (1024, 576), color=(200, 100, 50))
        
        # Versuche Video-Generation (sollte Fallback verwenden wenn Modell fehlt)
        print("   🎬 Attempting video generation...")
        start_time = time.time()
        
        frames = pipeline.img2video(test_image, num_frames=4)
        
        elapsed = time.time() - start_time
        print(f"   ✅ Generated {len(frames)} frames in {elapsed:.1f}s")
        
        # Prüfe ob Frames gültig sind
        if len(frames) > 0:
            first_frame = frames[0]
            print(f"   ✅ Frame size: {first_frame.size}")
            
            # Teste Video-Speicherung
            output_dir = ROOT / "outputs" / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / "test_fallback.gif"
            pipeline.save_video_gif(frames, output_path, fps=2)
            
            if output_path.exists():
                print(f"   ✅ Video saved: {output_path}")
                print(f"   📊 File size: {output_path.stat().st_size / 1024:.1f} KB")
            else:
                print("   ❌ Video file not created")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fallback generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Haupttest"""
    print("🚀 AndioMediaStudio Video Pipeline Test")
    print("=" * 50)
    
    # 1. Environment prüfen
    if not check_environment():
        print("\n❌ Environment check failed!")
        return False
    
    # 2. Modelle prüfen
    check_models()
    
    # 3. Pipeline testen
    pipeline = test_basic_pipeline()
    if not pipeline:
        print("\n❌ Pipeline test failed!")
        return False
    
    # 4. Modell-Erkennung testen
    test_model_detection(pipeline)
    
    # 5. Bild-Verarbeitung testen
    test_image = test_image_processing(pipeline)
    if not test_image:
        print("\n❌ Image processing test failed!")
        return False
    
    # 6. Fallback-Generation testen
    if test_fallback_generation(pipeline):
        print("\n✅ All tests passed!")
        print("\n💡 Next steps:")
        print("   1. Wait for SVD model download to complete")
        print("   2. Run this test again to verify real video generation")
        print("   3. Start server and test API endpoints")
        print("   4. Check outputs/videos/ for generated content")
        return True
    else:
        print("\n❌ Fallback generation test failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)