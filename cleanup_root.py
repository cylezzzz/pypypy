import os
import shutil

def cleanup_root_directory():
    """Räumt das Hauptverzeichnis auf - nur essenzielle Dateien behalten"""
    
    print("🧹 === HAUPTVERZEICHNIS AUFRÄUMEN ===")
    
    # Behalte nur diese essenziellen Dateien/Ordner
    keep_files = {
        # Core-System
        "start.py",
        "requirements.txt", 
        "README.md",
        "__init__.py",
        ".gitignore",
        
        # Ordner
        "models",
        "server", 
        "web",
        "config",
        "outputs",
        "workspace",
        "app",
        "desktop",
        "tools",
        ".venv"
    }
    
    # Test-/Entwicklungs-Dateien die gelöscht werden können
    cleanup_files = [
        "cleanup_models.py",
        "convert_realistic_vision.py", 
        "fix_model_registry.py",
        "repair_models.py",
        "test_clean_models.py",
        "test_models.py",
        "test_working_models.py",
        "model_inventory.json"
    ]
    
    # Generierte Test-Bilder
    test_images = [f for f in os.listdir(".") if f.startswith("test_") and f.endswith(".png")]
    
    print("🗑️ Entferne Test-/Entwicklungsdateien...")
    removed_count = 0
    
    for file in cleanup_files + test_images:
        if os.path.exists(file):
            try:
                if os.path.isfile(file):
                    os.remove(file)
                else:
                    shutil.rmtree(file)
                print(f"✅ Entfernt: {file}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Fehler bei {file}: {e}")
    
    print(f"\n✨ {removed_count} Dateien aufgeräumt!")
    
    # Zeige finale Struktur
    print("\n📁 Finale Struktur:")
    items = sorted(os.listdir("."))
    for item in items:
        if os.path.isdir(item):
            print(f"📁 {item}/")
        else:
            print(f"📄 {item}")

def create_simple_config():
    """Erstellt einfache Config nur für funktionierende Modelle"""
    
    config_content = """# AndioMediaStudio - Einfache Konfiguration
models:
  base_path: "models"
  
  # Nur funktionierende Modelle
  working_models:
    sdxl_base:
      path: "models/image/stable-diffusion-xl-base-1.0"
      type: "txt2img"
      pipeline: "StableDiffusionXLPipeline"
      enabled: true
      default: true
      
    sd15_checkpoint:
      path: "models/image/checkpoints/sd15.safetensors"
      type: "txt2img"  
      pipeline: "StableDiffusionPipeline"
      enabled: false  # Erst nach Test aktivieren
      format: "checkpoint"

# Server-Einstellungen
server:
  host: "127.0.0.1"
  port: 3000
  debug: true
  
# Performance
performance:
  prefer_gpu: true
  low_vram_mode: false
  enable_xformers: true
  
# Content-Generierung
generation:
  default_steps: 20
  default_guidance: 7.5
  default_size: [1024, 1024]
  max_batch_size: 1
"""
    
    os.makedirs("config", exist_ok=True)
    with open("config/settings.yaml", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("⚙️ Einfache Config erstellt: config/settings.yaml")

def verify_essential_files():
    """Prüft ob alle essenziellen Dateien vorhanden sind"""
    
    print("\n🔍 === VERIFIKATION ===")
    
    essential_files = [
        "start.py",
        "server/",
        "web/", 
        "models/",
        "config/"
    ]
    
    all_good = True
    for item in essential_files:
        if os.path.exists(item):
            print(f"✅ {item}")
        else:
            print(f"❌ FEHLT: {item}")
            all_good = False
    
    if all_good:
        print("\n🎯 Alle essenziellen Komponenten vorhanden!")
        print("Bereit zum Server-Start!")
    else:
        print("\n⚠️ Fehlende Komponenten gefunden!")
    
    return all_good

if __name__ == "__main__":
    cleanup_root_directory()
    create_simple_config()
    ready = verify_essential_files()
    
    if ready:
        print("\n🚀 === BEREIT ZUM START ===")
        print("Führe aus: python start.py --debug")
        print("Dann öffne: http://localhost:3000")
    else:
        print("\n⚠️ Reparaturen erforderlich vor Start")