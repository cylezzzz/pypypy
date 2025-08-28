# 🎨 AndioMediaStudio

**Ein universelles lokales KI-Medienstudio für Desktop & Heimnetzwerk**

AndioMediaStudio ist eine umfassende, selbst-gehostete AI-Mediengeneration-Platform ohne Inhaltseinschränkungen. Es kombiniert **Bild- und Video-Generierung**, **Bearbeitung**, **Posenerkennung**, **Lip-Sync**, **Outfit-Editing** und mehr — alles mit lokalen Modellen und maximaler kreativer Freiheit.

[![Version](https://img.shields.io/badge/version-0.5.0-blue.svg)](https://github.com/your-repo/andio-media-studio)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-supported-green.svg)](https://developer.nvidia.com/cuda-zone)

## ✨ Hauptfeatures

### 🎨 **Bildgenerierung**
- **Text → Bild** (Stable Diffusion, FLUX, SDXL)
- **Bild → Bild** Transformation mit vollständiger Kontrolle
- **Inpainting** mit automatischer Maskenerstellung (SAM/MediaPipe)
- **ControlNet** für Pose-Transfer und Strukturkontrolle
- **Batch-Generierung** mit verschiedenen Seeds
- **LoRA-Unterstützung** für personalisierte Styles

### 🎬 **Video-Generierung**
- **Text → Video** mit Stable Video Diffusion
- **Bild → Video** Animation statischer Bilder
- **Video → Video** Transformation bestehender Clips
- **Motion Transfer** zwischen Videos
- **Flexible Auflösungen** (16:9, 9:16, 1:1, Cinematic)
- **Frame-Rate Kontrolle** (8-30 fps)

### 💬 **Talking Head & Lip-Sync**
- **Wav2Lip** Integration für präzises Lip-Sync
- **SadTalker** für vollständige Gesichtsanimation
- **Audio-zu-Video** Synchronisation
- **Multi-Language** Unterstützung

### 🛡️ **Keine Einschränkungen**
- **Kein Content-Filter** - volle kreative Kontrolle
- **NSFW-Content erlaubt** - für Kunst und kreative Projekte
- **Lokale Verarbeitung** - deine Daten bleiben privat
- **Keine Cloud-Abhängigkeit** - funktioniert vollständig offline

## 🚀 Schnellstart

### Voraussetzungen
- **Python 3.10+**
- **NVIDIA GPU** mit 8GB+ VRAM empfohlen (CPU funktioniert auch)
- **Git** für Repository-Management
- **Node.js 20+** (optional für Desktop-App)

### Installation

1. **Repository klonen:**
```bash
git clone https://github.com/your-repo/AndioMediaStudio.git
cd AndioMediaStudio
```

2. **Python-Umgebung einrichten:**
```bash
# Virtuelle Umgebung erstellen
python -m venv .venv

# Aktivieren (Windows)
.venv\Scripts\activate
# Oder (Linux/Mac)
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

3. **Modelle herunterladen:**
```bash
# Beispiel: Stable Diffusion XL herunterladen
mkdir -p models/image
cd models/image
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
cd ../..
```

4. **Server starten:**
```bash
python start.py
```

5. **Studio öffnen:**
   - Im Browser: [http://localhost:3000](http://localhost:3000)
   - Im Netzwerk: `http://<deine-ip>:3000`

### Desktop-App (Optional)

```bash
cd desktop/electron
npm install
npm start
```

## 📁 Projektstruktur

```plaintext
AndioMediaStudio/
├── 📁 server/              # FastAPI-Backend & AI-Pipelines
│   ├── 📁 config/          # Konfigurationsdateien
│   ├── 📁 pipelines/       # AI-Generierungs-Pipelines
│   ├── 📁 registry/        # Modell-Management
│   └── 📄 app.py           # Haupt-API Server
├── 📁 web/                 # Web-Interface (HTML/CSS/JS)
│   ├── 📁 assets/          # Stylesheets & JavaScript
│   ├── 📄 index.html       # Dashboard
│   ├── 📄 create.html      # Bildgenerierung
│   ├── 📄 video.html       # Video Lab
│   ├── 📄 gallery.html     # Medien-Galerie
│   └── 📄 talking.html     # Talking Head Studio
├── 📁 desktop/electron/    # Desktop-App (Electron)
├── 📁 models/              # KI-Modelle (lokal)
│   ├── 📁 image/           # Bildmodelle (SD, SDXL, etc.)
│   ├── 📁 video/           # Videomodelle (SVD, AnimateDiff)
│   └── 📁 llm/             # Sprachmodelle (Ollama)
├── 📁 outputs/             # Generierte Inhalte
│   ├── 📁 images/          # Generierte Bilder
│   └── 📁 videos/          # Generierte Videos
├── 📁 workspace/           # Upload-Bereich
└── 📄 start.py             # Startup-Script
```

## 🤖 Unterstützte Modelle

### Bildmodelle
- **Stable Diffusion 1.5** - Klassiker für hochqualitative Bilder
- **Stable Diffusion XL** - Neueste Generation mit verbesserter Qualität
- **FLUX** - Cutting-edge Bildgenerierung
- **Custom Fine-Tunes** - Deine eigenen trainierten Modelle
- **LoRA Weights** - Lightweight Style-Anpassungen

### Video-Modelle
- **Stable Video Diffusion** - Text/Bild zu Video
- **AnimateDiff** - Bewegungsmodule für SD-Modelle
- **Custom Video Models** - Eigene trainierte Videomodelle

### Zusätzliche Features
- **ControlNet** - Pose, Depth, Canny Edge Control
- **SAM (Segment Anything)** - Automatische Objektsegmentierung
- **MediaPipe** - Echtzeit-Posenerkennung
- **Wav2Lip** - Präzises Audio-zu-Video Lip-Sync
- **SadTalker** - Vollständige Gesichtsanimation

## 🎯 Anwendungsfälle

### 🎨 **Kreative Projekte**
- Konzeptkunst und Illustration
- Charakter-Design für Games/Filme
- Storyboard-Entwicklung
- Künstlerische Experimente ohne Grenzen

### 🎬 **Content Creation**
- YouTube-Thumbnails und Banner
- Social Media Content
- Werbevisuals und Marketing
- Animierte GIFs und Memes

### 🏢 **Professionelle Anwendungen**
- Rapid Prototyping für Design
- Visualisierung von Ideen
- Präsentationsmaterial
- Produktvisualisierung

### 🔬 **Forschung & Entwicklung**
- AI-Modell-Experimente
- Generative AI Research
- Computer Vision Training
- Multimodale AI-Entwicklung

## ⚙️ Konfiguration

### Basis-Konfiguration (`server/config/settings.yaml`)
```yaml
# Server-Einstellungen
host: "0.0.0.0"
port: 3000
mode: "creative"  # Keine Inhaltsfilter

# Performance
prefer_gpu: true
max_parallel_workers: 2
enable_xformers: true

# Content-Policy
nsfw_allowed: true
safety_checker: false
content_filter: false
```

### Qualitäts-Presets (`server/config/presets.yaml`)
```yaml
image:
  FAST:    { steps: 15, guidance: 3.5, width: 768,  height: 1024 }
  BALANCED: { steps: 28, guidance: 6.0, width: 896,  height: 1152 }
  ULTRA:   { steps: 40, guidance: 7.5, width: 1024, height: 1344 }

video:
  FAST:    { duration: 4, fps: 12, steps: 15 }
  BALANCED: { duration: 6, fps: 16, steps: 25 }
  ULTRA:   { duration: 8, fps: 24, steps: 35 }
```

## 🔧 API-Dokumentation

### Bildgenerierung
```bash
curl -X POST http://localhost:3000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "task": "txt2img",
    "prompt": "ein wunderschöner Sonnenuntergang über den Bergen, ultrarealistisch, 8K",
    "quality": "ULTRA",
    "inputs": {
      "width": 1024,
      "height": 1344,
      "steps": 40,
      "guidance": 7.5,
      "seed": 42
    }
  }'
```

### Video-Generierung
```bash
curl -X POST http://localhost:3000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "task": "txt2video",
    "prompt": "ein majestätischer Adler gleitet durch Bergtäler, kinematisch",
    "quality": "BALANCED",
    "inputs": {
      "duration": 6,
      "fps": 16,
      "width": 1024,
      "height": 576
    }
  }'
```

### Direkte AI-Zugriff (für Experten)
```bash
curl -X POST http://localhost:3000/api/direct \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_type": "txt2img",
    "model_path": "image/stable-diffusion-xl-base-1.0",
    "prompt": "custom prompt with full control",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 9.0
  }'
```

## 🎮 Web-Interface Features

### 📊 **Dashboard**
- System-Status und GPU-Monitoring
- Aktive Jobs und Warteschlange
- Modell-Übersicht und -Management
- Direkte AI-Zugriffs-Konsole

### 🎨 **Create Studio**
- Intuitive Prompt-Eingabe mit Presets
- Drag & Drop Asset-Upload
- Live-Parameter-Kontrolle
- Batch-Generierung mit verschiedenen Seeds
- Automatische Maskenerstellung

### 🎬 **Video Lab**
- Multi-Format Video-Generierung
- Integrierter Video-Player mit Analyse-Tools
- Frame-Extraktion und Thumbnail-Generation
- Motion-Analyse und Stabilisierung
- Export in verschiedene Formate

### 🖼️ **Galerie**
- Erweiterte Filter- und Suchfunktionen
- Batch-Operationen (Download, Löschen)
- Metadaten-Anzeige mit Generierungs-Parametern
- Lightbox-Viewer für Bilder und Videos
- Drag & Drop Organisation

## 🔌 Erweiterte Features

### 🎭 **Auto-Masking**
```python
# MediaPipe Selfie Segmentation
await API.autoMask('path/to/image.jpg', 'mediapipe')

# SAM (Segment Anything Model) - wenn verfügbar
await API.autoMask('path/to/image.jpg', 'sam')
```

### 🤸 **Pose-Transfer**
```python
# ControlNet OpenPose
job = JobBuilder()
  .setTask('pose_transfer')
  .setPrompt('person in different outfit')
  .setInput('image_path', 'source.jpg')
  .setInput('pose_path', 'reference_pose.jpg')
  .execute()
```

### 💬 **Lip-Sync Integration**
```python
# Wav2Lip - Audio zu Video Synchronisation
job = JobBuilder()
  .setTask('lipsync')
  .setInput('face_video', 'person.mp4')
  .setInput('audio_path', 'speech.wav')
  .execute()
```

## 🛠️ Entwicklung & Anpassung

### Eigene Pipelines hinzufügen
```python
# server/pipelines/custom_pipeline.py
from .base_pipeline import BasePipeline

class CustomPipeline(BasePipeline):
    def __init__(self, base_dir: Path):
        super().__init__(base_dir)
    
    @torch.inference_mode()
    def generate(self, prompt: str, **kwargs):
        # Deine eigene Implementierung
        return images, metadata
```

### Custom Model Support
```python
# Modell-Scanner erweitern
def tag_by_name(name: str):
    if "my_custom_model" in name.lower():
        return ["txt2img", "custom", "special_feature"]
    return []
```

### Web-Interface anpassen
```javascript
// web/assets/custom.js
class CustomFeature {
    constructor() {
        this.setupCustomUI();
    }
    
    setupCustomUI() {
        // Deine eigenen UI-Komponenten
    }
}
```

## 📋 Systemanforderungen

### Minimum (CPU-Modus)
- **CPU:** Intel i5 oder AMD Ryzen 5
- **RAM:** 16GB System-Memory
- **Storage:** 50GB freier Speicherplatz
- **Python:** 3.10 oder höher

### Empfohlen (GPU-Beschleunigt)
- **GPU:** NVIDIA RTX 3070 oder besser (8GB+ VRAM)
- **CPU:** Intel i7 oder AMD Ryzen 7
- **RAM:** 32GB System-Memory
- **Storage:** 100GB+ SSD-Speicherplatz
- **CUDA:** 11.8 oder höher

### Optimal (Professionell)
- **GPU:** RTX 4090 oder A6000 (24GB+ VRAM)
- **CPU:** Intel i9 oder AMD Ryzen 9
- **RAM:** 64GB+ System-Memory
- **Storage:** 500GB+ NVMe SSD
- **Network:** Gigabit für Multi-Device Zugriff

## 🔒 Sicherheit & Datenschutz

### Lokale Verarbeitung
- Alle AI-Operationen laufen lokal auf deinem System
- Keine Daten werden an externe Server gesendet
- Vollständige Kontrolle über deine kreativen Inhalte

### Netzwerk-Sicherheit
```yaml
# Für Produktionsumgebungen anpassen
trusted_hosts: ["localhost", "192.168.1.*"]
cors_origins: ["http://localhost:3000"]
enable_https: true  # HTTPS in Produktion
```

### Content-Policy
```yaml
# Vollständige kreative Freiheit (Standard)
nsfw_allowed: true
safety_checker: false
content_filter: false

# Für öffentliche Installationen (optional)
nsfw_allowed: false
safety_checker: true
content_filter: true
```

## 🐛 Fehlerbehebung

### Häufige Probleme

**GPU wird nicht erkannt:**
```bash
# CUDA-Installation prüfen
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# PyTorch neu installieren mit CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Out of Memory Fehler:**
```yaml
# In settings.yaml
low_vram_mode: true
enable_cpu_offload: true
enable_vae_slicing: true
enable_attention_slicing: true
```

**Modelle werden nicht gefunden:**
```bash
# Modell-Verzeichnis prüfen
ls -la models/image/
ls -la models/video/

# Berechtigungen korrigieren
chmod -R 755 models/
```

**Langsame Generierung:**
```yaml
# Performance-Optimierungen
enable_xformers: true
compile_unet: true  # PyTorch 2.0+
max_parallel_workers: 1  # Für wenig VRAM
```

## 🤝 Beitragen

### Development Setup
```bash
# Development-Modus
git clone https://github.com/your-repo/AndioMediaStudio.git
cd AndioMediaStudio

# Pre-commit hooks installieren
pip install pre-commit
pre-commit install

# Tests ausführen
pytest tests/

# Code-Formatierung
black server/ web/
ruff server/ --fix
```

### Contribution Guidelines
1. **Fork** das Repository
2. **Branch** für dein Feature erstellen (`git checkout -b feature/amazing-feature`)
3. **Commit** deine Änderungen (`git commit -m 'Add amazing feature'`)
4. **Push** zum Branch (`git push origin feature/amazing-feature`)
5. **Pull Request** öffnen

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe [LICENSE](LICENSE) für Details.

## ⚠️ Rechtliche Hinweise

- **Nur für private und kreative Nutzung bestimmt**
- **Keine Inhaltsfilter** - Nutzer ist verantwortlich für generierten Content
- **Modell-Lizenzen beachten** - Verschiedene AI-Modelle haben eigene Lizenzbedingungen
- **Urheberrecht respektieren** - Keine Verletzung von Marken oder Persönlichkeitsrechten

## 🌟 Roadmap

### Version 0.6.0 (Geplant)
- [ ] **3D Model Generation** mit TriPo/InstantMesh
- [ ] **Advanced ControlNet** (Depth, Normal, Lineart)
- [ ] **Multi-GPU Support** für größere Installationen
- [ ] **Plugin-System** für eigene Erweiterungen

### Version 0.7.0 (Future)
- [ ] **Real-time Generation** mit LCM/Turbo-Modellen
- [ ] **Voice Cloning** Integration
- [ ] **Advanced Video Editing** Tools
- [ ] **Collaborative Features** für Teams

## 📞 Support

### Community
- **GitHub Discussions:** Für allgemeine Fragen und Ideas
- **GitHub Issues:** Für Bug-Reports und Feature-Requests
- **Discord:** [Community-Server beitreten](https://discord.gg/your-server)

### Professioneller Support
Für Enterprise-Installationen und kommerzielle Nutzung kontaktiere uns unter: support@andio-media-studio.com

---

**Made with ❤️ for the AI Art Community**

*AndioMediaStudio - Wo Kreativität keine Grenzen kennt* 🎨✨