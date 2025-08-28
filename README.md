
# 🎨 AndioMediaStudio

Ein universelles **lokales KI-Medienstudio** für Desktop & Heimnetzwerk.  
Es kombiniert **Bild- und Video-Generierung**, **Bearbeitung**, **Posenerkennung**, **Lip-Sync**, **Outfit-Editing** und mehr — alles mit lokalen Modellen und Agenten (z. B. [Ollama](https://ollama.ai)).

---

## 🚀 Features

- **Text → Bild** (Txt2Img)
- **Bild → Bild** (Img2Img)
- **Inpainting / Remover / Outfit-Change**  
- **Automatische Maskierung (SAM / MediaPipe)**
- **Posenerkennung & Pose-Transfer (ControlNet-OpenPose)**
- **Text → Video & Bild → Video** (Stable Video Diffusion, AnimateDiff)
- **Lip-Sync & Talking-Head** (Wav2Lip, SadTalker, vorbereitete Hooks)
- **Galerie & Player** (Web & Desktop)
- **Katalog / Store** – Übersicht aller lokal vorhandenen Modelle
- **Keine Filter** – volle kreative Kontrolle, privat nutzbar

---

## 📂 Projektstruktur

```plaintext
AndioMediaStudio/
├─ server/              # FastAPI-Backend & Pipelines
├─ web/                 # Web-Oberfläche (HTML/CSS/JS)
├─ desktop/electron/    # Electron-Wrapper für Desktop-App
├─ outputs/             # generierte Ergebnisse (Bilder/Videos)
├─ workspace/           # Uploads / Zwischendateien
└─ models/              # Lokale Modelle
```

### Modelle-Verzeichnis

```plaintext
models/
├─ image/
│  ├─ demo-sd15/
│  ├─ Realistic_Vision_V6.0_B1/
│  ├─ Realistic_Vision_V6.0_B1_noVAE/
│  ├─ stable-diffusion-xl-base-1.0/
│  ├─ stable-diffusion-xl-refiner-1.0/
│  ├─ tiny-sd/
│  ├─ sd15.safetensors
│  ├─ scheduler/
│  ├─ text_encoder_2/
│  ├─ tokenizer/
│  ├─ tokenizer_2/
│  └─ unet/
│
├─ llm/
│  └─ ollama.ai
│
└─ video/
   ├─ animatediff/
   ├─ AnimateDiff_Motion_Module_V3/
   ├─ Stable_Video_Diffusion_img2vid_/
   ├─ stable-video-diffusion-img2vid/
   ├─ stable-video-diffusion-img2vid-xt/
   ├─ wav2lip/        # hier wav2lip_gan.pth ablegen
   └─ sadtalker/      # SadTalker Pretrained-Files
```

ℹ️ Details findest du in [`models/README_models.txt`](models/README_models.txt).

---

## 🔧 Installation

### Voraussetzungen
- **Python 3.10+**
- **GPU mit CUDA** empfohlen (NVIDIA)
- **Node.js 20+** (nur wenn du Electron-Desktop nutzen willst)
- **Git** (für Repo-Management)
- Optional: **Ollama** (LLM-Agent)

### Setup
```bash
# Abhängigkeiten installieren
pip install -r requirements.txt

# Backend starten
python start.py
```

Danach im Browser:  
👉 [http://localhost:3000](http://localhost:3000)  
(oder von anderen Geräten im Heimnetz `http://<PC-IP>:3000`)

---

## 💻 Desktop-App (Electron)

```bash
cd desktop/electron
npm install
npm start
```

---

## 📡 API-Endpoints

- **POST `/api/jobs`** – Startet eine Aufgabe  
  Beispiel (Txt2Img):
  ```json
  {
    "task": "txt2img",
    "prompt": "ultrarealistisches Portrait, 85mm, volumetrisches Licht",
    "quality": "ULTRA",
    "model_preference": "Realistic_Vision_V6.0_B1"
  }
  ```

- **POST `/api/upload`** – Datei hochladen (landet in `workspace/`)  
- **GET `/api/models`** – Listet verfügbare Modelle  
- **GET `/api/outputs`** – Zeigt generierte Bilder/Videos  
- **POST `/api/auto_mask`** – Automatische Maske erzeugen (SAM / MediaPipe)

---

## ⚙️ Besondere Features

- **Auto-Maske**  
  Nutzt SAM-Checkpoint (`models/image/sam/*.pth`) oder fallback MediaPipe Selfie Segmentation.
- **ControlNet-Pose**  
  Erwartet ein OpenPose-ControlNet-Modell (`models/image/controlnet/…`).
- **Lip-Sync (Wav2Lip)**  
  `wav2lip_gan.pth` in `models/video/wav2lip/` einfügen.  
  Den Hook in `server/pipelines/lipsync_wav2lip.py` ggf. an dein Repo anpassen.
- **Talking-Head (SadTalker)**  
  SadTalker Pretrained-Modelle in `models/video/sadtalker/`.

---

## ⚠️ Hinweise

- **Dieses Projekt ist rein für private, kreative Nutzung gedacht.**
- Es gibt **keine Blockierungen oder Filter** – du hast volle Freiheit.
- Große Modelle werden **nicht ins GitHub-Repo hochgeladen** (siehe `.gitignore`).

---

## 📜 Lizenz

Privates Projekt – Nutzung auf eigene Verantwortung.
