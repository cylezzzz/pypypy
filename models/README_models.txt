
# Modelle-Verzeichnis – v4 (angepasst an deine Struktur)

Diese Ordnerstruktur ist vorbereitet, damit du deine vorhandenen Modelle einfach einfügen kannst:

models/
├─ image/
│  ├─ demo-sd15/                       → SD1.5 Demo-Modell
│  ├─ Realistic_Vision_V6.0_B1/        → Realistic Vision V6.0
│  ├─ Realistic_Vision_V6.0_B1_noVAE/  → Realistic Vision V6.0 ohne VAE
│  ├─ stable-diffusion-xl-base-1.0/    → SDXL Base
│  ├─ stable-diffusion-xl-refiner-1.0/ → SDXL Refiner
│  ├─ tiny-sd/                         → kleines Testmodell
│  ├─ scheduler/                       → Scheduler-Komponenten
│  ├─ text_encoder_2/                  → Text-Encoder
│  ├─ tokenizer/                       → Tokenizer
│  ├─ tokenizer_2/                     → zweiter Tokenizer
│  ├─ unet/                            → UNet-Komponenten
│  └─ sd15.safetensors                 → Einzelner Safetensors-Checkpoint (einfach hier ablegen)

├─ llm/
│  └─ ollama.ai                        → Ollama / LLM Konfigurationen

└─ video/
   ├─ animatediff/                     → AnimateDiff-Basis
   ├─ AnimateDiff_Motion_Module_V3/    → AnimateDiff Motion Modul
   ├─ Stable_Video_Diffusion_img2vid_/ → Stable Video Diffusion Variante
   ├─ stable-video-diffusion-img2vid/  → SVD Hauptmodell
   ├─ stable-video-diffusion-img2vid-xt/ → SVD XT-Modell
   ├─ wav2lip/                         → hier wav2lip_gan.pth einfügen
   └─ sadtalker/                       → SadTalker Pretrained-Files

## Hinweise
- Lege komplette Diffusers-Modelle mit `model_index.json` in den jeweiligen Ordnern ab.
- Einzeln heruntergeladene `.safetensors` oder `.ckpt` Dateien kannst du direkt in `models/image/` oder die passenden Unterordner legen.
- Für Wav2Lip: `wav2lip_gan.pth` in `models/video/wav2lip/` ablegen.
- Für SadTalker: kompletten Pretrained-Ordner in `models/video/sadtalker/` einfügen.

Beim Start von AndioMediaStudio wird dieses Verzeichnis gescannt und automatisch ein Katalog (`models/catalog.json`) erstellt.
