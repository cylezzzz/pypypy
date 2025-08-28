# Andio Media Studio – Full Local (Phase 2 ready)

Lokal, GPU-beschleunigt, mit echten Pipelines:
- **Txt→Img / Img→Img / Inpainting** (Diffusers)
- **Pose-Erkennung** (MediaPipe; Control-Guides vorbereitbar)
- **Txt→Video** (Stable Video Diffusion, wenn Modell im `models/video` vorhanden)
- **Galerie & Player**, **Model-Scanner**, **Electron-Desktop**

## Schnellstart
```bash
pip install -r requirements.txt
python start.py
# Browser: http://<PC>:3000
```
Optional Desktop:
```bash
cd desktop/electron && npm install && npm start
```

## Modelle
Lege deine Modelle hier ab:
- `models/image/` → Diffusers-Ordner (mit `model_index.json`) oder Gewichte (werden Ordner-weise erkannt)
- `models/video/` → SVD / Video-Modelle

Beim Start wird `models/catalog.json` generiert.

## API (kurz)
- `POST /api/jobs` mit JSON:
  - Bild:
    - `{"task":"txt2img","prompt":"...","quality":"ULTRA","model_preference":"<name>"}`
    - `{"task":"img2img","prompt":"...","inputs":{"image_path":"workspace/xyz.png"}}`
    - `{"task":"inpaint","prompt":"...","inputs":{"image_path":"workspace/xyz.png","mask_path":"workspace/mask.png"}}`
  - Video:
    - `{"task":"txt2video","prompt":"...","inputs":{"frames":25,"fps":16}}`
- `POST /api/upload` (multipart) → legt Datei in `workspace/` ab
- `GET /api/models` – gelistete Modelle
- `GET /api/outputs?kind=all|images|videos`

## Ollama-Agent
Der Agent nutzt **Ollama** (falls installiert) für Topic/NSFW-Klassifikation zur besseren Modellvorschlagsliste.
Keine Blocker – reine Empfehlung.

## Lip-Sync (Wav2Lip) & Talking (SadTalker)
Die Hooks sind vorbereitet, aber du musst dein lokales Repo/Checkpoint einbinden:
- Wav2Lip: Passe `server/pipelines/lipsync_wav2lip.py` an (Import aus deinem Repo, Checkpoint in `models/video/`).

## Hinweise
- Es gibt **keine Platzhalterbilder** – Pipelines erzeugen echte Ergebnisse, sofern kompatible Modelle vorhanden sind.
- Für Inpainting brauchst du eine **Maske** (schwarz/weiß). Du kannst sie extern erzeugen oder in Zukunft via SAM2 integrieren.
