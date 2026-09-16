# Security Platform — YOLOv11m + InsightFace

Real-time human detection, face recognition, restricted-zone monitoring, and intruder
alerting for uploaded video and live webcam feeds. A single Flask service pairs
**Ultralytics YOLOv11m** for person detection with an **InsightFace** face-recognition
pipeline (SCRFD → landmark alignment → ArcFace), streaming annotated frames and live
alerts to a browser console.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-black?style=flat-square&logo=flask)
![YOLOv11](https://img.shields.io/badge/YOLO-v11m-green?style=flat-square)
![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__l-orange?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red?style=flat-square&logo=opencv)

---

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Face recognition model setup](#face-recognition-model-setup)
- [Running the app](#running-the-app)
- [Using the console](#using-the-console)
- [Configuration reference](#configuration-reference)
- [API reference](#api-reference)
- [Data and storage](#data-and-storage)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Video and webcam input** — Process an uploaded file (MP4, AVI, MOV, MKV, any
  resolution) or a live camera stream, with an annotated preview rendered frame by frame.
- **YOLOv11m person detection** — High-accuracy human detection on every frame, with
  adjustable confidence and IoU (NMS) thresholds.
- **Face recognition** — A four-stage pipeline (person detection → face detection →
  5-point alignment → ArcFace embedding) matches faces against an enrolled database by
  cosine similarity.
- **Enrollment** — Register people from a single clear face photo and tag each as
  **Authorized** or **Blocklisted**; embeddings are stored in a local SQLite database.
- **Restricted zones** — Draw polygon zones directly on a video frame; people whose feet
  fall inside a zone are highlighted and logged separately.
- **Multi-tier alerting**, streamed live over Server-Sent Events (SSE):
  - **Blocklist match** — a known blocklisted person is detected.
  - **Zone intrusion** — a person enters a drawn restricted zone.
  - **Intruder** — an unrecognised person is detected (toggleable per run).
- **Auto-calibrate** — Samples frames from the uploaded video to recommend confidence and
  IoU settings.
- **Exports** — Download the annotated video and a per-frame detection CSV.
- **Live telemetry** — The console header reports real device (CUDA/CPU), FFmpeg, and
  face-model status pulled from the server's health endpoint.

---

## How it works

The recognition pipeline runs in four stages (`pipeline.py`):

| Stage | Task | Model |
|------:|------|-------|
| 1 | Person detection | YOLOv11m (`yolo11m.pt`) |
| 2 | Face detection | SCRFD (`det_10g.onnx`, from the `buffalo_l` pack) |
| 3 | Landmark alignment | InsightFace 5-point `norm_crop` |
| 4 | Embedding + match | ArcFace (`w600k_r50.onnx`) + cosine similarity |

Detected embeddings are compared against enrolled embeddings; a match above the cosine
similarity threshold (default `0.35`) resolves the person's identity and status, which
drives the on-frame annotation colour and any alert.

**Annotation colours:** green = person outside all zones · orange = person inside a zone ·
blue = authorized match · red = blocklist match · magenta = unknown intruder.

---

## Tech stack

- **Backend:** Flask, Flask-CORS
- **Detection:** Ultralytics YOLOv11m, OpenCV
- **Face recognition:** InsightFace (SCRFD + ArcFace) on ONNX Runtime
- **Video encoding:** `imageio-ffmpeg` (bundled FFmpeg) for browser-friendly H.264 output
- **Storage:** SQLite (enrolled people, embeddings, alerts)
- **Frontend:** Single-file HTML/CSS/JS console (`index.html`), no build step

---

## Project structure

```
.
├── app.py              # Flask entry point: creates app, registers blueprints, CLI runner
├── config.py           # All constants: colours, paths, thresholds, tuning parameters
├── analysis.py         # Frame analysis (Stage 1) and rendering (Stage 3)
├── geometry.py         # Zone geometry, calibration statistics, FFmpeg helpers
├── camera.py           # Camera source abstraction (local, network, phone push)
├── live_session.py     # Threaded live camera session (capture / inference / encode)
├── app_state.py        # Shared mutable state (jobs, session, faces cache, alert subs)
├── pipeline.py         # 4-stage detection + recognition pipeline and model loading
├── tracker.py          # Multi-object tracker with EMA smoothing and identity voting
├── alerts.py           # Edge-triggered alert rules engine
├── database.py         # SQLite schema and helpers (people, embeddings, alerts)
├── routes/             # Flask blueprints, one per route domain
│   ├── main_routes.py      # /, /phone, /health
│   ├── process_routes.py   # /process/*, /coords/*, /auto_calibrate
│   ├── webcam_routes.py    # /webcam/*, /camera/*
│   ├── enroll_routes.py    # /enroll, /people, /people/<id>
│   └── alert_routes.py     # /alerts, /alerts/clear, /alerts/stream
├── index.html          # Security console UI (served at /)
├── phone.html          # Phone camera page (served at /phone)
├── test_logic.py       # Dependency-free tests for tracker, alerts, and analysis logic
├── requirements.txt    # Python dependencies
├── yolo11m.pt          # YOLOv11m weights (auto-downloaded on first run if absent)
└── data/
    └── enrolled.db     # SQLite database (created on first run)
```

Face-recognition model files are **not** stored in the project. They live in the
InsightFace cache at `~/.insightface/models/buffalo_l/` — see
[Face recognition model setup](#face-recognition-model-setup).

---

## Prerequisites

- **Python 3.10 or newer**
- **pip** and, recommended, a virtual environment
- **~1 GB free disk** for models (YOLOv11m ≈ 40 MB, `buffalo_l` ≈ 290 MB)
- **Optional but recommended:** an NVIDIA GPU with CUDA for real-time performance. CPU-only
  works but is significantly slower.

---

## Installation

### 1. Get the code and create a virtual environment

```bash
# from the project folder
python -m venv .venv

# activate it
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows (PowerShell)
```

### 2. Install PyTorch

Install PyTorch first, matched to your hardware. Check your CUDA version with `nvidia-smi`.

```bash
# NVIDIA GPU, CUDA 12.x (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# NVIDIA GPU, CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only (slower)
pip install torch torchvision
```

### 3. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

> **ONNX Runtime and the GPU:** `requirements.txt` installs `onnxruntime` (CPU). If you
> have a GPU and want face recognition to use it, install the GPU build instead:
> `pip uninstall -y onnxruntime && pip install onnxruntime-gpu`. Install only one of the
> two — they conflict.

---

## Face recognition model setup

On first start the server attempts to download the `buffalo_l` InsightFace pack
automatically. YOLOv11m weights are fetched automatically by Ultralytics. If the automatic
download succeeds, you can skip this section.

If the download fails (a common cause is an interrupted or region-blocked GitHub transfer),
place the files manually — the loader scans the model folder and picks up the detection
(`det_*`) and recognition (`w600k_*`) ONNX files by pattern, so a manual copy is fully
supported.

### Manual installation

1. **Download `buffalo_l.zip` (~290 MB)** from either source:
   - GitHub: `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip`
   - SourceForge mirror: `https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download`

2. **Extract it** so the `.onnx` files sit directly inside the model folder:

   **Windows (PowerShell)** — note that File Explorer can't create the leading-dot folder,
   so use PowerShell:

   ```powershell
   $dest = "$env:USERPROFILE\.insightface\models\buffalo_l"
   New-Item -ItemType Directory -Force -Path $dest | Out-Null
   Expand-Archive -Path "$env:USERPROFILE\Downloads\buffalo_l.zip" -DestinationPath $dest -Force
   Get-ChildItem -Recurse $dest -Filter *.onnx | Select-Object Name, Length
   ```

   **Linux / macOS:**

   ```bash
   mkdir -p ~/.insightface/models/buffalo_l
   unzip ~/Downloads/buffalo_l.zip -d ~/.insightface/models/buffalo_l
   ls -la ~/.insightface/models/buffalo_l
   ```

3. **Verify the layout.** The folder should contain the ONNX files directly (not nested):

   ```
   ~/.insightface/models/buffalo_l/
   ├── det_10g.onnx        # face detector   (required)
   ├── w600k_r50.onnx      # face recognizer (required)
   ├── 1k3d68.onnx
   ├── 2d106det.onnx
   └── genderage.onnx
   ```

   If the files landed in a nested `buffalo_l/buffalo_l/` folder, move them up one level.
   The server also auto-corrects this on startup.

When the models are present, the console header shows **Face model: Ready** and the startup
log prints `Face detection + recognition ready.`

---

## Running the app

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

The server listens on `0.0.0.0:5000` and serves the console at `/`. A healthy startup log
looks like:

```
[INFO] Loading YOLOv11m...
[INFO] Loading face detector (det_10g.onnx)...
[INFO] Loading face recognizer (w600k_r50.onnx)...
[INFO] Face detection + recognition ready.
[INFO] CUDA: True
[INFO] Open this in your browser: http://localhost:5000
```

---

## Using the console

The interface has four tabs:

- **Video** — Upload a clip, adjust detection settings (or use **Auto-detect**),
  optionally draw restricted zones on the frame, toggle **Intruder detection**, then
  **Run detection**. A live annotated preview streams while processing; when it finishes,
  download the annotated video or the per-frame CSV.
- **Webcam** — Start the local camera for a live detection overlay.
- **Enroll** — Add a person from a clear face photo and set their status (Authorized or
  Blocklisted). Enrolled people are listed and can be deleted.
- **Alerts** — A live feed of blocklist, zone-intrusion, and intruder events, streamed over
  SSE. A red indicator appears on the tab when a new alert arrives.

---

## Configuration reference

Defaults live in `config.py` and `pipeline.py`.

| Setting | Default | Where | Notes |
|---------|---------|-------|-------|
| Server host / port | `0.0.0.0:5000` | `app.py` (`app.run`) | Change the `port=` argument to relocate |
| Confidence threshold | `0.40` | `pipeline.py` / UI | Clamped to `0.15–0.75` for auto-calibrate |
| IoU (NMS) threshold | `0.45` | `pipeline.py` / UI | Clamped to `0.30–0.75` for auto-calibrate |
| Recognition threshold | `0.35` | `pipeline.py` (`RECOG_THRESH`) | Cosine similarity for a face match |
| Calibration samples | `15` | `config.py` (`CALIBRATION_SAMPLES`) | Frames sampled by auto-detect |
| Model pack | `buffalo_l` | `pipeline.py` (`_MODEL_NAME`) | Loader matches `det_*` / `w600k_*` |

Box, zone, alert, authorized, and intruder annotation colours are defined as BGR constants
in `config.py`.

---

## API reference

All endpoints are served from the same origin as the console.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/` | Serve the console UI |
| `GET` | `/health` | Device, FFmpeg, and face-model status (drives the header telemetry) |
| `POST` | `/process/start` | Start a video job. Multipart form: `video` file + `conf`, `iou`, `thickness`, `font_size`, `zones` (JSON), `intruder_detection` |
| `GET` | `/process/stream/<job_id>` | SSE stream of annotated frames and progress |
| `GET` | `/process/status/<job_id>` | Job status snapshot |
| `GET` | `/process/result/<job_id>` | Download the annotated output video |
| `GET` | `/coords/<job_id>` | Download the per-frame detection CSV |
| `POST` | `/auto_calibrate` | Recommend `conf`/`iou` from sampled frames |
| `POST` | `/webcam/start` · `GET /webcam/status` · `POST /webcam/stop` · `GET /webcam/stream` | Live camera control and MJPEG-style stream |
| `POST` | `/enroll` | Enroll a person. Multipart form: `photo` file + `name`, `status` |
| `GET` | `/people` | List enrolled people |
| `DELETE` | `/people/<id>` | Remove a person |
| `GET` | `/alerts` · `POST /alerts/clear` · `GET /alerts/stream` | Read, clear, and live-stream alerts |

---

## Data and storage

- **Database:** `data/enrolled.db` (SQLite), created on first run, with three tables —
  `people`, `embeddings`, and `alerts`.
- **Working files:** uploads, encoded outputs, coordinate CSVs, and enrolment photos are
  written to a temporary directory **outside** the project, under your OS temp folder at
  `…/yolov11m_human_detection/`. These are scratch files and can be cleared safely.
- **Models:** cached in `~/.insightface/models/` (face) and the Ultralytics cache /
  project folder (`yolo11m.pt`).

---

## Troubleshooting

**"No face detected in photo" on every enrollment.**
Almost always means the face model isn't loaded, not that the photo is bad. Check
**http://localhost:5000/health** and read `face_detail.error`, or watch the server console.
See [Face recognition model setup](#face-recognition-model-setup) to install the model
files manually. If a photo genuinely has no clear, front-facing face, use a different image.

**Header shows "Face model: Not loaded" / amber banner.**
The `buffalo_l` files aren't in `~/.insightface/models/buffalo_l/`, or `insightface` /
`onnxruntime` isn't installed in the active environment. Verify with
`python -c "import insightface, onnxruntime; print('ok')"` and install the model files.

**Model download fails partway (empty error, stops before 100%).**
This is typically an interrupted GitHub transfer. Delete any partial folder
(`~/.insightface/models/buffalo_l`) and install the model manually, ideally via a browser
or `wget -c` so the download can resume, or use the SourceForge mirror.

**Annotated video won't play in the browser.**
FFmpeg wasn't available for H.264 re-encoding. `imageio-ffmpeg` ships a bundled binary; if
the header shows FFmpeg as unavailable, reinstall it with `pip install imageio-ffmpeg`. The
**Download video** button still works regardless.

**CUDA shows as CPU, or recognition is slow.**
Install the CUDA build of PyTorch (see [Installation](#installation)) and `onnxruntime-gpu`
for the face models. Confirm your driver with `nvidia-smi`. If `onnxruntime-gpu` errors on a
missing `cudnn`/`cublas` library, match its version to your installed CUDA/cuDNN, or fall
back to CPU `onnxruntime`.

**Webcam won't start.**
Grant the browser camera permission, close other apps using the camera, and make sure the
server host can access a camera device.

---

## License

Add your license here (for example, MIT). If you redistribute the models, review the
InsightFace and Ultralytics model licenses, which are separate from this project's code.
