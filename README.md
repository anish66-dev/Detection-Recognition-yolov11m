# Human Detection + Face Recognition (YOLOv11m + InsightFace)

This project provides a browser UI and Flask backend to:

1. Detect people in uploaded videos with YOLOv11m.
2. Recognize faces against reference photos using InsightFace buffalo_m models.
3. Export per-frame detection coordinates to CSV.

## What Runs in the Live App

Required runtime files:

- `index.html` (frontend)
- `app.py` (Flask API)
- `pipeline.py` (detection + face embedding)
- `video_test/reference_photos/` (your known-person reference images)
- `video_test/models/buffalo_m/` (InsightFace ONNX files)

Optional/offline scripts (not needed for live UI processing):

- `build_references.py`
- `analyze_all_videos.py`
- `generate_all_visuals.py`
- `evaluate_lfw_full.py`
- `plot_lfw_histogram.py`

## Prerequisites

- Python 3.11 or 3.12
- ffmpeg installed and available on PATH
- Optional NVIDIA GPU for faster inference

## Setup (Windows, venv inside this project)

Run all commands from the project root folder (important):

```powershell
cd .\human-detection-yolov11m
```

Create and activate local venv in this same folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch (choose one):

```powershell
# CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CUDA 11.8
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
# pip install torch torchvision
```

Install project dependencies:

```powershell
pip install -r requirements.txt
```

Install ffmpeg if needed:

```powershell
winget install --id Gyan.FFmpeg -e
```

## Add InsightFace Models (Required)

Your current pipeline expects these files:

- `video_test/models/buffalo_m/det_2.5g.onnx`
- `video_test/models/buffalo_m/w600k_r50.onnx`

Recommended method (download via InsightFace once, then copy):

```powershell
python -c "from insightface.app import FaceAnalysis; a=FaceAnalysis(name='buffalo_m'); a.prepare(ctx_id=-1, det_size=(640,640)); print('downloaded')"

$src = "$env:USERPROFILE\.insightface\models\buffalo_m"
$dst = "video_test\models\buffalo_m"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item "$src\det_2.5g.onnx" "$dst\" -Force
Copy-Item "$src\w600k_r50.onnx" "$dst\" -Force
```

Verify:

```powershell
Get-ChildItem "video_test\models\buffalo_m\*.onnx"
```

## Add Reference Photos

Put known-person images in:

- `video_test/reference_photos/`

At startup, `app.py` loads these images and builds an in-memory reference database.

## Run the App

Always run from project root so relative paths resolve correctly:

```powershell
cd .\human-detection-yolov11m
.\.venv\Scripts\Activate.ps1
python app.py
```

Then open `index.html` in Chrome/Edge.

Health endpoint:

- `http://localhost:5000/health`

You want:

- `recognition_ready: true`
- `reference_faces > 0`

## Recognition Frequency (Performance Control)

The UI includes a `Recognition stride` slider.

- `1` = run recognition every frame (slowest, most frequent updates)
- `5` = every 5th frame (faster)
- `10+` = much faster, but identity updates less often

This maps to backend field `recognition_every`.

## Output Folders

These are used by runtime and are auto-created:

- `uploads/` temporary uploaded videos
- `outputs/` processed video files
- `coords/` generated CSV files

## CSV Output Columns

Each detection row includes:

- `frame`, `person_id`
- `x1`, `y1`, `x2`, `y2`, `width_px`, `height_px`
- `confidence`
- `recognized_name`, `recognition_score`, `face_found`
- `frame_width`, `frame_height`

## Troubleshooting

If you see:

- `model_file ... det_2.5g.onnx should exist`
- `model_file ... w600k_r50.onnx should exist`

It usually means wrong current working directory. Run app from project root, not from another folder.

If you see ONNX warnings like CPU provider fallback, recognition still works but runs slower.

## Optional: GPU for InsightFace ONNX

If you want ONNX GPU inference later:

```powershell
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Then use `ctx_id=0` in `pipeline.py` model `prepare(...)` calls.

Integration of the following:

https://github.com/Chatradhara007/human-detection-yolov11m

https://github.com/P-Akshay-kumar/buffalo-recognition-demo
