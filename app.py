"""
YOLOv11m Human Detection Backend
- Fixes video codec to H264 for browser playback
- Saves bounding box coordinates to CSV
- Returns annotated video + CSV download link
"""

import cv2
import json
import uuid
import csv
import io
import os
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from pipeline import detect_persons, extract_all_faces, cosine_similarity

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
COORDS_DIR = Path("coords")
REFERENCE_DIR = Path("video_test/reference_photos")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
COORDS_DIR.mkdir(exist_ok=True)
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

RECOGNITION_THRESHOLD = float(os.environ.get("RECOGNITION_THRESHOLD", "0.35"))
REFERENCE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_reference_name(filename):
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip() or "Reference"


from pipeline import (
    detect_persons,
    extract_all_faces,
    cosine_similarity,
    detect_faces,
    align_face,
    get_embedding,
)

def extract_reference_embedding(image, image_name="reference"):
    """
    Extract an embedding from a reference image.

    Priority:
    1. Detect a face in the full image and embed that face.
    2. If face detection fails, try a last-resort resized crop so the database
       is never empty, but log that the image was weakly handled.
    """
    try:
        bboxes, kpss = detect_faces(image)
        if len(bboxes) > 0:
            best_idx = max(range(len(bboxes)), key=lambda i: bboxes[i][4])
            aligned = align_face(image, kpss[best_idx])
            embedding = get_embedding(aligned)
            return embedding
        print(f"[WARN] {image_name}: no face detected in full image")
    except Exception as e:
        print(f"[WARN] {image_name}: direct face extraction failed: {e}")

    try:
        h, w = image.shape[:2]
        if h > 0 and w > 0:
            fallback = cv2.resize(image, (112, 112))
            embedding = get_embedding(fallback)
            print(f"[WARN] {image_name}: used resized fallback embedding")
            return embedding
    except Exception as e:
        print(f"[WARN] {image_name}: fallback embedding failed: {e}")

    return None


def load_reference_database():
    database = []
    if not REFERENCE_DIR.exists():
        return database

    for image_path in sorted(REFERENCE_DIR.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in REFERENCE_EXTENSIONS:
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[WARN] {image_path.name}: failed to load, skipping")
            continue

        embedding = extract_reference_embedding(img, image_path.name)
        if embedding is None:
            print(f"[WARN] {image_path.name}: no face embedding found, skipping")
            continue

        database.append({
            "name": normalize_reference_name(image_path.name),
            "embedding": embedding,
        })
        print(f"[INFO] {image_path.name}: reference embedding loaded")

    return database
REFERENCE_DATABASE = load_reference_database()

print(f"[INFO] Shared YOLO detector ready. Reference faces loaded: {len(REFERENCE_DATABASE)}")

PERSON_CLASS_ID = 0
BOX_COLOR  = (0, 200, 80)
BOX_COLOR_MATCH = (72, 187, 120)
BOX_COLOR_UNKNOWN = (0, 165, 255)
LABEL_BG   = (0, 150, 60)
TEXT_COLOR = (255, 255, 255)


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def match_reference(embedding):
    best_name = "Unknown"
    best_score = -1.0

    for reference in REFERENCE_DATABASE:
        score = cosine_similarity(embedding, reference["embedding"])
        if score > best_score:
            best_score = score
            best_name = reference["name"]

    if best_score < RECOGNITION_THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score


def draw_boxes(frame, detections, recognition_map=None, thickness=2, font_scale=0.55):
    count = 0
    recognized_count = 0
    for det in detections:
        if int(det[5]) != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        conf = float(det[4])
        match = (recognition_map or {}).get((x1, y1, x2, y2))
        if match and match[0] != "Unknown":
            label = f"{match[0]} {match[1]:.2f}"
            box_color = BOX_COLOR_MATCH
            recognized_count += 1
        else:
            label = f"Person {conf:.2f}"
            box_color = BOX_COLOR_UNKNOWN if REFERENCE_DATABASE else BOX_COLOR
        count += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, 1)
        ly = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 4, ly + bl), LABEL_BG, -1)
        cv2.putText(frame, label, (x1 + 2, ly), font, font_scale, TEXT_COLOR, 1, cv2.LINE_AA)
    return frame, count, recognized_count


def overlay_hud(frame, frame_num, total, count, recognized_count=0):
    hud = f"Frame {frame_num}/{total}  |  Persons: {count}  |  Recognized: {recognized_count}"
    cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
    return frame


def reencode_to_h264(input_path, output_path):
    """Re-encode video to H264 using ffmpeg for browser compatibility."""
    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vcodec", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ], capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@app.route("/health", methods=["GET"])
def health():
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        ffmpeg = True
    except Exception:
        ffmpeg = False
    return jsonify({"status": "ok", "model": "yolo11m", "cuda": cuda_available(), "ffmpeg": ffmpeg, "recognition_ready": len(REFERENCE_DATABASE) > 0, "reference_faces": len(REFERENCE_DATABASE)})


@app.route("/process", methods=["POST"])
def process_video():
    
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400
    
    
    file        = request.files["video"]
    conf_thresh = float(request.form.get("conf", 0.40))
    iou_thresh  = float(request.form.get("iou", 0.45))
    box_thick   = int(request.form.get("thickness", 2))
    font_size   = int(request.form.get("font_size", 16))
    font_scale  = (font_size - 10) / (28 - 10) * 0.65 + 0.35

    recognition_every = int(request.form.get("recognition_every", 15))
    recognition_every = max(1, recognition_every)

    job_id       = str(uuid.uuid4())[:8]
    suffix       = Path(file.filename).suffix or ".mp4"
    input_path   = UPLOAD_DIR / f"{job_id}_input{suffix}"
    raw_path     = OUTPUT_DIR / f"{job_id}_raw.mp4"
    output_path  = OUTPUT_DIR / f"{job_id}_detected.mp4"
    coords_path  = COORDS_DIR / f"{job_id}_coords.csv"

    file.save(str(input_path))
    print(f"[INFO] Saved: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 400

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    device = "cuda" if cuda_available() else "cpu"

    print(f"[INFO] {width}x{height} @ {fps:.1f}fps | {total} frames | device={device}")

    # Write raw mp4v first, then re-encode to h264
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))

    total_det  = 0
    total_rec  = 0
    peak       = 0
    frame_num  = 0
    all_coords = []  # list of coord rows for CSV

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        detections = detect_persons(frame, conf=conf_thresh, iou=iou_thresh, device=device)
        recognition_map = {}
        
        should_run_recognition = (
            len(REFERENCE_DATABASE) > 0 and (frame_num % recognition_every == 0)
        )
        face_results = extract_all_faces(frame, person_boxes=detections) if should_run_recognition else []

        for result in face_results:
            person_bbox = tuple(map(int, result["person_bbox"]))
            name, score = match_reference(result["embedding"])
            recognition_map[person_bbox] = (name, score)

        for det_index, det in enumerate(detections, start=1):
            b = det[:4]
            c = det[4]
            bbox_key = tuple(map(int, b))
            matched_name, matched_score = recognition_map.get(bbox_key, ("Unknown", -1.0))
            if matched_name != "Unknown":
                total_rec += 1
            # Save coords: frame, person_id, x1, y1, x2, y2, conf, width, height
            all_coords.append({
                "frame": frame_num,
                "person_id": det_index,
                "x1": int(b[0]), "y1": int(b[1]),
                "x2": int(b[2]), "y2": int(b[3]),
                "width_px": int(b[2]-b[0]),
                "height_px": int(b[3]-b[1]),
                "confidence": round(float(c), 4),
                "recognized_name": matched_name,
                "recognition_score": round(float(matched_score), 4) if matched_score >= 0 else "",
                "face_found": 1 if bbox_key in recognition_map else 0,
                "frame_width": width,
                "frame_height": height
            })

        frame, count, recognized_count = draw_boxes(frame, detections, recognition_map, box_thick, font_scale)
        frame        = overlay_hud(frame, frame_num, total, count, recognized_count)
        total_det   += count
        peak         = max(peak, count)
        writer.write(frame)

        if frame_num % 30 == 0:
            print(f"  [frame {frame_num}/{total}] persons: {count}")

    cap.release()
    writer.release()

    # Save CSV coordinates file
    with open(str(coords_path), "w", newline="") as csvfile:
        fieldnames = ["frame","person_id","x1","y1","x2","y2","width_px","height_px","confidence","recognized_name","recognition_score","face_found","frame_width","frame_height"]
        writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(all_coords)
    print(f"[INFO] Saved coords: {coords_path} ({len(all_coords)} rows)")

    # Try re-encoding to H264 for browser playback
    print("[INFO] Re-encoding to H264 for browser compatibility...")
    h264_ok = reencode_to_h264(raw_path, output_path)

    if h264_ok:
        print("[INFO] H264 encoding successful")
        final_path = output_path
        raw_path.unlink(missing_ok=True)
    else:
        print("[WARN] ffmpeg not found — sending raw mp4v (may not play in browser)")
        final_path = raw_path

    # Clean up input
    input_path.unlink(missing_ok=True)

    print(f"[INFO] Done. frames={frame_num} detections={total_det} peak={peak}")

    # Read into memory and send
    with open(str(final_path), "rb") as f:
        video_bytes = f.read()

    stats = json.dumps({
        "frames": frame_num,
        "total_detections": int(total_det),
        "recognized_detections": int(total_rec),
        "peak_per_frame": int(peak),
        "job_id": job_id,
        "coords_rows": len(all_coords),
        "reference_faces": len(REFERENCE_DATABASE)
    })

    response = make_response(video_bytes)
    response.status_code = 200
    response.headers["Content-Type"] = "video/mp4"
    response.headers["Content-Disposition"] = 'inline; filename="detected.mp4"'
    response.headers["Content-Length"] = str(len(video_bytes))
    response.headers["X-Stats"] = stats
    response.headers["X-Job-Id"] = job_id
    response.headers["Access-Control-Expose-Headers"] = "X-Stats, X-Job-Id"
    return response


@app.route("/coords/<job_id>", methods=["GET"])
def get_coords(job_id):
    """Download the CSV coordinates file for a completed job."""
    # Sanitize job_id
    job_id = job_id.replace("..", "").replace("/", "").replace("\\", "")
    coords_path = COORDS_DIR / f"{job_id}_coords.csv"
    if not coords_path.exists():
        return jsonify({"error": "Coordinates file not found"}), 404
    with open(str(coords_path), "rb") as f:
        csv_bytes = f.read()
    response = make_response(csv_bytes)
    response.headers["Content-Type"] = "text/csv"
    response.headers["Content-Disposition"] = f'attachment; filename="detections_{job_id}.csv"'
    response.headers["Content-Length"] = str(len(csv_bytes))
    return response


if __name__ == "__main__":
    print(f"[INFO] CUDA: {cuda_available()}")
    print("[INFO] Server starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)