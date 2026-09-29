"""
process_routes.py — video processing job routes.

Includes the background worker, SSE preview stream, status/result endpoints,
coordinate CSV download, and auto-calibration.
"""

import cv2
import json
import uuid
import csv
import base64
import time
import queue
import threading
import numpy as np
from pathlib import Path

from flask import Blueprint, request, jsonify, make_response, Response

import pipeline
import tracker as tracking
from alerts import AlertEngine
from analysis import analyse_frame, render_frame, count_zones, overlay_hud
from geometry import (
    normalize_zones, reencode_to_h264,
    sample_frame_indices, compute_brightness, compute_blur_sharpness,
    iou_xyxy, otsu_split,
)
from config import (
    UPLOAD_DIR, OUTPUT_DIR, COORDS_DIR,
    RECOG_STRIDE_VIDEO, PREVIEW_SSE_FPS,
    CONF_MIN, CONF_MAX, IOU_MIN, IOU_MAX, CALIBRATION_SAMPLES,
)
from app_state import _jobs, has_enrollment, make_alert_sink

bp = Blueprint("process", __name__)


# ============================================================
# BACKGROUND WORKER
# ============================================================

def _process_worker(job_id, input_path, conf_thresh, iou_thresh, thickness, font_scale,
                    zones, raw_path, output_path, coords_path, intruder_detection=True):
    job = _jobs[job_id]
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        job['status']['error'] = "Cannot open video"
        job['status']['done'] = True
        job['frame_queue'].put(None)
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))

    # Per-job tracker and alert state — never shared with the live session.
    trk = tracking.Tracker()
    frame_ref = [0]
    engine = AlertEngine(make_alert_sink(job_id, frame_ref))

    from analytics.crowd_counter import CrowdCounter
    from analytics.crowd_monitor import CrowdMonitor
    from analytics.direction_analyzer import DirectionAnalyzer
    crowd_counter    = CrowdCounter()
    crowd_monitor    = CrowdMonitor()
    direction_analyzer = DirectionAnalyzer()

    total_det   = 0
    peak        = 0
    frame_num   = 0
    all_coords  = []
    zone_peak   = {z["name"]: 0 for z in zones}
    zone_total  = {z["name"]: 0 for z in zones}

    job['status']['total_frames'] = total

    preview_t_last = time.time()
    preview_fps = 0.0
    last_preview_push = 0.0
    preview_interval = 1.0 / PREVIEW_SSE_FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        frame_ref[0] = frame_num

        # Video time, not wall time: how fast the machine processes the file
        # must not change the results it produces.
        now = frame_num / fps

        records, detections = analyse_frame(
            frame, trk, conf_thresh, iou_thresh, zones, now,
            do_recognition=(frame_num % RECOG_STRIDE_VIDEO == 0 or frame_num == 1),
        )
        engine.evaluate(records, now,
                        intruder_detection=intruder_detection,
                        has_enrollment=has_enrollment())

        crowd_stats = crowd_counter.update(
            records, zones, now,
            detections=detections,
            frame_shape=frame.shape,
        )
        direction = direction_analyzer.update(trk.tracks)
        crowd_monitor.evaluate(
            crowd_stats["current_count"], crowd_stats["zones"], now,
            dominant_direction=direction,
        )

        zone_counts = count_zones(records, zones)
        count = len(records)

        frame = render_frame(frame, records, zones, thickness, font_scale)
        frame = overlay_hud(frame, frame_num, total, count, zone_counts)

        total_det += count
        peak = max(peak, count)
        for name, n in zone_counts.items():
            zone_total[name] += n
            zone_peak[name] = max(zone_peak[name], n)

        writer.write(frame)

        for rec in records:
            x1, y1, x2, y2 = rec["bbox"]
            all_coords.append({
                "frame": frame_num,
                "track_id": rec["track_id"],
                "person_id": rec["track_id"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width_px": x2 - x1,
                "height_px": y2 - y1,
                "confidence": round(rec["conf"], 4),
                "frame_width": width,
                "frame_height": height,
                "zone": rec["zone"] or "",
                "matched_name": rec["identity_name"] or "",
                "matched_status": rec["identity_status"] or "",
                "identity_state": rec["identity_state"],
            })

        now_wall = time.time()
        elapsed = now_wall - preview_t_last
        if elapsed > 0:
            preview_fps = round(1.0 / elapsed, 1)
        preview_t_last = now_wall

        job['status']['frame_num'] = frame_num
        job['status']['progress_pct'] = int((frame_num / total) * 100) if total else 0
        job['status']['peak'] = peak
        job['status']['count'] = count
        job['status']['zone_counts'] = zone_counts
        job['status']['preview_fps'] = preview_fps

        # Throttle the SSE preview. Base64-encoding every frame just to have the
        # browser drop most of them wastes real CPU that YOLO needs.
        if now_wall - last_preview_push >= preview_interval:
            last_preview_push = now_wall
            preview_width = min(480, width)
            preview_height = max(1, int(height * (preview_width / width)))
            preview_frame = cv2.resize(frame, (preview_width, preview_height))

            ok, jpeg = cv2.imencode('.jpg', preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 45])
            if ok:
                stats_json = json.dumps({
                    "frame": frame_num, "total": total, "count": count,
                    "zones": zone_counts, "fps": preview_fps
                })
                try:
                    job['frame_queue'].put_nowait({
                        "frame": base64.b64encode(jpeg).decode(),
                        "stats": stats_json,
                    })
                except queue.Full:
                    pass  # Browser is behind — drop rather than block YOLO

    cap.release()
    writer.release()

    with open(str(coords_path), "w", newline="") as csvfile:
        fieldnames = ["frame","track_id","person_id","x1","y1","x2","y2","width_px","height_px",
                      "confidence","frame_width","frame_height","zone","matched_name",
                      "matched_status","identity_state"]
        writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(all_coords)

    job['status']['encoding'] = True
    h264_ok = reencode_to_h264(raw_path, output_path)

    if h264_ok:
        raw_path.unlink(missing_ok=True)
    else:
        job['status']['ffmpeg_fail'] = True

    input_path.unlink(missing_ok=True)

    job['status']['done'] = True
    crowd_final = crowd_counter.get_stats()
    monitor_final = crowd_monitor.get_status()
    job['status']['stats'] = {
        "frames": frame_num,
        "total_detections": int(total_det),
        "peak_per_frame": int(peak),
        "unique_people": max(0, trk.next_id - 1),
        "coords_rows": len(all_coords),
        "zones": [{"name": n, "peak": zone_peak[n], "total_frames_occupied": zone_total[n]} for n in zone_peak],
        "crowd": {
            **crowd_final,
            "trend":             monitor_final["trend"],
            "dominant_direction": monitor_final["dominant_direction"],
            "status":            monitor_final["status"],
        }
    }
    job['frame_queue'].put(None)  # EOF sentinel


# ============================================================
# ROUTES
# ============================================================

@bp.route("/process/start", methods=["POST"])
def process_start():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file = request.files["video"]
    conf_thresh = float(request.form.get("conf", 0.40))
    iou_thresh  = float(request.form.get("iou", 0.45))
    box_thick   = int(request.form.get("thickness", 2))
    font_size   = int(request.form.get("font_size", 16))
    font_scale  = (font_size - 10) / (28 - 10) * 0.65 + 0.35

    try:
        zones_input = json.loads(request.form.get("zones", "[]"))
    except (json.JSONDecodeError, TypeError):
        zones_input = []

    intruder_detection = request.form.get("intruder_detection", "1") == "1"

    job_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename).suffix or ".mp4"

    input_path   = UPLOAD_DIR / f"{job_id}_input{suffix}"
    raw_path     = OUTPUT_DIR / f"{job_id}_raw.mp4"
    output_path  = OUTPUT_DIR / f"{job_id}_detected.mp4"
    coords_path  = COORDS_DIR / f"{job_id}_coords.csv"

    file.save(str(input_path))

    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    zones = normalize_zones(zones_input, width, height)

    _jobs[job_id] = {
        "frame_queue": queue.Queue(maxsize=30),
        "status": {
            "job_id": job_id,
            "progress_pct": 0,
            "done": False,
            "encoding": False,
            "total_frames": 0,
            "frame_num": 0,
        },
        "paths": {
            "output_path": output_path,
            "raw_path": raw_path,
            "coords_path": coords_path
        }
    }

    t = threading.Thread(target=_process_worker, args=(
        job_id, input_path, conf_thresh, iou_thresh, box_thick, font_scale, zones,
        raw_path, output_path, coords_path, intruder_detection
    ))
    t.start()
    _jobs[job_id]['thread'] = t

    return jsonify({"job_id": job_id})


@bp.route("/process/stream/<job_id>", methods=["GET"])
def process_stream(job_id):
    if job_id not in _jobs:
        return "Job not found", 404

    def generate():
        q = _jobs[job_id]['frame_queue']
        while True:
            item = q.get()
            if item is None:
                break
            yield f"event: stats\ndata: {item['stats']}\n\n"
            yield f"event: frame\ndata: {item['frame']}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@bp.route("/process/status/<job_id>", methods=["GET"])
def process_status(job_id):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(_jobs[job_id]['status'])


@bp.route("/process/result/<job_id>", methods=["GET"])
def process_result(job_id):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404

    job = _jobs[job_id]
    if not job['status']['done']:
        return jsonify({"error": "Job not finished"}), 400

    final_path = job['paths']['output_path']
    if job['status'].get('ffmpeg_fail'):
        final_path = job['paths']['raw_path']

    with open(str(final_path), "rb") as f:
        video_bytes = f.read()

    response = make_response(video_bytes)
    response.status_code = 200
    response.headers["Content-Type"] = "video/mp4"
    response.headers["Content-Disposition"] = f'inline; filename="detected_{job_id}.mp4"'

    stats_json = json.dumps(job['status'].get('stats', {}))
    response.headers["X-Stats"] = stats_json
    response.headers["X-Job-Id"] = job_id
    response.headers["Access-Control-Expose-Headers"] = "X-Stats, X-Job-Id"

    return response


@bp.route("/process", methods=["POST"])
def process_video():
    return jsonify({"error": "Please use /process/start flow instead"}), 400


@bp.route("/coords/<job_id>", methods=["GET"])
def get_coords(job_id):
    job_id = job_id.replace("..", "").replace("/", "").replace("\\", "")
    coords_path = COORDS_DIR / f"{job_id}_coords.csv"
    if not coords_path.exists():
        return jsonify({"error": "Coordinates file not found"}), 404
    with open(str(coords_path), "rb") as f:
        csv_bytes = f.read()
    response = make_response(csv_bytes)
    response.headers["Content-Type"] = "text/csv"
    response.headers["Content-Disposition"] = f'attachment; filename="detections_{job_id}.csv"'
    return response


@bp.route("/auto_calibrate", methods=["POST"])
def auto_calibrate():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file = request.files["video"]
    job_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename).suffix or ".mp4"
    tmp_path = UPLOAD_DIR / f"{job_id}_calib{suffix}"
    file.save(str(tmp_path))

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        tmp_path.unlink(missing_ok=True)
        return jsonify({"error": "Cannot open video"}), 400

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    sample_idxs = sample_frame_indices(total)

    confidences, per_frame_counts, overlap_ratios = [], [], []
    brightness_vals, blur_vals = [], []

    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_vals.append(compute_brightness(gray))
        blur_vals.append(compute_blur_sharpness(gray))

        # merge_duplicates=False: this measures raw detector statistics, so the
        # duplicate-suppression pass must not distort the numbers.
        detections = pipeline.detect_persons(frame, 0.05, 0.5, merge_duplicates=False)
        frame_boxes = []
        for det in detections:
            confidences.append(float(det[4]))
            frame_boxes.append(det[:4])

        per_frame_counts.append(len(frame_boxes))
        if len(frame_boxes) >= 2:
            pair_overlaps = [
                iou_xyxy(frame_boxes[i], frame_boxes[j])
                for i in range(len(frame_boxes))
                for j in range(i + 1, len(frame_boxes))
            ]
            pair_overlaps = [o for o in pair_overlaps if o > 0]
            if pair_overlaps:
                overlap_ratios.append(float(np.mean(pair_overlaps)))

    cap.release()
    tmp_path.unlink(missing_ok=True)

    reasoning = []

    split = otsu_split(confidences)
    rec_conf = split if split is not None else 0.40
    if split is None:
        reasoning.append("Too few candidate detections in the sampled frames — starting from the standard baseline.")

    avg_brightness = float(np.mean(brightness_vals)) if brightness_vals else 128.0
    avg_blur       = float(np.mean(blur_vals)) if blur_vals else 200.0

    if avg_brightness < 70:
        rec_conf -= 0.07
        reasoning.append(f"Dark footage (avg brightness {avg_brightness:.0f}/255) — lowered confidence so dim detections aren't missed.")
    elif avg_brightness > 190:
        rec_conf += 0.03
        reasoning.append(f"Bright, high-contrast footage (avg brightness {avg_brightness:.0f}/255) — raised confidence slightly.")

    if avg_blur < 80:
        rec_conf -= 0.05
        reasoning.append(f"Footage looks soft/blurry (sharpness {avg_blur:.0f}) — lowered confidence to compensate.")

    rec_conf = float(np.clip(rec_conf, CONF_MIN, CONF_MAX))

    avg_persons  = float(np.mean(per_frame_counts)) if per_frame_counts else 0.0
    avg_overlap  = float(np.mean(overlap_ratios)) if overlap_ratios else 0.0

    if avg_overlap > 0.25 or avg_persons > 8:
        rec_iou = 0.65
        reasoning.append(f"Crowded scene (avg {avg_persons:.1f} people/frame, box overlap {avg_overlap:.2f}) — raised IoU so adjacent people aren't merged into one box.")
    elif avg_overlap > 0.12 or avg_persons > 4:
        rec_iou = 0.50
        reasoning.append(f"Moderate crowd density (avg {avg_persons:.1f} people/frame) — using a balanced IoU.")
    else:
        rec_iou = 0.40
        reasoning.append(f"Sparse scene (avg {avg_persons:.1f} people/frame) — tighter IoU to clean up duplicate boxes.")

    rec_iou = float(np.clip(rec_iou, IOU_MIN, IOU_MAX))

    if not reasoning:
        reasoning.append("Typical lighting and crowd density — using balanced defaults.")

    return jsonify({
        "recommended_conf": round(rec_conf, 2),
        "recommended_iou": round(rec_iou, 2),
        "reasoning": reasoning,
        "details": {
            "sampled_frames": len(sample_idxs),
            "avg_persons_per_frame": round(avg_persons, 2),
            "avg_brightness": round(avg_brightness, 1),
            "avg_blur_sharpness": round(avg_blur, 1),
            "avg_overlap": round(avg_overlap, 3),
            "total_candidate_detections": len(confidences)
        }
    })
