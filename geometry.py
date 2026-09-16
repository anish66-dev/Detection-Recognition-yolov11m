"""
geometry.py — zone geometry, calibration statistics, and video encoding helpers.

Pure functions with no Flask, no global state, and no ML model dependencies.
"""

import subprocess
import cv2
import numpy as np

import tracker as tracking
from config import CALIBRATION_SAMPLES


def point_in_polygon(x, y, polygon):
    n = len(polygon)
    if n < 3: return False
    inside = False
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    xinters = x1 if y1 == y2 else (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    if x1 == x2 or x <= xinters:
                        inside = not inside
        x1, y1 = x2, y2
    return inside


def normalize_zones(zones_input, width, height):
    zones = []
    for z in zones_input or []:
        pts = z.get("points", [])
        if len(pts) >= 3:
            pixel_pts = [(float(p[0]) * width, float(p[1]) * height) for p in pts]
            zones.append({"name": z.get("name", "Zone"), "points": pixel_pts})
    return zones


def sample_frame_indices(total_frames, n_samples=CALIBRATION_SAMPLES):
    if total_frames <= 0: return []
    if total_frames <= n_samples: return list(range(total_frames))
    step = total_frames / n_samples
    return [int(i * step) for i in range(n_samples)]


def compute_brightness(gray_frame): return float(gray_frame.mean())
def compute_blur_sharpness(gray_frame): return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


def iou_xyxy(box_a, box_b):
    return tracking.iou(box_a, box_b)


def otsu_split(values):
    if len(values) < 6: return None
    arr = np.clip(np.array(values, dtype=np.float32) * 255, 0, 255).astype(np.uint8)
    hist = cv2.calcHist([arr], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total == 0: return None
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b, weight_b, max_var, best_t = 0.0, 0.0, 0.0, 0
    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0: continue
        weight_f = total - weight_b
        if weight_f == 0: break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > max_var:
            max_var, best_t = var_between, t
    return best_t / 255.0


def get_ffmpeg_executable():
    import shutil
    cmd = shutil.which("ffmpeg")
    if cmd: return cmd
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def reencode_to_h264(input_path, output_path):
    ffmpeg_cmd = get_ffmpeg_executable()
    try:
        result = subprocess.run([
            ffmpeg_cmd, "-y",
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
