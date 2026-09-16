"""
pipeline.py — OS-portable 4-stage face recognition pipeline.

Stages:
  1. Person detection     — YOLO11m
  2. Face detection       — SCRFD (via buffalo_m)
  3. Landmark alignment   — 5-point norm_crop (via InsightFace)
  4. Embedding extraction — ArcFace w600k_r50 (via buffalo_m)

Works on Windows, Linux, macOS — no hardcoded paths.
Models auto-download via InsightFace and ultralytics on first run.
"""

import cv2
import shutil
import numpy as np
import urllib.request
from pathlib import Path
from ultralytics import YOLO

from tracker import merge_duplicate_boxes

# Constants
PERSON_CLASS_ID = 0
CONF_THRESH_DEFAULT = 0.40
IOU_THRESH_DEFAULT = 0.45

# Recognition uses a hysteresis band rather than a single threshold.
# A single cut-off makes a known person flip to "INTRUDER" the moment motion
# blur knocks similarity from 0.36 to 0.34. Between the two values the result
# is "uncertain": the label is held and no alert is raised.
RECOG_THRESH = 0.35        # >= this  -> confidently a match
RECOG_THRESH_LOW = 0.28    # <  this  -> confidently NOT a match

# Path.home() resolves correctly on Windows, Linux, and macOS.
# buffalo_l is the standard, highest-quality pack and the most reliably hosted.
_MODEL_NAME = "buffalo_l"
_MODEL_DIR = Path.home() / ".insightface" / "models" / _MODEL_NAME

# Shared model references (loaded once)
_yolo = None
_det = None   # SCRFD face detector
_rec = None   # ArcFace recognizer

# Records *why* face models failed to load, so the UI/logs can show the real reason
# instead of a misleading green checkmark.
_load_error = None


_device = None


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device():
    """
    Resolved once and cached. ``detect_persons`` used to call
    ``_cuda_available()`` on every single frame, which imports torch and probes
    the driver inside the hot loop for a value that cannot change.
    """
    global _device
    if _device is None:
        _device = "cuda" if _cuda_available() else "cpu"
    return _device


def _find_model_files():
    """
    Locate the detection and recognition ONNX files inside the buffalo_m folder.

    Filenames differ between InsightFace bundles/versions, so we scan the folder
    instead of hardcoding names. Detection models are named ``det_*.onnx`` and
    recognition models ``w600k_*.onnx`` (e.g. w600k_r50 or w600k_mbf). Landmark
    and attribute models (1k3d68, 2d106det, genderage) are deliberately ignored.

    Returns:
        (det_path, rec_path) — Paths, or (None, None) if not found.
    """
    if not _MODEL_DIR.exists():
        return None, None
    onnx = sorted(_MODEL_DIR.glob("*.onnx"))
    det = next((p for p in onnx if p.name.startswith("det_")), None)
    rec = next((p for p in onnx if p.name.startswith("w600k")), None)
    return det, rec


def _ensure_models():
    """
    Make sure buffalo_m ONNX files exist locally, downloading if needed.

    Returns True ONLY when both a detection and a recognition model are actually
    present on disk afterwards — not just because a download was attempted.
    """
    global _load_error

    det, rec = _find_model_files()
    if det and rec:
        return True

    # Try auto-download via FaceAnalysis
    try:
        from insightface.app import FaceAnalysis
        print(f"[INFO] Downloading {_MODEL_NAME} model bundle (~290MB, first run only)...")
        app = FaceAnalysis(name=_MODEL_NAME)
        app.prepare(ctx_id=-1, det_size=(640, 640))

        # Fix nested folder issue that can occur on first extraction
        nested = _MODEL_DIR / _MODEL_NAME
        if nested.exists() and nested.is_dir():
            for f in nested.iterdir():
                shutil.move(str(f), str(_MODEL_DIR / f.name))
            nested.rmdir()
    except Exception as e:
        _load_error = f"model download failed: {e}"
        print(f"[ERROR] Failed to download InsightFace models: {e}")
        return False

    det, rec = _find_model_files()
    if not (det and rec):
        present = [p.name for p in _MODEL_DIR.glob("*.onnx")] if _MODEL_DIR.exists() else []
        _load_error = (
            f"buffalo_m folder present but no det_*/w600k_* ONNX found in {_MODEL_DIR}. "
            f"Files there: {present}"
        )
        print(f"[ERROR] {_load_error}")
        return False

    print("[INFO] buffalo_m model files ready.")
    return True


def _load_models():
    """Load all models once (expensive — done once per process, not per frame)."""
    global _yolo, _det, _rec, _load_error

    if _yolo is None:
        try:
            print("[INFO] Loading YOLOv11m...")
            _yolo = YOLO("yolo11mtrained.pt")
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}")

    if _det is not None and _rec is not None:
        return

    try:
        from insightface.model_zoo import model_zoo
    except ImportError:
        _load_error = "insightface package not installed"
        print("[WARN] insightface package not found. Face recognition disabled.")
        return

    if not _ensure_models():
        print("[WARN] InsightFace models not available. Face recognition disabled.")
        return

    det_path, rec_path = _find_model_files()
    ctx_id = 0 if _cuda_available() else -1

    try:
        if _det is None and det_path is not None:
            print(f"[INFO] Loading face detector ({det_path.name})...")
            _det = model_zoo.get_model(str(det_path))
            _det.prepare(ctx_id=ctx_id, input_size=(640, 640))

        if _rec is None and rec_path is not None:
            print(f"[INFO] Loading face recognizer ({rec_path.name})...")
            _rec = model_zoo.get_model(str(rec_path))
            _rec.prepare(ctx_id=ctx_id)
    except Exception as e:
        _load_error = f"model initialisation failed: {e}"
        print(f"[ERROR] Failed to initialise InsightFace models: {e}")
        return

    if _det is None or _rec is None:
        _load_error = _load_error or "detector or recognizer did not initialise"
        print(f"[WARN] Face models incomplete: {_load_error}")
    else:
        _load_error = None
        print("[INFO] Face detection + recognition ready.")


def models_ready():
    """True only when BOTH detection and recognition models are loaded in memory."""
    return _det is not None and _rec is not None


def get_status():
    """Real, in-memory model status for the /health endpoint and diagnostics."""
    return {
        "yolo": _yolo is not None,
        "face_detector": _det is not None,
        "face_recognizer": _rec is not None,
        "ready": models_ready(),
        "error": _load_error,
    }


# ── Stage 1: Person Detection ─────────────────────────────────────────────────

def detect_persons(frame, conf=CONF_THRESH_DEFAULT, iou=IOU_THRESH_DEFAULT,
                   merge_duplicates=True):
    """
    Stage 1: YOLO11m person detection.

    Args:
        merge_duplicates: run a second, class-agnostic pass that collapses
            boxes sitting on the same body. YOLO's own NMS still lets duplicates
            through at low ``conf`` with high ``iou`` — and those duplicates are
            what make one person render as two differently-labelled boxes.
            Pass False when measuring raw detector statistics (calibration).

    Returns:
        List of [x1, y1, x2, y2, confidence].
    """
    if _yolo is None:
        return []

    results = _yolo(frame, classes=[PERSON_CLASS_ID], conf=conf, iou=iou,
                    verbose=False, device=get_device())

    detections = []
    for r in results:
        if r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for b, c in zip(boxes, confs):
                detections.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)])

    if merge_duplicates and len(detections) > 1:
        detections = merge_duplicate_boxes(detections)
    return detections


# ── Stage 2: Face Detection (SCRFD) ───────────────────────────────────────────

def detect_faces_scrfd(img):
    """
    Stage 2: SCRFD face detection + 5-point landmark localization.

    Args:
        img: BGR image (full frame or person crop).

    Returns:
        (bboxes, kpss) — face boxes [x1,y1,x2,y2,conf] and 5x2 landmark arrays.
        Returns (np.array([]), np.array([])) if no detector available.
    """
    if _det is None:
        return np.array([]), np.array([])

    bboxes, kpss = _det.detect(img, max_num=0, metric="default")
    return bboxes, kpss


# ── Stage 3: Alignment ────────────────────────────────────────────────────────

def align_face(img, kps):
    """
    Stage 3: Geometric alignment using 5 landmarks → 112x112 crop.

    Args:
        img: BGR image.
        kps: 5x2 array of (x, y) landmark coordinates.

    Returns:
        112x112x3 aligned face image.
    """
    from insightface.utils import face_align
    return face_align.norm_crop(img, kps, image_size=112)


# ── Stage 4: Embedding ────────────────────────────────────────────────────────

def get_embedding_from_aligned(aligned_face):
    """
    Stage 4: ArcFace embedding extraction + L2 normalization.

    Args:
        aligned_face: 112x112x3 BGR image from align_face().

    Returns:
        512-dimensional unit-norm float32 embedding vector.
    """
    if _rec is None:
        return None

    emb = _rec.get_feat(aligned_face).flatten()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


# ── Full Pipeline ──────────────────────────────────────────────────────────────

def extract_all_faces(img):
    """
    Full 4-stage pipeline on a single image.

    Runs YOLO person detection, then per-person: SCRFD face detection,
    5-point alignment, ArcFace embedding extraction.

    Args:
        img: BGR image as numpy array.

    Returns:
        List of dicts, one per successfully processed face:
            {
                "person_bbox": [x1, y1, x2, y2, conf],
                "face_bbox":   np.ndarray [x1, y1, x2, y2, confidence] (global coords),
                "embedding":   np.ndarray shape (512,), unit-norm,
            }
        Empty list if no faces found.
    """
    if _det is None or _rec is None:
        return []

    results = []
    person_boxes = detect_persons(img)

    for det in person_boxes:
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        conf = float(det[4])
        person_crop = img[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        bboxes, kpss = detect_faces_scrfd(person_crop)
        if len(bboxes) == 0:
            # Person detected but no face found — still record it
            results.append({
                "person_bbox": [x1, y1, x2, y2, conf],
                "face_bbox": None,
                "embedding": None,
            })
            continue

        # Pick highest-confidence face
        best_idx = int(np.argmax(bboxes[:, 4]))
        kps = kpss[best_idx]

        aligned = align_face(person_crop, kps)
        embedding = get_embedding_from_aligned(aligned)

        # Convert crop-local face coordinates back to full-frame coordinates
        local_face_bbox = bboxes[best_idx].copy()
        global_face_bbox = np.array([
            local_face_bbox[0] + x1,
            local_face_bbox[1] + y1,
            local_face_bbox[2] + x1,
            local_face_bbox[3] + y1,
            local_face_bbox[4]  # Keep detection confidence score
        ], dtype=np.float32)

        results.append({
            "person_bbox": [x1, y1, x2, y2, conf],
            "face_bbox": global_face_bbox,
            "embedding": embedding,
        })

    return results


# ── Legacy compatibility functions ─────────────────────────────────────────────

def detect_faces(frame):
    """
    Detect every face in a full frame and embed each one.

    Returns:
        List of ``{"bbox": [x1, y1, x2, y2], "conf": float, "embedding": ndarray}``
        in full-frame coordinates. Plain dicts — the previous version built
        throwaway classes with ``type('Face', (), {...})()`` per face, per frame.
    """
    if _det is None or _rec is None:
        return []

    try:
        bboxes, kpss = detect_faces_scrfd(frame)
    except Exception as e:
        print(f"[WARN] Face detection failed: {e}")
        return []

    if len(bboxes) == 0:
        return []

    faces = []
    for i in range(len(bboxes)):
        try:
            aligned = align_face(frame, kpss[i])
            emb = get_embedding_from_aligned(aligned)
        except Exception:
            continue
        if emb is None:
            continue
        faces.append({
            "bbox": [float(v) for v in bboxes[i][:4]],
            "conf": float(bboxes[i][4]) if len(bboxes[i]) > 4 else 1.0,
            "embedding": emb,
        })
    return faces


def get_embedding(img):
    """
    Given an image containing a face, returns the embedding of the largest face found.
    Used for enrollment.
    """
    if _det is None or _rec is None:
        return None

    bboxes, kpss = detect_faces_scrfd(img)
    if len(bboxes) == 0:
        return None

    # Pick largest face by area
    if len(bboxes) > 1:
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        best_idx = int(np.argmax(areas))
    else:
        best_idx = 0

    aligned = align_face(img, kpss[best_idx])
    return get_embedding_from_aligned(aligned)


def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two 1D numpy arrays."""
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def match_face(face_embedding, enrolled_faces, threshold=RECOG_THRESH):
    """
    Find best match in enrolled_faces.
    enrolled_faces: list of dicts {"id":..., "name":..., "status":..., "embedding":...}
    Returns (best_match_dict, similarity_score) or (None, best_sim)
    """
    if face_embedding is None or not enrolled_faces:
        return None, 0.0

    best_match = None
    best_sim = -1

    for person in enrolled_faces:
        sim = cosine_similarity(face_embedding, person["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_match = person

    if best_sim >= threshold:
        return best_match, float(best_sim)
    return None, float(best_sim)


def match_face_state(face_embedding, enrolled_faces,
                     high=RECOG_THRESH, low=RECOG_THRESH_LOW):
    """
    Match with a hysteresis band instead of a single hard threshold.

    Returns:
        (match_or_None, similarity, state) where state is:
          'known'     — similarity >= high, this is the person
          'unknown'   — similarity <  low, definitely nobody enrolled
          'uncertain' — in between; caller should hold its previous label and
                        must not raise an intruder alert

    The middle band is what stops a known person being flagged as an intruder
    for the one frame they turn their head.
    """
    if face_embedding is None:
        return None, 0.0, "uncertain"
    if not enrolled_faces:
        # Nobody enrolled: this is a configuration state, not an intrusion.
        return None, 0.0, "uncertain"

    best_match, best_sim = None, -1.0
    for person in enrolled_faces:
        sim = cosine_similarity(face_embedding, person["embedding"])
        if sim > best_sim:
            best_sim, best_match = sim, person

    best_sim = float(best_sim)
    if best_sim >= high:
        return best_match, best_sim, "known"
    if best_sim < low:
        return None, best_sim, "unknown"
    return None, best_sim, "uncertain"


# Initialize models on module load
_load_models()