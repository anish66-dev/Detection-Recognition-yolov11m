"""
Full recognition pipeline: YOLO11 person detection -> SCRFD face detection
-> 5-point landmark alignment -> ArcFace (buffalo_m) embedding.
"""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from insightface.model_zoo import model_zoo
from insightface.utils import face_align


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "video_test" / "models" / "buffalo_m"
yolo_model = YOLO("yolo11m.pt")

_FACE_DET_MODEL = None
_REC_MODEL = None


def detect_persons(img, conf=0.25, iou=0.45, device=None):
    results = yolo_model(img, classes=[0], conf=conf, iou=iou, verbose=False, device=device)
    detections = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        for box, conf_value, class_value in zip(boxes, confs, classes):
            detections.append([*box, conf_value, class_value])
    return detections


def get_face_det_model():
    global _FACE_DET_MODEL
    if _FACE_DET_MODEL is None:
        _FACE_DET_MODEL = model_zoo.get_model(str(MODEL_DIR / "det_2.5g.onnx"))
        _FACE_DET_MODEL.prepare(ctx_id=-1, input_size=(640, 640))
    return _FACE_DET_MODEL


def get_rec_model():
    global _REC_MODEL
    if _REC_MODEL is None:
        _REC_MODEL = model_zoo.get_model(str(MODEL_DIR / "w600k_r50.onnx"))
        _REC_MODEL.prepare(ctx_id=-1)
    return _REC_MODEL


def detect_faces(img):
    bboxes, kpss = get_face_det_model().detect(img, max_num=0, metric="default")
    return bboxes, kpss


def align_face(img, kps):
    return face_align.norm_crop(img, kps, image_size=112)


def get_embedding(aligned_face):
    emb = get_rec_model().get_feat(aligned_face).flatten()
    emb = emb / np.linalg.norm(emb)
    return emb


def extract_all_faces(img, person_boxes=None):
    results = []
    if person_boxes is None:
        person_boxes = detect_persons(img)

    for person_box in person_boxes:
        x1, y1, x2, y2 = map(int, person_box[:4])
        person_crop = img[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        bboxes, kpss = detect_faces(person_crop)
        if len(bboxes) == 0:
            continue

        best_idx = np.argmax(bboxes[:, 4])
        kps = kpss[best_idx]

        aligned = align_face(person_crop, kps)
        embedding = get_embedding(aligned)

        results.append({
            "person_bbox": (x1, y1, x2, y2),
            "face_bbox": bboxes[best_idx],
            "embedding": embedding,
        })

    return results


def cosine_similarity(emb1, emb2):
    return float(np.dot(emb1, emb2))