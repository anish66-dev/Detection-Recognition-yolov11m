"""
analysis.py — frame analysis and rendering stages.

Stage 1 (analyse_frame) determines what is true about each person in the frame.
Stage 3 (render_frame) draws that information onto the frame.

Stage 2 (alerting) lives in alerts.py and runs between these two.
"""

import cv2
import numpy as np

import pipeline
import tracker as tracking
from geometry import point_in_polygon
from config import (
    FACE_CONTAINMENT_MIN,
    BOX_COLOR, LABEL_BG,
    ZONE_BOX_COLOR, ZONE_LABEL_BG,
    ALERT_BOX_COLOR, ALERT_LABEL_BG,
    AUTH_BOX_COLOR, AUTH_LABEL_BG,
    INTRUDER_BOX_COLOR, INTRUDER_LABEL_BG,
    ZONE_LINE_COLOR, TEXT_COLOR,
)
import app_state


# ============================================================
# STAGE 1 — ANALYSIS  (facts only: no drawing, no alerting)
# ============================================================

def assign_faces_to_persons(person_boxes, faces):
    """
    One-to-one assignment of detected faces to person boxes.

    The previous implementation took the first face whose *centre* fell inside a
    person box and stopped there, with no exclusivity. When two people overlap,
    person B's face centre sits inside person A's box, so A was labelled with
    B's identity — and both boxes could claim the same face, which is exactly
    how one human ended up drawn as a plain person box *and* a magenta INTRUDER
    box simultaneously.

    Here every (person, face) pair is scored by how much of the face box lies
    inside the person box, and pairs are consumed greedily best-first, so each
    face and each person is used at most once.

    Args:
        person_boxes: list of [x1, y1, x2, y2] in frame coordinates.
        faces: list of dicts from ``pipeline.detect_faces``.

    Returns:
        dict mapping person index -> face dict.
    """
    pairs = []
    for pi, pbox in enumerate(person_boxes):
        for fi, face in enumerate(faces):
            score = tracking.containment(face["bbox"], pbox)
            if score >= FACE_CONTAINMENT_MIN:
                pairs.append((score, pi, fi))
    pairs.sort(reverse=True)

    assigned, used_p, used_f = {}, set(), set()
    for score, pi, fi in pairs:
        if pi in used_p or fi in used_f:
            continue
        assigned[pi] = faces[fi]
        used_p.add(pi)
        used_f.add(fi)
    return assigned


def zone_for_box(bbox, zones):
    """
    Zone containing this person, using the bottom-centre of the box as the
    ground contact point. Returns the zone name, or None.
    """
    x1, y1, x2, y2 = bbox
    ref_x, ref_y = (x1 + x2) / 2.0, y2
    for z in zones:
        if point_in_polygon(ref_x, ref_y, z["points"]):
            return z["name"]
    return None


def analyse_frame(frame, trk, conf_thresh, iou_thresh, zones, now,
                  do_recognition=True):
    """
    Stage 1: work out what is true about this frame. Pure analysis.

    Runs detection, updates the tracker, then attaches identity votes and zone
    membership to each track. Draws nothing and raises no alerts.

    Args:
        trk: the ``tracking.Tracker`` owned by this video source.
        now: seconds — wall clock for live sources, frame_num/fps for files.
        do_recognition: False on frames where face recognition is skipped.
            Identity persists on the track, so skipping is invisible in output.

    Returns:
        (records, detections) where:
          records    — list of per-person record dicts (see ``Track.to_record``).
          detections — raw YOLO detection list [[x1,y1,x2,y2,conf], ...] so
                       callers can pass it to the density estimator without
                       running YOLO a second time.
    """
    detections = pipeline.detect_persons(frame, conf_thresh, iou_thresh)
    trk.update(detections, now)

    live_tracks = trk.active()

    if do_recognition and live_tracks:
        faces = pipeline.detect_faces(frame)
        if faces:
            boxes = [t.predict(now) for t in live_tracks]
            assigned = assign_faces_to_persons(boxes, faces)
            for pi, face in assigned.items():
                match, sim, state = pipeline.match_face_state(
                    face["embedding"], app_state._enrolled_faces_cache
                )
                live_tracks[pi].observe_identity(
                    match["name"] if match else None,
                    match["status"] if match else None,
                    sim, state,
                )

    for t in live_tracks:
        t.zone = zone_for_box(t.predict(now), zones)

    return [t.to_record(now) for t in live_tracks], detections


# ============================================================
# STAGE 3 — RENDER  (pixels only: no analysis, no alerting)
# ============================================================

def _style_for(rec):
    """
    Box colour and label text for one record.

    Attributes are independent, but a box has only one colour, so the *display*
    applies a priority order. The zone is still appended to the label, so an
    authorized person inside a zone reads "Ravi | Vault" in blue rather than
    losing the zone information entirely.
    """
    state = rec["identity_state"]
    status = rec["identity_status"]

    if state == "known" and status == "blocklisted":
        return ALERT_BOX_COLOR, ALERT_LABEL_BG, f"{rec['identity_name']} {rec['identity_sim']:.2f}"
    if state == "unknown":
        return INTRUDER_BOX_COLOR, INTRUDER_LABEL_BG, f"INTRUDER {rec['identity_sim']:.2f}"
    if state == "known" and status == "authorized":
        return AUTH_BOX_COLOR, AUTH_LABEL_BG, f"{rec['identity_name']} {rec['identity_sim']:.2f}"
    if rec["zone"]:
        return ZONE_BOX_COLOR, ZONE_LABEL_BG, f"Person {rec['conf']:.2f}"
    return BOX_COLOR, LABEL_BG, f"Person {rec['conf']:.2f}"


def render_frame(frame, records, zones, thickness, font_scale, show_ids=True):
    """Stage 3: draw records and zones onto the frame. Mutates and returns it."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    for rec in records:
        x1, y1, x2, y2 = rec["bbox"]
        b_color, l_bg, label = _style_for(rec)

        if rec["zone"]:
            label += f" | {rec['zone']}"
        if show_ids:
            label = f"#{rec['track_id']} " + label

        cv2.rectangle(frame, (x1, y1), (x2, y2), b_color, thickness)
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, 1)
        ly = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 4, ly + bl), l_bg, -1)
        cv2.putText(frame, label, (x1 + 2, ly), font, font_scale, TEXT_COLOR, 1, cv2.LINE_AA)

    for z in zones:
        pts = np.array(z["points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=ZONE_LINE_COLOR, thickness=2)
        label_x, label_y = int(z["points"][0][0]), max(int(z["points"][0][1]) - 10, 14)
        cv2.putText(frame, z["name"], (label_x, label_y), font, 0.6, ZONE_LINE_COLOR, 2, cv2.LINE_AA)

    return frame


def count_zones(records, zones):
    counts = {z["name"]: 0 for z in zones}
    for rec in records:
        if rec["zone"] in counts:
            counts[rec["zone"]] += 1
    return counts


def overlay_hud(frame, frame_num, total, count, zone_counts=None, extra=None):
    if total:
        hud = f"Frame {frame_num}/{total}  |  Persons: {count}"
    else:
        hud = f"LIVE  |  Persons: {count}"

    if zone_counts:
        hud += "  |  " + "  ".join(f"{name}: {n}" for name, n in zone_counts.items())
    if extra:
        hud += f"  |  {extra}"
    cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
    return frame
