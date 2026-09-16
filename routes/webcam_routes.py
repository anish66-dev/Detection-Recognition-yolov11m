"""
webcam_routes.py — live camera and mobile push-frame routes.
"""

import json
import time

from flask import Blueprint, request, jsonify, Response

from camera import CameraSource, _push_buffer
from live_session import LiveSession
import app_state

bp = Blueprint("webcam", __name__)


# ============================================================
# WEBCAM / LIVE ROUTES
# ============================================================

@bp.route("/webcam/start", methods=["POST"])
def webcam_start():
    with app_state._session_lock:
        if app_state._session is not None and app_state._session.running:
            return jsonify({"status": "already running", **app_state._session.status()})

        conf = float(request.form.get("conf", 0.40))
        iou = float(request.form.get("iou", 0.45))
        thickness = int(request.form.get("thickness", 2))
        font_size = int(request.form.get("font_size", 16))
        font_scale = (font_size - 10) / (28 - 10) * 0.65 + 0.35
        try:
            zones_input = json.loads(request.form.get("zones", "[]"))
        except (json.JSONDecodeError, TypeError):
            zones_input = []

        intruder_detection = request.form.get("intruder_detection", "1") == "1"
        source_spec = request.form.get("source", "0")

        session = LiveSession(source_spec, conf, iou, thickness, font_scale,
                              zones_input, intruder_detection)
        if not session.start():
            return jsonify({"error": session.error or "Failed to start camera"}), 400

        app_state._session = session

    # Give capture a moment so the first status call is meaningful.
    time.sleep(0.6)
    if app_state._session.error:
        err = app_state._session.error
        app_state._session.stop()
        return jsonify({"error": err}), 400

    return jsonify({"status": "started", **app_state._session.status()})


@bp.route("/webcam/status", methods=["GET"])
def webcam_status():
    if app_state._session is None:
        return jsonify({"running": False, "error": None, "has_frame": False})
    return jsonify(app_state._session.status())


@bp.route("/webcam/stop", methods=["POST"])
def webcam_stop():
    with app_state._session_lock:
        if app_state._session is not None:
            app_state._session.stop()
            app_state._session = None
    return jsonify({"status": "stopped"})


@bp.route("/webcam/stream", methods=["GET"])
def webcam_stream():
    if app_state._session is None or not app_state._session.running:
        return "Webcam not started. Call /webcam/start first.", 400
    return Response(app_state._session.frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ============================================================
# MOBILE CAMERA ROUTES
# ============================================================

@bp.route("/camera/probe", methods=["POST"])
def camera_probe():
    """
    Test a camera spec before starting a session, so the UI can report a wrong
    phone IP immediately instead of failing once streaming has begun.
    """
    spec = request.form.get("source", "0")
    src = CameraSource(spec)
    if not src.open():
        return jsonify({"ok": False, "error": src.error}), 400

    if src.kind == "push":
        src.release()
        return jsonify({"ok": True, "kind": "push",
                        "note": "Open /phone on the handset to start sending frames."})

    ok, frame = src.read()
    if not ok or frame is None:
        src.release()
        return jsonify({"ok": False, "error": "Opened the source but received no frames."}), 400

    w, h = src.dimensions(frame)
    src.release()
    return jsonify({"ok": True, "kind": src.kind, "width": w, "height": h})


@bp.route("/camera/push", methods=["POST"])
def camera_push():
    """
    Ingest one JPEG frame from the phone browser page.

    Accepts either a multipart 'frame' file or a raw image/jpeg body.
    """
    if "frame" in request.files:
        data = request.files["frame"].read()
    else:
        data = request.get_data()

    if not data:
        return jsonify({"error": "empty frame"}), 400
    if not _push_buffer.put_jpeg(data):
        return jsonify({"error": "could not decode frame"}), 400
    return jsonify({"ok": True})


@bp.route("/camera/push/status", methods=["GET"])
def camera_push_status():
    return jsonify({"alive": _push_buffer.alive})


@bp.route("/camera/hosts", methods=["GET"])
def camera_hosts():
    """
    LAN addresses this server is reachable at, so the UI can show the exact URL
    to open on the phone instead of making the user hunt for it.
    """
    import socket
    addrs = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            addrs.add(info[4][0])
    except Exception:
        pass
    try:
        # Reaches nothing, but makes the OS pick the outbound interface.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        addrs.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    addrs.discard("127.0.0.1")

    port = request.host.split(":")[-1] if ":" in request.host else "5000"
    scheme = "https" if request.is_secure else "http"
    return jsonify({
        "addresses": sorted(addrs),
        "phone_urls": [f"{scheme}://{a}:{port}/phone" for a in sorted(addrs)],
        "secure": request.is_secure,
    })
