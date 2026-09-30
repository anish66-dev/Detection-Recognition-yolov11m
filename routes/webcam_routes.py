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

@bp.route("/cameras/start", methods=["POST"])
def cameras_start():
    camera_id = request.form.get("camera_id", "LIVE")
    if app_state.get_session(camera_id) is not None and app_state.get_session(camera_id).running:
        return jsonify({"status": "already running", **app_state.get_session(camera_id).status()})

    conf = float(request.form.get("conf", 0.60))
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

    session = LiveSession(camera_id, source_spec, conf, iou, thickness, font_scale,
                          zones_input, intruder_detection)
    if not session.start():
        return jsonify({"error": session.error or "Failed to start camera"}), 400

    app_state.add_session(camera_id, session)

    # Give capture a moment so the first status call is meaningful.
    time.sleep(0.6)
    if session.error:
        err = session.error
        session.stop()
        app_state.remove_session(camera_id)
        return jsonify({"error": err}), 400

    return jsonify({"status": "started", "camera_id": camera_id, **session.status()})

@bp.route("/cameras", methods=["GET"])
def cameras_list():
    return jsonify({"cameras": app_state.active_camera_ids()})

@bp.route("/cameras/<camera_id>/status", methods=["GET"])
def cameras_status(camera_id):
    session = app_state.get_session(camera_id)
    if session is None:
        return jsonify({"running": False, "error": None, "has_frame": False}), 404
    return jsonify(session.status())

@bp.route("/cameras/<camera_id>/stop", methods=["POST"])
def cameras_stop(camera_id):
    session = app_state.get_session(camera_id)
    if session is not None:
        session.stop()
        app_state.remove_session(camera_id)
    return jsonify({"status": "stopped", "camera_id": camera_id})

@bp.route("/cameras/<camera_id>/config", methods=["POST"])
def cameras_config(camera_id):
    session = app_state.get_session(camera_id)
    if session is None:
        return jsonify({"error": "Camera not found"}), 404
        
    if "zones" in request.form:
        try:
            zones_input = json.loads(request.form.get("zones", "[]"))
            session.zones_input = zones_input
            session.zones = [] # forces normalization on next frame
        except Exception as e:
            return jsonify({"error": str(e)}), 400
            
    if "intruder_detection" in request.form:
        session.intruder_detection = request.form.get("intruder_detection") == "1"
        
    return jsonify({"status": "ok"})

@bp.route("/cameras/stop-all", methods=["POST"])
def cameras_stop_all():
    for cam_id in app_state.active_camera_ids():
        session = app_state.get_session(cam_id)
        if session is not None:
            session.stop()
            app_state.remove_session(cam_id)
    return jsonify({"status": "all stopped"})

@bp.route("/cameras/<camera_id>/stream", methods=["GET"])
def cameras_stream(camera_id):
    session = app_state.get_session(camera_id)
    if session is None or not session.running:
        return "Camera not started.", 400
    return Response(session.frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@bp.route("/cameras/<camera_id>/snapshot", methods=["GET"])
def cameras_snapshot(camera_id):
    session = app_state.get_session(camera_id)
    if session is None or not session.running:
        return "Camera not started.", 400
    
    with session._jpeg_lock:
        if session._jpeg is None:
            return "No frame yet.", 400
        frame_bytes = session._jpeg
        
    return Response(frame_bytes, mimetype='image/jpeg')

@bp.route("/cameras/<camera_id>/zones", methods=["GET"])
def cameras_get_zones(camera_id):
    session = app_state.get_session(camera_id)
    if session is None:
        return jsonify({"error": "Camera not found"}), 404
    return jsonify({
        "zones": session.zones_input or [],
        "intruder_detection": session.intruder_detection
    })


# ── Backward Compatibility ──

@bp.route("/webcam/start", methods=["POST"])
def webcam_start():
    # Force ID to LIVE for legacy calls
    d = request.form.to_dict()
    d["camera_id"] = "LIVE"
    request.form = request.form.__class__(d)
    return cameras_start()

@bp.route("/webcam/status", methods=["GET"])
def webcam_status():
    session = app_state.get_session("LIVE")
    if session is None:
        return jsonify({"running": False, "error": None, "has_frame": False})
    return jsonify(session.status())

@bp.route("/webcam/stop", methods=["POST"])
def webcam_stop():
    return cameras_stop("LIVE")

@bp.route("/webcam/stream", methods=["GET"])
def webcam_stream():
    return cameras_stream("LIVE")


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
