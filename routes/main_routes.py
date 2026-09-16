"""
main_routes.py — top-level page and health routes.
"""

import subprocess

from flask import Blueprint, jsonify, send_from_directory

import pipeline
from config import PROJECT_DIR
from geometry import get_ffmpeg_executable
import app_state

bp = Blueprint("main", __name__)


@bp.route("/", methods=["GET"])
def serve_frontend():
    return send_from_directory(PROJECT_DIR, "index.html")


@bp.route("/phone", methods=["GET"])
def serve_phone_page():
    """Camera page the phone opens. See CameraSource for how frames come back."""
    return send_from_directory(PROJECT_DIR, "phone.html")


@bp.route("/health", methods=["GET"])
def health():
    ffmpeg_cmd = get_ffmpeg_executable()
    try:
        subprocess.run([ffmpeg_cmd, "-version"], capture_output=True, timeout=5)
        ffmpeg = True
    except Exception:
        ffmpeg = False

    # Report whether the face models are ACTUALLY loaded in memory — not just
    # whether the download folder exists.
    face_status = pipeline.get_status()

    return jsonify({
        "status": "ok",
        "model": "yolo11m",
        "cuda": pipeline.get_device() == "cuda",
        "ffmpeg": ffmpeg,
        "insightface": face_status["ready"],
        "face_detail": face_status,
        "enrolled_people": len(app_state._enrolled_faces_cache),
    })
