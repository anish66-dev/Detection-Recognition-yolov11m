"""
enroll_routes.py — face enrollment and people management routes.
"""

import uuid
import cv2
from pathlib import Path

from flask import Blueprint, request, jsonify

import database
import pipeline
from config import PHOTOS_DIR
from app_state import refresh_faces_cache

bp = Blueprint("enroll", __name__)


@bp.route("/enroll", methods=["POST"])
def enroll():
    if "photo" not in request.files:
        return jsonify({"error": "No photo file"}), 400

    name = request.form.get("name")
    status = request.form.get("status", "authorized")

    if not name:
        return jsonify({"error": "Name is required"}), 400

    file = request.files["photo"]
    suffix = Path(file.filename).suffix or ".jpg"
    photo_filename = f"{uuid.uuid4()}{suffix}"
    photo_path = PHOTOS_DIR / photo_filename
    file.save(str(photo_path))

    img = cv2.imread(str(photo_path))
    if img is None:
        photo_path.unlink()
        return jsonify({"error": "Invalid image format"}), 400

    # If the face models never loaded, EVERY photo fails — surface that clearly
    # instead of blaming the user's photo.
    if not pipeline.models_ready():
        photo_path.unlink()
        model_status = pipeline.get_status()
        return jsonify({
            "error": "Face model not loaded on the server. Check the server console logs.",
            "detail": model_status.get("error"),
        }), 503

    embedding = pipeline.get_embedding(img)
    if embedding is None:
        photo_path.unlink()
        return jsonify({"error": "No face detected in photo"}), 400

    person_id = database.add_person(name, status, photo_path, embedding)
    refresh_faces_cache()

    return jsonify({
        "status": "success",
        "person_id": person_id,
        "name": name,
        "role": status
    })


@bp.route("/people", methods=["GET"])
def list_people():
    people = database.get_all_people()
    return jsonify(people)


@bp.route("/people/<int:person_id>", methods=["DELETE"])
def delete_person(person_id):
    database.delete_person(person_id)
    refresh_faces_cache()
    return jsonify({"status": "deleted"})
