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
import app_state

bp = Blueprint("enroll", __name__)


@bp.route("/enroll", methods=["POST"])
def enroll():
    files = request.files.getlist("photo")
    if not files:
        return jsonify({"error": "No photo file"}), 400

    name = request.form.get("name")
    status = request.form.get("status", "authorized")

    if not name:
        return jsonify({"error": "Name is required"}), 400

    if not pipeline.models_ready():
        model_status = pipeline.get_status()
        return jsonify({
            "error": "Face model not loaded on the server. Check the server console logs.",
            "detail": model_status.get("error"),
        }), 503

    import numpy as np
    embeddings = []
    first_photo_path = None

    for file in files:
        suffix = Path(file.filename).suffix or ".jpg"
        photo_filename = f"{uuid.uuid4()}{suffix}"
        photo_path = PHOTOS_DIR / photo_filename
        file.save(str(photo_path))

        img = cv2.imread(str(photo_path))
        if img is None:
            photo_path.unlink()
            continue

        emb = pipeline.get_embedding(img)
        if emb is not None:
            embeddings.append(emb)
            if first_photo_path is None:
                first_photo_path = photo_path
            else:
                photo_path.unlink() # We only keep the first image on disk
        else:
            photo_path.unlink()

    if not embeddings:
        return jsonify({"error": "No faces detected in any provided photos"}), 400

    # Average embeddings to create a more robust representation
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    # 1. Face Deduplication Check
    # Check if this face matches anyone already enrolled (similarity > 0.75)
    match, sim = pipeline.match_face(avg_embedding, app_state._enrolled_faces_cache, threshold=0.75)
    if match:
        if first_photo_path:
            first_photo_path.unlink()
        return jsonify({
            "error": f"Duplicate found: This person is already enrolled as '{match['name']}' "
                     f"(similarity: {sim:.2f}). Please update the existing profile instead."
        }), 409

    token = str(uuid.uuid4())
    person_id = database.add_person(name, status, first_photo_path, avg_embedding, token)
    app_state.refresh_faces_cache()

    return jsonify({
        "status": "success",
        "person_id": person_id,
        "name": name,
        "role": status,
        "token": token
    })


@bp.route("/people", methods=["GET"])
def list_people():
    people = database.get_all_people()
    return jsonify(people)


@bp.route("/persons/token/<token>", methods=["GET"])
def get_person_by_token(token):
    person = database.get_person_by_token(token)
    if not person:
        return jsonify({"error": "Person not found"}), 404
    return jsonify(person)


@bp.route("/people/<int:person_id>", methods=["DELETE"])
def delete_person(person_id):
    database.delete_person(person_id)
    app_state.refresh_faces_cache()
    return jsonify({"status": "deleted"})


@bp.route("/people/<int:person_id>/status", methods=["POST"])
def update_person_status(person_id):
    data = request.json
    if not data or "status" not in data:
        return jsonify({"error": "Missing status"}), 400
    if data["status"] not in ("authorized", "blocklisted"):
        return jsonify({"error": "Invalid status"}), 400
        
    database.set_person_status(person_id, data["status"])
    app_state.refresh_faces_cache()
    return jsonify({"status": "success", "new_status": data["status"]})


@bp.route("/zones/authorizations", methods=["GET"])
def get_all_zone_authorizations():
    auths = database.get_all_zone_authorizations()
    return jsonify(auths)


@bp.route("/zones/<path:zone_name>/authorized", methods=["GET"])
def get_zone_authorized(zone_name):
    pids = database.get_zone_authorized_persons(zone_name)
    return jsonify({"zone": zone_name, "authorized_person_ids": pids})


@bp.route("/zones/<path:zone_name>/authorize", methods=["POST"])
def set_zone_authorized(zone_name):
    data = request.json
    if not data or "person_ids" not in data:
        return jsonify({"error": "Missing person_ids array"}), 400
    
    database.set_zone_authorized_persons(zone_name, data["person_ids"])
    app_state.refresh_zone_authorizations_cache()
    return jsonify({"status": "success", "zone": zone_name, "authorized_person_ids": data["person_ids"]})
