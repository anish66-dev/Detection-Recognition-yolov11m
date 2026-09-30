"""
analytics_routes.py — crowd analytics API endpoints.

GET  /analytics/crowd  — full crowd analytics state (all new fields included)
POST /analytics/reset  — clear all accumulated session analytics and restart
"""

from flask import Blueprint, jsonify
import app_state

bp = Blueprint("analytics", __name__)


@bp.route("/analytics/crowd", defaults={"camera_id": "LIVE"}, methods=["GET"])
@bp.route("/analytics/crowd/<camera_id>", methods=["GET"])
def crowd_stats(camera_id):
    session = app_state.get_session(camera_id)
    if session is None:
        return jsonify({"error": "Camera not found or not running"}), 404

    stats  = session.crowd_counter.get_stats()
    status = session.crowd_monitor.get_status()

    # Merge monitor state into stats dict
    stats["status"]             = status["status"]
    stats["trend"]              = status["trend"]
    stats["dominant_direction"] = status["dominant_direction"]
    stats["thresholds"]         = status["thresholds"]
    stats["events"]             = status["events"]

    # Build the zone table: count + crowd status per zone
    zones_merged = {}
    for zone, count in stats.get("zones", {}).items():
        z_status = status["zone_status"].get(zone, "NORMAL")
        zones_merged[zone] = {"count": count, "status": z_status}
    stats["zones"] = zones_merged

    return jsonify(stats)


@bp.route("/analytics/reset", methods=["POST"])
def reset_analytics():
    """
    Clear all accumulated session-level analytics and restart counters.

    Clears: headcount history, peak/avg/min, trend history, direction
    history, zone statistics, crowd events, density-estimator state.

    Does NOT clear: model weights, enrolled faces, alert DB records,
    application configuration, or per-job video processing state.
    """
    app_state.reset_analytics()
    return jsonify({"status": "reset", "message": "Analytics reset successfully."})
