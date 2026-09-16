"""
alert_routes.py — alert retrieval, clearing, and live SSE stream routes.
"""

import json
import queue

from flask import Blueprint, jsonify, Response

import database
import app_state

bp = Blueprint("alerts", __name__)


@bp.route("/alerts", methods=["GET"])
def get_alerts():
    alerts = database.get_recent_alerts(limit=50)
    return jsonify(alerts)


@bp.route("/alerts/clear", methods=["POST"])
def clear_alerts():
    database.clear_alerts()
    return jsonify({"status": "cleared"})


@bp.route("/alerts/stream", methods=["GET"])
def stream_alerts():
    q = queue.Queue(maxsize=10)
    app_state._alert_subscribers.append(q)

    def generate():
        try:
            while True:
                alert = q.get()
                yield f"data: {json.dumps(alert)}\n\n"
        except GeneratorExit:
            if q in app_state._alert_subscribers:
                app_state._alert_subscribers.remove(q)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
