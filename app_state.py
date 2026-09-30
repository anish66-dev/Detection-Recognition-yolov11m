"""
app_state.py — shared mutable application state.

Global state that was scattered across the top of app.py, consolidated here
so that route modules, the live session, and the processing worker can all
import it without circular dependencies.
"""

import queue
import threading

import database
def reset_analytics():
    """
    Clear all accumulated session analytics and restart from a clean state.
    """
    with _session_lock:
        for session in _sessions.values():
            session.crowd_counter.reset()
            session.crowd_monitor.reset()
            session.direction_analyzer.reset()


# ── Video processing jobs ────────────────────────────────────────────────────

_jobs = {}


# ── Alert SSE subscribers ────────────────────────────────────────────────────

_alert_subscribers = []


# ── Enrolled faces cache ─────────────────────────────────────────────────────

_enrolled_faces_cache = []
zone_authorizations_cache = {}

def refresh_faces_cache():
    global _enrolled_faces_cache
    _enrolled_faces_cache = database.get_all_embeddings()
    
def refresh_zone_authorizations_cache():
    global zone_authorizations_cache
    zone_authorizations_cache = database.get_all_zone_authorizations()

refresh_faces_cache()
refresh_zone_authorizations_cache()


def has_enrollment():
    return len(_enrolled_faces_cache) > 0


# ── Alert broadcasting ───────────────────────────────────────────────────────

def push_alert_to_subscribers(alert_dict):
    # Iterate over a copy so concurrent SSE disconnects don't mutate the list mid-loop
    for q in list(_alert_subscribers):
        try:
            q.put_nowait(alert_dict)
        except queue.Full:
            pass


def make_alert_sink(job_id, frame_ref):
    """
    Build the callback the AlertEngine fires through.

    ``frame_ref`` is a single-element list so the worker can keep updating the
    current frame number without rebuilding the sink every frame.
    """
    def sink(name, similarity, alert_type):
        display_name = f"[{job_id}] {name}"
        alert = database.save_alert(display_name, similarity, job_id, frame_ref[0],
                                    alert_type=alert_type)
        push_alert_to_subscribers(alert)
        return alert
    return sink


_sessions = {}
_session_lock = threading.Lock()

def get_session(cam_id):
    with _session_lock:
        return _sessions.get(cam_id)

def add_session(cam_id, session):
    with _session_lock:
        _sessions[cam_id] = session

def remove_session(cam_id):
    with _session_lock:
        if cam_id in _sessions:
            del _sessions[cam_id]

def all_sessions():
    with _session_lock:
        return list(_sessions.values())

def active_camera_ids():
    with _session_lock:
        return list(_sessions.keys())
