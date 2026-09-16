"""
app_state.py — shared mutable application state.

Global state that was scattered across the top of app.py, consolidated here
so that route modules, the live session, and the processing worker can all
import it without circular dependencies.
"""

import queue
import threading

import database


# ── Video processing jobs ────────────────────────────────────────────────────

_jobs = {}


# ── Alert SSE subscribers ────────────────────────────────────────────────────

_alert_subscribers = []


# ── Enrolled faces cache ─────────────────────────────────────────────────────

_enrolled_faces_cache = []


def refresh_faces_cache():
    global _enrolled_faces_cache
    _enrolled_faces_cache = database.get_all_embeddings()


refresh_faces_cache()


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
        alert = database.save_alert(name, similarity, job_id, frame_ref[0],
                                    alert_type=alert_type)
        push_alert_to_subscribers(alert)
        return alert
    return sink


# ── Live session state ────────────────────────────────────────────────────────

_session = None
_session_lock = threading.Lock()
