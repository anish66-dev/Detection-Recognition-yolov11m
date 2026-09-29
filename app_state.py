"""
app_state.py — shared mutable application state.

Global state that was scattered across the top of app.py, consolidated here
so that route modules, the live session, and the processing worker can all
import it without circular dependencies.
"""

import queue
import threading

import database
from analytics.crowd_counter    import CrowdCounter
from analytics.crowd_monitor    import CrowdMonitor
from analytics.direction_analyzer import DirectionAnalyzer

# ── Crowd Analytics ──────────────────────────────────────────────────────────

_crowd_counter    = CrowdCounter()
_crowd_monitor    = CrowdMonitor()
_direction_analyzer = DirectionAnalyzer()


def reset_analytics():
    """
    Clear all accumulated session analytics and restart from a clean state.

    What is reset:
      - headcount history (current / peak / avg / min / estimated)
      - crowd trend history
      - crowd direction history
      - zone statistics
      - crowd events log
      - density estimator smoothing state

    What is NOT reset:
      - model weights / configuration
      - alert database records
      - enrolled faces
      - live session / video job state
      - application configuration
    """
    _crowd_counter.reset()
    _crowd_monitor.reset()
    _direction_analyzer.reset()


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
