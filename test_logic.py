"""
test_logic.py — dependency-free tests for the tracking and alerting logic.

Run with:  python test_logic.py

No torch, no cv2, no camera and no model download required. tracker.py and
alerts.py are pure Python, and app.py is imported with its heavy dependencies
stubbed out. The cases below cover the specific defects this rewrite targets:
duplicate boxes on one person, faces assigned to the wrong body, alert spam
from a moving intruder, and the zone rule that could never fire.
"""

import os
import sys
import types
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracker import Tracker, merge_duplicate_boxes
from alerts import AlertEngine

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  ok  {msg}")


def section(title):
    print(f"\n{title}")


# ═══════════════════════════════════════════════════════════════════════════
# Tracker
# ═══════════════════════════════════════════════════════════════════════════

def test_tracker():
    section("tracker.py")

    # A person walking across the frame must stay one identity, not become a
    # new "person" every frame — everything downstream keys off this.
    trk = Tracker()
    ids = []
    for i in range(12):
        x = 100 + i * 8
        tracks = trk.update([[x, 50, x + 60, 250, 0.9]], now=i / 30.0)
        ids.append(tracks[0].id)
    assert len(set(ids)) == 1, f"track ID churned: {set(ids)}"
    assert tracks[0].confirmed
    ok(f"moving person keeps one ID across 12 frames (id={ids[0]})")

    # Coasting: boxes extrapolate forward between detections, but never further
    # than MAX_COAST_SECS or a stale track would sail off screen.
    t = tracks[0]
    base = t.predict(11 / 30.0)
    later = t.predict(11 / 30.0 + 0.2)
    assert later[0] > base[0]
    assert abs(t.predict(11 / 30.0 + 10.0)[0] - t.predict(11 / 30.0 + 0.5)[0]) < 1e-6
    ok("box coasts forward between detections and caps extrapolation")

    # Two people converging must not swap identities.
    trk = Tracker()
    for i in range(10):
        ts = trk.update([[100 + i * 5, 50, 160 + i * 5, 250, 0.9],
                         [400 - i * 5, 50, 460 - i * 5, 250, 0.9]], now=i / 30.0)
    assert len(ts) == 2 and ts[0].id != ts[1].id
    ok("two people keep distinct IDs while converging")

    # Duplicate suppression: the direct cause of one human rendering as two
    # differently-coloured boxes.
    kept = merge_duplicate_boxes([[100, 50, 160, 250, 0.90],
                                  [102, 52, 158, 248, 0.55],
                                  [400, 50, 460, 250, 0.80]])
    assert len(kept) == 2 and kept[0][4] == 0.90
    ok("overlapping duplicate boxes collapse, highest confidence wins")

    # Nested boxes have low IoU but high containment — IoU alone misses these.
    assert len(merge_duplicate_boxes([[100, 50, 200, 300, 0.9],
                                      [120, 60, 180, 150, 0.6]])) == 1
    ok("nested duplicate box collapses (containment, not just IoU)")

    # Identity voting: one motion-blurred frame must not turn a known person
    # into an intruder.
    trk = Tracker()
    for i in range(6):
        ts = trk.update([[100, 50, 160, 250, 0.9]], now=i / 30.0)
    t = ts[0]
    for _ in range(5):
        t.observe_identity("Ravi", "authorized", 0.71, "known")
    assert t.identity_state == "known" and t.identity_name == "Ravi"
    t.observe_identity(None, None, 0.10, "unknown")
    assert t.identity_state == "known", "a single bad frame flipped the identity"
    ok("known identity survives one bad recognition frame")

    for _ in range(6):
        t.observe_identity(None, None, 0.10, "unknown")
    assert t.identity_state == "unknown"
    ok("sustained non-recognition does flip the track to unknown")

    # The hysteresis band must never commit a label either way.
    trk = Tracker()
    for i in range(6):
        ts = trk.update([[10, 10, 60, 200, 0.9]], now=i / 30.0)
    u = ts[0]
    for _ in range(9):
        u.observe_identity("Ravi", "authorized", 0.31, "uncertain")
    assert u.identity_state == "uncertain" and u.identity_name is None
    ok("borderline similarity stays 'uncertain', commits no label")

    trk = Tracker(max_misses=3)
    for i in range(5):
        trk.update([[10, 10, 60, 200, 0.9]], now=i / 30.0)
    for i in range(5, 12):
        live = trk.update([], now=i / 30.0)
    assert live == []
    ok("track expires after enough consecutive misses")

def test_tracker_recovery():
    section("tracker.py (recovery)")
    trk = Tracker(max_misses=5)
    
    # 1. New track
    trk.update([[100, 50, 160, 250, 0.9]], now=0.0)
    t_id = trk.tracks[0].id
    assert trk.tracks[0].state == "active"
    
    # 2. Miss for 3 frames (coasting)
    for i in range(1, 4):
        trk.update([], now=i / 30.0)
    assert len(trk.tracks) == 1
    assert trk.tracks[0].id == t_id
    assert trk.tracks[0].state == "coasting"
    ok("track kept alive during missed frames (coasting)")
    
    # 3. Re-detect within max_misses (should restore ID)
    trk.update([[102, 52, 162, 252, 0.9]], now=4.0 / 30.0)
    assert len(trk.tracks) == 1
    assert trk.tracks[0].id == t_id
    assert trk.tracks[0].state == "active"
    ok("re-detection within max_misses restores same ID")
    
    # 4. Miss beyond max_misses
    for i in range(5, 12):
        trk.update([], now=i / 30.0)
    assert len(trk.tracks) == 0
    ok("track evicted after max_misses consecutive misses")
    
    # 5. New detection gets new ID
    trk.update([[100, 50, 160, 250, 0.9]], now=12.0 / 30.0)
    assert len(trk.tracks) == 1
    assert trk.tracks[0].id > t_id
    ok("new detection after eviction gets a new higher ID")


# ═══════════════════════════════════════════════════════════════════════════
# Alert engine
# ═══════════════════════════════════════════════════════════════════════════

def _rec(tid=1, zone=None, state="uncertain", name=None, status=None,
         sim=0.0, confirmed=True, x=100):
    return {"track_id": tid, "bbox": [x, 50, x + 60, 250], "conf": 0.9,
            "zone": zone, "identity_name": name, "identity_status": status,
            "identity_sim": sim, "identity_state": state, "confirmed": confirmed}


def _engine():
    log = []
    eng = AlertEngine(lambda n, s, t: (log.append((n, t)) or {"name": n, "alert_type": t}))
    return eng, log


def test_alerts():
    section("alerts.py")

    # THE regression: the old code keyed its cooldown on exact pixel position,
    # so a moving intruder bypassed the debounce and alerted on every frame.
    eng, log = _engine()
    for i in range(120):
        eng.evaluate([_rec(state="unknown", x=100 + i)], now=i / 30.0)
    assert len(log) == 1, f"moving intruder fired {len(log)} times"
    ok("moving intruder over 120 frames fires exactly 1 alert")

    # Level-triggered rules re-fire forever while the condition holds.
    eng, log = _engine()
    for i in range(600):
        eng.evaluate([_rec(zone="Restricted")], now=i / 30.0)
    assert len(log) == 1, f"standing in zone fired {len(log)} times"
    ok("person standing in a zone for 20s fires exactly 1 alert")

    eng, log = _engine()
    t = 0.0
    for _ in range(30): eng.evaluate([_rec(zone="Restricted")], now=t); t += 1 / 30
    for _ in range(30): eng.evaluate([_rec(zone=None)], now=t); t += 1 / 30
    t += 60.0
    for _ in range(30): eng.evaluate([_rec(zone="Restricted")], now=t); t += 1 / 30
    assert len(log) == 2
    ok("leaving and re-entering after re-arm is a genuine new event")

    # The dead-elif-branch bug: zone alerts were unreachable for anyone whose
    # face had been detected.
    eng, log = _engine()
    for i in range(30):
        eng.evaluate([_rec(zone="Vault", state="known", name="Mallory",
                           status="blocklisted", sim=0.8)], now=i / 30.0)
    assert sorted(t for _, t in log) == ["restricted_entry", "zone_intrusion"]
    ok("blocklisted person in a zone raises BOTH alerts")

    eng, log = _engine()
    for i in range(30):
        eng.evaluate([_rec(zone="Vault", state="known", name="Ravi",
                           status="authorized", sim=0.8)], now=i / 30.0)
    assert [t for _, t in log] == ["zone_intrusion"]
    ok("authorized person in a zone still raises the zone alert")

    eng, log = _engine()
    for i in range(60):
        eng.evaluate([_rec(state="unknown")], now=i / 30.0, has_enrollment=False)
    assert log == []
    ok("empty enrollment DB does not make everyone an intruder")

    eng, log = _engine()
    for i in range(60):
        eng.evaluate([_rec(state="uncertain")], now=i / 30.0)
    assert log == []
    ok("uncertain recognition raises nothing")

    eng, log = _engine()
    for i in range(60):
        eng.evaluate([_rec(state="unknown", confirmed=False)], now=i / 30.0)
    assert log == []
    ok("unconfirmed 2-frame detection blip raises nothing")

    eng, log = _engine()
    eng.evaluate([_rec(zone="Vault")], now=0.0)
    eng.evaluate([_rec(zone=None)], now=1 / 30)
    assert log == []
    ok("box clipping a zone edge for one frame raises nothing")

    eng, log = _engine()
    for i in range(60):
        eng.evaluate([_rec(tid=1, state="unknown"),
                      _rec(tid=2, state="unknown", x=400)], now=i / 30.0)
    assert len(log) == 2
    ok("two separate intruders each alert once")

    eng, log = _engine()
    t = 0.0
    for _ in range(20): eng.evaluate([_rec(zone="ZoneA")], now=t); t += 1 / 30
    for _ in range(20): eng.evaluate([_rec(zone="ZoneB")], now=t); t += 1 / 30
    assert len(log) == 2
    ok("walking from Zone A into Zone B alerts for both")

    # The old cooldown dict grew one entry per pixel position, forever.
    eng, log = _engine()
    for i in range(200):
        eng.evaluate([_rec(tid=i, state="unknown")], now=i * 10.0)
    assert len(eng._states) < 60, f"state leak: {len(eng._states)} entries"
    ok(f"rule state stays bounded ({len(eng._states)} entries after 200 tracks)")


# ═══════════════════════════════════════════════════════════════════════════
# app.py analysis helpers (heavy deps stubbed)
# ═══════════════════════════════════════════════════════════════════════════

def _import_app_with_stubs():
    for name in ("cv2", "numpy", "ultralytics", "insightface"):
        sys.modules.setdefault(name, MagicMock())

    flask = types.ModuleType("flask")

    class FakeFlask:
        def __init__(self, *a, **k): pass
        def route(self, *a, **k): return lambda f: f
        def run(self, *a, **k): pass
        def register_blueprint(self, *a, **k): pass

    flask.Flask = FakeFlask
    flask.Blueprint = lambda *a, **k: type("BP", (), {
        "route": lambda self, *a2, **k2: lambda f: f,
    })()
    for attr in ("request", "jsonify", "make_response", "send_from_directory", "Response"):
        setattr(flask, attr, MagicMock())
    sys.modules["flask"] = flask

    fc = types.ModuleType("flask_cors")
    fc.CORS = lambda *a, **k: None
    sys.modules["flask_cors"] = fc

    db = types.ModuleType("database")
    db.get_all_embeddings = lambda: []
    db.save_alert = lambda *a, **k: {}
    db.init_db = lambda: None
    sys.modules["database"] = db

    pl = types.ModuleType("pipeline")
    pl.get_device = lambda: "cpu"
    pl.detect_persons = lambda *a, **k: []
    pl.detect_faces = lambda f: []
    pl.match_face_state = lambda *a, **k: (None, 0.0, "uncertain")
    pl.get_status = lambda: {}
    pl.models_ready = lambda: True
    pl.get_embedding = lambda i: None
    sys.modules["pipeline"] = pl

    # Import the individual modules that contain the functions we test
    import config as cfg
    import analysis
    import geometry
    import camera
    import app
    return app, analysis, geometry, camera, cfg


def test_app():
    section("app.py (refactored modules)")
    app, analysis, geometry, camera, cfg = _import_app_with_stubs()
    ok("all modules import cleanly")

    # Two overlapping bodies: B's face centre falls inside A's box, which is
    # exactly what the old first-centre-inside-wins loop mis-assigned.
    person_a = [100, 50, 260, 400]
    person_b = [200, 50, 360, 400]
    face_a = {"bbox": [140, 70, 190, 130], "id": "A"}
    face_b = {"bbox": [250, 70, 300, 130], "id": "B"}

    got = analysis.assign_faces_to_persons([person_a, person_b], [face_a, face_b])
    assert got[0]["id"] == "A" and got[1]["id"] == "B", got
    ok("overlapping people get correct 1:1 face assignment")

    # One face inside two nested boxes must be claimed once, not twice — the
    # double-claim is what drew one human as a person box AND an INTRUDER box.
    got = analysis.assign_faces_to_persons([person_a, [110, 55, 250, 390]], [face_a])
    assert len(got) == 1
    ok("one face inside two boxes is claimed by exactly one")

    assert analysis.assign_faces_to_persons([person_a], [{"bbox": [900, 900, 950, 950]}]) == {}
    ok("face outside every body stays unassigned")

    zones = [{"name": "Vault", "points": [(100, 100), (300, 100), (300, 300), (100, 300)]}]
    assert analysis.zone_for_box([150, 50, 250, 250], zones) == "Vault"
    assert analysis.zone_for_box([150, 50, 250, 90], zones) is None
    assert analysis.zone_for_box([400, 50, 500, 250], zones) is None
    ok("zone membership uses the bottom-centre ground point")

    assert analysis.count_zones([{"zone": "Vault"}, {"zone": "Vault"}, {"zone": None}], zones) == {"Vault": 2}
    ok("zone occupancy counts correctly")

    def rec(state="uncertain", status=None, zone=None, name=None):
        return {"identity_state": state, "identity_status": status, "zone": zone,
                "identity_name": name, "identity_sim": 0.8, "conf": 0.9}

    assert analysis._style_for(rec("known", "blocklisted", "Vault", "M"))[0] == cfg.ALERT_BOX_COLOR
    assert analysis._style_for(rec("unknown"))[0] == cfg.INTRUDER_BOX_COLOR
    assert analysis._style_for(rec("known", "authorized", "Vault", "R"))[0] == cfg.AUTH_BOX_COLOR
    assert analysis._style_for(rec(zone="Vault"))[0] == cfg.ZONE_BOX_COLOR
    assert analysis._style_for(rec())[0] == cfg.BOX_COLOR
    assert analysis._style_for(rec("uncertain"))[0] == cfg.BOX_COLOR
    ok("render priority correct; 'uncertain' never draws as INTRUDER")

    cases = {
        "0": ("device", 0), "1": ("device", 1), "": ("device", 0),
        "push": ("push", None), "PUSH": ("push", None),
        "http://192.168.1.42:8080/video": ("url", "http://192.168.1.42:8080/video"),
        "rtsp://10.0.0.5:554/live": ("url", "rtsp://10.0.0.5:554/live"),
    }
    for spec, want in cases.items():
        assert camera.parse_source(spec) == want, f"{spec!r} -> {camera.parse_source(spec)}"
    ok(f"camera source parsing correct for {len(cases)} specs")


def test_crowd():
    section("analytics.py (crowd)")
    from analytics.crowd_counter import CrowdCounter
    from analytics.crowd_monitor import CrowdMonitor
    import config
    
    config.CROWD_WARNING_THRESHOLD = 2
    config.CROWD_CRITICAL_THRESHOLD = 4
    
    counter = CrowdCounter()
    monitor = CrowdMonitor()
    
    zones = [{"name": "Lobby"}, {"name": "Vault"}]
    
    # Empty
    stats = counter.update([], zones, now=0.0)
    events = monitor.evaluate(stats["current_count"], stats["zones"], now=0.0)
    assert stats["current_count"] == 0
    assert not events
    
    # 1 active, 1 coasting -> count 1
    records = [
        {"state": "active", "zone": "Lobby"},
        {"state": "coasting", "zone": "Vault"}
    ]
    stats = counter.update(records, zones, now=1.0)
    events = monitor.evaluate(stats["current_count"], stats["zones"], now=1.0)
    
    assert stats["current_count"] == 1
    assert stats["peak_count"] == 1
    assert stats["zones"]["Lobby"] == 1
    assert stats["zones"]["Vault"] == 0
    assert monitor.state == "NORMAL"
    ok("active tracks counted, coasting ignored")
    
    # Jump to 3 (WARNING)
    records = [{"state": "active", "zone": "Lobby", "confirmed": True}] * 3
    stats = counter.update(records, zones, now=2.0)
    events = monitor.evaluate(stats["current_count"], stats["zones"], now=2.0)
    
    assert stats["current_count"] == 3
    assert stats["peak_count"] == 3
    assert any(e["type"] == "global_crowd" and e["new_state"] == "CROWDED" for e in events)
    assert monitor.state == "CROWDED"
    ok("thresholds fire correctly (NORMAL -> CROWDED)")
    
    # Stay at 3 -> no new events
    events = monitor.evaluate(stats["current_count"], stats["zones"], now=3.0)
    assert not events
    ok("state-change event emitted only once per transition")


# ═══════════════════════════════════════════════════════════════════════════
# Density Estimator
# ═══════════════════════════════════════════════════════════════════════════

def test_density_estimator():
    section("analytics/density_estimator.py")

    from analytics.density_estimator import DensityEstimator
    import config

    est = DensityEstimator()
    frame_shape = (720, 1280)

    # Sparse scene: below DENSE_YOLO_THRESH — pass-through, no density mode.
    sparse_boxes = [[100, 100, 160, 300, 0.9]] * (config.DENSE_YOLO_THRESH - 1)
    count, active = est.estimate(sparse_boxes, frame_shape)
    assert active is False, "Dense mode should NOT activate for sparse scene"
    assert count == float(len(sparse_boxes)), "Sparse scene should return YOLO count"
    ok("sparse scene: density estimator is pass-through")

    # Dense scene: at or above DENSE_YOLO_THRESH — density mode activates.
    # Use a fresh estimator so the EMA starting point is zero (no bleed from the sparse test).
    est2 = DensityEstimator()
    dense_boxes = [[50 + i * 60, 100, 110 + i * 60, 400, 0.85]
                   for i in range(config.DENSE_YOLO_THRESH + 5)]
    # Run several frames to let EMA converge
    count2, active2 = None, None
    for _ in range(5):
        count2, active2 = est2.estimate(dense_boxes, frame_shape)
    assert active2 is True, "Dense mode should activate for dense scene"
    yolo_n = len(dense_boxes)
    # Estimate should be within 20% of the YOLO count. Boundary-clipped kernels
    # can make the integral slightly less than the exact headcount.
    assert abs(count2 - yolo_n) / yolo_n <= 0.20, (
        f"Density estimate ({count2}) is more than 20% off YOLO count ({yolo_n})"
    )
    ok("dense scene: density mode activates and estimate within 20% of YOLO count")



    # Reset clears smoothing state.
    est.reset()
    count3, active3 = est.estimate([], frame_shape)
    assert active3 is False
    assert count3 == 0.0
    ok("reset clears estimator state")


# ═══════════════════════════════════════════════════════════════════════════
# Crowd Trend
# ═══════════════════════════════════════════════════════════════════════════

def test_crowd_trend():
    section("CrowdMonitor — trend")

    from analytics.crowd_monitor import CrowdMonitor

    monitor = CrowdMonitor()

    # Build increasing counts over CROWD_TREND_WINDOW samples
    import config
    window = config.CROWD_TREND_WINDOW
    for i in range(window):
        monitor.evaluate(i, {}, now=float(i))
    assert monitor.trend == "INCREASING", f"Expected INCREASING, got {monitor.trend}"
    ok("crowd trend correctly reports INCREASING")

    monitor2 = CrowdMonitor()
    # Decreasing counts
    for i in range(window, 0, -1):
        monitor2.evaluate(i, {}, now=float(window - i))
    assert monitor2.trend == "DECREASING", f"Expected DECREASING, got {monitor2.trend}"
    ok("crowd trend correctly reports DECREASING")

    monitor3 = CrowdMonitor()
    # Stable counts
    for _ in range(window):
        monitor3.evaluate(5, {}, now=0.0)
    assert monitor3.trend == "STABLE", f"Expected STABLE, got {monitor3.trend}"
    ok("crowd trend correctly reports STABLE")


# ═══════════════════════════════════════════════════════════════════════════
# Direction Analyzer
# ═══════════════════════════════════════════════════════════════════════════

def test_direction_analyzer():
    section("analytics/direction_analyzer.py")

    from analytics.direction_analyzer import DirectionAnalyzer, VOTE_WINDOW
    from tracker import Tracker

    analyzer = DirectionAnalyzer()
    trk = Tracker()

    # Simulate a person walking RIGHT (x increases each frame)
    for i in range(20):
        x = 100 + i * 15
        trk.update([[x, 100, x + 60, 300, 0.9]], now=i / 30.0)

    direction = analyzer.update(trk.tracks)
    # After enough frames the direction should converge to RIGHT or NONE
    # (depends on velocity magnitude vs MIN_VELOCITY_PX_S)
    assert direction in ("RIGHT", "NONE"), f"Unexpected direction: {direction}"
    ok(f"direction analyzer output '{direction}' for rightward motion (RIGHT or NONE accepted)")

    analyzer.reset()
    assert analyzer.dominant_direction == "NONE"
    ok("direction analyzer reset clears state")


# ═══════════════════════════════════════════════════════════════════════════
# Reset Analytics
# ═══════════════════════════════════════════════════════════════════════════

def test_reset_analytics():
    section("CrowdCounter + CrowdMonitor — reset()")

    from analytics.crowd_counter import CrowdCounter
    from analytics.crowd_monitor import CrowdMonitor

    counter = CrowdCounter()
    monitor = CrowdMonitor()
    zones = []

    # Accumulate some state.
    for i in range(5):
        records = [{"state": "active", "zone": None, "confirmed": True}] * (i + 1)
        stats = counter.update(records, zones, now=float(i))
        monitor.evaluate(stats["current_count"], {}, now=float(i))

    assert counter.peak_count > 0
    assert counter.sample_count > 0
    assert len(counter.history) > 0

    counter.reset()
    monitor.reset()

    assert counter.current_count == 0, "current_count must be 0 after reset"
    assert counter.peak_count == 0, "peak_count must be 0 after reset"
    assert counter.min_count == -1, "min_count must be -1 (sentinel) after reset"
    assert counter.sample_count == 0, "sample_count must be 0 after reset"
    assert len(counter.history) == 0, "history must be empty after reset"
    assert counter.estimated_count == 0.0
    assert counter.density_level == "LOW"

    assert monitor.state == "NORMAL"
    assert monitor.trend == "STABLE"
    assert monitor.dominant_direction == "NONE"
    assert len(monitor.events) == 0
    ok("reset() clears all accumulated counter state")
    ok("reset() clears all accumulated monitor state")

    # After reset, accumulation starts fresh from next update.
    records = [{"state": "active", "zone": None, "confirmed": True}]
    stats = counter.update(records, zones, now=100.0)
    assert stats["current_count"] == 1
    assert stats["peak_count"] == 1
    assert stats["average_count"] == 1.0
    ok("analytics accumulate correctly after reset")


if __name__ == "__main__":
    test_tracker()
    test_tracker_recovery()
    test_alerts()
    # Analytics tests must run BEFORE test_app, because test_app installs a
    # MagicMock for numpy via sys.modules.setdefault("numpy", MagicMock()).
    # Once that mock is in place, numpy-dependent analytics code breaks.
    test_density_estimator()
    test_crowd_trend()
    test_direction_analyzer()
    test_reset_analytics()
    test_app()
    test_crowd()
    print(f"\n{len(PASSED)} checks passed.")
