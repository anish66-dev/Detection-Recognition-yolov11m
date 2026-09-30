"""
alerts.py — edge-triggered alert rules.

The problem this replaces
-------------------------
The previous logic debounced intruder alerts with a key built from the person's
pixel position::

    alert_key = f"intruder:{x1}_{y1}"   # "approximate position"

x1/y1 are exact pixels, so a person who moved one pixel produced a brand-new
key, skipped the cooldown entirely, and fired an alert on *every frame*. The
cooldown dict also grew one entry per pixel position and was never pruned.

Two changes fix it:

  * **Key on track ID.** A person is one entity over time, not a new one each
    frame. This requires the tracker, which is why the two land together.
  * **Trigger on transitions, not on states.** Fire when someone *enters* a
    zone, not for every frame they stand in it. A level-triggered rule with a
    5-second cooldown still pings forever while the condition holds.

Rules are also independent rather than mutually exclusive: a blocklisted person
walking into a restricted zone raises *both* a restricted-entry and a zone
alert. In the old if/elif chain the zone branch was last and therefore
unreachable for anyone whose face had been detected.

Dependency-free and clock-injected so it can be tested without a camera.
"""

# A confirmed condition must hold this many analysed frames before firing.
# Suppresses single-frame flickers (a box clipping a zone edge, one blurred face).
CONFIRM_FRAMES = {
    "restricted_entry": 2,
    "intruder": 5,
    "zone_intrusion": 3,
}

# Once fired for a track, the same alert type will not fire again for that track
# until this many seconds have passed *and* the condition has lapsed and returned.
REARM_SECS = 45.0

# Guards against unbounded growth if tracks churn quickly.
MAX_TRACKED_STATES = 512


class _TrackState:
    """
    Per-track rule state.

    Latching is keyed by *(alert type, subject)* rather than alert type alone.
    The subject is the zone name or the matched person's name, and it matters:
    someone walking from Zone A straight into Zone B is a new event that must
    fire immediately, even though "zone_intrusion" fired moments ago.
    """

    __slots__ = ("streaks", "fired_at", "latched", "last_seen")

    def __init__(self, now):
        self.streaks = {}     # alert_type -> consecutive confirming frames
        self.fired_at = {}    # "type|subject" -> timestamp of last fire
        self.latched = {}     # alert_type -> subject currently latched high
        self.last_seen = now


class AlertEngine:
    """
    Stateful rule evaluator. One instance per video source.

    Args:
        sink: callable ``(name, similarity, alert_type) -> alert_dict | None``.
            Injected so tests do not touch the database. app.py wires this to
            ``database.save_alert`` plus the SSE broadcast.
    """

    def __init__(self, sink, rearm_secs=REARM_SECS):
        self._sink = sink
        self._rearm = rearm_secs
        self._states = {}

    # ── rule predicates ──────────────────────────────────────────────────────
    # Each returns (should_fire, display_name, similarity, subject) for one
    # record. They are evaluated independently — no elif chain, so a single
    # person can satisfy several at once.

    @staticmethod
    def _rule_restricted_entry(rec):
        if rec["identity_state"] == "known" and rec["identity_status"] == "blocklisted":
            name = rec["identity_name"]
            if rec.get("zone"):
                name = f"{name} (in {rec['zone']})"
            return True, name, rec["identity_sim"], rec["identity_name"]
        return False, None, 0.0, None

    @staticmethod
    def _rule_intruder(rec, intruder_detection, has_enrollment):
        # With nobody enrolled, every face is "unrecognised" and the whole frame
        # would alert. That is a misconfiguration, not an intrusion.
        if not intruder_detection or not has_enrollment:
            return False, None, 0.0, None
        if rec["identity_state"] == "unknown":
            name = "Unknown Intruder"
            if rec.get("zone"):
                name = f"{name} (in {rec['zone']})"
            return True, name, rec["identity_sim"], "unknown"
        return False, None, 0.0, None

    @staticmethod
    def _rule_zone_intrusion(rec):
        zone = rec.get("zone")
        if zone:
            # Check zone authorization
            import app_state
            authorized = app_state.zone_authorizations_cache.get(zone, [])
            pid = rec.get("identity_id")
            
            if pid and pid in authorized:
                return False, None, 0.0, None
                
            name = rec.get("identity_name") or "Unknown Person"
            return True, f"{name} entered restricted {zone}", 0.0, zone
        return False, None, 0.0, None

    # ── evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, records, now, intruder_detection=True, has_enrollment=True):
        """
        Evaluate all rules against this frame's records and emit any alerts.

        Args:
            records: per-person dicts from ``Track.to_record()``.
            now: seconds; wall clock for live sources, frame_num/fps for files.

        Returns:
            List of alert dicts produced by the sink this frame (usually empty).
        """
        fired = []

        for rec in records:
            # Unconfirmed tracks are noise — a spurious two-frame detection must
            # never raise an alarm.
            if not rec.get("confirmed"):
                continue

            tid = rec["track_id"]
            state = self._states.get(tid)
            if state is None:
                state = self._states[tid] = _TrackState(now)
            state.last_seen = now

            candidates = (
                ("restricted_entry",) + self._rule_restricted_entry(rec),
                ("intruder",) + self._rule_intruder(rec, intruder_detection, has_enrollment),
                ("zone_intrusion",) + self._rule_zone_intrusion(rec),
            )

            for alert_type, should, name, sim, subject in candidates:
                if not should:
                    # Condition lapsed: unlatch so a genuine re-entry can fire.
                    state.streaks[alert_type] = 0
                    state.latched.pop(alert_type, None)
                    continue

                # Same rule but a different subject (Zone A -> Zone B, or a
                # re-identified person) is a distinct event: restart the streak
                # and drop the old latch rather than swallowing it.
                if state.latched.get(alert_type) not in (None, subject):
                    state.streaks[alert_type] = 0
                    state.latched.pop(alert_type, None)

                streak = state.streaks.get(alert_type, 0) + 1
                state.streaks[alert_type] = streak
                if streak < CONFIRM_FRAMES.get(alert_type, 1):
                    continue

                # Latched high already — this is the same ongoing event.
                if state.latched.get(alert_type) == subject:
                    continue

                key = f"{alert_type}|{subject}"
                last = state.fired_at.get(key)
                if last is not None and (now - last) < self._rearm:
                    continue

                state.latched[alert_type] = subject
                state.fired_at[key] = now
                alert = self._sink(name, sim, alert_type)
                if alert is not None:
                    fired.append(alert)

        self._prune(now, {r["track_id"] for r in records})
        return fired

    def _prune(self, now, live_ids):
        """Drop state for tracks that are long gone, so memory stays bounded."""
        if len(self._states) <= MAX_TRACKED_STATES:
            stale = [
                tid for tid, st in self._states.items()
                if tid not in live_ids and (now - st.last_seen) > self._rearm * 2
            ]
        else:
            stale = sorted(self._states, key=lambda t: self._states[t].last_seen)
            stale = stale[: len(self._states) - MAX_TRACKED_STATES]
        for tid in stale:
            self._states.pop(tid, None)

    def reset(self):
        self._states.clear()
