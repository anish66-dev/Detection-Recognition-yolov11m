"""
analytics/crowd_monitor.py — crowd state monitor.

Evaluates the current crowd state (NORMAL / CROWDED / CRITICAL) using the
same threshold logic as before, and now additionally computes:

  trend             — INCREASING / STABLE / DECREASING
                      Uses a rolling history window so a single anomalous
                      frame cannot flip the trend.

  dominant_direction — delegated to DirectionAnalyzer; exposed here so the
                       monitor is the single source of truth for the UI.

  events            — rolling list of crowd-state transitions (global + per
                      zone) for the event feed in the dashboard.

All fields are cleared by reset() which does NOT touch model configuration.
"""

import collections
import config
from config import CROWD_TREND_WINDOW

# Minimum fractional change between the two halves of the trend window to
# classify the trend as INCREASING or DECREASING (not STABLE).
_TREND_CHANGE_FRAC = 0.10   # 10 % relative change

# Maximum number of crowd events retained in the events log.
_MAX_EVENTS = 100


class CrowdMonitor:
    def __init__(self):
        self.state       = "NORMAL"
        self.zone_states: dict = {}
        self.trend       = "STABLE"
        self.dominant_direction = "NONE"
        self.events: collections.deque = collections.deque(maxlen=_MAX_EVENTS)

        # Internal trend history — keeps (timestamp, count) pairs.
        self._trend_history: collections.deque = collections.deque(
            maxlen=CROWD_TREND_WINDOW
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, current_count, zone_counts, now,
                 dominant_direction: str = "NONE"):
        """
        Evaluate crowd state and update all derived analytics.

        Args:
            current_count:      YOLO-tracked active person count.
            zone_counts:        dict {zone_name: count}.
            now:                timestamp.
            dominant_direction: from DirectionAnalyzer (optional).

        Returns:
            List of new crowd-state-change event dicts (may be empty).
        """
        new_events = []

        # ── Global status ─────────────────────────────────────────────
        new_state = self._get_status(current_count)
        if new_state != self.state:
            ev = {
                "type":      "global_crowd",
                "old_state": self.state,
                "new_state": new_state,
                "count":     current_count,
                "timestamp": now,
            }
            new_events.append(ev)
            self.events.append(ev)
            self.state = new_state

        # ── Zone status ───────────────────────────────────────────────
        for zone, count in zone_counts.items():
            z_state     = self._get_status(count)
            old_z_state = self.zone_states.get(zone, "NORMAL")
            if z_state != old_z_state:
                ev = {
                    "type":      "zone_crowd",
                    "zone":      zone,
                    "old_state": old_z_state,
                    "new_state": z_state,
                    "count":     count,
                    "timestamp": now,
                }
                new_events.append(ev)
                self.events.append(ev)
                self.zone_states[zone] = z_state

        # ── Trend ─────────────────────────────────────────────────────
        self._trend_history.append((now, current_count))
        self.trend = self._compute_trend()

        # ── Direction ─────────────────────────────────────────────────
        self.dominant_direction = dominant_direction

        return new_events

    def get_status(self) -> dict:
        return {
            "status":     self.state,
            "trend":      self.trend,
            "dominant_direction": self.dominant_direction,
            "thresholds": {
                "warning":  config.CROWD_WARNING_THRESHOLD,
                "critical": config.CROWD_CRITICAL_THRESHOLD,
            },
            "zone_status": {z: s for z, s in self.zone_states.items()},
            "events":      list(self.events)[-20:],  # last 20 for the API
        }

    def reset(self):
        """Clear accumulated session state. Configuration is untouched."""
        self.state      = "NORMAL"
        self.zone_states.clear()
        self.trend      = "STABLE"
        self.dominant_direction = "NONE"
        self.events.clear()
        self._trend_history.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_status(self, count: int) -> str:
        if count >= config.CROWD_CRITICAL_THRESHOLD:
            return "CRITICAL"
        if count >= config.CROWD_WARNING_THRESHOLD:
            return "CROWDED"
        return "NORMAL"

    def _compute_trend(self) -> str:
        """
        Classify INCREASING / STABLE / DECREASING using the rolling history.

        Strategy: split the window into a first half and a second half;
        compare the mean count of each half. A relative change above
        _TREND_CHANGE_FRAC is called increasing or decreasing; otherwise
        stable.

        With fewer than 4 samples we cannot compute a meaningful trend so
        we return STABLE.
        """
        history = list(self._trend_history)
        n = len(history)
        if n < 4:
            return "STABLE"

        mid   = n // 2
        first  = [v for _, v in history[:mid]]
        second = [v for _, v in history[mid:]]

        mean_first  = sum(first)  / max(1, len(first))
        mean_second = sum(second) / max(1, len(second))

        if mean_first == 0:
            # Avoid divide-by-zero when the scene was empty
            return "INCREASING" if mean_second > 0 else "STABLE"

        change_frac = (mean_second - mean_first) / mean_first
        if change_frac > _TREND_CHANGE_FRAC:
            return "INCREASING"
        if change_frac < -_TREND_CHANGE_FRAC:
            return "DECREASING"
        return "STABLE"
