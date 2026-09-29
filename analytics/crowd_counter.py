"""
analytics/crowd_counter.py — crowd headcount aggregator.

Keeps two separate counts:
  current_count   — active YOLO-tracked individuals (existing behaviour)
  tracked_count   — confirmed tracks only (hits >= MIN_HITS)
  estimated_count — Gaussian density-estimator supplement for dense scenes

The density estimator is only active when YOLO count >= DENSE_YOLO_THRESH.
In sparse scenes estimated_count == current_count (pass-through, no overhead).

density_level summarises spatial density as LOW / MEDIUM / HIGH / CRITICAL
using the per-pixel density derived from the density-map integral and frame
area. This deliberately does NOT use the raw YOLO count alone.
"""

import collections
from config import CROWD_HISTORY_LEN, DENSE_YOLO_THRESH
from analytics.density_estimator import DensityEstimator

# Density thresholds: estimated crowd per 100,000 pixels of frame area.
# Calibrated for typical CCTV resolutions (720p, 1080p).
_DENSITY_HIGH     = 15   # people / 100k pixels → HIGH
_DENSITY_CRITICAL = 30   # people / 100k pixels → CRITICAL
_DENSITY_MEDIUM   = 5    # people / 100k pixels → MEDIUM


class CrowdCounter:
    def __init__(self):
        self.current_count   = 0   # active YOLO-tracked (confirmed + unconfirmed)
        self.tracked_count   = 0   # confirmed tracks only
        self.estimated_count = 0.0 # density-estimator output (float)
        self.density_active  = False  # True when estimator is in dense mode

        self.peak_count      = 0
        self.min_count       = -1
        self.total_count_sum = 0
        self.sample_count    = 0
        self.history         = collections.deque(maxlen=CROWD_HISTORY_LEN)
        self.zone_counts     = {}
        self.density_level   = "LOW"

        self._estimator      = DensityEstimator()
        self._frame_shape    = (720, 1280)  # default; updated on first call

    # ------------------------------------------------------------------

    def update(self, records, zones, now, detections=None, frame_shape=None):
        """
        Update all counters from the current frame's analysis results.

        Args:
            records:      list of per-track record dicts (from analyse_frame).
            zones:        list of zone dicts (for zone-count breakdown).
            now:          timestamp (wall clock or frame_num/fps).
            detections:   raw YOLO detection list [[x1,y1,x2,y2,conf], ...]
                          If None, uses records as a fallback.
            frame_shape:  (H, W) or (H, W, C). Used for pixel-density calc.
        """
        # ── Track-level counts ────────────────────────────────────────
        active_records  = [r for r in records if r["state"] == "active"]
        confirmed_records = [r for r in active_records if r.get("confirmed", False)]

        self.current_count = len(active_records)
        self.tracked_count = len(confirmed_records)

        # ── Density estimation ────────────────────────────────────────
        if frame_shape is not None:
            self._frame_shape = frame_shape[:2]  # (H, W)

        # Use raw detections when available; fall back to records.
        bbox_source = detections if detections is not None else records
        estimated, self.density_active = self._estimator.estimate(
            bbox_source, self._frame_shape
        )
        self.estimated_count = estimated

        # ── Density level ─────────────────────────────────────────────
        self.density_level = self._compute_density_level(
            self.estimated_count, self._frame_shape
        )

        # ── Running statistics (use current_count for continuity) ─────
        self.history.append((now, self.current_count))
        self.peak_count = max(self.peak_count, self.current_count)
        if self.min_count == -1:
            self.min_count = self.current_count
        else:
            self.min_count = min(self.min_count, self.current_count)
        self.total_count_sum += self.current_count
        self.sample_count    += 1

        # ── Zone counts ───────────────────────────────────────────────
        zone_counts = {z["name"]: 0 for z in zones} if zones else {}
        for r in records:
            if r["state"] == "active" and r["zone"] in zone_counts:
                zone_counts[r["zone"]] += 1
        self.zone_counts = zone_counts

        return self.get_stats()

    # ------------------------------------------------------------------

    def get_stats(self):
        return {
            "current_count":   self.current_count,
            "tracked_count":   self.tracked_count,
            "estimated_count": self.estimated_count,
            "density_active":  self.density_active,
            "density_level":   self.density_level,
            "average_count":   round(self.total_count_sum / self.sample_count, 1)
                               if self.sample_count > 0 else 0,
            "peak_count":      self.peak_count,
            "minimum_count":   self.min_count if self.min_count != -1 else 0,
            "history":         list(self.history),
            "zones":           self.zone_counts,
        }

    def reset(self):
        """Clear all accumulated session analytics. Model state is preserved."""
        self.current_count   = 0
        self.tracked_count   = 0
        self.estimated_count = 0.0
        self.density_active  = False
        self.peak_count      = 0
        self.min_count       = -1
        self.total_count_sum = 0
        self.sample_count    = 0
        self.history.clear()
        self.zone_counts     = {}
        self.density_level   = "LOW"
        self._estimator.reset()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_density_level(estimated_count: float, frame_shape: tuple) -> str:
        """
        Compute a density level label from crowd count and frame area.

        Using spatial density (people per area) rather than a raw headcount
        makes the threshold meaningful across different resolutions and zoom
        levels — a count of 15 is dense for a hallway but sparse for a
        stadium wide-shot.
        """
        H, W = frame_shape[0], frame_shape[1]
        area_100k = max(1.0, H * W / 100_000.0)
        ppl_per_100k = estimated_count / area_100k

        if ppl_per_100k >= _DENSITY_CRITICAL:
            return "CRITICAL"
        if ppl_per_100k >= _DENSITY_HIGH:
            return "HIGH"
        if ppl_per_100k >= _DENSITY_MEDIUM:
            return "MEDIUM"
        return "LOW"
