"""
analytics/direction_analyzer.py — dominant crowd-direction estimator.

Uses the EMA-smoothed velocity stored on each Track object (populated by the
existing Tracker.update() → Track.update() call chain in tracker.py).

No additional per-frame computation is needed at the detection level: the
velocities are already maintained as a by-product of tracking.

Algorithm
---------
1. Collect all confirmed tracks that have been seen for at least
   CROWD_DIRECTION_MIN_FRAMES hits.
2. For each such track read (vx, vy) from track.vel[:2] (x-velocity of bbox
   left edge — a good proxy for centre-x velocity because the box size is
   stable while a person walks).
3. Aggregate: sum vx across all tracks → Σvx; sum vy → Σvy.
4. Classify dominant axis: whichever of |Σvx| vs |Σvy| is larger.
5. Within that axis: vx > 0 → RIGHT, vx < 0 → LEFT, vy > 0 → DOWN, vy < 0 → UP.
6. A vote ("LEFT", "RIGHT", "UP", "DOWN", or "NONE") is recorded in a short
   deque.  The mode of the deque is the reported direction — this prevents
   one-frame velocity spikes from flipping the direction.

Minimum motion threshold
------------------------
If the aggregate velocity magnitude is below MIN_VELOCITY_PX_S (pixels/second)
the scene is classified as STATIONARY / NONE.
"""

import math
import collections

from config import CROWD_DIRECTION_MIN_FRAMES

# Pixels-per-second aggregate velocity magnitude below which we call it NONE.
MIN_VELOCITY_PX_S = 3.0

# Size of the majority-vote smoothing window (in analytics update steps).
VOTE_WINDOW = 10


def _classify_vector(vx: float, vy: float) -> str:
    """Map a 2-D velocity vector to a cardinal direction label."""
    mag = math.sqrt(vx * vx + vy * vy)
    if mag < MIN_VELOCITY_PX_S:
        return "NONE"
    if abs(vx) >= abs(vy):
        return "RIGHT" if vx > 0 else "LEFT"
    return "DOWN" if vy > 0 else "UP"


def _mode(iterable):
    """Return the most common element, or 'NONE' if empty."""
    counts: dict = {}
    for v in iterable:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.__getitem__) if counts else "NONE"


class DirectionAnalyzer:
    """
    Aggregate tracker velocities into a dominant crowd-movement direction.

    Keep one instance per video source (mirrors the Tracker / CrowdCounter
    per-source pattern).
    """

    def __init__(self):
        self._votes: collections.deque = collections.deque(maxlen=VOTE_WINDOW)
        self.dominant_direction: str = "NONE"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tracks: list) -> str:
        """
        Compute the dominant direction from the current live tracks.

        Args:
            tracks: the list of Track objects from Tracker.tracks (or
                    Tracker.active()). Both are fine — unconfirmed tracks are
                    filtered out internally.

        Returns:
            The current dominant direction string:
            "NONE" | "LEFT" | "RIGHT" | "UP" | "DOWN"
        """
        sum_vx = 0.0
        sum_vy = 0.0
        n = 0

        for t in tracks:
            # Only use confirmed tracks that have been seen long enough.
            if not t.confirmed or t.hits < CROWD_DIRECTION_MIN_FRAMES:
                continue
            # track.vel = [vx_left, vy_top, vx_right, vy_bottom]
            # Centre-x velocity ≈ mean of left-edge and right-edge x velocities.
            vx = (t.vel[0] + t.vel[2]) * 0.5
            vy = (t.vel[1] + t.vel[3]) * 0.5
            sum_vx += vx
            sum_vy += vy
            n += 1

        if n == 0:
            vote = "NONE"
        else:
            avg_vx = sum_vx / n
            avg_vy = sum_vy / n
            vote = _classify_vector(avg_vx, avg_vy)

        self._votes.append(vote)
        self.dominant_direction = _mode(self._votes)
        return self.dominant_direction

    def reset(self):
        """Clear accumulated direction history."""
        self._votes.clear()
        self.dominant_direction = "NONE"

    def get_status(self) -> dict:
        return {"dominant_direction": self.dominant_direction}
