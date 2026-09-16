"""
tracker.py — lightweight multi-object tracker with temporal smoothing.

Why this exists
---------------
Detection alone gives you a fresh, unrelated set of boxes every frame. That
causes three separate problems downstream:

  * boxes jitter and flicker, because nothing links frame N to frame N+1;
  * alerts cannot be de-duplicated, because there is no stable identity to
    debounce against (keying on pixel coordinates re-fires on every movement);
  * the display is pinned to the inference rate, because there is nothing to
    draw between detections.

A tracker fixes all three. Each detection is associated with an existing track,
box coordinates are smoothed with an EMA, and a per-track velocity lets the
renderer *coast* boxes forward between inference ticks so the video stays smooth
even when YOLO only manages 6 fps.

Deliberately dependency-free (no numpy, no scipy) so it stays fast on small
inputs and is trivially unit-testable.

Time base
---------
Every method takes an explicit timestamp instead of calling time.time(). For a
live camera pass wall-clock seconds; for a video file pass ``frame_num / fps``.
Both then behave identically, and video processing stays deterministic.
"""

from collections import deque

# Association / lifecycle defaults
IOU_MATCH_THRESH = 0.30   # below this, a detection cannot continue a track
MAX_MISSES       = 12     # frames a track survives unmatched before deletion
MIN_HITS         = 3      # matches before a track is shown/alerted on
EMA_ALPHA        = 0.6    # box smoothing: higher = more responsive, less smooth
VEL_ALPHA        = 0.4    # velocity smoothing
MAX_COAST_SECS   = 0.5    # never extrapolate further than this into the future

# Identity voting
IDENTITY_WINDOW  = 9      # recognition observations retained per track
IDENTITY_MIN_VOTES = 3    # votes needed before a label is committed


def iou(a, b):
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def containment(inner, outer):
    """
    Fraction of ``inner``'s area that lies inside ``outer``.

    Used to assign a face box to a person box. Plain IoU is useless here — a
    face is tiny relative to a body, so even a perfect assignment scores ~0.05.
    """
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


def merge_duplicate_boxes(boxes, iou_thresh=0.75, contain_thresh=0.85):
    """
    Class-agnostic pass that collapses duplicate person boxes.

    YOLO's built-in NMS runs per detection head and, at a low confidence
    threshold with a high NMS IoU, still lets two or three boxes survive on the
    same body. Those duplicates are what makes a single person appear as both a
    plain "Person" box and a magenta "INTRUDER" box at the same time.

    A box is dropped when it overlaps a higher-confidence box beyond
    ``iou_thresh``, or when it is almost entirely swallowed by one
    (``contain_thresh``) — the nested case IoU alone misses.

    Args:
        boxes: iterable of [x1, y1, x2, y2, conf, ...]; extra trailing fields
            are preserved on the survivors.

    Returns:
        Filtered list, ordered by descending confidence.
    """
    ordered = sorted(boxes, key=lambda b: -float(b[4]))
    kept = []
    for box in ordered:
        redundant = False
        for k in kept:
            if iou(box, k) >= iou_thresh or containment(box, k) >= contain_thresh:
                redundant = True
                break
        if not redundant:
            kept.append(box)
    return kept


class Track:
    """One tracked person, persisting across frames."""

    def __init__(self, track_id, bbox, conf, now):
        self.id = track_id
        self.bbox = [float(v) for v in bbox[:4]]
        self.vel = [0.0, 0.0, 0.0, 0.0]
        self.conf = float(conf)

        self.hits = 1
        self.misses = 0
        self.created_at = now
        self.updated_at = now

        # Identity is a property of the *track*, not of a single frame. Voting
        # over a window stops the label flickering between a real name and
        # INTRUDER when one frame happens to be motion-blurred.
        self._votes = deque(maxlen=IDENTITY_WINDOW)

        # Attributes owned by the analysis stage; independent of each other by
        # design, so a person can be both "in a zone" and "blocklisted".
        self.zone = None
        self.identity_name = None
        self.identity_status = None
        self.identity_sim = 0.0
        self.identity_state = "unseen"  # unseen | known | unknown | uncertain

    @property
    def confirmed(self):
        """True once the track has been seen enough to be trustworthy."""
        return self.hits >= MIN_HITS

    def update(self, bbox, conf, now):
        """Fold a matched detection into the track with EMA smoothing."""
        dt = max(1e-3, now - self.updated_at)
        prev = list(self.bbox)

        for i in range(4):
            self.bbox[i] = EMA_ALPHA * float(bbox[i]) + (1.0 - EMA_ALPHA) * prev[i]
            inst_vel = (self.bbox[i] - prev[i]) / dt
            self.vel[i] = VEL_ALPHA * inst_vel + (1.0 - VEL_ALPHA) * self.vel[i]

        self.conf = float(conf)
        self.hits += 1
        self.misses = 0
        self.updated_at = now

    def mark_missed(self):
        self.misses += 1

    def predict(self, now):
        """
        Box extrapolated to ``now`` — the renderer's view of where this person
        is *right now*, even though the last detection is a few frames old.

        Extrapolation is capped at MAX_COAST_SECS; past that a stale track would
        sail off the frame rather than simply sitting still.
        """
        dt = min(max(0.0, now - self.updated_at), MAX_COAST_SECS)
        if dt <= 0:
            return list(self.bbox)
        return [self.bbox[i] + self.vel[i] * dt for i in range(4)]

    # ── identity voting ──────────────────────────────────────────────────────

    def observe_identity(self, name, status, sim, state):
        """
        Record one recognition observation. ``state`` is 'known', 'unknown' or
        'uncertain'; uncertain observations are stored but never win a vote,
        which is what gives recognition its hysteresis band.
        """
        self._votes.append((name, status, float(sim), state))
        self._commit_identity()

    def _commit_identity(self):
        knowns, unknowns = {}, 0
        best_sim = {}
        for name, status, sim, state in self._votes:
            if state == "known" and name:
                knowns[(name, status)] = knowns.get((name, status), 0) + 1
                best_sim[(name, status)] = max(best_sim.get((name, status), 0.0), sim)
            elif state == "unknown":
                unknowns += 1

        top_key, top_votes = None, 0
        for key, votes in knowns.items():
            if votes > top_votes:
                top_key, top_votes = key, votes

        if top_key and top_votes >= IDENTITY_MIN_VOTES:
            self.identity_name, self.identity_status = top_key
            self.identity_sim = best_sim[top_key]
            self.identity_state = "known"
        elif unknowns >= IDENTITY_MIN_VOTES and top_votes < IDENTITY_MIN_VOTES:
            self.identity_name, self.identity_status = None, None
            self.identity_sim = 0.0
            self.identity_state = "unknown"
        else:
            # Not enough evidence either way — hold fire rather than guess.
            if self.identity_state != "known":
                self.identity_state = "uncertain"

    def to_record(self, now=None):
        """Flat dict consumed by the alert-rule and render stages."""
        box = self.predict(now) if now is not None else list(self.bbox)
        return {
            "track_id": self.id,
            "bbox": [int(round(v)) for v in box],
            "conf": self.conf,
            "zone": self.zone,
            "identity_name": self.identity_name,
            "identity_status": self.identity_status,
            "identity_sim": self.identity_sim,
            "identity_state": self.identity_state,
            "confirmed": self.confirmed,
        }


class Tracker:
    """
    Greedy-IoU multi-object tracker. One instance per video source — never
    share one between the webcam and a file job, or their IDs will collide.
    """

    def __init__(self, iou_thresh=IOU_MATCH_THRESH, max_misses=MAX_MISSES):
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.tracks = []
        self._next_id = 1

    @property
    def next_id(self):
        """ID the next new track will get; ``next_id - 1`` is how many distinct
        people have been seen so far."""
        return self._next_id

    def update(self, detections, now):
        """
        Associate ``detections`` ([x1,y1,x2,y2,conf]) with existing tracks.

        Greedy highest-IoU-first rather than Hungarian: with a handful of people
        per frame the assignments are identical, and it avoids a scipy dependency.

        Returns the list of live tracks (including unconfirmed ones).
        """
        dets = list(detections)

        # Score every (track, detection) pair, then consume greedily so each
        # detection claims at most one track and vice versa.
        pairs = []
        for ti, track in enumerate(self.tracks):
            predicted = track.predict(now)
            for di, det in enumerate(dets):
                score = iou(predicted, det)
                if score >= self.iou_thresh:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        used_tracks, used_dets = set(), set()
        for score, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            self.tracks[ti].update(dets[di][:4], dets[di][4], now)
            used_tracks.add(ti)
            used_dets.add(di)

        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.mark_missed()

        for di, det in enumerate(dets):
            if di not in used_dets:
                self.tracks.append(Track(self._next_id, det[:4], det[4], now))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks

    def active(self, include_unconfirmed=False):
        """Tracks worth drawing: confirmed, and not currently coasting too long."""
        return [
            t for t in self.tracks
            if (t.confirmed or include_unconfirmed) and t.misses <= self.max_misses
        ]

    def reset(self):
        self.tracks = []
        self._next_id = 1
