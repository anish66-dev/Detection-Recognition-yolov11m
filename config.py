"""
config.py — application-wide constants and directory setup.

Every tuning knob, colour definition, path, and threshold that was scattered
at the top of app.py lives here so modules can import only what they need
without pulling in Flask or heavy ML dependencies.
"""

import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# IMPORTANT: these live OUTSIDE the project folder, in the OS temp directory.
BASE_TMP_DIR = Path(tempfile.gettempdir()) / "yolov11m_human_detection"
UPLOAD_DIR = BASE_TMP_DIR / "uploads"
OUTPUT_DIR = BASE_TMP_DIR / "outputs"
COORDS_DIR = BASE_TMP_DIR / "coords"
PHOTOS_DIR = BASE_TMP_DIR / "photos"

for d in [UPLOAD_DIR, OUTPUT_DIR, COORDS_DIR, PHOTOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PERSON_CLASS_ID = 0
BOX_COLOR       = (0, 200, 80)     # green   - person, nothing special
LABEL_BG        = (0, 150, 60)
ZONE_BOX_COLOR  = (0, 140, 255)    # orange  - person inside a zone
ZONE_LABEL_BG   = (0, 100, 210)
ALERT_BOX_COLOR = (0, 0, 255)      # red     - blocklisted person
ALERT_LABEL_BG  = (0, 0, 200)
AUTH_BOX_COLOR  = (255, 144, 30)   # blue    - authorized person
AUTH_LABEL_BG   = (200, 100, 0)
INTRUDER_BOX_COLOR = (200, 0, 200) # magenta - unrecognised face
INTRUDER_LABEL_BG  = (150, 0, 150)
ZONE_LINE_COLOR = (0, 220, 255)
TEXT_COLOR      = (255, 255, 255)

CONF_MIN, CONF_MAX = 0.15, 0.75
IOU_MIN, IOU_MAX   = 0.30, 0.75
CALIBRATION_SAMPLES = 15

# Tracking lifecycle
TRACK_MAX_MISSES = 30    # frames track survives unmatched (~1s at 30fps)
TRACK_MIN_HITS   = 3     # detections before a track is confirmed

# Crowd analytics
CROWD_WARNING_THRESHOLD = 10   # count >= this → CROWDED
CROWD_CRITICAL_THRESHOLD = 20  # count >= this → CRITICAL
CROWD_HISTORY_LEN       = 300  # samples kept in the rolling window

# Crowd density estimation — adaptive activation
# When YOLO detects this many or more people, the Gaussian density estimator
# supplements the raw YOLO count to account for occlusion / missed detections.
DENSE_YOLO_THRESH = 8

# Crowd trend — number of history samples to use for INCREASING/STABLE/DECREASING.
# At ~0.5 Hz analytics polling this is roughly 60 seconds of history.
CROWD_TREND_WINDOW = 30

# Crowd direction — minimum track hits before a track contributes a direction vote.
# Prevents single-frame ghosts from skewing the dominant-direction calculation.
CROWD_DIRECTION_MIN_FRAMES = 5

# A face box must have at least this much of its area inside a person box
# before we accept that they belong to the same human.
FACE_CONTAINMENT_MIN = 0.60

# Face recognition is far more expensive than detection. Because identity now
# lives on the track rather than on the frame, it can run on a subset of frames
# without the label flickering.
RECOG_STRIDE_VIDEO = 3
RECOG_STRIDE_LIVE = 4

# Live pipeline tuning
LIVE_DISPLAY_FPS = 25          # render/encode rate; independent of inference rate
LIVE_JPEG_QUALITY = 72
PREVIEW_SSE_FPS = 12           # upload-preview push rate over SSE
