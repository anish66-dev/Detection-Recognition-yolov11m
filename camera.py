"""
camera.py — camera source abstraction.

Uniform interface over local capture devices, network streams (IP camera apps),
and push frames from the phone browser page.
"""

import cv2
import numpy as np
import platform
import time
import threading


class PushBuffer:
    """
    Holds the most recent frame POSTed to /camera/push by a phone browser.

    Only the newest frame is kept. Queueing them would build exactly the latency
    we are trying to remove — in a live view, a late frame is worthless.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._last_rx = 0.0

    def put_jpeg(self, jpeg_bytes):
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        with self._lock:
            self._frame = frame
            self._seq += 1
            self._last_rx = time.time()
        return True

    def get(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def alive(self):
        """True if the phone has sent something in the last few seconds."""
        return (time.time() - self._last_rx) < 5.0

    def reset(self):
        with self._lock:
            self._frame = None
            self._seq = 0
            self._last_rx = 0.0


_push_buffer = PushBuffer()


def parse_source(spec):
    """
    Interpret a camera spec string.

    Returns (kind, value) where kind is 'device', 'url' or 'push':
      "" / "0" / "1"            -> local capture device index
      "push"                    -> frames arriving from the /phone page
      "http://…", "rtsp://…"    -> network camera (a phone running an IP-camera app)
    """
    spec = (spec or "0").strip()
    if spec.lower() == "push":
        return "push", None
    if spec.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
        return "url", spec
    try:
        return "device", int(spec)
    except ValueError:
        # Anything else is treated as a URL; cv2 will report if it can't open it.
        return "url", spec


class CameraSource:
    """
    Uniform read interface over a local device, a network camera, or push frames.

    Network sources get automatic reconnection: a phone on Wi-Fi drops frames
    when the screen dims or the connection hiccups, and that should reconnect
    rather than kill the session.
    """

    def __init__(self, spec):
        self.kind, self.value = parse_source(spec)
        self.spec = spec
        self.cap = None
        self.error = None
        self._reconnects = 0

    def open(self):
        if self.kind == "push":
            _push_buffer.reset()
            return True

        if self.kind == "device":
            # On Windows CAP_DSHOW avoids a multi-second probe delay.
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0
            self.cap = cv2.VideoCapture(self.value, backend)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            self.cap = cv2.VideoCapture(self.value)

        if not self.cap or not self.cap.isOpened():
            self.error = (
                "Cannot open camera. Is it connected and not in use by another app?"
                if self.kind == "device" else
                f"Cannot open stream: {self.value}. Check the phone's IP and port, "
                "and that both devices are on the same Wi-Fi network."
            )
            return False

        # THE latency fix for live sources. Without this the driver queues
        # frames, so every read returns older footage and the display falls
        # further behind reality the longer the session runs.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.error = None
        return True

    def read(self):
        if self.kind == "push":
            frame = _push_buffer.get()
            if frame is None:
                return False, None
            return True, frame

        if self.cap is None:
            return False, None

        ok, frame = self.cap.read()
        if ok:
            self._reconnects = 0
            return True, frame

        # Network streams get automatic infinite reconnections.
        if self.kind == "url":
            self._reconnects += 1
            if self._reconnects % 10 == 1:
                print(f"[WARN] Stream dropped, reconnect attempt {self._reconnects}...")
            try:
                self.cap.release()
            except Exception:
                pass
            time.sleep(1.0)
            if self.open():
                return self.read()
            return False, None
            
        return False, None

    def dimensions(self, fallback_frame=None):
        if self.kind != "push" and self.cap is not None:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return w, h
        if fallback_frame is not None:
            return fallback_frame.shape[1], fallback_frame.shape[0]
        return 640, 480

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
