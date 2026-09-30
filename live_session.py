"""
live_session.py — threaded live camera session.

Three cooperating threads instead of one serial loop:

    capture   — reads as fast as the source allows and keeps only the newest
                frame. Never blocked by inference.
    inference — runs detection + recognition on whatever the newest frame
                is, at whatever rate the hardware sustains.
    encode    — renders and JPEG-encodes at a steady display rate, coasting
                tracked boxes forward between inference results.

The old design ran all three in sequence, so display FPS was pinned to YOLO
FPS and every skipped inference was also a skipped display frame. Splitting
them keeps the video smooth at 25 fps even when detection manages only 6.
"""

import cv2
import time
import threading

import pipeline
import tracker as tracking
import app_state
from alerts import AlertEngine
from camera import CameraSource
from geometry import normalize_zones
from analysis import analyse_frame, render_frame, count_zones, overlay_hud
from config import LIVE_DISPLAY_FPS, LIVE_JPEG_QUALITY, RECOG_STRIDE_LIVE
from app_state import has_enrollment, make_alert_sink
from analytics.crowd_counter import CrowdCounter
from analytics.crowd_monitor import CrowdMonitor
from analytics.direction_analyzer import DirectionAnalyzer


class LiveSession:

    def __init__(self, camera_id, source_spec, conf, iou, thickness, font_scale,
                 zones_input, intruder_detection):
        self.camera_id = camera_id
        self.source = CameraSource(source_spec)
        self.conf = conf
        self.iou = iou
        self.thickness = thickness
        self.font_scale = font_scale
        self.zones_input = zones_input
        self.zones = []
        self.intruder_detection = intruder_detection

        self.running = False
        self.error = None

        self.tracker = tracking.Tracker()
        # The inference thread mutates tracker.tracks while the encode thread
        # iterates it, so every access has to be guarded.
        self._trk_lock = threading.Lock()
        self._frame_ref = [0]
        self.engine = AlertEngine(make_alert_sink(self.camera_id, self._frame_ref))

        self.crowd_counter = CrowdCounter()
        self.crowd_monitor = CrowdMonitor()
        self.direction_analyzer = DirectionAnalyzer()

        self._raw_lock = threading.Lock()
        self._raw = None          # newest captured frame
        self._raw_seq = 0

        self._rec_lock = threading.Lock()
        self._records = []

        self._jpeg_lock = threading.Lock()
        self._jpeg = None
        self._jpeg_seq = 0

        self._threads = []
        self.infer_fps = 0.0
        self.display_fps = 0.0

    # ── thread bodies ────────────────────────────────────────────────────────

    def _capture_loop(self):
        while self.running:
            ok, frame = self.source.read()
            if not ok:
                if self.source.kind in ("push", "url"):
                    # Phone hasn't sent anything yet, or URL is reconnecting. Not fatal.
                    time.sleep(0.05)
                    continue
                self.error = self.source.error or "Camera stopped sending frames."
                break
                
            # If the phone sends 4K, it chokes YOLO and JPEG encoding. Cap at 720p/1080p-ish.
            if frame.shape[1] > 1280:
                scale = 1280 / frame.shape[1]
                frame = cv2.resize(frame, (1280, int(frame.shape[0] * scale)))
                
            with self._raw_lock:
                self._raw = frame
                self._raw_seq += 1
        self.running = False

    def _latest_raw(self):
        with self._raw_lock:
            if self._raw is None:
                return None, 0
            return self._raw.copy(), self._raw_seq

    def _inference_loop(self):
        last_seq = -1
        frame_num = 0
        t_last = time.time()

        while self.running:
            frame, seq = self._latest_raw()
            if frame is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq
            frame_num += 1
            self._frame_ref[0] = frame_num

            if not self.zones:
                h, w = frame.shape[:2]
                self.zones = normalize_zones(self.zones_input, w, h)

            now = time.time()
            do_recog = (frame_num % RECOG_STRIDE_LIVE == 0 or frame_num == 1)
            
            # 1. Heavy ML Inference (Releases GIL, runs entirely outside the tracking lock)
            pre_det = pipeline.detect_persons(frame, self.conf, self.iou)
            pre_faces = pipeline.detect_faces(frame) if do_recog else None

            # 2. Fast tracking & state update (Inside the lock)
            with self._trk_lock:
                records, detections = analyse_frame(
                    frame, self.tracker, self.conf, self.iou, self.zones, now,
                    do_recognition=do_recog,
                    precomputed_detections=pre_det,
                    precomputed_faces=pre_faces
                )
            self.engine.evaluate(records, now,
                                 intruder_detection=self.intruder_detection,
                                 has_enrollment=has_enrollment())

            # Pass raw detections (already computed by analyse_frame — no second
            # YOLO run) and frame shape to the density estimator.
            stats = self.crowd_counter.update(
                records, self.zones, now,
                detections=detections,
                frame_shape=frame.shape,
            )

            # Direction analysis uses the tracker velocity vectors directly.
            direction = self.direction_analyzer.update(self.tracker.tracks)

            events = self.crowd_monitor.evaluate(
                stats["current_count"], stats["zones"], now,
                dominant_direction=direction,
            )
            for event in events:
                # Reuse the alert broadcasting channel for crowd alerts
                if event["type"] == "global_crowd":
                    msg = f"Crowd level changed to {event['new_state']} ({event['count']} people)"
                else:
                    msg = f"Zone {event['zone']} changed to {event['new_state']} ({event['count']} people)"
                
                if event["new_state"] in ("CROWDED", "CRITICAL"):
                    import database
                    alert_dict = database.save_alert(
                        person_name=msg,
                        similarity=0.0,
                        job_id=self.camera_id,
                        frame_num=frame_num,
                        alert_type="crowd"
                    )
                    alert_dict["time_str"] = "LIVE"
                    app_state.push_alert_to_subscribers(alert_dict)
                else:
                    app_state.push_alert_to_subscribers({
                        "id": 0, "alert_type": "crowd", "person_name": msg,
                        "similarity": 0, "timestamp": now, "time_str": "LIVE", "frame_num": frame_num,
                        "job_id": self.camera_id
                    })

            with self._rec_lock:
                self._records = records

            dt = now - t_last
            if dt > 0:
                self.infer_fps = round(0.7 * self.infer_fps + 0.3 * (1.0 / dt), 1)
            t_last = now

    def _encode_loop(self):
        interval = 1.0 / LIVE_DISPLAY_FPS
        t_last = time.time()

        while self.running:
            start = time.time()
            frame, seq = self._latest_raw()
            if frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            # Coast tracked boxes to *now* rather than redrawing them where they
            # were at the last inference tick. This is what removes the visible
            # stutter between detections.
            with self._trk_lock:
                records = [t.to_record(now) for t in self.tracker.active()]

            zone_counts = count_zones(records, self.zones)
            frame = render_frame(frame, records, self.zones, self.thickness, self.font_scale)
            frame = overlay_hud(frame, 0, 0, len(records), zone_counts,
                                extra=f"det {self.infer_fps:.0f}fps")

            ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, LIVE_JPEG_QUALITY])
            if ok:
                with self._jpeg_lock:
                    self._jpeg = jpeg.tobytes()
                    self._jpeg_seq += 1

            dt = now - t_last
            if dt > 0:
                self.display_fps = round(0.7 * self.display_fps + 0.3 * (1.0 / dt), 1)
            t_last = now

            slack = interval - (time.time() - start)
            if slack > 0:
                time.sleep(slack)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        if not self.source.open():
            self.error = self.source.error
            return False

        self.running = True
        for target in (self._capture_loop, self._inference_loop, self._encode_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)
        return True

    def stop(self):
        self.running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        self.source.release()

    def frames(self):
        """
        MJPEG generator that yields only when a *new* frame exists.

        The previous version had no sleep and no new-frame check on its success
        path, so it re-sent the same JPEG in a tight loop — thousands of
        identical frames per second, saturating the socket and burning a core
        that inference needed. That alone accounted for much of the lag.
        """
        last_seq = -1
        idle_deadline = time.time() + 10.0

        while self.running:
            with self._jpeg_lock:
                frame, seq = self._jpeg, self._jpeg_seq

            if frame is not None and seq != last_seq:
                last_seq = seq
                idle_deadline = time.time() + 10.0
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                if time.time() > idle_deadline:
                    break
                time.sleep(0.005)

    def status(self):
        return {
            "running": self.running,
            "error": self.error,
            "has_frame": self._jpeg is not None,
            "source": self.source.spec,
            "source_kind": self.source.kind,
            "infer_fps": self.infer_fps,
            "display_fps": self.display_fps,
            "tracks": len(self._records),
        }
