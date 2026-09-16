"""
Integrated YOLOv11m Security Platform
- YOLOv11m human detection with multi-object tracking
- InsightFace face recognition, voted per track
- Edge-triggered alerting
- Live sources: local webcam, phone over Wi-Fi (IP camera app or browser push)

Frame handling is split into three stages that used to be one function:

    analyse_frame()  ->  records   (what is true about each person)
    AlertEngine      ->  alerts    (what is worth telling someone about)
    render_frame()   ->  pixels    (what it looks like)

Keeping them separate is what allows a person to be both "in a restricted zone"
and "on the blocklist" at the same time. The previous single if/elif chain made
those mutually exclusive, which silently disabled zone alerts for anyone whose
face had been detected.
"""

from flask import Flask
from flask_cors import CORS

import pipeline
import app_state
from routes.main_routes import bp as main_bp
from routes.process_routes import bp as process_bp
from routes.webcam_routes import bp as webcam_bp
from routes.enroll_routes import bp as enroll_bp
from routes.alert_routes import bp as alert_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(main_bp)
app.register_blueprint(process_bp)
app.register_blueprint(webcam_bp)
app.register_blueprint(enroll_bp)
app.register_blueprint(alert_bp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv11m Security Platform")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--https", action="store_true",
        help="Serve over ad-hoc HTTPS. Required only for the /phone "
             "browser-camera page: browsers grant getUserMedia on secure "
             "origins only, so over plain http:// on a LAN IP the phone's "
             "camera stays blocked. Needs 'pip install pyopenssl'. Not needed "
             "for IP-camera apps.",
    )
    args = parser.parse_args()

    print(f"[INFO] Device: {pipeline.get_device()}")
    print(f"[INFO] Enrolled people: {len(app_state._enrolled_faces_cache)}")
    if not app_state._enrolled_faces_cache:
        print("[WARN] Nobody enrolled — intruder detection stays off until at "
              "least one person is enrolled.")

    scheme = "https" if args.https else "http"
    print(f"[INFO] Open this in your browser: {scheme}://localhost:{args.port}")
    print(f"[INFO] Phone camera page:        {scheme}://<this-machine-ip>:{args.port}/phone")

    ssl_ctx = "adhoc" if args.https else None
    # threaded=True is required for SSE and MJPEG streaming
    app.run(host=args.host, port=args.port, debug=False, threaded=True, ssl_context=ssl_ctx)
