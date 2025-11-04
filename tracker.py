#!/usr/bin/env python3
# === Imports ===
# Core vision / math / utils
import cv2
import time
import math
import numpy as np

# Web server (control API + MJPEG page)
from flask import Flask, Response, request, render_template_string
from flask_cors import CORS  # allow calls from the WebRTC page

# Concurrency primitives
from threading import Thread, Lock, Condition
import threading

# CLI args / timestamps / small GUI dialogs for file/duration picking
import argparse
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, simpledialog

# WebRTC stack (aiortc) for low-latency viewing
import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
from fractions import Fraction

# If you connect a local HDMI monitor to the Pi, set this True to see a window
SHOW_LOCAL = True  # True only when debugging with a monitor

# === Flask App Initialization ===
# Flask hosts the MJPEG stream (:5000/stream.mjpg), the control UI (:5000/control),
# and simple control endpoints (/command, /select_point, /nudge, /set_target_mode).
app = Flask(__name__)
CORS(app)  # allow the WebRTC page (on :8080) to POST to :5000 endpoints

# === Global shared state (protected by frame_ready) ===
# IMPORTANT CONCURRENCY RULE:
#   Reader threads (camera/file) update ONLY current_frame (+ playback telemetry).
#   The main loop renders overlays → THEN publishes output_frame.
#   WebRTC & MJPEG read output_frame. This avoids flicker.
output_frame = None               # Final frame after overlays → served to MJPEG / WebRTC
current_frame = None              # Raw latest frame from camera/file (no overlays)
command_from_remote = None        # One-letter command from web UI: 'r','s','q'
bbox = None                       # Current CSRT tracking box (x, y, w, h)
tracking = False                  # Tracking on/off flag
tracker = None                    # OpenCV tracker object
bMoovingTgt = False               # Target type (False=fixed/large box, True=moving/small box)

# --- Bounding-box size clamp (prevents explosion when target gets close) ---
MAX_BB_WIDTH = 120
MAX_BB_HEIGHT = 120

# Playback controls (used only in playback mode)
playback_rate = 1.0     # 0.25, 0.5, 1.0, 2.0, 4.0 ...
seek_to_msec = None     # when set (int), reader seeks to this timestamp (ms) ASAP
playback_ctrl_lock = Lock()

# Playback telemetry (reader updates; main loop reads to sync trackbars)
playback_duration_ms = 0.0
playback_pos_ms = 0.0

# Local-UI (OpenCV) trackbar flags (playback-only)
_trackbar_ready = False
_suppress_trackbar_cb = False

# Thread sync for frame sharing between producer (reader) and consumers (MJPEG/WebRTC)
frame_lock = Lock()
frame_ready = Condition(frame_lock)

# ============ Camera/File Reader (unified) ============
# It only sets current_frame & playback telemetry and notifies frame_ready.
cap = None
picam2 = None
_reader_thread = None
_stop_reader = threading.Event()

def _reader_playback(path, loop=False):
    """
    Video file reader that respects playback_rate and seek_to_msec globals.
    Updates playback_pos_ms and playback_duration_ms for UI sync.
    """
    global cap, current_frame, playback_rate, seek_to_msec
    global playback_duration_ms, playback_pos_ms

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    base_delay = 1.0 / fps
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    playback_duration_ms = (total_frames / fps * 1000.0) if total_frames > 0 else 0.0

    print(f"[INFO] Playback started ({fps:.1f} fps) duration≈{playback_duration_ms/1000:.2f}s")

    while not _stop_reader.is_set():
        with playback_ctrl_lock:
            if seek_to_msec is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(seek_to_msec))
                seek_to_msec = None
            rate = max(0.1, float(playback_rate))

        ok, frame = cap.read()
        if not ok:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[INFO] Playback ended")
                break

        playback_pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        with frame_ready:
            current_frame = frame
            frame_ready.notify_all()

        # Adjust pacing
        if rate > 1.0:
            # skip frames for >1×
            frames_to_skip = int(rate) - 1
            for _ in range(frames_to_skip):
                cap.grab()
            time.sleep(base_delay * 0.25)
        else:
            time.sleep(base_delay / rate)

def _reader_live_picam():
    """
    Continuously read frames from PiCamera2 and publish into current_frame.
    Avoid any overlay work here; overlays happen in the main loop.
    """
    global current_frame, picam2
    print("[INFO] Live reader started (PiCamera2)")
    while not _stop_reader.is_set():
        frame = picam2.capture_array()
        if frame is None:
            continue
        with frame_ready:
            current_frame = frame  # raw frame only
            frame_ready.notify_all()

# === HTML Page for /control (MJPEG) ===
# Lightweight page to 1) watch MJPEG stream and 2) send commands/select points.
# For lowest latency viewing use the WebRTC page served on :8080.
HTML_PAGE = """
<!doctype html>
<html>
  <head>
    <title>Tracker Control</title>
    <style>
      body { font-family: sans-serif; text-align: center; background-color: #f0f0f0; }
      h1 { margin-top: 20px; }
      #video { margin-top: 20px; border: 4px solid #333; width: 640px; height: 480px; cursor: crosshair; }
      button { padding: 10px 20px; margin: 10px; font-size: 16px; background-color: #4CAF50;
               color: white; border: none; border-radius: 5px; cursor: pointer; }
      button.quit { background-color: #f44336; }
      button:hover { opacity: 0.8; }
      .note { margin-top: 10px; color: #555; font-size: 14px; }
    </style>
  </head>
  <body>
    <h1>Tracker Remote Control</h1>

    <!-- Persistent MJPEG stream (higher latency than WebRTC, but trivial to view) -->
    <img id="video" src="/stream.mjpg" width="640" height="480" />

    <div>
      <button onclick="sendCommand('r')">Reset Tracker</button>
      <button onclick="sendCommand('s')">Stop Tracking</button>
      <button class="quit" onclick="sendCommand('q')">Quit</button>
    </div>

    <div class="note">
      Tip: For lower latency video, open <code>http://HOST:8080/</code> (WebRTC) and click "Start".
      Clicking video here or there selects the target.
    </div>

    <script>
      const img = document.getElementById('video');

      // Send click coordinates to the server (selects a new bbox around that point)
      img.addEventListener('click', (event) => {
        const rect = img.getBoundingClientRect();
        const x = Math.floor(event.clientX - rect.left);
        const y = Math.floor(event.clientY - rect.top);
        fetch('/select_point', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: `x=${x}&y=${y}`
        });
      });

      // Send control commands (reset, stop, quit) to the Flask app
      function sendCommand(cmd) {
        fetch('/command', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: 'cmd=' + cmd
        });
      }
    </script>
  </body>
</html>
"""

# === MAVLink Setup ===
# If connected, we send normalized pitch/yaw as RC overrides.
# If not present, we just print debug values.
mavlink_enabled = False
try:
    from pymavlink import mavutil
    connection = mavutil.mavlink_connection('/dev/serial0', baud=57600)
    connection.wait_heartbeat(timeout=5)
    connection.arducopter_arm()  # Attempt to arm the autopilot (safe to remove if undesired)
    connection.mav.set_mode_send(
        connection.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        0
    )
    mavlink_enabled = True
    print("[MAVLink] Connected and heartbeat received.")
except Exception as e:
    print(f"[WARNING] MAVLink not connected: {e}")
    connection = None

def send_attitude(pitch, yaw):
    """
    Push pitch/yaw commands via MAVLink (RC override).
    When MAVLink isn't connected, prints debug for tuning.
    pitch/yaw are in radians; mapping to PWM is heuristic and may need tuning.
    """
    if not mavlink_enabled:
        print(f"[DEBUG] Pitch: {math.degrees(pitch):.2f}deg, Yaw: {math.degrees(yaw):.2f}deg")
        return
    try:
        # Map small angles to PWM deltas (±~500 around 1500). Adjust to your vehicle.
        pitch_pwm = int(1500 + pitch * 500)
        yaw_pwm   = int(1500 + yaw   * 500)
        connection.mav.rc_channels_override_send(
            connection.target_system, connection.target_component,
            pitch_pwm, yaw_pwm, 0, 0, 0, 0, 0, 0
        )
        print(f"[MAVLink] Sent pitch PWM: {pitch_pwm}, yaw PWM: {yaw_pwm}")
    except Exception as e:
        print(f"[MAVLink] Failed to send attitude: {e}")

# === Command-line arguments setup ===
# mode:
#   live     → use PiCamera2
#   record   → record live camera to AVI for given duration
#   playback → play from a file (optionally loop)
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['live', 'record', 'playback'], default='live')
parser.add_argument('--video', help='Path to video file for playback')
parser.add_argument('--duration', type=int, help='Duration to record (seconds)')
parser.add_argument('--loop', action='store_true', help='Loop video in playback mode')
args = parser.parse_args()

# If recording and duration not provided on CLI, ask with a small dialog (Tk)
if args.mode == 'record' and not args.duration:
    root = tk.Tk()
    root.withdraw()
    duration = simpledialog.askinteger("Recording Duration", "How many seconds to record?",
                                       minvalue=1, maxvalue=3600)
    if not duration:
        print("[ERROR] No duration selected. Exiting.")
        exit(1)
    args.duration = duration
    print(f"[INFO] Recording duration set to {args.duration} seconds")

# === Input Setup: file playback or live camera (each spawns a reader thread) ===
if args.mode == 'playback':
    if not args.video:
        # Optional file-pick dialog for convenience when not given on CLI
        root = tk.Tk()
        root.withdraw()
        args.video = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.avi *.mp4 *.mov *.mkv"), ("All files", "*.*")]
        )
        if not args.video:
            print("[ERROR] No file selected. Exiting...")
            exit(1)
    print(f"[INFO] Playback mode from file: {args.video} (loop={args.loop})")
    _stop_reader.clear()
    _reader_thread = Thread(target=_reader_playback, args=(args.video, args.loop), daemon=True)
    _reader_thread.start()
else:
    # Live Pi camera (tuned for 640x480 @ ~30 FPS to keep latency & CPU in check)
    from picamera2 import Picamera2
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"  # gets you BGR after capture_array
    picam2.configure("preview")
    picam2.set_controls({"FrameRate": 30})  # keep latency down
    picam2.start()
    time.sleep(0.3)
    _stop_reader.clear()
    _reader_thread = Thread(target=_reader_live_picam, daemon=True)
    _reader_thread.start()

# === CSRT tracker params/tuning ===
# We bias parameters depending on moving vs fixed target:
#  - moving: more scales, faster scale learning, wider search
#  - fixed:  fewer scales, slower adaptation (reduces jitter)
def make_csrt_params(moving: bool):
    p = cv2.TrackerCSRT_Params()

    # Appearance features (robustness vs CPU)
    for name, val in dict(
        use_hog=True,
        use_color_names=True,
        use_gray=False,
        use_rgb=True,
        use_channel_weights=True,
        use_segmentation=False,     # turn True if supported and you want more robustness
        window_function="hann",
        kaiser_alpha=3.2,
        hog_clip=2.0,
        histogram_bins=16,
        background_ratio=2          # smaller = tighter background sampling
    ).items():
        if hasattr(p, name): setattr(p, name, val)

    if moving:
        suggested = dict(
            number_of_scales=55,
            scale_step=1.02,
            scale_lr=0.65,
            scale_sigma_factor=0.30,
            scale_model_max_area=1024
        )
    else:
        suggested = dict(
            number_of_scales=33,
            scale_step=1.03,
            scale_lr=0.15,
            scale_sigma_factor=0.25,
            scale_model_max_area=512
        )
    for name, val in suggested.items():
        if hasattr(p, name): setattr(p, name, val)

    # Discrimination vs speed
    if hasattr(p, "admm_iterations"):
        p.admm_iterations = 6 if moving else 9
    if hasattr(p, "template_size"):
        p.template_size = 200 if moving else 160
    if hasattr(p, "filter_lr"):
        p.filter_lr = 0.25 if moving else 0.08
    return p

def create_csrt_tracker(moving: bool):
    """
    Create a CSRT tracker with tuned params. Some OpenCV builds don't accept
    params in the constructor, so we gracefully fall back.
    """
    try:
        params = make_csrt_params(moving)
        return cv2.TrackerCSRT_create(params)
    except TypeError:
        t = cv2.TrackerCSRT_create()
        if hasattr(t, "set"):
            params = make_csrt_params(moving)
            for k, v in params.__dict__.items():
                try:
                    getattr(t, k)  # probe
                    setattr(t, k, v)
                except Exception:
                    pass
        return t

# === Video Recording Setup (mode=record) ===
writer = None
record_start_time = None
if args.mode == 'record':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_filename = f"RecordingsMahat/recording_{timestamp}.avi"
    writer = cv2.VideoWriter(video_filename, fourcc, 60.0, (640, 480))  # 60 fps container OK
    record_start_time = time.time()
    print(f"[INFO] Recording to {video_filename}")

# === Local debug window (optional) ===
if SHOW_LOCAL:
    cv2.namedWindow("Tracker")

# ---- Trackbar callbacks (playback-only) ----
def _on_seek_trackbar(pos):
    """Trackbar 'position' callback: pos is 0..1000 → absolute ms in file."""
    global seek_to_msec, _suppress_trackbar_cb
    if _suppress_trackbar_cb or playback_duration_ms <= 0:
        return
    frac = pos / 1000.0
    with playback_ctrl_lock:
        seek_to_msec = int(frac * playback_duration_ms)

def _on_rate_trackbar(val):
    """Trackbar 'rate x0.01' callback: val is 10..800 → 0.10x..8.00x."""
    global playback_rate, _suppress_trackbar_cb
    if _suppress_trackbar_cb:
        return
    r = max(0.1, min(8.0, val / 100.0))
    with playback_ctrl_lock:
        playback_rate = r

# Create playback trackbars once window exists (playback-only)
if SHOW_LOCAL and args.mode == 'playback':
    cv2.createTrackbar('position', 'Tracker', 0, 1000, _on_seek_trackbar)
    cv2.createTrackbar('rate x0.01', 'Tracker', int(100), 800, _on_rate_trackbar)  # start 1.00x
    _trackbar_ready = True

def draw_rectangle(event, x, y, flags, param):
    """
    Local GUI selection: left-click initializes a new tracker centered at (x,y).
    Box size depends on target mode:
      moving  → 30x30
      fixed   → 80x80
    """
    global bbox, tracking, tracker, current_frame, bMoovingTgt
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        frame_for_init = current_frame.copy()
        if bMoovingTgt:
            w, h = 30, 30
        else:
            w, h = 80, 80
        bbox = (max(0, x - w//2), max(0, y - h//2), w, h)
        tracker = create_csrt_tracker(bMoovingTgt)
        tracker.init(frame_for_init, bbox)
        tracking = True
        print(f"[INFO] Tracker initialized ({'MOVING' if bMoovingTgt else 'FIXED'}) at ({x}, {y}), box {w}x{h}")

if SHOW_LOCAL:
    cv2.setMouseCallback("Tracker", draw_rectangle)

# JPEG quality for MJPEG streaming (lower → faster encode; WebRTC ignores this)
JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 60]

# === MJPEG generator ===
def generate_stream():
    """
    Yields a multipart/x-mixed-replace JPEG stream from output_frame.
    This is the higher-latency legacy view; useful as a fallback.
    """
    global output_frame
    while True:
        with frame_ready:
            if output_frame is None:
                frame_ready.wait(timeout=0.05)
            frame = None if output_frame is None else output_frame.copy()
        if frame is None:
            continue
        ok, buffer = cv2.imencode('.jpg', frame, JPEG_PARAMS)
        if not ok:
            continue
        chunk = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + chunk + b'\r\n')

# === Flask routes (control + MJPEG) ===
@app.route('/stream.mjpg')
def mjpeg():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # simple alias to MJPEG stream
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control')
def control():
    # You can enable in playback too; currently blocked by choice.
    if args.mode == 'playback':
        return "Control not available in playback mode", 403
    return render_template_string(HTML_PAGE)

@app.route('/command', methods=['POST'])
def command():
    """
    Accepts 'r' (reset), 's' (stop), 'q' (quit). One-character commands.
    """
    global command_from_remote
    cmd = request.form.get("cmd")
    if cmd in ['r', 's', 'q']:
        command_from_remote = cmd
        return "OK", 200
    return "Invalid", 400

@app.route('/select_point', methods=['POST'])
def select_point():
    """
    Initialize a new tracker around a clicked point from the web UI.
    Box size depends on target mode (moving/fixed).
    """
    global bbox, tracking, tracker, current_frame, bMoovingTgt
    try:
        x = int(request.form.get("x"))
        y = int(request.form.get("y"))
        if current_frame is None:
            return "No frame", 400
        if bMoovingTgt:
            w, h = 30, 30
        else:
            w, h = 80, 80

        tracking = False
        bbox = None
        tracker = None

        bbox = (max(0, x - w//2), max(0, y - h//2), w, h)
        tracker = create_csrt_tracker(bMoovingTgt)
        tracker.init(current_frame, bbox)
        tracking = True
        print(f"[INFO] Tracker initialized ({'MOVING' if bMoovingTgt else 'FIXED'}) from remote click at ({x}, {y}), box {w}x{h}")
        return "OK", 200
    except Exception as e:
        return f"Error: {e}", 400

@app.route('/nudge', methods=['POST'])
def nudge():
    """
    Nudge current bbox by (dx,dy) pixels and re-init the tracker on the current frame.
    This gives precise keyboard/GUI micro-adjustments without reselecting.
    """
    global bbox, tracking, tracker, current_frame
    try:
        dx = int(request.form.get("dx", 0))
        dy = int(request.form.get("dy", 0))
        if current_frame is None:
            return "No frame", 400

        h, w = current_frame.shape[:2]

        # If no bbox yet, create a small centered box to nudge from
        if bbox is None:
            bw = bh = 60
            cx = w // 2
            cy = h // 2
        else:
            x, y, bw, bh = map(int, bbox)
            cx = x + bw // 2
            cy = y + bh // 2

        # Apply and clamp
        cx = max(bw // 2, min(w - bw // 2, cx + dx))
        cy = max(bh // 2, min(h - bh // 2, cy + dy))
        x_new = int(max(0, min(w - bw, cx - bw // 2)))
        y_new = int(max(0, min(h - bh, cy - bh // 2)))
        bbox_new = (x_new, y_new, bw, bh)

        # Re-init tracker on the *current* frame for immediate feedback
        tracking = False
        tracker = None
        tracker_local = create_csrt_tracker(bMoovingTgt)
        tracker_local.init(current_frame, bbox_new)
        tracker = tracker_local
        bbox = bbox_new
        tracking = True
        print(f"[INFO] Nudged bbox by ({dx},{dy}) -> {bbox}")
        return "OK", 200
    except Exception as e:
        print(f"[ERROR] Nudge failed: {e}")
        return f"Error: {e}", 400

@app.route('/set_target_mode', methods=['POST'])
def set_target_mode():
    """
    Toggle between fixed vs moving target.
    The UI button posts bMoovingTgt=1/0 to this endpoint.
    """
    global bMoovingTgt
    val = request.form.get("bMoovingTgt", "0")
    bMoovingTgt = (val == "1")
    print(f"[INFO] Target mode set to: {'MOVING' if bMoovingTgt else 'FIXED'}")
    return "OK", 200

# === Launch Flask in separate thread ===
# NOTE: we run Flask (control + MJPEG) on :5000 in a background thread.
flask_thread = Thread(target=lambda: app.run(host="0.0.0.0", port=5000, threaded=True))
flask_thread.daemon = True
flask_thread.start()

# === WebRTC HTML (vertical rail) ===
# Low-latency viewer served by aiohttp on :8080 (separate from Flask).
WEBRTC_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>WebRTC Viewer</title>
    <style>
      :root { --w: 640px; --h: 480px; }
      * { box-sizing: border-box; }
      body { font-family: sans-serif; background:#f0f0f0; margin:0; padding:24px; }
      h1 { margin:0 0 14px 0; text-align:center; }
      .layout { display:grid; grid-template-columns: minmax(var(--w), 1fr) 220px; gap:18px; align-items:start; justify-content:center; max-width:1200px; margin:0 auto; }
      .video-panel { display:flex; flex-direction:column; align-items:center; }
      #video { width: var(--w); height: var(--h); background:#000; border:4px solid #333; cursor:crosshair; }
      #status { margin-top:8px; color:#444; font-size:14px; min-height:1.2em; }
      .hint { margin-top:6px; color:#666; font-size:12px; text-align:center; }
      .hint kbd { background:#eee; border:1px solid #ccc; border-bottom-width:2px; padding:2px 6px; border-radius:4px; }
      .rail { display:flex; flex-direction:column; align-items:stretch; gap:10px; }
      .btn { padding:10px 14px; border:0; border-radius:8px; font-size:15px; cursor:pointer; color:white; box-shadow:0 2px 6px rgba(0,0,0,.15); }
      .btn.secondary { background:#607D8B; }
      .btn.go { background:#4CAF50; }
      .btn.stop { background:#4CAF50; opacity:.9; }
      .btn.quit { background:#f44336; }
      .btn.toggle { background:#2196F3; }
      .btn:active { transform: translateY(1px); }
      .dpad { margin-top:8px; display:grid; grid-template-columns:56px 56px 56px; grid-template-rows:56px 56px 56px; gap:8px; justify-content:center; }
      .dpad button { width:56px; height:56px; background:#9E9E9E; border:0; border-radius:10px; color:#fff; font-size:18px; cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,.15); }
      .dpad .blank { visibility:hidden; }
      @media (max-width:980px){ .layout{ grid-template-columns:1fr;} .rail{ flex-direction:row; flex-wrap:wrap; justify-content:center;} }
    </style>
  </head>
  <body>
    <h1>WebRTC Live Video</h1>
    <div class="layout">
      <div class="video-panel">
        <video id="video" autoplay playsinline></video>
        <div id="status"></div>
        <div class="hint">
          Click the video to select a target. Use arrow keys to nudge by 5px
          (<kbd>Shift</kbd>=10px, <kbd>Alt</kbd>=1px). <kbd>M</kbd> toggles Fixed/Moving.
          R/S/Q for Reset/Stop/Quit.
        </div>
      </div>
      <div class="rail">
        <button id="startBtn" class="btn secondary">Start</button>
        <button class="btn go"   onclick="sendCmd('r')">Reset (R)</button>
        <button class="btn stop" onclick="sendCmd('s')">Stop (S)</button>
        <button class="btn quit" onclick="sendCmd('q')">Quit (Q)</button>
        <button id="tgtBtn" class="btn toggle" onclick="toggleTarget()">Target: Fixed (M)</button>
        <div class="dpad">
          <span class="blank"></span>
          <button title="Up"    onclick="nudge(0,-5)">▲</button>
          <span class="blank"></span>
          <button title="Left"  onclick="nudge(-5,0)">◀</button>
          <span class="blank"></span>
          <button title="Right" onclick="nudge(5,0)">▶</button>
          <span class="blank"></span>
          <button title="Down"  onclick="nudge(0,5)">▼</button>
          <span class="blank"></span>
        </div>
      </div>
    </div>
    <script>
      // Minimal single-page app for low-latency viewing and control
      const video = document.getElementById('video');
      const startBtn = document.getElementById('startBtn');
      const statusEl = document.getElementById('status');
      const tgtBtn = document.getElementById('tgtBtn');

      function setStatus(msg){ statusEl.textContent = msg; }

      // Control API to Flask (:5000)
      async function sendCmd(cmd){
        try{
          const resp = await fetch('http://' + location.hostname + ':5000/command', {
            method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
            body:'cmd=' + encodeURIComponent(cmd)
          });
          setStatus(resp.ok ? 'Sent command: ' + cmd : 'Command failed: ' + cmd);
        }catch(e){ setStatus('Command error: ' + e); }
      }

      async function nudge(dx, dy){
        try{
          const resp = await fetch('http://' + location.hostname + ':5000/nudge', {
            method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
            body:`dx=${dx}&dy=${dy}`
          });
          setStatus(resp.ok ? `Nudged (${dx}, ${dy})` : `Nudge failed (${dx}, ${dy})`);
        }catch(e){ setStatus('Nudge error: ' + e); }
      }

      // Click-to-select absolute point (server creates a bbox around it)
      video.addEventListener('click', async (e)=>{
        const r = video.getBoundingClientRect();
        const x = Math.floor(e.clientX - r.left);
        const y = Math.floor(e.clientY - r.top);
        try{
          const resp = await fetch('http://' + location.hostname + ':5000/select_point', {
            method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
            body:`x=${x}&y=${y}`
          });
          setStatus(resp.ok ? `Selected (${x}, ${y})` : `Select failed (${x}, ${y})`);
        }catch(e){ setStatus('Select error: ' + e); }
      });

      // Toggle target mode (fixed/moving)
      let movingTgt = false; // default: Fixed
      async function toggleTarget(){
        movingTgt = !movingTgt;
        tgtBtn.textContent = 'Target: ' + (movingTgt ? 'Moving' : 'Fixed') + ' (M)';
        try{
          const resp = await fetch('http://' + location.hostname + ':5000/set_target_mode', {
            method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
            body:'bMoovingTgt=' + (movingTgt ? 1 : 0)
          });
          setStatus(resp.ok ? ('Mode: ' + (movingTgt ? 'MOVING' : 'FIXED')) : 'Mode set failed');
        }catch(e){ setStatus('Mode error: ' + e); }
      }

      // Keyboard shortcuts
      window.addEventListener('keydown', (e)=>{
        const step = e.shiftKey ? 10 : (e.altKey ? 1 : 5);
        if (e.key === 'ArrowRight'){ nudge(step, 0); e.preventDefault(); }
        if (e.key === 'ArrowLeft'){ nudge(-step, 0); e.preventDefault(); }
        if (e.key === 'ArrowUp'){ nudge(0, -step); e.preventDefault(); }
        if (e.key === 'ArrowDown'){ nudge(0, step); e.preventDefault(); }
        if (e.key === 'm' || e.key === 'M') toggleTarget();
        if (e.key === 'r' || e.key === 'R') sendCmd('r');
        if (e.key === 's' || e.key === 'S') sendCmd('s');
        if (e.key === 'q' || e.key === 'Q') sendCmd('q');
      });

      // WebRTC handshake with aiohttp server (:8080)
      async function start(){
        try{
          const pc = new RTCPeerConnection();
          pc.ontrack = (ev)=>{ video.srcObject = ev.streams[0]; setStatus('Streaming…'); };
          const offer = await pc.createOffer({ offerToReceiveVideo: true });
          await pc.setLocalDescription(offer);
          const resp = await fetch('/offer', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
          });
          const answer = await resp.json();
          await pc.setRemoteDescription(answer);
          setStatus('Connected via WebRTC');
        }catch(e){ setStatus('WebRTC error: ' + e); }
      }
      startBtn.addEventListener('click', start);
    </script>
  </body>
</html>
"""

# === WebRTC track over global output_frame ===
class GlobalFrameTrack(MediaStreamTrack):
    """
    A media track that pulls from output_frame at a fixed target FPS.
    NOTE: WebRTC is pull-based; recv() is awaited each frame.
    """
    kind = "video"

    def __init__(self, target_fps=30):
        super().__init__()
        self.time_base = Fraction(1, 90000)  # 90 kHz clock (standard for video pts)
        self.frame_interval = 1.0 / target_fps
        self._last_ts = 0.0  # last send time (seconds, wall clock)

    async def recv(self) -> VideoFrame:
        global output_frame
        # Pace frames roughly to target_fps
        now = time.time()  # TIP: switch to time.monotonic() if you want immunity to clock jumps
        if self._last_ts:
            to_sleep = self.frame_interval - (now - self._last_ts)
            if to_sleep > 0:
                await asyncio.sleep(to_sleep)
        self._last_ts = time.time()

        # Grab latest rendered frame
        with frame_ready:
            if output_frame is None:
                frame_ready.wait(timeout=0.05)
            frame = None if output_frame is None else output_frame.copy()

        if frame is None:
            # No frame available → signal upstream to try again
            raise MediaStreamError("No frame available")

        # Construct an AV VideoFrame (BGR24) with proper timestamps
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts = int(self._last_ts * 90000)
        vf.time_base = self.time_base
        return vf

# Active peer connections (for cleanup on shutdown)
pcs = set()

async def webrtc_index(request):
    # Serve the WebRTC HTML page
    return web.Response(text=WEBRTC_HTML, content_type="text/html")

async def webrtc_offer(request):
    """
    Standard WebRTC offer/answer exchange:
      - Create RTCPeerConnection
      - Attach our GlobalFrameTrack
      - Set remote offer, create and return local answer
    """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            pcs.discard(pc)

    pc.addTrack(GlobalFrameTrack(target_fps=30))
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

async def on_webrtc_shutdown(app):
    # Close any lingering peer connections on server stop
    await asyncio.gather(*[pc.close() for pc in pcs])

def start_webrtc_server():
    """
    aiohttp-based WebRTC signaling server running in a background thread on :8080.
    handle_signals=False is required because signals can only be registered in main thread.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app_webrtc = web.Application()
    app_webrtc.on_shutdown.append(on_webrtc_shutdown)
    app_webrtc.router.add_get("/", webrtc_index)
    app_webrtc.router.add_post("/offer", webrtc_offer)
    web.run_app(app_webrtc, host="0.0.0.0", port=8080, handle_signals=False)

# Launch the WebRTC server in a background thread
webrtc_thread = Thread(target=start_webrtc_server, daemon=True)
webrtc_thread.start()

# === Main Loop (render & publish) ===
# Wait for frames from the reader thread, run tracking, draw overlays, publish to output_frame.
while True:
    # Wait for a new current_frame from reader
    with frame_ready:
        if current_frame is None:
            frame_ready.wait(timeout=0.02)
        frame = None if current_frame is None else current_frame.copy()

    if frame is None:
        # Still allow key handling if local window is open
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    h, w = frame.shape[:2]

    # Handle one-letter commands from the web control
    if command_from_remote == 'r':
        tracking = False
        bbox = None
        tracker = None
        print("[INFO] Tracker reset from remote")
        command_from_remote = None
    elif command_from_remote == 's':
        tracking = False
        command_from_remote = None
    elif command_from_remote == 'q':
        print("[INFO] Quit requested from remote")
        break

    # === Tracker update & overlay ===
    if tracking and tracker is not None:
        try:
            success, bbox = tracker.update(frame)
            if success:
                x, y, bw, bh = map(int, bbox)
                cx, cy = x + bw // 2, y + bh // 2

                # === Limit bounding box size (prevent growing beyond max) ===
                if bw > MAX_BB_WIDTH or bh > MAX_BB_HEIGHT:
                    scale_w = MAX_BB_WIDTH / bw
                    scale_h = MAX_BB_HEIGHT / bh
                    scale = min(scale_w, scale_h)
                    new_bw = int(bw * scale)
                    new_bh = int(bh * scale)
                    x = max(0, cx - new_bw // 2)
                    y = max(0, cy - new_bh // 2)
                    bw, bh = new_bw, new_bh
                    bbox = (x, y, bw, bh)
                    tracker = create_csrt_tracker(bMoovingTgt)
                    tracker.init(frame, bbox)
                    print(f"[INFO] BB limited to {bw}x{bh} (max {MAX_BB_WIDTH}x{MAX_BB_HEIGHT})")

                # Center offsets for attitude mapping
                dx = cx - w // 2
                dy = cy - h // 2
                norm_dx = dx / w
                norm_dy = dy / h

                # Simple FOV→angle mapping (heuristic; tune to your camera FOV)
                yaw   =  norm_dx * math.radians(60)   # assume ~60° HFOV
                pitch = -norm_dy * math.radians(45)   # assume ~45° VFOV
                send_attitude(pitch, yaw)

                # Choose colors by target mode (just for quick visual cue)
                if bMoovingTgt:
                    box_color = (0, 0, 255)   # Red for moving target
                    cross_color = (0, 0, 255)
                else:
                    box_color = (255, 0, 0)   # Blue for fixed target
                    cross_color = (255, 0, 0)

                # Draw bbox & crosshair
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 2)
                cv2.line(frame, (cx - 10, cy), (cx + 10, cy), cross_color, 1)
                cv2.line(frame, (cx, cy - 10), (cx, cy + 10), cross_color, 1)
            else:
                # Tracking lost → keep frame visible and annotate
                cv2.putText(frame, "Tracking lost", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            # Fail-safe: drop tracking on any exception (e.g., tracker internal error)
            print(f"[ERROR] Tracker update failed: {e}")
            tracking = False

    # === Optional: write to file if in record mode ===
    if args.mode == 'record' and writer is not None:
        writer.write(frame)
        if args.duration and (time.time() - record_start_time >= args.duration):
            print("[INFO] Reached recording duration, exiting.")
            break

    # Timestamp overlay (useful to eyeball latency across paths)
    cv2.putText(frame, datetime.now().strftime('%H:%M:%S.%f')[:-3],
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    # --- Local UI: keep trackbars synced (playback mode only) ---
    if SHOW_LOCAL and args.mode == 'playback' and _trackbar_ready and playback_duration_ms > 0:
        try:
            _suppress_trackbar_cb = True
            pos_frac = max(0.0, min(1.0, playback_pos_ms / playback_duration_ms))
            cv2.setTrackbarPos('position', 'Tracker', int(pos_frac * 1000))
            with playback_ctrl_lock:
                cv2.setTrackbarPos('rate x0.01', 'Tracker', int(playback_rate * 100.0))
        finally:
            _suppress_trackbar_cb = False

    # === Publish the final frame ===
    with frame_ready:
        output_frame = frame.copy()
        frame_ready.notify_all()

    # Local debug preview window (safe even if no window; waitKey returns -1)
    if SHOW_LOCAL:
        cv2.imshow("Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        tracking = False
        bbox = None
        tracker = None
        print("[INFO] Tracker reset from Pi")

    # --- Playback keyboard controls (still available) ---
    elif args.mode == 'playback':
        if key == ord('f'):   # faster
            with playback_ctrl_lock:
                playback_rate = min(playback_rate * 2.0, 8.0)
            print(f"[PLAYBACK] Speed {playback_rate:.1f}×")

        elif key == ord('s'): # slower
            with playback_ctrl_lock:
                playback_rate = max(playback_rate / 2.0, 0.25)
            print(f"[PLAYBACK] Speed {playback_rate:.2f}×")

        elif key == ord('1'): # normal speed
            with playback_ctrl_lock:
                playback_rate = 1.0
            print("[PLAYBACK] Speed reset to 1×")

        elif key == ord('j'): # rewind ~5 s
            if cap is not None:
                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                with playback_ctrl_lock:
                    seek_to_msec = max(0, pos - 5000)
                print(f"[PLAYBACK] Seek −5 s")

        elif key == ord('l'): # forward ~5 s
            if cap is not None:
                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                with playback_ctrl_lock:
                    seek_to_msec = pos + 5000
                print(f"[PLAYBACK] Seek +5 s")

# === Cleanup ===
cv2.destroyAllWindows()
_stop_reader.set()
if _reader_thread and _reader_thread.is_alive():
    _reader_thread.join(timeout=1.0)
if cap:
    cap.release()
if args.mode != 'playback' and picam2 is not None:
    picam2.close()
if args.mode == 'record' and writer:
    writer.release()
