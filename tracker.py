#!/usr/bin/env python3
# === Imports ===
import cv2
import time
import math
import numpy as np
from flask import Flask, Response, request, render_template_string
from threading import Thread, Lock, Condition
import argparse
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, simpledialog
from flask_cors import CORS

# WebRTC / aiohttp deps
import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
from fractions import Fraction

SHOW_LOCAL = True  # True only when debugging with a monitor

# === Flask App Initialization ===
app = Flask(__name__)
CORS(app)  # allow the WebRTC page to post to /select_point on :5000

output_frame = None               # Frame to be streamed over HTTP / WebRTC
command_from_remote = None        # Command from web interface (reset, stop, quit)
current_frame = None              # Latest frame from camera
bbox = None                       # Bounding box of tracked object
tracking = False                  # Tracking state flag
tracker = None                    # OpenCV tracker object

# Thread sync for sharing frames with WebRTC
frame_lock = Lock()
frame_ready = Condition(frame_lock)

# === HTML Page for /control ===
HTML_PAGE = """
<!doctype html>
<html>
  <head>
    <title>Tracker Control</title>
    <style>
      body {
        font-family: sans-serif;
        text-align: center;
        background-color: #f0f0f0;
      }
      h1 { margin-top: 20px; }
      #video {
        margin-top: 20px;
        border: 4px solid #333;
        width: 640px;
        height: 480px;
        cursor: crosshair;
      }
      button {
        padding: 10px 20px; margin: 10px;
        font-size: 16px; background-color: #4CAF50;
        color: white; border: none; border-radius: 5px; cursor: pointer;
      }
      button.quit { background-color: #f44336; }
      button:hover { opacity: 0.8; }
      .note { margin-top: 10px; color: #555; font-size: 14px; }
    </style>
  </head>
  <body>
    <h1>Tracker Remote Control</h1>

    <!-- Persistent MJPEG stream -->
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

      // Send click coordinates to the server
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

      // Send control commands (reset, stop, quit)
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
mavlink_enabled = False  # True if MAVLink connection is established
try:
    from pymavlink import mavutil
    connection = mavutil.mavlink_connection('/dev/serial0', baud=57600)
    connection.wait_heartbeat(timeout=5)
    connection.arducopter_arm()  # Attempt to arm the autopilot
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

# === Send pitch/yaw as MAVLink override or debug print ===
def send_attitude(pitch, yaw):
    if not mavlink_enabled:
        print(f"[DEBUG] Pitch: {math.degrees(pitch):.2f}deg, Yaw: {math.degrees(yaw):.2f}deg")
        return
    try:
        pitch_pwm = int(1500 + pitch * 500)
        yaw_pwm = int(1500 + yaw * 500)
        connection.mav.rc_channels_override_send(
            connection.target_system,
            connection.target_component,
            pitch_pwm, yaw_pwm, 0, 0, 0, 0, 0, 0
        )
        print(f"[MAVLink] Sent pitch PWM: {pitch_pwm}, yaw PWM: {yaw_pwm}")
    except Exception as e:
        print(f"[MAVLink] Failed to send attitude: {e}")

# === Command-line arguments setup ===
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['live', 'record', 'playback'], default='live')
parser.add_argument('--video', help='Path to video file for playback')
parser.add_argument('--duration', type=int, help='Duration to record (seconds)')
args = parser.parse_args()

# === If recording mode and no duration provided, ask user via GUI ===
if args.mode == 'record' and not args.duration:
    root = tk.Tk()
    root.withdraw()  # Hide root Tkinter window
    duration = simpledialog.askinteger("Recording Duration", "How many seconds to record?", minvalue=1, maxvalue=3600)
    if not duration:
        print("[ERROR] No duration selected. Exiting.")
        exit(1)
    args.duration = duration
    print(f"[INFO] Recording duration set to {args.duration} seconds")

# === Camera or Video Input Setup ===
from picamera2 import Picamera2
if args.mode == 'playback':
    if not args.video:
        root = tk.Tk()
        root.withdraw()
        args.video = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.avi *.mp4 *.mov *.mkv"), ("All files", "*.*")]
        )
        if not args.video:
            print("[ERROR] No file selected. Exiting...")
            exit(1)
    cap = cv2.VideoCapture(args.video)

else:
    # Initialize the Pi Camera for live capture
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.set_controls({"FrameRate": 30})  # keep latency down
    picam2.start()
    time.sleep(0.3)


def make_csrt_params(moving: bool):
    p = cv2.TrackerCSRT_Params()

    # Robust appearance features (keep on for both; turn off only if starving for CPU)
    for name, val in dict(
        use_hog=True,
        use_color_names=True,
        use_gray=False,
        use_rgb=True,
        use_channel_weights=True,
        use_segmentation=False,  # turn True if your build supports it and you want more robustness
        window_function="hann",
        kaiser_alpha=3.2,  # used if window_function='kaiser'
        hog_clip=2.0,
        histogram_bins=16,
        background_ratio=2  # smaller = tighter background sampling
    ).items():
        if hasattr(p, name): setattr(p, name, val)

    # Scale space & adaptation (key for motion vs fixed)
    if moving:
        suggested = dict(
            number_of_scales=55,       # more scales → wider search
            scale_step=1.02,           # finer steps
            scale_lr=0.65,             # faster scale learning
            scale_sigma_factor=0.30,   # wider scale kernel
            scale_model_max_area=1024  # allow larger scale model if available
        )
    else:
        suggested = dict(
            number_of_scales=33,       # fewer scales → less jitter
            scale_step=1.03,           # slightly coarser
            scale_lr=0.15,             # slow scale learning (stable size)
            scale_sigma_factor=0.25,
            scale_model_max_area=512
        )
    for name, val in suggested.items():
        if hasattr(p, name): setattr(p, name, val)

    # Optimization / discrimination
    # More ADMM iterations = better discrimination (slower).
    if hasattr(p, "admm_iterations"):
        p.admm_iterations = 6 if moving else 9

    # Template / search behaviour
    # (Some builds expose template_size, we keep modest to avoid big CPU)
    if hasattr(p, "template_size"):
        p.template_size = 200 if moving else 160

    # Some OpenCV builds expose 'filter_lr' (general model learning rate); if present, use it.
    if hasattr(p, "filter_lr"):
        p.filter_lr = 0.25 if moving else 0.08

    return p

def create_csrt_tracker(moving: bool):
    try:
        params = make_csrt_params(moving)
        return cv2.TrackerCSRT_create(params)
    except TypeError:
        # Fallback if this OpenCV build doesn't accept params in the ctor
        t = cv2.TrackerCSRT_create()
        # Some builds allow setting attributes on t.params; guarded just in case.
        if hasattr(t, "set"):
            params = make_csrt_params(moving)
            for k, v in params.__dict__.items():
                try:
                    getattr(t, k)  # touch
                    setattr(t, k, v)
                except Exception:
                    pass
        return t






# === Video Recording Setup ===
writer = None
record_start_time = None
if args.mode == 'record':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_filename = f"RecordingsMahat/recording_{timestamp}.avi"
    writer = cv2.VideoWriter(video_filename, fourcc, 60.0, (640, 480))
    record_start_time = time.time()
    print(f"[INFO] Recording to {video_filename}")

# === Create OpenCV window for visual tracker feedback ===
if SHOW_LOCAL:
    cv2.namedWindow("Tracker")

# === Mouse callback for Pi GUI tracker selection ===
def draw_rectangle(event, x, y, flags, param):
    global bbox, tracking, tracker, current_frame, bMoovingTgt
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        frame_for_init = current_frame.copy()
        if bMoovingTgt:
            w, h = 30, 30   # smaller box for moving target
        else:
            w, h = 80, 80   # larger box for stationary target
        bbox = (max(0, x - w//2), max(0, y - h//2), w, h)
        tracker = create_csrt_tracker(bMoovingTgt)

#        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame_for_init, bbox)
        tracking = True
        print(f"[INFO] Tracker initialized ({'MOVING' if bMoovingTgt else 'FIXED'}) at ({x}, {y}), box {w}x{h}")


# Bind mouse callback only if showing local window
if SHOW_LOCAL:
    cv2.setMouseCallback("Tracker", draw_rectangle)

# Tune JPEG quality for faster encode (50–70 is a good range)
JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 60]

# === Stream generator for video feed (MJPEG) ===
def generate_stream():
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

# === Flask routes ===
@app.route('/stream.mjpg')
def mjpeg():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control')
def control():
    if args.mode == 'playback':
        return "Control not available in playback mode", 403
    return render_template_string(HTML_PAGE)

@app.route('/command', methods=['POST'])
def command():
    global command_from_remote
    cmd = request.form.get("cmd")
    if cmd in ['r', 's', 'q']:
        command_from_remote = cmd
        return "OK", 200
    return "Invalid", 400

@app.route('/select_point', methods=['POST'])
def select_point():
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

#        tracker = cv2.TrackerCSRT_create()
        tracker.init(current_frame, bbox)
        tracking = True
        print(f"[INFO] Tracker initialized ({'MOVING' if bMoovingTgt else 'FIXED'}) from remote click at ({x}, {y}), box {w}x{h}")
        return "OK", 200
    except Exception as e:
        return f"Error: {e}", 400


@app.route('/nudge', methods=['POST'])
def nudge():
    """Nudge the current tracking box by (dx, dy) pixels and re-init the tracker."""
    global bbox, tracking, tracker, current_frame
    try:
        dx = int(request.form.get("dx", 0))
        dy = int(request.form.get("dy", 0))
        if current_frame is None:
            return "No frame", 400

        h, w = current_frame.shape[:2]

        # If we have no bbox yet, create one centered first
        if bbox is None:
            bw = bh = 60
            cx = w // 2
            cy = h // 2
        else:
            x, y, bw, bh = map(int, bbox)
            cx = x + bw // 2
            cy = y + bh // 2

        # Apply nudge
        cx = max(bw // 2, min(w - bw // 2, cx + dx))
        cy = max(bh // 2, min(h - bh // 2, cy + dy))

        # Rebuild clamped bbox
        x_new = int(max(0, min(w - bw, cx - bw // 2)))
        y_new = int(max(0, min(h - bh, cy - bh // 2)))
        bbox_new = (x_new, y_new, bw, bh)

        # Re-init tracker on current frame
        tracking = False
        tracker = None
        tracker_local = create_csrt_tracker(bMoovingTgt)

        #tracker_local = cv2.TrackerCSRT_create()
        tracker_local.init(current_frame, bbox_new)
        tracker = tracker_local
        bbox = bbox_new
        tracking = True

        print(f"[INFO] Nudged bbox by ({dx},{dy}) -> {bbox}")
        return "OK", 200
    except Exception as e:
        print(f"[ERROR] Nudge failed: {e}")
        return f"Error: {e}", 400
   
# Toggle state (ברירת מחדל: מטרה קבועה)
bMoovingTgt = False  # שים לב לשם בדיוק כפי שביקשת

@app.route('/set_target_mode', methods=['POST'])
def set_target_mode():
    global bMoovingTgt
    val = request.form.get("bMoovingTgt", "0")
    bMoovingTgt = (val == "1")
    print(f"[INFO] Target mode set to: {'MOVING' if bMoovingTgt else 'FIXED'}")
    return "OK", 200



# === Launch Flask in separate thread ===
flask_thread = Thread(target=lambda: app.run(host="0.0.0.0", port=5000, threaded=True))
flask_thread.daemon = True
flask_thread.start()

WEBRTC_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>WebRTC Viewer</title>
    <style>
      :root { --w: 640px; --h: 480px; }
      body { font-family: sans-serif; text-align:center; background:#f0f0f0; margin:0; padding:24px; }
      h1 { margin: 0 0 10px 0; }
      #video { width: var(--w); height: var(--h); background:#000; border:4px solid #333; cursor:crosshair; }
      .controls { margin: 12px 0; display: inline-flex; gap: 10px; flex-wrap: wrap; align-items: center; justify-content: center; }
      button {
        padding: 10px 14px; border: 0; border-radius: 8px; font-size: 15px; cursor: pointer;
        background:#4CAF50; color:white; box-shadow: 0 2px 6px rgba(0,0,0,.15);
      }
      button.secondary { background:#607D8B; }
      button.quit { background:#f44336; }
      button.toggle { background:#2196F3; }
      button:active { transform: translateY(1px); }

      /* D-pad */
      .dpad { display: inline-grid; grid-template-columns: 48px 48px 48px; grid-template-rows: 48px 48px 48px; gap:6px; margin-left: 8px; }
      .dpad button { width:48px; height:48px; background:#9E9E9E; }
      .dpad .blank { visibility:hidden; }

      #status { margin-top: 8px; color:#444; font-size: 14px; min-height: 1.2em; }
      .hint { margin-top:6px; color:#666; font-size: 12px; }
      .hint kbd { background:#eee; border:1px solid #ccc; border-bottom-width:2px; padding:2px 6px; border-radius:4px;}
    </style>
  </head>
  <body>
    <h1>WebRTC Live Video</h1>
    <video id="video" autoplay playsinline></video>

    <div class="controls">
      <button id="startBtn" class="secondary">Start</button>
      <button onclick="sendCmd('r')">Reset (R)</button>
      <button onclick="sendCmd('s')">Stop (S)</button>
      <button class="quit" onclick="sendCmd('q')">Quit (Q)</button>

      <!-- Toggle target type -->
      <button id="tgtBtn" class="toggle" onclick="toggleTarget()">Target: Fixed (M)</button>

      <!-- D-pad -->
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

    <div id="status"></div>
    <div class="hint">
      Click the video to select a target. Use arrow keys to nudge by 5px
      (<kbd>Shift</kbd>=10px, <kbd>Alt</kbd>=1px). <kbd>M</kbd> toggles Fixed/Moving.
      R/S/Q for Reset/Stop/Quit.
    </div>

    <script>
      const video = document.getElementById('video');
      const startBtn = document.getElementById('startBtn');
      const statusEl = document.getElementById('status');
      const tgtBtn = document.getElementById('tgtBtn');

      function setStatus(msg) { statusEl.textContent = msg; }

      async function sendCmd(cmd) {
        try {
          const resp = await fetch('http://' + location.hostname + ':5000/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: 'cmd=' + encodeURIComponent(cmd)
          });
          setStatus(resp.ok ? 'Sent command: ' + cmd : 'Command failed: ' + cmd);
        } catch (e) { setStatus('Command error: ' + e); }
      }

      // Nudge API
      async function nudge(dx, dy) {
        try {
          const resp = await fetch('http://' + location.hostname + ':5000/nudge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `dx=${dx}&dy=${dy}`
          });
          setStatus(resp.ok ? `Nudged (${dx}, ${dy})` : `Nudge failed (${dx}, ${dy})`);
        } catch (e) { setStatus('Nudge error: ' + e); }
      }

      // Click-to-select absolute point
      video.addEventListener('click', async (e) => {
        const r = video.getBoundingClientRect();
        const x = Math.floor(e.clientX - r.left);
        const y = Math.floor(e.clientY - r.top);
        try {
          const resp = await fetch('http://' + location.hostname + ':5000/select_point', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `x=${x}&y=${y}`
          });
          setStatus(resp.ok ? `Selected (${x}, ${y})` : `Select failed (${x}, ${y})`);
        } catch (e) { setStatus('Select error: ' + e); }
      });

      // Moving/Fixed toggle
      let movingTgt = false; // default: Fixed
      async function toggleTarget() {
        movingTgt = !movingTgt;
        tgtBtn.textContent = 'Target: ' + (movingTgt ? 'Moving' : 'Fixed') + ' (M)';
        try {
          const resp = await fetch('http://' + location.hostname + ':5000/set_target_mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: 'bMoovingTgt=' + (movingTgt ? 1 : 0)
          });
          setStatus(resp.ok ? ('Mode: ' + (movingTgt ? 'MOVING' : 'FIXED')) : 'Mode set failed');
        } catch (e) { setStatus('Mode error: ' + e); }boun
      }

      // Keyboard: arrows nudge; Shift=10px, Alt=1px (default 5px); M toggles mode
      window.addEventListener('keydown', (e) => {
        const step = e.shiftKey ? 10 : (e.altKey ? 1 : 5);
        if (e.key === 'ArrowRight') { nudge(step, 0);  e.preventDefault(); }
        if (e.key === 'ArrowLeft')  { nudge(-step, 0); e.preventDefault(); }
        if (e.key === 'ArrowUp')    { nudge(0, -step); e.preventDefault(); }
        if (e.key === 'ArrowDown')  { nudge(0, step);  e.preventDefault(); }
        if (e.key === 'm' || e.key === 'M') toggleTarget();
        if (e.key === 'r' || e.key === 'R') sendCmd('r');
        if (e.key === 's' || e.key === 'S') sendCmd('s');
        if (e.key === 'q' || e.key === 'Q') sendCmd('q');
      });

      // WebRTC start
      async function start() {
        try {
          const pc = new RTCPeerConnection();
          pc.ontrack = (ev) => { video.srcObject = ev.streams[0]; setStatus('Streaming…'); };
          const offer = await pc.createOffer({ offerToReceiveVideo: true });
          await pc.setLocalDescription(offer);
          const resp = await fetch('/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
          });
          const answer = await resp.json();
          await pc.setRemoteDescription(answer);
          setStatus('Connected via WebRTC');
        } catch (e) { setStatus('WebRTC error: ' + e); }
      }
      startBtn.addEventListener('click', start);
    </script>
  </body>
</html>
"""


class GlobalFrameTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, target_fps=30):
        super().__init__()
        self.time_base = Fraction(1, 90000)  # 90 kHz clock
        self.frame_interval = 1.0 / target_fps
        self._last_ts = 0.0

    async def recv(self) -> VideoFrame:
        global output_frame
        now = time.time()
        if self._last_ts:
            to_sleep = self.frame_interval - (now - self._last_ts)
            if to_sleep > 0:
                await asyncio.sleep(to_sleep)
        self._last_ts = time.time()

        with frame_ready:
            if output_frame is None:
                frame_ready.wait(timeout=0.05)
            frame = None if output_frame is None else output_frame.copy()

        if frame is None:
            raise MediaStreamError("No frame available")

        # output_frame is BGR already; use bgr24 (no extra conversion)
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts = int(self._last_ts * 90000)
        vf.time_base = self.time_base
        return vf

pcs = set()

async def webrtc_index(request):
    return web.Response(text=WEBRTC_HTML, content_type="text/html")

async def webrtc_offer(request):
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
    await asyncio.gather(*[pc.close() for pc in pcs])

def start_webrtc_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app_webrtc = web.Application()
    app_webrtc.on_shutdown.append(on_webrtc_shutdown)
    app_webrtc.router.add_get("/", webrtc_index)
    app_webrtc.router.add_post("/offer", webrtc_offer)
    #web.run_app(app_webrtc, host="0.0.0.0", port=8080)
    web.run_app(app_webrtc, host="0.0.0.0", port=8080, handle_signals=False)


# Launch the WebRTC server in a background thread
webrtc_thread = Thread(target=start_webrtc_server, daemon=True)
webrtc_thread.start()

# === Main Loop ===
while True:
    # Capture frame from video or camera
    if args.mode == 'playback':
        ret, frame = cap.read()
        if not ret:
            break
    else:
        frame = picam2.capture_array()

    if frame is None:
        continue

    current_frame = frame.copy()
    h, w = frame.shape[:2]

    # Handle commands from remote control
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

    # Update tracker
    if tracking and tracker is not None:
        try:
            success, bbox = tracker.update(frame)
            if success:
                x, y, bw, bh = map(int, bbox)
                cx, cy = x + bw // 2, y + bh // 2
                dx = cx - w // 2
                dy = cy - h // 2
                norm_dx = dx / w
                norm_dy = dy / h
                yaw = norm_dx * math.radians(60)
                pitch = -norm_dy * math.radians(45)
                send_attitude(pitch, yaw)

                # Draw overlay
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
                cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)
            else:
                cv2.putText(frame, "Tracking lost", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            print(f"[ERROR] Tracker update failed: {e}")
            tracking = False

    # Write frame if recording
    if args.mode == 'record' and writer is not None:
        writer.write(frame)
        if args.duration and (time.time() - record_start_time >= args.duration):
            print("[INFO] Reached recording duration, exiting.")
            break

    # Optional: timestamp overlay (helps eyeball latency on both outputs)
    cv2.putText(frame, datetime.now().strftime('%H:%M:%S.%f')[:-3],
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Publish the latest frame for MJPEG & WebRTC
    with frame_ready:
        output_frame = frame.copy()
        frame_ready.notify_all()

    # Local debug window
    if SHOW_LOCAL:
        cv2.imshow("Tracker", frame)

    # Keyboard handling (safe even if no window; returns -1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        tracking = False
        bbox = None
        tracker = None
        print("[INFO] Tracker reset from Pi")

# === Cleanup ===
cv2.destroyAllWindows()
if args.mode == 'record' and writer:
    writer.release()
if args.mode != 'playback':
    picam2.close()
