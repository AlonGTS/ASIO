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
    global bbox, tracking, tracker, current_frame
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        frame_for_init = current_frame.copy()
        w, h = 60, 60
        bbox = (max(0, x - w//2), max(0, y - h//2), w, h)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame_for_init, bbox)
        tracking = True
        print(f"[INFO] Tracker initialized from Pi click at ({x}, {y})")

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
    global bbox, tracking, tracker, current_frame
    try:
        x = int(request.form.get("x"))
        y = int(request.form.get("y"))
        w, h = 60, 60
        if current_frame is None:
            return "No frame", 400
        tracking = False
        bbox = None
        tracker = None
        bbox = (max(0, x - w//2), max(0, y - h//2), w, h)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(current_frame, bbox)
        tracking = True
        print(f"[INFO] Tracker initialized from remote click at ({x}, {y})")
        return "OK", 200
    except Exception as e:
        return f"Error: {e}", 400

# === Launch Flask in separate thread ===
flask_thread = Thread(target=lambda: app.run(host="0.0.0.0", port=5000, threaded=True))
flask_thread.daemon = True
flask_thread.start()

# === WebRTC (aiortc + aiohttp) ===
WEBRTC_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>WebRTC Viewer</title>
    <style>
      body { font-family: sans-serif; text-align:center; background:#f0f0f0; }
      video { width:640px; height:480px; background:#000; border:4px solid #333; cursor:crosshair; }
      button { margin-top:12px; padding:10px 16px; }
    </style>
  </head>
  <body>
    <h1>WebRTC Live Video</h1>
    <video id="v" autoplay playsinline></video><br/>
    <button id="start">Start</button>

    <script>
      const video = document.getElementById('v');
      const startBtn = document.getElementById('start');

      // Click-to-select using your existing Flask endpoint on :5000
      video.addEventListener('click', (e) => {
        const r = video.getBoundingClientRect();
        const x = Math.floor(e.clientX - r.left);
        const y = Math.floor(e.clientY - r.top);
        fetch('http://' + location.hostname + ':5000/select_point', {
          method:'POST',
          headers:{'Content-Type':'application/x-www-form-urlencoded'},
          body:`x=${x}&y=${y}`
        });
      });

      async function start() {
        const pc = new RTCPeerConnection();
        pc.ontrack = (ev) => { video.srcObject = ev.streams[0]; };
        const offer = await pc.createOffer({ offerToReceiveVideo: true });
        await pc.setLocalDescription(offer);
        const resp = await fetch('/offer', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({sdp: pc.localDescription.sdp, type: pc.localDescription.type})
        });
        const answer = await resp.json();
        await pc.setRemoteDescription(answer);
      }
      startBtn.onclick = start;
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
