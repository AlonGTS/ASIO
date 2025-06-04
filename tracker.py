# === Imports ===
import cv2
import time
import math
import numpy as np
from flask import Flask, Response, request, render_template_string
from threading import Thread

# === Flask App Initialization ===
app = Flask(__name__)
output_frame = None               # Frame to be streamed over HTTP
command_from_remote = None       # Command from web interface (reset, stop, quit)
current_frame = None             # Latest frame from camera
bbox = None                      # Bounding box of tracked object
tracking = False                 # Tracking state flag
tracker = None                   # OpenCV tracker object

# === MAVLink Setup ===
mavlink_enabled = False          # Flag if MAVLink connection is active
try:
    from pymavlink import mavutil
    connection = mavutil.mavlink_connection('/dev/serial0', baud=57600)
    connection.wait_heartbeat(timeout=5)
    connection.arducopter_arm()  # Attempt to arm the autopilot (if applicable)
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

# === Function to send pitch/yaw as PWM via MAVLink or print for debug ===
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

# === HTML UI page served at /control ===
HTML_PAGE = """
<!doctype html>
<html>
  <head>
    <title>Tracker Control</title>
    <style>
      body { font-family: sans-serif; text-align: center; background-color: #f0f0f0; }
      h1 { margin-top: 20px; }
      #videoCanvas { margin-top: 20px; border: 4px solid #333; width: 640px; height: 480px; cursor: crosshair; }
      button { padding: 10px 20px; margin: 10px; font-size: 16px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
      button.quit { background-color: #f44336; }
      button:hover { opacity: 0.8; }
    </style>
  </head>
  <body>
    <h1>Tracker Remote Control</h1>
    <canvas id="videoCanvas" width="640" height="480" onclick="sendPoint(event)"></canvas>
    <div>
      <button onclick="sendCommand('r')">Reset Tracker</button>
      <button onclick="sendCommand('s')">Stop Tracking</button>
      <button class="quit" onclick="sendCommand('q')">Quit</button>
    </div>
    <script>
      const canvas = document.getElementById("videoCanvas");
      const ctx = canvas.getContext("2d");
      function refreshFrame() {
        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          setTimeout(refreshFrame, 100);
        };
        img.src = "/";
      }
      function sendCommand(cmd) {
        fetch('/command', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: 'cmd=' + cmd
        });
      }
      function sendPoint(event) {
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor(event.clientX - rect.left);
        const y = Math.floor(event.clientY - rect.top);
        fetch('/select_point', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: `x=${x}&y=${y}`
        });
      }
      refreshFrame();
    </script>
  </body>
</html>
"""

# === Flask routes ===
@app.route('/')
def index():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control')
def control():
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
        # Reset tracker before new point selection
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

# === Stream generator for video feed ===
def generate_stream():
    global output_frame
    while True:
        if output_frame is not None:
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# === Launch Flask in separate thread ===
flask_thread = Thread(target=lambda: app.run(host="0.0.0.0", port=5000, threaded=True))
flask_thread.daemon = True
flask_thread.start()

# === Camera Setup ===
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(2)

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

cv2.setMouseCallback("Tracker", draw_rectangle)

# === Main Loop ===
while True:
    frame = picam2.capture_array()
    if frame is None:
        continue
    current_frame = frame.copy()
    h, w = frame.shape[:2]

    # Handle remote commands
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
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
                cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)
            else:
                cv2.putText(frame, "Tracking lost", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            print(f"[ERROR] Tracker update failed: {e}")
            tracking = False

    output_frame = frame.copy()
    cv2.imshow("Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        tracking = False
        bbox = None
        tracker = None
        print("[INFO] Tracker reset from Pi")

cv2.destroyAllWindows()
picam2.close()
