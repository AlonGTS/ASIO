import cv2
import time
import math
import numpy as np
import argparse
import datetime
import os

# === Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["live", "record", "video"], required=True, help="Select operation mode")
parser.add_argument("--video", help="Path to video file for --mode video or output file for --mode record")
parser.add_argument("--duration", type=int, default=10, help="Duration (in seconds) for recording mode")
args = parser.parse_args()

# === MAVLink Setup (safe for testing without FC) ===
mavlink_enabled = False
try:
    from pymavlink import mavutil
    connection = mavutil.mavlink_connection('udpout:127.0.0.1:14550', autoreconnect=True)
    connection.wait_heartbeat(timeout=5)
    mavlink_enabled = True
    print("[MAVLink] Connected and heartbeat received.")
except Exception as e:
    print(f"[WARNING] MAVLink not connected: {e}")
    connection = None

# === FOV for Attitude Conversion ===
FOV_X_DEG = 60
FOV_Y_DEG = 45

def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]

def send_attitude(pitch, yaw):
    if not mavlink_enabled:
        print(f"[DEBUG] Would send pitch: {math.degrees(pitch):.2f}°deg, yaw: {math.degrees(yaw):.2f}deg (MAVLink disabled)")
        return

    quat = euler_to_quaternion(0, pitch, yaw)
    connection.mav.set_attitude_target_send(
        0, 1, 1,
        0b00000100,
        quat,
        0, 0, 0,
        0
    )
    print(f"[MAVLink] Sent pitch: {math.degrees(pitch):.2f}deg, yaw: {math.degrees(yaw):.2f}deg")

# === Mode: Record ===
if args.mode == "record":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        exit()

    output_path = args.video or f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in range(int(fps * args.duration)):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] Recording saved to {output_path}")
    exit()

# === Mode: Video or Live Tracking ===
video_source = args.video if args.mode == "video" else 0
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("[ERROR] Failed to open video source")
    exit()

active_tracker = "CSRT"
tracker = cv2.TrackerCSRT_create()
tracking = False
bbox = None
frame_for_init = None
drawing = False
ix, iy = -1, -1
current_frame = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bbox, tracking, tracker, frame_for_init, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        frame_for_init = current_frame.copy()
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        bbox = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if bbox and bbox[2] > 10 and bbox[3] > 10:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame_for_init, bbox)
            tracking = True
            print(f"[INFO] Tracking started with bbox: {bbox}")

cv2.namedWindow("Tracker")
cv2.setMouseCallback("Tracker", draw_rectangle)

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_frame = frame.copy()
    h, w = frame.shape[:2]

    cv2.putText(frame, f"Active Tracker: {active_tracker}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if args.mode == "live":
        cv2.putText(frame, "Draw ROI | Press 'r' to reset | 'q' to quit",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if drawing and bbox is not None:
        x, y, bw, bh = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 255), 2)

    if tracking:
        start_time = time.time()
        success, bbox = tracker.update(frame)
        end_time = time.time()
        elapsed_time = end_time - start_time

        if success:
            x, y, bw, bh = map(int, bbox)
            cx, cy = x + bw // 2, y + bh // 2
            dx = cx - w // 2
            dy = cy - h // 2

            norm_dx = dx / w
            norm_dy = dy / h

            yaw_rad = norm_dx * math.radians(FOV_X_DEG)
            pitch_rad = -norm_dy * math.radians(FOV_Y_DEG)

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)

            cv2.putText(frame, f"dx: {dx}, dy: {dy}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {math.degrees(pitch_rad):.2f}°, Yaw: {math.degrees(yaw_rad):.2f}°",
                        (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"{(int)(1000*elapsed_time)} ms, {(int)(1/elapsed_time)} fps", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            send_attitude(pitch_rad, yaw_rad)
        else:
            cv2.putText(frame, "Tracking lost", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        tracking = False
        bbox = None
        tracker = cv2.TrackerCSRT_create()
        print("[INFO] Tracker reset.")

cap.release()
cv2.destroyAllWindows()
