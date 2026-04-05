import cv2
import time
import argparse
import datetime
import os
import subprocess

# === Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["live", "record", "video"], default="video", help="Select operation mode")
parser.add_argument("--video", help="Path to video file for --mode video or output file for --mode record")
parser.add_argument("--duration", type=int, default=10, help="Duration (in seconds) for recording mode")
args = parser.parse_args()

def pick_video_file():
    script = 'set f to choose file with prompt "Select video file" of type {"public.movie","com.apple.quicktime-movie","public.mpeg-4","public.avi"}\nreturn POSIX path of f'
    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    return result.stdout.strip() or None

# === Mode: Record ===
if args.mode == "record":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        exit()

    output_path = args.video or f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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

BOX_SIZE = 80
cv2.namedWindow("Tracker")

# === Video loop (re-opens file picker when video ends) ===
video_path = args.video if args.mode == "video" else None

while True:
    # --- Open source ---
    if args.mode == "video":
        if not video_path:
            video_path = pick_video_file()
        if not video_path:
            print("[ERROR] No video file selected")
            break
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Failed to open video source")
        break

    tracker = cv2.TrackerCSRT_create()
    tracking = False
    bbox = None
    frame_for_init = None
    current_frame = None

    def draw_rectangle(event, x, y, flags, param):
        global bbox, tracking, tracker, frame_for_init, current_frame
        if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
            half = BOX_SIZE // 2
            bbox = (x - half, y - half, BOX_SIZE, BOX_SIZE)
            frame_for_init = current_frame.copy()
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame_for_init, bbox)
            tracking = True
            print(f"[INFO] Tracking started at point ({x}, {y}) with bbox: {bbox}")

    cv2.setMouseCallback("Tracker", draw_rectangle)

    # --- Playback loop ---
    quit_all = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = frame.copy()
        h, w = frame.shape[:2]

        label = "GTS tracking"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        cv2.putText(frame, label, ((w - label_w) // 2, label_h + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        if tracking:
            start_time = time.time()
            success, bbox = tracker.update(frame)
            elapsed_time = time.time() - start_time

            if success:
                x, y, bw, bh = map(int, bbox)
                cx, cy = x + bw // 2, y + bh // 2
                dx = cx - w // 2
                dy = cy - h // 2

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 0, 255), 1)
                cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 0, 255), 1)

                cv2.putText(frame, f"dx: {dx}, dy: {dy}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"{int(1000*elapsed_time)} ms, {int(1/elapsed_time)} fps", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Tracking lost", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Tracker", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            quit_all = True
            break
        elif key & 0xFF == ord('r'):
            tracking = False
            bbox = None
            tracker = cv2.TrackerCSRT_create()
            print("[INFO] Tracker reset.")
        elif key & 0xFF == ord('n') and args.mode == "video":
            video_path = None
            break
        elif tracking and bbox is not None and current_frame is not None:
            # Arrow keys: shift bbox by 5 pixels and reinit tracker
            # macOS: 63232/63233/63234/63235 — Linux: 82/84/81/83
            dx_shift, dy_shift = 0, 0
            if key == 0:    # Up
                dy_shift = -5
            elif key == 1:  # Down
                dy_shift = 5
            elif key == 2:  # Left
                dx_shift = -5
            elif key == 3:  # Right
                dx_shift = 5
            if dx_shift or dy_shift:
                x, y, bw, bh = map(int, bbox)
                bbox = (x + dx_shift, y + dy_shift, bw, bh)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(current_frame, bbox)
                print(f"[INFO] Bbox shifted by ({dx_shift}, {dy_shift}) → {bbox}")

    cap.release()

    if quit_all or args.mode != "video":
        break

    # Video ended or 'n' pressed — pick a new file
    video_path = None

cv2.destroyAllWindows()
