#!/usr/bin/env python3
# === Imports ===
# Core vision / math / utils
import cv2
import time
import math
import numpy as np

# Concurrency primitives
from threading import Thread, Lock, Condition
import threading
from types import SimpleNamespace

# CLI args / timestamps / small GUI dialogs for file/duration picking
import tomllib
from pathlib import Path

_HERE = Path(__file__).parent
import argparse
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, simpledialog

import webrtc_server
from gts_tracker import GTSTracker

# Load configuration
with open(_HERE / "config.toml", "rb") as _f:
    _cfg = tomllib.load(_f)

SHOW_LOCAL    = _cfg["display"]["show_local"]
MAX_BB_WIDTH  = _cfg["tracking"]["max_bb_width"]
MAX_BB_HEIGHT = _cfg["tracking"]["max_bb_height"]
MAIN_SIZES    = [tuple(s) for s in _cfg["camera"]["main_sizes"]]
LORES_SIZES   = [tuple(s) for s in _cfg["camera"]["lores_sizes"]]

# Shared frame buffer for WebRTC
frame_buffer = webrtc_server.FrameBuffer()

# === Global shared state (protected by frame_ready) ===
output_frame = None               # Final frame after overlays → served to MJPEG / WebRTC

# Mutable state shared between main loop, reader threads, and Flask/WebRTC
state = SimpleNamespace(
    current_frame   = None,   # Raw latest frame from camera/file (no overlays)
    command_from_remote = None,   # One-letter command from web UI: 'r','s','q'
    bbox            = None,   # Current CSRT tracking box (MAIN coords: x, y, w, h)
    tracking        = False,  # Tracking on/off flag
    tracker         = None,   # OpenCV tracker object (runs on LORES frame)
    bMoovingTgt     = False,  # Target type (False=fixed, True=moving)
    lores_size      = None,   # Filled after config load below
)

_main_idx = 0       # index into MAIN_SIZES
_lores_idx = 0      # index into LORES_SIZES

main_size = list(MAIN_SIZES[_main_idx])    # [W, H] for capture/preview/output
lores_size = list(LORES_SIZES[_lores_idx]) # [W, H] for tracking
state.lores_size = lores_size

# Playback controls (used only in playback mode)
playback_rate = 1.0
seek_to_msec = None
playback_ctrl_lock = Lock()

# Playback telemetry (reader updates; main loop reads to sync trackbars)
playback_duration_ms = 0.0
playback_pos_ms = 0.0

# Local-UI (OpenCV) trackbar flags (playback-only)
_trackbar_ready = False
_suppress_trackbar_cb = False

# FPS meter (rough)
_prev_ts = time.time()
_fps_alpha = 0.9
_est_fps = 0.0

# Thread sync for frame sharing between producer (reader) and consumers (MJPEG/WebRTC)
frame_lock = Lock()
frame_ready = Condition(frame_lock)

# ============ Camera/File Reader (unified) ============
cap = None
picam2 = None
_reader_thread = None
_stop_reader = threading.Event()

def _reader_playback(path, loop=False):
    """
    Video file reader that respects playback_rate and seek_to_msec globals.
    Updates playback_pos_ms and playback_duration_ms for UI sync.
    """
    global cap, playback_rate, seek_to_msec
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
            state.current_frame = frame
            frame_ready.notify_all()

        # Adjust pacing
        if rate > 1.0:
            frames_to_skip = int(rate) - 1
            for _ in range(frames_to_skip):
                cap.grab()
            time.sleep(base_delay * 0.25)
        else:
            time.sleep(base_delay / rate)

def _init_live_camera():
    """(Re)create and start PiCamera2 with the current main_size."""
    global picam2
    from picamera2 import Picamera2
    if picam2 is not None:
        try: picam2.stop()
        except Exception: pass
        try: picam2.close()
        except Exception: pass
        picam2 = None

    picam2 = Picamera2()
    w, h = main_size
    picam2.preview_configuration.main.size = (int(w), int(h))
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.set_controls({"FrameRate": 30})
    picam2.start()
    print(f"[LIVE] Camera started MAIN={w}x{h}")

def _reader_live_picam():
    """Continuously read frames from PiCamera2 and publish into current_frame."""
    global picam2
    print("[INFO] Live reader started (PiCamera2)")
    while not _stop_reader.is_set():
        frame = picam2.capture_array()
        if frame is None:
            continue
        with frame_ready:
            state.current_frame = frame  # raw MAIN frame only
            frame_ready.notify_all()

def _restart_reader_live():
    """Stop live reader, reinit camera (for new MAIN size), and restart reader."""
    global _reader_thread
    _stop_reader.set()
    if _reader_thread and _reader_thread.is_alive():
        _reader_thread.join(timeout=1.0)
    _stop_reader.clear()
    _init_live_camera()
    _reader_thread = Thread(target=_reader_live_picam, daemon=True)
    _reader_thread.start()


# === MAVLink Setup ===
import mavlink_client
mavlink_client.connect()

# === Command-line arguments setup ===
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['live', 'record', 'playback'], default='live')
parser.add_argument('--video', help='Path to video file for playback')
parser.add_argument('--duration', type=int, help='Duration to record (seconds)')
parser.add_argument('--loop', action='store_true', help='Loop video in playback mode')
args = parser.parse_args()

# If recording and duration not provided on CLI, ask with a small dialog (Tk)
if args.mode == 'record' and not args.duration:
    root = tk.Tk(); root.withdraw()
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
        root = tk.Tk(); root.withdraw()
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
    _stop_reader.clear()
    _init_live_camera()
    _reader_thread = Thread(target=_reader_live_picam, daemon=True)
    _reader_thread.start()

# === GTS Tracker (compiled module) ===
def create_csrt_tracker(moving: bool):
    return GTSTracker(mode="moving" if moving else "fixed")

# === Video Recording Setup (mode=record) ===
writer = None
record_start_time = None
if args.mode == 'record':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_filename = f"RecordingsMahat/recording_{timestamp}.avi"
    writer = cv2.VideoWriter(video_filename, fourcc, 60.0, (640, 480))
    record_start_time = time.time()
    print(f"[INFO] Recording to {video_filename}")

# === Local debug window (optional) ===
if SHOW_LOCAL:
    cv2.namedWindow("Tracker")

# ---- Trackbar callbacks (playback-only) ----
def _on_seek_trackbar(pos):
    """
    OpenCV trackbar callback for playback seek.
    Converts the trackbar position (0–1000) to a millisecond timestamp
    and stores it in seek_to_msec for the reader thread to act on.
    Suppressed while the main loop is updating the trackbar programmatically.
    """
    global seek_to_msec, _suppress_trackbar_cb
    if _suppress_trackbar_cb or playback_duration_ms <= 0:
        return
    frac = pos / 1000.0
    with playback_ctrl_lock:
        seek_to_msec = int(frac * playback_duration_ms)

def _on_rate_trackbar(val):
    """
    OpenCV trackbar callback for playback speed.
    Maps the trackbar integer value to a playback rate in the range [0.1, 8.0]×.
    Suppressed while the main loop is syncing the trackbar to the current rate.
    """
    global playback_rate, _suppress_trackbar_cb
    if _suppress_trackbar_cb:
        return
    r = max(0.1, min(8.0, val / 100.0))
    with playback_ctrl_lock:
        playback_rate = r

if SHOW_LOCAL and args.mode == 'playback':
    cv2.createTrackbar('position', 'Tracker', 0, 1000, _on_seek_trackbar)
    cv2.createTrackbar('rate x0.01', 'Tracker', int(100), 800, _on_rate_trackbar)
    _trackbar_ready = True

# === Mouse callback (local window) ===
def draw_rectangle(event, x, y, flags, param):
    """
    Local GUI selection: left-click initializes a new tracker centered at (x,y) in MAIN coords.
    """
    if event == cv2.EVENT_LBUTTONDOWN and state.current_frame is not None:
        frame_for_init = state.current_frame.copy()
        w, h = (30, 30) if state.bMoovingTgt else (80, 80)
        x0 = max(0, x - w//2); y0 = max(0, y - h//2)
        bbox_main = (x0, y0, w, h)
        # Convert MAIN → LORES
        mw, mh = frame_for_init.shape[1], frame_for_init.shape[0]
        lw, lh = state.lores_size
        sx = lw / mw; sy = lh / mh
        xb = int(x0 * sx); yb = int(y0 * sy)
        wb = max(2, int(w * sx)); hb = max(2, int(h * sy))
        lores_frame = cv2.resize(frame_for_init, (lw, lh), interpolation=cv2.INTER_LINEAR)

        tracker_local = create_csrt_tracker(state.bMoovingTgt)
        tracker_local.init(lores_frame, (xb, yb, wb, hb))

        state.tracker = tracker_local
        state.bbox = bbox_main
        state.tracking = True
        print(f"[INFO] Tracker init (MAIN) at ({x},{y}), box {w}x{h} | LORES {lw}x{lh}")

if SHOW_LOCAL:
    cv2.setMouseCallback("Tracker", draw_rectangle)


import flask_app
app = flask_app.create_app(state, create_csrt_tracker)

# === Launch Flask in separate thread ===
flask_thread = Thread(target=lambda: app.run(host="0.0.0.0", port=5000, threaded=True))
flask_thread.daemon = True
flask_thread.start()

# === WebRTC server (background thread) ===
webrtc_thread = Thread(target=webrtc_server.start, args=(frame_buffer,), daemon=True)
webrtc_thread.start()

# === Helpers to cycle resolutions (LIVE mode only) ===
def _cycle_main(delta):
    """
    Step the MAIN capture resolution up (+1) or down (-1) through MAIN_SIZES.
    Restarts the live camera reader at the new resolution. Live mode only.
    """
    global _main_idx, main_size
    _main_idx = (_main_idx + delta) % len(MAIN_SIZES)
    main_size = list(MAIN_SIZES[_main_idx])
    print(f"[LIVE] Reconfig MAIN → {main_size[0]}x{main_size[1]} (restart reader)")
    _restart_reader_live()

def _cycle_lores(delta):
    """
    Step the LORES tracking resolution up (+1) or down (-1) through LORES_SIZES.
    Takes effect on the next tracker initialization; does not restart the camera.
    """
    global _lores_idx
    _lores_idx = (_lores_idx + delta) % len(LORES_SIZES)
    state.lores_size = list(LORES_SIZES[_lores_idx])
    print(f"[TRACK] LORES → {state.lores_size[0]}x{state.lores_size[1]}")

# === Main Loop (render & publish) ===
while True:
    # Wait for a new current_frame from reader
    with frame_ready:
        if state.current_frame is None:
            frame_ready.wait(timeout=0.02)
        frame = None if state.current_frame is None else state.current_frame.copy()

    if frame is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    mh, mw = frame.shape[:2]
    lw, lh = state.lores_size
    sx_m2l = lw / mw
    sy_m2l = lh / mh
    sx_l2m = mw / lw
    sy_l2m = mh / lh

    # Handle commands
    if state.command_from_remote == 'r':
        state.tracking = False; state.bbox = None; state.tracker = None
        print("[INFO] Tracker reset from remote")
        state.command_from_remote = None
    elif state.command_from_remote == 's':
        state.tracking = False
        state.command_from_remote = None
    elif state.command_from_remote == 'q':
        print("[INFO] Quit requested from remote")
        break

    # Tracking on LORES
    lores_frame = cv2.resize(frame, (lw, lh), interpolation=cv2.INTER_LINEAR)

    if state.tracking and state.tracker is not None:
        try:
            success, bbox_lo = state.tracker.update(lores_frame)
            if success:
                xl, yl, wl, hl = map(int, bbox_lo)
                x = int(xl * sx_l2m); y = int(yl * sy_l2m)
                bw = max(2, int(wl * sx_l2m)); bh = max(2, int(hl * sy_l2m))
                cx, cy = x + bw // 2, y + bh // 2

                # Clamp bbox growth
                if bw > MAX_BB_WIDTH or bh > MAX_BB_HEIGHT:
                    scale_w = MAX_BB_WIDTH / bw
                    scale_h = MAX_BB_HEIGHT / bh
                    scale = min(scale_w, scale_h)
                    new_bw = max(2, int(bw * scale))
                    new_bh = max(2, int(bh * scale))
                    x = max(0, min(mw - new_bw, cx - new_bw // 2))
                    y = max(0, min(mh - new_bh, cy - new_bh // 2))
                    bw, bh = new_bw, new_bh
                    xb = int(x * sx_m2l); yb = int(y * sy_m2l)
                    wb = max(2, int(bw * sx_m2l)); hb = max(2, int(bh * sy_m2l))
                    state.tracker = create_csrt_tracker(state.bMoovingTgt)
                    state.tracker.init(lores_frame, (xb, yb, wb, hb))
                    print(f"[INFO] BB limited to {bw}x{bh} (max {MAX_BB_WIDTH}x{MAX_BB_HEIGHT})")

                state.bbox = (x, y, bw, bh)

                # Center offsets for attitude mapping (MAIN coords)
                dx = cx - mw // 2
                dy = cy - mh // 2
                norm_dx = dx / mw
                norm_dy = dy / mh

                # Simple FOV→angle mapping (heuristic; tune to your camera FOV)
                yaw   =  norm_dx * math.radians(60)   # ~60° HFOV
                pitch = -norm_dy * math.radians(45)   # ~45° VFOV
                yaw_err   =  norm_dx * math.radians(60)   # rad
                pitch_err = -norm_dy * math.radians(45)   # rad

                mavlink_client.send_vision_error(pitch_err, yaw_err)


                # Box visuals
                if state.bMoovingTgt:
                    box_color = (0, 0, 255)
                    cross_color = (0, 0, 255)
                else:
                    box_color = (255, 0, 0)
                    cross_color = (255, 0, 0)

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 2)
                cv2.line(frame, (cx - 10, cy), (cx + 10, cy), cross_color, 1)
                cv2.line(frame, (cx, cy - 10), (cx, cy + 10), cross_color, 1)
            else:
                cv2.putText(frame, "Tracking lost", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            print(f"[ERROR] Tracker update failed: {e}")
            state.tracking = False

    # Write to file if in record mode
    if args.mode == 'record' and writer is not None:
        writer.write(frame)
        if args.duration and (time.time() - record_start_time >= args.duration):
            print("[INFO] Reached recording duration, exiting.")
            break

    # FPS estimate
    now = time.time()
    dt = max(1e-6, now - _prev_ts)
    _prev_ts = now
    inst_fps = 1.0 / dt
    _est_fps = _fps_alpha * _est_fps + (1.0 - _fps_alpha) * inst_fps if _est_fps > 0 else inst_fps

    # Overlay text
    stamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    overlay1 = f"{stamp}"
    overlay2 = f"MAIN {mw}x{mh} | TRACK {lw}x{lh} | {int(_est_fps)} FPS"
    cv2.putText(frame, overlay1, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, overlay2, (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    if args.mode == 'live':
        cv2.putText(frame, "Keys: z/x MAIN -, +   c/v TRACK -, +", (8, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    # Playback UI sync
    if SHOW_LOCAL and args.mode == 'playback' and _trackbar_ready and playback_duration_ms > 0:
        try:
            _suppress_trackbar_cb = True
            pos_frac = max(0.0, min(1.0, playback_pos_ms / playback_duration_ms))
            cv2.setTrackbarPos('position', 'Tracker', int(pos_frac * 1000))
            with playback_ctrl_lock:
                cv2.setTrackbarPos('rate x0.01', 'Tracker', int(playback_rate * 100.0))
        finally:
            _suppress_trackbar_cb = False

    # Publish final frame
    with frame_ready:
        output_frame = frame.copy()
        frame_ready.notify_all()
    frame_buffer.put(output_frame)

    # Local window
    if SHOW_LOCAL:
        cv2.imshow("Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        state.tracking = False; state.bbox = None; state.tracker = None
        print("[INFO] Tracker reset from Pi")
    elif args.mode == 'live':
        if key == ord('x'):   _cycle_main(+1)
        elif key == ord('z'): _cycle_main(-1)
        elif key == ord('v'): _cycle_lores(+1)
        elif key == ord('c'): _cycle_lores(-1)
    elif args.mode == 'playback':
        if key == ord('f'):
            with playback_ctrl_lock: playback_rate = min(playback_rate * 2.0, 8.0)
            print(f"[PLAYBACK] Speed {playback_rate:.1f}×")
        elif key == ord('s'):
            with playback_ctrl_lock: playback_rate = max(playback_rate / 2.0, 0.25)
            print(f"[PLAYBACK] Speed {playback_rate:.2f}×")
        elif key == ord('1'):
            with playback_ctrl_lock: playback_rate = 1.0
            print("[PLAYBACK] Speed reset to 1×")
        elif key == ord('j'):
            if cap is not None:
                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                with playback_ctrl_lock: seek_to_msec = max(0, pos - 5000)
                print(f"[PLAYBACK] Seek −5 s")
        elif key == ord('l'):
            if cap is not None:
                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                with playback_ctrl_lock: seek_to_msec = pos + 5000
                print(f"[PLAYBACK] Seek +5 s")

# === Cleanup ===
cv2.destroyAllWindows()
_stop_reader.set()
if _reader_thread and _reader_thread.is_alive():
    _reader_thread.join(timeout=1.0)
if cap:
    cap.release()
if args.mode != 'playback' and picam2 is not None:
    try: picam2.stop()
    except Exception: pass
    try: picam2.close()
    except Exception: pass
if args.mode == 'record' and writer:
    writer.release()
