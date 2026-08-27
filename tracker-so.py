#!/usr/bin/env python3
# === Imports ===
# Core vision / math / utils
import cv2
import time
import math
import numpy as np
import psutil
import collections
import csv

# Concurrency primitives
from threading import Thread, Lock, Condition
import threading
from types import SimpleNamespace

# CLI args / timestamps / small GUI dialogs for file/duration picking
import fcntl
import os
import signal
import socket
import struct
import subprocess
import sys
import tomllib
from pathlib import Path

_HERE = Path(__file__).parent
import argparse
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog

import webrtc_server
from gts_tracker import GTSTracker


def _choose_video_file():
    """Native macOS file picker via AppleScript, run in a completely
    separate `osascript` process — deliberately NOT Tkinter. A Tk dialog
    launched from a process VS Code's debugger spawned proved unreliable in
    practice (the GCS window went unresponsive, spinning-cursor beachball,
    clicks not registering) — likely a focus/window-server interaction
    specific to how debugpy launches its debuggee. Running the dialog in an
    unrelated process sidesteps that outright, and this process never
    touches Tk/Cocoa itself, so it also doesn't get registered as a second,
    unlabeled "Python" app in the Dock for the rest of its life the way
    even a destroyed Tk window does."""
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'POSIX path of (choose file of type {"avi","mp4","mov","mkv"} '
             'with prompt "Select video file")'],
            capture_output=True, text=True, timeout=120,
        )
        path = result.stdout.strip()
        return path if path and result.returncode == 0 else None
    except Exception as e:
        print(f"[ERROR] File picker failed: {e}")
        return None


# ── Tracking Quality Monitor ─────────────────────────────────────────────────
class TrackingQualityMonitor:
    """
    Per-frame confidence score [0.0–1.0] computed on top of CSRT.

    CSRT's internal PSR is not exposed via OpenCV's Python bindings; this
    class approximates it from three observable signals:

      • Frame-to-frame NCC (55 %)  — compares the current ROI patch to the
        patch from the PREVIOUS frame.  Between consecutive frames the scale
        change is tiny even when the drone is approaching the target, so NCC
        stays high on correct tracking and drops sharply on a drift event.
        (Comparing to the *initial* template would fail as the drone closes in.)

      • Velocity gate        (30 %)  — penalises bbox-centre jumps > 15 % of
        the frame's larger dimension in a single step (teleportation = drift).

      • Size-change gate     (15 %)  — penalises sudden bbox area changes
        larger than 4× in one frame (unphysical growth/shrink).

    Thresholds
    ----------
    score ≥ SCORE_GOOD        → green  box, MAVLink control enabled
    score ≥ SCORE_UNCERTAIN   → orange box, MAVLink still enabled (visible warn)
    score <  SCORE_UNCERTAIN  → red    box, MAVLink suppressed
    score <  SCORE_UNCERTAIN for BAD_FRAMES_LIMIT consecutive frames → tracking broken
    """

    SCORE_GOOD        = 0.60
    SCORE_UNCERTAIN   = 0.40
    BAD_FRAMES_LIMIT  = 1      # 1 bad frame breaks tracking immediately
    _TMPL_SIZE        = (64, 64)   # canonical patch size for NCC

    def __init__(self):
        self._prev_patch = None    # grayscale 64×64 from previous frame
        self._prev_cx    = None
        self._prev_cy    = None
        self._prev_area  = None
        self.score       = 1.0
        self.bad_frames  = 0      # consecutive frames below SCORE_UNCERTAIN

    # ── public API ────────────────────────────────────────────────────────

    def init(self, frame: np.ndarray, bbox: tuple):
        """
        Capture the first appearance patch and reset history.
        Call this on the first successful update after a new tracker is
        initialised (pass lores frame + lores bbox).
        """
        x, y, w, h = (int(v) for v in bbox)
        patch = self._safe_crop(frame, x, y, w, h)
        if patch is not None and patch.size > 0:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            self._prev_patch = cv2.resize(gray, self._TMPL_SIZE)
        else:
            self._prev_patch = None
        self._prev_cx   = x + w // 2
        self._prev_cy   = y + h // 2
        self._prev_area = max(1, w * h)
        self.score      = 1.0

    def update(self, frame: np.ndarray, bbox: tuple) -> float:
        """
        Compute confidence for this frame after a successful CSRT update.
        frame and bbox must be in lores coordinate space.
        Returns score ∈ [0.0, 1.0] and advances internal state.
        """
        if bbox is None:
            self.score = 0.0
            return self.score

        xl, yl, wl, hl = (int(v) for v in bbox)
        cx, cy         = xl + wl // 2, yl + hl // 2
        curr_area      = max(1, wl * hl)

        # ── Frame-to-frame NCC ────────────────────────────────────────────
        patch = self._safe_crop(frame, xl, yl, wl, hl)
        if patch is not None and patch.size > 0:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            curr = cv2.resize(gray, self._TMPL_SIZE)
            if self._prev_patch is not None:
                res       = cv2.matchTemplate(curr, self._prev_patch, cv2.TM_CCOEFF_NORMED)
                ncc_score = float(np.clip(res[0, 0], 0.0, 1.0))
            else:
                ncc_score = 0.5          # no previous patch yet → neutral
            self._prev_patch = curr      # roll forward
        else:
            ncc_score = 0.5              # patch out of bounds → neutral

        # ── Velocity gate ─────────────────────────────────────────────────
        if self._prev_cx is not None:
            fh, fw    = frame.shape[:2]
            jump      = math.hypot(cx - self._prev_cx, cy - self._prev_cy)
            max_jump  = max(fw, fh) * 0.15
            vel_score = float(np.clip(1.0 - jump / max_jump, 0.0, 1.0))
        else:
            vel_score = 1.0
        self._prev_cx, self._prev_cy = cx, cy

        # ── Size-change gate ──────────────────────────────────────────────
        if self._prev_area is not None:
            ratio      = max(curr_area, self._prev_area) / min(curr_area, self._prev_area)
            # ratio 1.0 → score 1.0 | ratio ≥ 4.0 → score 0.0 (linear)
            size_score = float(np.clip(1.0 - (ratio - 1.0) / 3.0, 0.0, 1.0))
        else:
            size_score = 1.0
        self._prev_area = curr_area

        # ── Combined score ────────────────────────────────────────────────
        self.score = 0.55 * ncc_score + 0.30 * vel_score + 0.15 * size_score
        return self.score

    def reset(self):
        self._prev_patch = None
        self._prev_cx    = self._prev_cy = None
        self._prev_area  = None
        self.score       = 1.0
        self.bad_frames  = 0

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _safe_crop(frame, x, y, w, h):
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]


# Load configuration
with open(_HERE / "config.toml", "rb") as _f:
    _cfg = tomllib.load(_f)

SHOW_LOCAL    = _cfg["display"]["show_local"]
MAX_BB_WIDTH  = _cfg["tracking"]["max_bb_width"]
MAX_BB_HEIGHT = _cfg["tracking"]["max_bb_height"]
# White-target refinement has two phases: brightness-blob detection while
# the target is small/medium in frame, and — once close/large enough that
# the "+" mark is clearly resolved — dark-cross-centroid detection instead
# (blob detection alone degrades up close: the sheet fills much of the box
# with uneven brightness/glare, and it starts landing on a corner instead of
# the sheet's true center). This timer switches from the first phase to the
# second this many seconds after launch — 0 = never switch (stays on plain
# blob detection for the whole flight; tune this per typical launch-to-
# touchdown duration). Falls back to the old key name for compatibility with
# existing per-Pi config files across the fleet.
WHITE_TARGET_CLOSE_RANGE_AFTER_S = _cfg["tracking"].get(
    "white_target_close_range_after_s",
    _cfg["tracking"].get("white_target_disable_after_s", 0))
MAIN_SIZES    = [tuple(s) for s in _cfg["camera"]["main_sizes"]]
LORES_SIZES   = [tuple(s) for s in _cfg["camera"]["lores_sizes"]]
_CAM_IDLE_FPS   = _cfg["camera"].get("idle_fps", 5)
_CAM_ACTIVE_FPS = _cfg["camera"].get("active_fps", 30)

_LOG_CFG          = _cfg.get("logging", {})
_BASELINE_ENABLED = _LOG_CFG.get("baseline_enabled", False)
_BASELINE_DIR     = _LOG_CFG.get("baseline_dir", "logs")

def _get_iface_ip(iface: str) -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            return socket.inet_ntoa(
                fcntl.ioctl(s.fileno(), 0x8915,
                            struct.pack('256s', iface[:15].encode()))[20:24]
            )
    except OSError:
        return None

_net_iface  = _cfg["network"]["interface"]
_net        = _cfg["network"][_net_iface]
BIND_IP     = _get_iface_ip(_net_iface) or _net["bind_ip"]
_VIDEO_MODE = _cfg["network"].get("video_mode", "jpeg_udp")
GCS_IP     = None   # learned dynamically when GCS announces itself over the command channel
print(f"[NET] interface={_net_iface}  bind={BIND_IP}  gcs=<waiting for GCS hello>")

# Shared frame buffer for WebRTC
frame_buffer = webrtc_server.FrameBuffer()

# === Global shared state ===
# Mutable state shared between main loop, reader threads, and Flask/WebRTC
state = SimpleNamespace(
    current_frame   = None,   # Raw latest frame from camera/file (no overlays)
    frame_gen       = 0,      # bumped by reader threads on every new frame; lets the
                              # main loop tell "new frame" from "same frame, re-read"
    command_from_remote = None,   # One-letter command from web UI: 'r','s','q'
    bbox            = None,   # Current GTS tracking box (MAIN coords: x, y, w, h)
    tracking        = False,  # Tracking on/off flag
    tracker         = None,   # OpenCV tracker object (runs on LORES frame)
    bMoovingTgt     = False,  # Target type (False=fixed, True=moving)
    lores_size      = None,   # Filled after config load below
    last_init_source = None,  # Why the current tracker was (re)created — set by whichever
                               # code path creates it, just before assigning state.tracker.
                               # One of: "click", "local_click", "nudge", "resize",
                               # "bbox_clamp". Consumed by the baseline logger's session_source.
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
_playback_ended = False   # set by reader thread when video finishes (non-loop mode)
_current_video_path = None  # last opened file — used for restart

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
state.frame_lock = frame_lock   # expose to flask_app for safe current_frame snapshots

# Short rolling history of raw MAIN frames, keyed by state.frame_gen, so a
# GCS click that echoes back the frame_gen it was actually looking at can be
# applied to that historical frame (and replayed forward) instead of
# whatever frame is live on the Pi by the time the click round-trips back.
# maxlen covers ~2s at active_fps — comfortably more than one round trip.
_FRAME_HISTORY_LEN = max(30, _CAM_ACTIVE_FPS * 2)
_frame_history_lock = Lock()
_frame_history = collections.deque(maxlen=_FRAME_HISTORY_LEN)  # [(gen, frame), ...] oldest first

def _get_frame_history(min_gen):
    """(gen, frame) pairs for min_gen and everything captured after it, still
    in the buffer, oldest first. None if min_gen has already aged out (round
    trip took too long) or was never captured."""
    with _frame_history_lock:
        snapshot = list(_frame_history)
    if not snapshot or snapshot[0][0] > min_gen:
        return None
    result = [(g, f) for g, f in snapshot if g >= min_gen]
    return result if result else None

# ============ Camera/File Reader (unified) ============
cap = None
picam2 = None
_reader_thread = None
_stop_reader = threading.Event()
_camera_active = False   # False = idle (low fps, power-save); True = full fps. GCS-controlled.

# GCS-controlled: when True, the white/bright-blob aim refinement, periodic
# bbox recentering onto it, and the independent no-blob drift check are all
# active (see _refine_aim_point and its call site). When False, tracking
# behaves exactly as it did before any of that — plain bbox center only.
_white_target_enabled = False
_wt_close_range = False   # far/medium-range (blob) vs close-range (cross) phase — initialized
                           # here, well before flask_thread.start(), so the /status endpoint's
                           # aim_phase lambda can never reference this before it exists (a real
                           # NameError race: Flask can start serving requests, including from an
                           # already-polling GCS, well before the main loop reaches its first
                           # iteration, since real Pi hardware init — camera, MAVProxy — happens
                           # in between and takes real time)

def _set_white_target(enabled: bool):
    global _white_target_enabled
    _white_target_enabled = enabled
    print(f"[WT]   White-target mode {'ON' if enabled else 'OFF'}")

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
                print("[INFO] Playback ended — press R to restart, O to open new file")
                _playback_ended = True
                break

        playback_pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        with frame_ready:
            state.current_frame = frame
            state.frame_gen += 1
            gen = state.frame_gen
            frame_ready.notify_all()
        with _frame_history_lock:
            _frame_history.append((gen, frame))

        # Adjust pacing
        if rate > 1.0:
            frames_to_skip = int(rate) - 1
            for _ in range(frames_to_skip):
                cap.grab()
            time.sleep(base_delay * 0.25)
        else:
            time.sleep(base_delay / rate)

def _restart_playback(new_path=None):
    """
    Stop the current playback reader thread and start a fresh one.
    Pass new_path to switch to a different file; omit to replay the current one.
    Resets tracking state and playback-ended flag.
    """
    global _reader_thread, _playback_ended, _current_video_path, seek_to_msec
    # Stop the old reader
    _stop_reader.set()
    if _reader_thread and _reader_thread.is_alive():
        _reader_thread.join(timeout=2.0)
    _stop_reader.clear()

    path = new_path or _current_video_path
    if not path:
        return
    _current_video_path = path
    _playback_ended = False
    seek_to_msec = None

    # Clear tracking so operator re-selects on the new/rewound video
    state.tracking = False
    state.tracker  = None
    state.bbox     = None

    _reader_thread = Thread(target=_reader_playback, args=(path, args.loop), daemon=True)
    _reader_thread.start()
    print(f"[PLAYBACK] Restarted: {path}")

def _pick_full_fov_sensor_mode(picam2, min_fps):
    """
    Pick the smallest (fastest-reading-out) sensor mode whose crop still spans
    the WHOLE sensor array — same full FOV as the native resolution — instead
    of hardcoding the full pixel array as output_size. On IMX708 the full-res
    4608x2592 mode is full-FOV but caps out at ~14 fps; the 2x2-binned
    2304x1296 mode is *also* full-FOV (same crop_limits) and reaches ~56 fps.
    Falls back to the full-resolution mode if no full-FOV mode hits min_fps.
    """
    full_w, full_h = picam2.sensor_resolution
    full_fov = [m for m in picam2.sensor_modes
                if tuple(m["crop_limits"][2:]) == (full_w, full_h)]
    if not full_fov:
        return (full_w, full_h)
    full_fov.sort(key=lambda m: m["size"][0] * m["size"][1])  # smallest/fastest first
    for m in full_fov:
        if m["fps"] >= min_fps:
            return m["size"]
    return max(full_fov, key=lambda m: m["fps"])["size"]

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
    # Pick a sensor mode covering the full array (consistent FOV) fast enough
    # for active_fps, instead of always forcing the full-res ~14fps mode.
    sensor_size = _pick_full_fov_sensor_mode(picam2, _CAM_ACTIVE_FPS)
    config = picam2.create_preview_configuration(
        main={"size": (int(w), int(h)), "format": "RGB888"},
        sensor={"output_size": sensor_size},
    )
    picam2.configure(config)
    fps = _CAM_ACTIVE_FPS if _camera_active else _CAM_IDLE_FPS
    picam2.set_controls({"FrameRate": fps})
    picam2.start()
    print(f"[LIVE] Camera started MAIN={w}x{h} sensor={sensor_size} @ {fps}fps "
          f"({'active' if _camera_active else 'idle'})")

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
            state.frame_gen += 1
            gen = state.frame_gen
            frame_ready.notify_all()
        with _frame_history_lock:
            _frame_history.append((gen, frame))

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

def _set_camera_active(active: bool):
    """
    Explicit GCS-triggered toggle: raise/lower the live camera's capture FPS.
    Idle (low fps) keeps the Pi's CPU/heat down while waiting; a GCS command
    bumps it to full fps for responsive tracking. MAVLink keeps flowing at
    the mavproxy level regardless of this setting.
    """
    global _camera_active
    _camera_active = active
    if picam2 is None:
        return
    fps = _CAM_ACTIVE_FPS if active else _CAM_IDLE_FPS
    try:
        picam2.set_controls({"FrameRate": fps})
        print(f"[LIVE] Camera → {fps} fps ({'active' if active else 'idle'})")
    except Exception as e:
        print(f"[LIVE] Failed to set framerate {fps}: {e}")


# === CLI args (moved ahead of MAVLink/Serial Setup below, which needs
# args.mode to skip mavproxy/serial entirely in playback/record mode — there's
# no flight controller to talk to there, and start_mavproxy() launches a real
# subprocess against a real serial port that only exists on the Pi) ===
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['live', 'record', 'playback'], default='live')
parser.add_argument('--video', help='Path to video file for playback')
parser.add_argument('--duration', type=int, help='Duration to record (seconds)')
parser.add_argument('--loop', action='store_true', help='Loop video in playback mode')
args = parser.parse_args()

# === MAVLink / Serial Setup ===
# set_autopilot() just records a string, harmless in any mode. start_mavproxy()
# and connect() are the two calls that actually touch hardware/subprocesses —
# skipped outside 'live' since every mavlink_client function is already
# internally gated on _enabled (stays False here), so playback/record mode
# just runs with MAVLink silently disabled instead of crashing on a Mac/no-Pi
# host (no mavproxy install, no /dev/serial/by-id path).
import mavlink_client
_mav_cfg = _cfg["mavlink"]
AUTOPILOT = _mav_cfg.get("autopilot", "custom")
mavlink_client.set_autopilot(AUTOPILOT)
if args.mode == 'live':
    mavlink_client.start_mavproxy(
        pixhawk_port  = _mav_cfg["pixhawk_port"],
        pixhawk_baud  = _mav_cfg["pixhawk_baud"],
        gcs_port      = _mav_cfg["gcs_port"],
        local_port    = _mav_cfg["local_port"],
        extra_outputs = _mav_cfg.get("extra_outputs", []),
        mavproxy_path = _mav_cfg.get("mavproxy_path"),
    )
    mavlink_client.connect(
        url=f"udpin:0.0.0.0:{_mav_cfg['local_port']}",
        fallback_url=f"udpout:{GCS_IP}:{_mav_cfg['gcs_port']}",
    )
else:
    print(f"[MAVLink] Skipped in --mode {args.mode} (no flight controller to connect to)")

# Ensure MAVProxy is killed on SIGTERM (terminal closed) and SIGHUP,
# not just on normal exit / Ctrl-C (which atexit already handles).
def _shutdown(signum, frame):
    print(f"\n[Tracker] Signal {signum} received — shutting down.")
    try:
        _stop_recording()      # finalize any active recording before exit
    except NameError:
        pass                   # recording not yet initialised (early signal)
    try:
        if _baseline_queue is not None:
            _baseline_queue.put(None)   # stop sentinel — worker flushes+closes the CSV
            _baseline_thread.join(timeout=1.0)
    except NameError:
        pass                   # baseline logger not yet initialised (early signal)
    mavlink_client._stop_mavproxy()
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGHUP, _shutdown)

# === CPU usage/temp monitor (for GCS HUD) ===
# Sampled in a dedicated thread so /status never blocks on psutil's 1s interval.
_cpu_percent = 0.0
_cpu_temp_c  = None

def _read_cpu_temp_c():
    """Pi SoC temperature in °C, or None if unavailable (e.g. non-Pi host)."""
    try:
        zones = psutil.sensors_temperatures()
        zone = zones.get("cpu_thermal") or next(iter(zones.values()), None)
        if zone:
            return zone[0].current
    except Exception:
        pass
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None

def _cpu_monitor():
    global _cpu_percent, _cpu_temp_c
    psutil.cpu_percent(interval=None)   # prime the internal counter
    while True:
        _cpu_percent = psutil.cpu_percent(interval=1.0)
        _cpu_temp_c  = _read_cpu_temp_c()

Thread(target=_cpu_monitor, daemon=True).start()


# Precomputed FOV constants (avoid recomputing math.radians every frame)
_HFOV_RAD = math.radians(60)   # ~60° horizontal FOV
_VFOV_RAD = math.radians(45)   # ~45° vertical FOV

# PX4 rate-mode gain — px4 mode only, does not touch custom mode's _mav_x/_mav_y.
# _px4_pitch_err/_px4_yaw_err are sent as body-RATE setpoints (see
# mavlink_client.send_attitude_target), but their raw FOV-derived value tops
# out at the frame edge around ±22.5° (pitch) / ±30° (yaw) — nowhere near a
# meaningful rate. This gain scales that up so a target pinned at the frame
# edge commands ~100°/s on the larger (yaw) axis; MAX_ANGLE in
# mavlink_client.py still clamps above that as a hard ceiling.
_PX4_RATE_GAIN = math.radians(100) / (_HFOV_RAD / 2)

# (args parsed earlier, ahead of the MAVLink/Serial Setup block above)

# If recording and duration not provided on CLI, ask with a small dialog (Tk)
if args.mode == 'record' and not args.duration:
    root = tk.Tk(); root.withdraw()
    duration = simpledialog.askinteger("Recording Duration", "How many seconds to record?",
                                       minvalue=1, maxvalue=3600)
    root.destroy()   # otherwise this process keeps a hidden Tk window alive for its whole
                      # lifetime, which registers it as a second, unlabeled "Python" app in
                      # the Dock alongside gcs.py's "Mahat GCS" — confusing, and can steal
                      # keyboard/mouse focus from the actual GCS window
    if not duration:
        print("[ERROR] No duration selected. Exiting.")
        exit(1)
    args.duration = duration
    print(f"[INFO] Recording duration set to {args.duration} seconds")

# === Input Setup: file playback or live camera (each spawns a reader thread) ===
if args.mode == 'playback':
    # Also falls through to the picker for a non-empty but nonexistent path
    # (e.g. a stale/typo'd --video).
    if not args.video or not os.path.isfile(args.video):
        if args.video:
            print(f"[INFO] '{args.video}' not found — opening file picker")
        args.video = _choose_video_file()
        if not args.video:
            print("[ERROR] No file selected. Exiting...")
            exit(1)
    print(f"[INFO] Playback mode from file: {args.video} (loop={args.loop})")
    _current_video_path = args.video
    _stop_reader.clear()
    _reader_thread = Thread(target=_reader_playback, args=(args.video, args.loop), daemon=True)
    _reader_thread.start()
else:
    _stop_reader.clear()
    _init_live_camera()
    _reader_thread = Thread(target=_reader_live_picam, daemon=True)
    _reader_thread.start()

# === GTS Tracker (compiled module) ===
def create_gts_tracker(moving: bool):
    return GTSTracker(mode="moving" if moving else "fixed")

# Finds the white/bright blob within the tracker's bbox (e.g. a target
# sheet or marker sitting in a loose box that also covers plain terrain).
# Used both to refine the displayed aim point and to periodically re-center
# the tracker's own box onto it (see the periodic-resync block in the main
# loop, which reuses the same reinit approach as flask_app.py's /nudge
# endpoint). Picks the BRIGHTEST sufficiently-large blob (99th-percentile
# threshold keeps this specific to genuinely bright things, not just
# generally-lighter terrain texture) — this works the same way whether the
# target is a faint dot at long range or fills most of the box up close.
# The area cap is intentionally generous (not "small blob only"): during a
# landing approach the target grows continuously from a speck to filling
# most of the frame, and a tight area cap excludes the real target once
# it's grown past it — leaving only small brightness-noise fragments as
# candidates, which can land anywhere including box corners. Only when the
# "bright" region is nearly the entire crop (no real feature left to find,
# just uniform brightness) does this fall back to the plain bbox center.
_BLOB_PERCENTILE    = 99.0   # top ~1% brightest pixels = candidate blob pixels
_BLOB_MIN_AREA      = 2      # px — smaller than this is more likely sensor noise
_BLOB_MAX_AREA_FRAC = 0.90   # essentially the whole crop being "bright" isn't a real blob
_BLOB_MIN_CONTRAST  = 15     # blob must be at least this much brighter than the crop average

_STICKY_MAX_JUMP_FRAC = 0.6   # prev_point search radius, as a fraction of max(bw, bh)

def _refine_aim_point(frame, x, y, bw, bh, fallback_cx, fallback_cy, prev_point=None):
    """Return (cx, cy, found) — the centroid of a qualifying bright blob
    inside bbox (x,y,bw,bh) and found=True, or (fallback_cx, fallback_cy,
    False) — the plain bbox center — if no such blob is confidently present
    this frame.

    prev_point (MAIN coords, optional): when given, prefer whichever
    qualifying blob is closest to it — falling back to the single brightest
    blob only if none are within _STICKY_MAX_JUMP_FRAC of it (e.g. no
    prev_point yet, or the target genuinely moved further than that this
    frame). Without this, the pick is a fresh, memoryless "brightest blob
    right now" decision every frame — if more than one bright thing is in
    the box at once (the real target plus some incidental bright terrain/
    clutter), tiny frame-to-frame brightness differences can flip which one
    wins, making the point hop between them instead of tracking one object."""
    if bw < 6 or bh < 6:
        return fallback_cx, fallback_cy, False
    crop = frame[y:y + bh, x:x + bw]
    if crop.size == 0:
        return fallback_cx, fallback_cy, False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    # +1: a percentile that lands exactly on the background level (e.g. a
    # near-uniform crop) would otherwise pull most of the background into
    # the "bright" mask along with the blob.
    thresh = np.percentile(gray, _BLOB_PERCENTILE) + 1
    bright = (gray >= thresh).astype(np.uint8) * 255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return fallback_cx, fallback_cy, False

    area_crop = bw * bh
    bg_mean = gray.mean()
    candidates = []   # (label_idx, mean_val)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < _BLOB_MIN_AREA or a / area_crop > _BLOB_MAX_AREA_FRAC:
            continue
        mean_val = gray[labels == i].mean()
        if mean_val - bg_mean < _BLOB_MIN_CONTRAST:
            continue
        candidates.append((i, mean_val))
    if not candidates:
        return fallback_cx, fallback_cy, False

    if prev_point is not None:
        # Once locked onto something, proximity to where we were is the
        # PRIMARY criterion — pick whichever qualifying candidate is closest
        # to prev_point, not whichever is brightest. Brightness/contrast/size
        # already qualified this pool above; ranking by brightness AGAIN here
        # is what caused the flip-flop in the first place — two candidates
        # close together (the real target + some incidental bright thing)
        # would keep swapping places as whichever was marginally brighter
        # changed frame to frame, even with a distance-based prefilter.
        pcx, pcy = prev_point
        max_jump = max(bw, bh) * _STICKY_MAX_JUMP_FRAC
        dists = [(i, math.hypot(x + centroids[i][0] - pcx, y + centroids[i][1] - pcy))
                 for i, _ in candidates]
        nearest_idx, nearest_dist = min(dists, key=lambda t: t[1])
        if nearest_dist <= max_jump:
            best_idx = nearest_idx
        else:
            # Nothing plausible near the last position (real jump, or first
            # lock after a gap) — fall back to brightest overall.
            best_idx, _ = max(candidates, key=lambda t: t[1])
    else:
        best_idx, _ = max(candidates, key=lambda t: t[1])
    cx_i, cy_i = centroids[best_idx]
    return x + int(round(cx_i)), y + int(round(cy_i)), True

# Close-range alternative to _refine_aim_point: once the target is close/
# large enough that the "+" mark itself is clearly resolved, brightness-blob
# detection degrades (the sheet fills much of the box with uneven brightness/
# glare and starts winning on a corner instead of the sheet's true center).
# This finds the dark "+" mark's centroid within the sheet instead — a
# centroid is rotation- and scale-invariant by construction, so unlike
# template matching this needs no reference photo, no per-flight
# calibration, and no scale/rotation search grid (real footage shows the
# mark at visibly different rotations frame to frame, which would make plain
# template matching either fail or need an expensive multi-angle search).
_CROSS_MIN_AREA_FRAC = _cfg["tracking"].get("aim_cross_min_area_frac", 0.01)
_CROSS_MAX_AREA_FRAC = _cfg["tracking"].get("aim_cross_max_area_frac", 0.35)
_CROSS_MIN_CONTRAST  = _cfg["tracking"].get("aim_cross_min_contrast", 20)

def _find_cross_centroid(frame, x, y, bw, bh, fallback_cx, fallback_cy):
    """Return (cx, cy, found) — the centroid of a dark "+"-shaped mark found
    within the bright sheet inside bbox (x,y,bw,bh), or (fallback_cx,
    fallback_cy, False) if no sheet-like region or no confidently-present
    mark is found this frame."""
    if bw < 6 or bh < 6:
        return fallback_cx, fallback_cy, False
    crop = frame[y:y + bh, x:x + bw]
    if crop.size == 0:
        return fallback_cx, fallback_cy, False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    # Trim a small edge margin before analysis — a hard crop boundary can
    # carry encoding/interpolation artifact pixels (e.g. a stray near-black
    # or near-white edge row/column) that don't represent real scene content
    # but can hijack Otsu's threshold choice if left in (observed: a single
    # pure-black pixel column at a crop edge pulled Otsu's threshold to the
    # very bottom of the range, misclassifying almost the whole crop as
    # "bright" and picking up that same artifact as the "dark cross").
    _EDGE_TRIM = 3
    t = _EDGE_TRIM
    if gray.shape[0] > 2 * t + 4 and gray.shape[1] > 2 * t + 4:
        gray_in = gray[t:-t, t:-t]
    else:
        gray_in = gray
        t = 0

    # Stage 1: find the sheet itself — largest sufficiently-bright connected
    # component in the crop, via Otsu (adapts to the crop's own bright/dark
    # split) so a loose box that also covers darker background clutter can't
    # have that clutter mistaken for the mark. No upper bound on how much of
    # the crop this covers: close range (where this function runs) is
    # exactly when the box has grown to closely fit the target, so the sheet
    # legitimately filling nearly the whole crop is the expected good case,
    # not a sign of "nothing distinct here."
    _, bright_raw = cv2.threshold(gray_in, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    frac = (bright_raw > 0).mean()
    if frac < 0.03:
        return fallback_cx, fallback_cy, False
    filled = cv2.morphologyEx(bright_raw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(filled, connectivity=8)
    if n <= 1:
        return fallback_cx, fallback_cy, False
    sheet_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sheet_extent = labels == sheet_idx

    # Stage 2: find the dark cross strictly INSIDE the sheet (erode first so
    # the sheet's own edge pixels aren't mistaken for the mark).
    interior = cv2.erode(sheet_extent.astype(np.uint8) * 255, np.ones((7, 7), np.uint8)) > 0
    if interior.sum() == 0:
        return fallback_cx, fallback_cy, False
    dark_in_sheet = interior & (bright_raw == 0)
    dark_frac = dark_in_sheet.sum() / interior.sum()
    if dark_frac < _CROSS_MIN_AREA_FRAC or dark_frac > _CROSS_MAX_AREA_FRAC:
        return fallback_cx, fallback_cy, False
    ys_i, xs_i = np.nonzero(interior)
    vals = gray_in[ys_i, xs_i]
    dark_sel = dark_in_sheet[ys_i, xs_i]
    if vals[~dark_sel].mean() - vals[dark_sel].mean() < _CROSS_MIN_CONTRAST:
        return fallback_cx, fallback_cy, False   # not enough contrast — likely just noise/shadow
    dys, dxs = np.nonzero(dark_in_sheet)
    return x + t + int(round(dxs.mean())), y + t + int(round(dys.mean())), True

# === Video Recording Setup (shared by 'record' mode and live toggle) ===
import queue as _queue_mod

writer = None
record_queue = None
record_thread = None
record_start_time = None

# Dynamic recording state (live-mode toggle)
_recording      = False
_rec_writer     = None
_rec_queue      = None
_rec_thread     = None
_rec_start_time = None
_rec_filename   = None
_rec_lock       = Lock()

def _record_worker(proc, queue):
    """Drain frame bytes from *queue* into ffmpeg stdin; None sentinel stops the loop."""
    while True:
        frame_bytes = queue.get()
        if frame_bytes is None:
            break
        try:
            proc.stdin.write(frame_bytes)
        except BrokenPipeError:
            print("[REC] ffmpeg pipe closed unexpectedly")
            break
    try:
        proc.stdin.close()
    except Exception:
        pass
    proc.wait()

def _start_recording():
    """Start a new recording session (safe to call even if already recording)."""
    global _recording, _rec_writer, _rec_queue, _rec_thread, _rec_start_time, _rec_filename
    with _rec_lock:
        if _recording:
            print("[REC] Already recording")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _rec_filename = f"RecordingsMahat/recording_{timestamp}.mp4"
        w, h = main_size[0], main_size[1]
        # frag_keyframe+empty_moov → every GOP is a self-contained fragment written
        # immediately; the file is playable / recoverable even after a hard reboot.
        _rec_writer = subprocess.Popen([
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', '30',
            '-i', '-',
            '-vcodec', 'libx264', '-preset', 'ultrafast', '-crf', '28',
            '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
            _rec_filename
        ], stdin=subprocess.PIPE)
        _rec_queue  = _queue_mod.Queue(maxsize=30)
        _rec_thread = Thread(target=_record_worker, args=(_rec_writer, _rec_queue), daemon=True)
        _rec_thread.start()
        _rec_start_time = time.time()
        _recording = True
        print(f"[REC] ▶ Started → {_rec_filename}  ({w}x{h})")

def _stop_recording():
    """Finalize and close the current recording session."""
    global _recording, _rec_writer, _rec_queue, _rec_thread, _rec_start_time
    with _rec_lock:
        if not _recording:
            return
        _recording = False
        if _rec_queue is not None:
            _rec_queue.put(None)          # signal worker to stop
        if _rec_thread is not None:
            _rec_thread.join(timeout=8.0) # wait for ffmpeg to finalize
        dur = time.time() - _rec_start_time if _rec_start_time else 0
        print(f"[REC] ■ Stopped after {dur:.1f}s → {_rec_filename}")
        _rec_writer = None
        _rec_queue  = None
        _rec_thread = None
        _rec_start_time = None

if args.mode == 'record':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"RecordingsMahat/recording_{timestamp}.mp4"
    w, h = main_size[0], main_size[1]
    writer = subprocess.Popen([
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', '30',
        '-i', '-',
        '-vcodec', 'libx264', '-preset', 'ultrafast', '-crf', '28',
        '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
        video_filename
    ], stdin=subprocess.PIPE)
    record_queue = _queue_mod.Queue(maxsize=30)
    record_thread = Thread(target=_record_worker, args=(writer, record_queue), daemon=True)
    record_thread.start()
    record_start_time = time.time()
    print(f"[INFO] Recording to {video_filename}  ({w}x{h})")

# === Local debug window (optional) ===
if SHOW_LOCAL:
    cv2.namedWindow("Tracker")

# ---- Trackbar callbacks (playback-only) ----
def _on_seek_trackbar(pos):
    """
    OpenCV trackbar callback for playback seek.
    Converts the trackbar position (0–1000) to a millisecond timestamp
    and stores it in seek_to_msec for the reader thread to act on.
    If playback has ended, automatically restarts the reader at the new position.
    Suppressed while the main loop is updating the trackbar programmatically.
    """
    global seek_to_msec, _suppress_trackbar_cb
    if _suppress_trackbar_cb or playback_duration_ms <= 0:
        return
    frac = pos / 1000.0
    with playback_ctrl_lock:
        seek_to_msec = int(frac * playback_duration_ms)
    if _playback_ended:
        _restart_playback()   # reader is gone — restart it; it will seek on first iteration

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

        tracker_local = create_gts_tracker(state.bMoovingTgt)
        tracker_local.init(lores_frame, (xb, yb, wb, hb))

        state.last_init_source = "local_click"
        state.tracker = tracker_local
        state.bbox = bbox_main
        state.tracking = True
        print(f"[INFO] Tracker init (MAIN) at ({x},{y}), box {w}x{h} | LORES {lw}x{lh}")

if SHOW_LOCAL:
    cv2.setMouseCallback("Tracker", draw_rectangle)


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

# Launch button: "custom" mode just flips the launch flag the Simulink app
# reads over NAMED_VALUE_FLOAT; "px4" mode actually arms/disarms the FC
# (mode switch already happened at connect time, see set_guided_mode()).
def _launch_fn(v=None):
    launch = v if v is not None else not mavlink_client._launched
    if AUTOPILOT == "px4":
        if launch:
            time.sleep(1)  # give the FC a beat to settle into OFFBOARD before arming
            mavlink_client.arm()
        else:
            mavlink_client.disarm()
    else:
        mavlink_client.set_launch(launch)

import flask_app
app = flask_app.create_app(
    state, create_gts_tracker,
    cycle_main_fn      = _cycle_main if args.mode == 'live' else None,
    cycle_lores_fn     = _cycle_lores,
    launch_fn          = _launch_fn,
    get_launch_state_fn= lambda: mavlink_client._launched,
    toggle_record_fn   = lambda: _stop_recording() if _recording else _start_recording(),
    get_record_state_fn= lambda: _recording,
    set_fps_fn         = _set_camera_active if args.mode == 'live' else None,
    get_fps_state_fn   = lambda: _camera_active,
    get_cpu_fn         = lambda: _cpu_percent,
    get_cpu_temp_fn    = lambda: _cpu_temp_c,
    get_frame_history_fn = _get_frame_history if args.mode in ('live', 'playback') else None,
    # Lambdas, not bare references: set_video_mode/_current_video_mode are
    # defined later in this file (the video-stream section) — deferring the
    # name lookup to call time (well after the whole module has loaded)
    # avoids a NameError here at import time.
    set_video_mode_fn  = lambda mode: set_video_mode(mode),
    get_video_mode_fn  = lambda: _current_video_mode,
    set_white_target_fn = _set_white_target,
    get_white_target_fn  = lambda: _white_target_enabled,
    get_aim_phase_fn = lambda: (
        "cross" if _white_target_enabled and _wt_close_range else
        "blob"  if _white_target_enabled else
        "off"),
)

# === Launch Flask in separate thread (production WSGI server, not Werkzeug's dev server) ===
# flask_port defaults to 5000 (unset in config.toml on every Pi in the fleet)
# but is overridable — mainly for local/offline testing on a Mac, where
# macOS's own AirPlay Receiver squats on 5000 by default.
_FLASK_PORT = _cfg["network"].get("flask_port", 5000)
from waitress import serve as _waitress_serve
print(f"[Flask]  http://{BIND_IP}:{_FLASK_PORT}  (waitress)")
flask_thread = Thread(target=lambda: _waitress_serve(app, host="0.0.0.0", port=_FLASK_PORT, threads=8, _quiet=True))
flask_thread.daemon = True
flask_thread.start()

# === Video stream (mode selected by config.toml video_mode, switchable live
#     between jpeg_udp and h264_udp via /set_video_mode — see below) ===
#
# webrtc is deliberately NOT part of the switchable set: it's a different
# connection model entirely (HTTP/ICE signaling, not just a UDP port), and
# isn't the "comms degraded, drop to something loss-tolerant" case this
# exists for. If video_mode is "webrtc", it just starts once, as before,
# and /set_video_mode has nothing to switch away from.

import socket as _socket   # needed by command channel regardless of video_mode

_UDP_PORT = _cfg["network"].get("gcs_udp_port", 5600)   # shared by jpeg_udp and h264_udp

_video_thread     = None   # currently running jpeg_udp/h264_udp sender thread, or None
_video_stop_event = None   # its stop signal, or None
_current_video_mode = None  # "jpeg_udp" / "h264_udp" (whatever's actually running), or None
                             # if video_mode is "webrtc"/unknown — those aren't switchable

def _start_jpeg_udp():
    """(Re)start the JPEG-over-UDP sender. Its own thread + socket, torn
    down cleanly by _stop_video_stream() before anything else starts on
    the same port."""
    global _video_thread, _video_stop_event
    _jpeg_quality = _cfg["network"].get("gcs_jpeg_quality", 40)
    _stream_width = _cfg["network"].get("gcs_stream_width", 480)
    # Encode+send rate to the GCS is independent of capture/tracking fps — the
    # operator's video view doesn't need every frame, but the tracker/MAVLink
    # loop does. Throttling this is the cheapest way to cut CPU without
    # slowing down tracking. 0 = uncapped (encode every published frame).
    _stream_fps      = _cfg["network"].get("gcs_stream_fps", 15)
    _stream_interval = (1.0 / _stream_fps) if _stream_fps > 0 else 0.0

    _udp_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    _udp_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_SNDBUF, 1 << 20)
    _jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, _jpeg_quality]
    _udp_max = 65400

    stop_event = threading.Event()

    def _udp_stream_worker():
        last_gen      = -1
        _t0           = time.time()
        _sent         = 0
        _last_send_ts = 0.0
        print(f"[UDP]  stream worker ready — waiting for GCS to announce "
              f"(cap {_stream_fps if _stream_fps > 0 else 'uncapped'} fps)")
        while not stop_event.is_set():
            frame, gen = frame_buffer.get(last_gen=last_gen, timeout=0.1)
            if frame is None:
                continue
            last_gen = gen

            if GCS_IP is None:
                continue

            _now_gate = time.time()
            if _stream_interval > 0 and (_now_gate - _last_send_ts) < _stream_interval:
                continue   # skip encode+send for this frame — capture/tracking keep running at full fps
            _last_send_ts = _now_gate

            _sent += 1
            _now = time.time()
            if _now - _t0 >= 5.0:
                print(f"[UDP]  {_sent / (_now - _t0):.1f} fps  ({_sent} frames in {_now-_t0:.1f}s)  → {GCS_IP}")
                _t0, _sent = _now, 0

            h_f, w_f = frame.shape[:2]
            if w_f > _stream_width:
                scale = _stream_width / w_f
                frame = cv2.resize(frame, (_stream_width, int(h_f * scale)), interpolation=cv2.INTER_LINEAR)

            ok, buf = cv2.imencode('.jpg', frame, _jpeg_params)
            if not ok:
                continue
            # 4-byte big-endian frame_gen header + JPEG bytes. GCS must strip
            # the first 4 bytes before decoding, and echo that value back as
            # frame_gen on /select_point so the Pi can init tracking against
            # the frame the operator actually clicked on (see
            # STABILIZATION.md — "the stale-click problem").
            data = struct.pack('>I', gen & 0xFFFFFFFF) + buf.tobytes()
            if len(data) > _udp_max:
                continue
            try:
                _udp_sock.sendto(data, (GCS_IP, _UDP_PORT))
            except Exception as e:
                print(f"[UDP]  send error: {e}")

        try:
            _udp_sock.close()
        except Exception:
            pass
        print("[UDP]  stream worker stopped")

    _video_thread = Thread(target=_udp_stream_worker, daemon=True)
    _video_thread.start()
    _video_stop_event = stop_event

def _start_h264_udp():
    global _video_thread, _video_stop_event
    import h264_udp_server
    _h264_kbps = _cfg["network"].get("h264_bitrate_kbps", 2000)
    _video_thread, _video_stop_event = h264_udp_server.start(
        frame_buffer,
        gcs_ip_getter = lambda: GCS_IP,
        port          = _UDP_PORT,
        stream_width  = _cfg["network"].get("gcs_stream_width", 480),
        stream_fps    = _cfg["network"].get("gcs_stream_fps", 15),
        bitrate_kbps  = _h264_kbps,
        gop_seconds   = _cfg["network"].get("h264_gop_seconds", 0.5),
        rtp_payload   = _cfg["network"].get("h264_rtp_payload", 1200),
    )
    print(f"[H264]  udp://<gcs>:{_UDP_PORT}  bitrate={_h264_kbps}kbps  "
          f"(hardware encoder used if available, software fallback otherwise)")

def _stop_video_stream():
    """Signal whichever of jpeg_udp/h264_udp is running to stop, and wait
    (briefly) for its socket/encoder to actually release the port."""
    global _video_thread, _video_stop_event
    if _video_stop_event is not None:
        _video_stop_event.set()
    if _video_thread is not None:
        _video_thread.join(timeout=2.0)
    _video_thread = None
    _video_stop_event = None

def set_video_mode(mode):
    """Live-switch between jpeg_udp and h264_udp — called from
    flask_app.py's /set_video_mode, itself reached via gcs.py's VIDEO
    button over the UDP command channel. A real operational need: H.264
    depends on reference frames, so packet loss over a degraded link can
    corrupt everything until the next keyframe; jpeg_udp sends independent
    per-frame JPEGs, where a lost packet only ever costs that one frame.
    Returns True/False so the Flask route can report success."""
    global _current_video_mode
    if mode not in ("jpeg_udp", "h264_udp"):
        print(f"[VIDEO] Ignoring unsupported mode for live switch: {mode!r}")
        return False
    if _current_video_mode is None:
        print(f"[VIDEO] Not currently in a switchable mode (video_mode="
              f"{_VIDEO_MODE!r} at startup) — ignoring switch request")
        return False
    if mode == _current_video_mode:
        return True
    print(f"[VIDEO] Switching {_current_video_mode} -> {mode}")
    _stop_video_stream()
    (_start_jpeg_udp if mode == "jpeg_udp" else _start_h264_udp)()
    _current_video_mode = mode
    return True

if _VIDEO_MODE == "webrtc":
    _webrtc_kbps = _cfg["network"].get("webrtc_bitrate_kbps", 0)
    print(f"[WebRTC] http://{BIND_IP}:8080  bitrate={'unconstrained' if _webrtc_kbps == 0 else str(_webrtc_kbps)+'kbps'}")
    webrtc_thread = Thread(
        target=webrtc_server.start,
        args=(frame_buffer,),
        kwargs={"host": BIND_IP, "target_bitrate_kbps": _webrtc_kbps},
        daemon=True,
    )
    webrtc_thread.start()

elif _VIDEO_MODE == "jpeg_udp":
    _start_jpeg_udp()
    _current_video_mode = "jpeg_udp"

elif _VIDEO_MODE == "h264_udp":
    _start_h264_udp()
    _current_video_mode = "h264_udp"

else:
    print(f"[WARN] Unknown video_mode '{_VIDEO_MODE}' in config.toml — no video stream started")

# === UDP command channel (broadcast-based — works through AP isolation) ===
# Mac sends JSON command datagrams to the subnet broadcast address.
# Pi listens on this port and forwards them to the local Flask API,
# so all existing command logic is reused without duplication.
import json as _json
import requests as _req

_CMD_PORT = _cfg["network"].get("gcs_cmd_port", 5601)
_cmd_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
_cmd_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
_cmd_sock.bind(('', _CMD_PORT))

def _udp_cmd_listener():
    global GCS_IP
    print(f"[CMD]  listening for commands on UDP:{_CMD_PORT}")
    while True:
        try:
            data, addr = _cmd_sock.recvfrom(4096)

            # Learn / update GCS IP dynamically from sender address
            if GCS_IP != addr[0]:
                print(f"[NET]  GCS IP {'learned' if GCS_IP is None else 'updated'}: {addr[0]}  (was {GCS_IP})")
                GCS_IP = addr[0]

            msg = _json.loads(data.decode())
            ep  = msg.pop("endpoint", None)
            if ep:   # "hello" heartbeats have no endpoint — skip the Flask call
                print(f"[CMD]  ← {addr[0]}  {ep}  {msg}")
                _req.post(f"http://127.0.0.1:{_FLASK_PORT}/{ep}", data=msg, timeout=1)
        except Exception as e:
            if "timed out" not in str(e).lower():
                print(f"[CMD]  {e}")

Thread(target=_udp_cmd_listener, daemon=True).start()

# === Baseline instrumentation (Step A — TRACKER_ACCURACY_ROADMAP.md) ===
# Per-frame CSV log of what the tracker already computes (bbox center,
# confidence, processing time) so Experiment 1's baseline metrics can be
# derived after the fact, without changing tracking behavior. Writes happen
# on a dedicated thread off a bounded queue — logging never blocks the
# tracking loop; rows are dropped (not queued up) if the disk falls behind.
_BASELINE_FIELDS = [
    "wall_ts", "session_id", "session_source", "frame_gen",
    "main_w", "main_h", "lores_w", "lores_h",
    "success", "cx", "cy", "bbox_w", "bbox_h",
    "center_dx", "center_dy", "center_dist",
    "update_ms", "tq_score", "bad_frames", "drift_event", "inst_fps",
]

_baseline_queue = None
_baseline_thread = None

def _baseline_worker(path, q):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_BASELINE_FIELDS)
        writer.writeheader()
        f.flush()
        while True:
            row = q.get()
            if row is None:   # shutdown sentinel
                break
            writer.writerow(row)
            f.flush()   # rows are ~30/s at most — fine to flush every write

def _log_baseline(row: dict):
    if _baseline_queue is None:
        return
    try:
        _baseline_queue.put_nowait(row)
    except _queue_mod.Full:
        pass   # never block the tracking loop for logging

if _BASELINE_ENABLED:
    _baseline_out_dir = _HERE / _BASELINE_DIR
    _baseline_out_dir.mkdir(parents=True, exist_ok=True)
    _baseline_path = _baseline_out_dir / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    _baseline_queue = _queue_mod.Queue(maxsize=500)
    _baseline_thread = Thread(target=_baseline_worker, args=(_baseline_path, _baseline_queue), daemon=True)
    _baseline_thread.start()
    print(f"[BASELINE] logging per-frame tracking metrics to {_baseline_path}")

# === Main Loop (render & publish) ===
# Cached scale factors — recomputed only when resolution changes
_cached_dims = (0, 0, 0, 0)   # (mw, mh, lw, lh)
sx_m2l = sy_m2l = sx_l2m = sy_l2m = 1.0

# Tracking quality monitor (state persists across frames)
_tq_monitor      = TrackingQualityMonitor()
_last_tracker_id = None   # detect tracker replacement from ANY init path
_tq_needs_init   = True   # capture first patch on next successful update

_last_frame_gen = -1   # last state.frame_gen this loop has already processed

# Baseline-logging state (Step A)
_log_session_id = 0            # increments on every new tracker instance
_log_session_source = "unknown"      # why the current session started — see state.last_init_source
_log_prev_cx = _log_prev_cy = None   # previous frame's MAIN-coord center, for jump distance

# Independent drift signal: the correlation tracker's own quality score only
# checks "does this still look like what I saw last frame" — a smooth lock
# onto the wrong thing (e.g. rocks/clutter that happens to look self-similar
# frame to frame) can pass that check indefinitely while still being wrong.
# Since the actual target is known to be a bright/white blob (or, close
# range, a dark cross within one), treat "nothing found in or near the box
# for many consecutive frames" as drift too, even when the tracker's own
# score looks fine.
_no_aim_streak     = 0
_NO_AIM_DRIFT_LIMIT = 10   # consecutive not-found frames before treating this as drift

_blob_center_countdown = 0   # frames left until the next periodic re-center-on-blob
_BLOB_CENTER_EVERY_N_FRAMES = 15   # how often to resync the box onto the blob, in frames

_aim_smooth = None   # (cx, cy) EMA of the aim point in MAIN coords; None until first found
_AIM_SMOOTH_ALPHA = 0.35   # higher = more responsive/less smooth, lower = smoother/more lag
_AIM_SMOOTH_SNAP_FRAC = 0.25   # jump beyond this fraction of box size = real motion, snap
                                # to it immediately instead of easing (avoids lagging the
                                # smoothed point outside a fast-moving target)

_launch_ts   = None    # time.time() of the False→True launch edge; None until launched
_was_launched = False  # previous frame's launched state, to detect that edge
_wt_phase_announced = False   # edge-detects the phase switch for a one-time log line

while True:
    # Wait for a genuinely NEW frame (by generation, not just "not None") —
    # otherwise, once current_frame is set once, this would spin re-processing
    # and re-publishing the same stale frame as fast as the CPU allows,
    # completely ignoring the camera's actual capture rate (incl. idle fps).
    with frame_ready:
        if state.frame_gen == _last_frame_gen:
            frame_ready.wait(timeout=0.02)
        if state.frame_gen == _last_frame_gen or state.current_frame is None:
            frame = None
        else:
            frame = state.current_frame.copy()
            _last_frame_gen = state.frame_gen

    if frame is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    mh, mw = frame.shape[:2]
    lw, lh = state.lores_size
    lores_frame = cv2.resize(frame, (lw, lh), interpolation=cv2.INTER_NEAREST)

    # Track the launch edge so white-target mode can switch from blob
    # detection to close-range cross detection late in the flight (see
    # WHITE_TARGET_CLOSE_RANGE_AFTER_S) — blob detection degrades once the
    # target is close/large (starts landing on a corner instead of the
    # sheet's true center), which is exactly when the "+" mark is clearly
    # resolved enough for cross detection to take over.
    _now_launched = mavlink_client._launched
    if _now_launched and not _was_launched:
        _launch_ts = time.time()
    elif not _now_launched:
        _launch_ts = None
    _was_launched = _now_launched

    _wt_close_range = (WHITE_TARGET_CLOSE_RANGE_AFTER_S > 0 and _launch_ts is not None
                        and (time.time() - _launch_ts) >= WHITE_TARGET_CLOSE_RANGE_AFTER_S)
    if _wt_close_range and _white_target_enabled and not _wt_phase_announced:
        print(f"[WT]   {WHITE_TARGET_CLOSE_RANGE_AFTER_S:.0f}s since launch — "
              f"switching white-target refinement to close-range cross detection")
    _wt_phase_announced = _wt_close_range
    _wt_active = _white_target_enabled

    if (mw, mh, lw, lh) != _cached_dims:
        old_mw, old_mh = _cached_dims[0], _cached_dims[1]
        sx_m2l = lw / mw; sy_m2l = lh / mh
        sx_l2m = mw / lw; sy_l2m = mh / lh
        _cached_dims = (mw, mh, lw, lh)

        # Reinit tracker at new resolution to prevent tracking point drift
        if state.tracking and state.tracker is not None and state.bbox is not None and old_mw > 0:
            ox, oy, obw, obh = map(int, state.bbox)
            ncx = (ox + obw / 2) / old_mw
            ncy = (oy + obh / 2) / old_mh
            new_cx = int(ncx * mw)
            new_cy = int(ncy * mh)
            bw = max(2, int(obw * mw / old_mw))
            bh = max(2, int(obh * mh / old_mh))
            x0 = max(0, min(mw - bw, new_cx - bw // 2))
            y0 = max(0, min(mh - bh, new_cy - bh // 2))
            xb = int(x0 * sx_m2l); yb = int(y0 * sy_m2l)
            wb = max(2, int(bw * sx_m2l)); hb = max(2, int(bh * sy_m2l))
            new_tracker = create_gts_tracker(state.bMoovingTgt)
            new_tracker.init(lores_frame, (xb, yb, wb, hb))
            state.last_init_source = "resize"
            state.tracker = new_tracker
            state.bbox = (x0, y0, bw, bh)
            print(f"[INFO] Tracker reinitialized after resolution change: MAIN {mw}x{mh} LORES {lw}x{lh}")

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

    # Tracking on LORES (lores_frame computed above, before dims check)
    _mav_x, _mav_y = 100.0, 100.0  # sentinel: not tracking ("custom" mode)
    _px4_pitch_err, _px4_yaw_err = 0.0, 0.0  # neutral hold ("px4" mode)

    # Detect tracker replacement from ANY init path (flask, mouse, resolution change)
    if state.tracker is not None and id(state.tracker) != _last_tracker_id:
        _last_tracker_id = id(state.tracker)
        _tq_needs_init   = True
        _tq_monitor.reset()
        _log_session_id += 1
        _log_session_source = state.last_init_source or "unknown"
        _log_prev_cx = _log_prev_cy = None
        _no_aim_streak = 0
        _aim_smooth = None

    if state.tracking and state.tracker is not None:
        try:
            _upd_t0 = time.perf_counter()
            success, bbox_lo = state.tracker.update(lores_frame)
            _update_ms = (time.perf_counter() - _upd_t0) * 1000.0
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
                    state.tracker = create_gts_tracker(state.bMoovingTgt)
                    state.tracker.init(lores_frame, (xb, yb, wb, hb))
                    state.last_init_source = "bbox_clamp"
                    # This path bypasses the generic "detect tracker replacement" block
                    # above (already ran this iteration), so update its bookkeeping here too.
                    _last_tracker_id = id(state.tracker)
                    _tq_needs_init   = True      # new tracker → re-capture patch
                    _tq_monitor.reset()
                    _log_session_id += 1
                    _log_session_source = "bbox_clamp"
                    _log_prev_cx = _log_prev_cy = None
                    print(f"[INFO] BB limited to {bw}x{bh} (max {MAX_BB_WIDTH}x{MAX_BB_HEIGHT})")

                state.bbox = (x, y, bw, bh)
                # White-target mode gates ALL of: aim-point refinement (display +
                # MAVLink), periodic bbox recentering onto it, and the independent
                # no-aim drift check below. Off = plain bbox-center tracking,
                # exactly as before any of this existed. When on, which DETECTOR
                # runs depends on range: blob detection far/medium range, dark-
                # cross-centroid detection close range (see WHITE_TARGET_CLOSE_
                # RANGE_AFTER_S) — the latter is what stays accurate once the
                # target is close/large, where blob detection starts landing on
                # a corner of the sheet instead of its true center.
                if _wt_active:
                    if _wt_close_range:
                        aim_cx, aim_cy, _aim_found = _find_cross_centroid(frame, x, y, bw, bh, cx, cy)
                    else:
                        aim_cx, aim_cy, _aim_found = _refine_aim_point(
                            frame, x, y, bw, bh, cx, cy, prev_point=_aim_smooth)
                    _no_aim_streak = 0 if _aim_found else _no_aim_streak + 1

                    # Periodically resync the tracker's own box onto the aim point —
                    # same reinit approach as flask_app.py's /nudge endpoint (shift
                    # bbox position, keep its size, reinit the tracker there), but
                    # only every _BLOB_CENTER_EVERY_N_FRAMES frames, not continuously
                    # — a fixed, simple cadence rather than continuously second-
                    # guessing an otherwise-healthy track.
                    _blob_center_countdown -= 1
                    if _blob_center_countdown <= 0:
                        _blob_center_countdown = _BLOB_CENTER_EVERY_N_FRAMES
                        if _aim_found and (aim_cx != cx or aim_cy != cy):
                            nx = max(0, min(mw - bw, aim_cx - bw // 2))
                            ny = max(0, min(mh - bh, aim_cy - bh // 2))
                            xb = int(nx * sx_m2l); yb = int(ny * sy_m2l)
                            wb = max(2, int(bw * sx_m2l)); hb = max(2, int(bh * sy_m2l))
                            state.tracker = create_gts_tracker(state.bMoovingTgt)
                            state.tracker.init(lores_frame, (xb, yb, wb, hb))
                            state.last_init_source = "aim_center"
                            state.bbox = (nx, ny, bw, bh)
                            x, y, cx, cy = nx, ny, aim_cx, aim_cy
                            aim_cx, aim_cy = cx, cy
                            _last_tracker_id = id(state.tracker)
                            _tq_needs_init   = True
                            _tq_monitor.reset()
                            _log_session_id += 1
                            _log_session_source = "aim_center"
                            _log_prev_cx = _log_prev_cy = None

                    # Smooth the aim point so small frame-to-frame noise is
                    # damped, WITHOUT lagging behind real fast motion (e.g. a
                    # fast approach/descent shifting the target a lot between
                    # frames): plain exponential smoothing always trails by a
                    # proportional gap, which — under fast real motion — can
                    # leave the smoothed point outside the target entirely
                    # while the raw detection is still correctly on it. So:
                    # ease toward small moves, but SNAP straight to the raw
                    # point once it has moved far enough to be real motion
                    # rather than noise, instead of easing toward it too.
                    if _aim_found:
                        if _aim_smooth is None:
                            _aim_smooth = (aim_cx, aim_cy)
                        else:
                            _jump = math.hypot(aim_cx - _aim_smooth[0], aim_cy - _aim_smooth[1])
                            if _jump > _AIM_SMOOTH_SNAP_FRAC * max(bw, bh):
                                _aim_smooth = (aim_cx, aim_cy)   # real motion — don't lag behind it
                            else:
                                _aim_smooth = (
                                    _AIM_SMOOTH_ALPHA * aim_cx + (1 - _AIM_SMOOTH_ALPHA) * _aim_smooth[0],
                                    _AIM_SMOOTH_ALPHA * aim_cy + (1 - _AIM_SMOOTH_ALPHA) * _aim_smooth[1],
                                )
                        aim_cx, aim_cy = int(round(_aim_smooth[0])), int(round(_aim_smooth[1]))
                    elif _aim_smooth is not None:
                        aim_cx, aim_cy = int(round(_aim_smooth[0])), int(round(_aim_smooth[1]))
                else:
                    aim_cx, aim_cy = cx, cy
                    _aim_found = False
                    _no_aim_streak = 0

                # Tracking quality: init on first frame, update on all subsequent ones
                if _tq_needs_init:
                    _tq_monitor.init(lores_frame, (xl, yl, wl, hl))
                    _tq_needs_init = False
                else:
                    _tq_monitor.update(lores_frame, (xl, yl, wl, hl))
                tq = _tq_monitor.score

                # Center offsets for attitude mapping (MAIN coords) — uses the
                # refined aim point (the "+" mark when resolvable, else the
                # plain bbox center), not the raw bbox center directly.
                dx = aim_cx - mw // 2
                dy = aim_cy - mh // 2
                norm_dx = dx / mw
                norm_dy = dy / mh

                # Normalize to -1..1: 0 = centred, ±1 = target at frame edge
                pitch_norm = -norm_dy / (_VFOV_RAD / 2)
                yaw_norm   =  norm_dx / (_HFOV_RAD / 2)
                # Count consecutive bad frames — break tracking if drift sustained
                if tq < TrackingQualityMonitor.SCORE_UNCERTAIN:
                    _tq_monitor.bad_frames += 1
                else:
                    _tq_monitor.bad_frames = 0   # good frame resets the counter
                _bad_frames_for_log = _tq_monitor.bad_frames   # snapshot before a drift reset zeroes it

                _no_aim_drift = _no_aim_streak >= _NO_AIM_DRIFT_LIMIT
                _drift_event = (_tq_monitor.bad_frames >= TrackingQualityMonitor.BAD_FRAMES_LIMIT
                                 or _no_aim_drift)
                if _drift_event and not _wt_active:
                    # White-target mode off → behave exactly as before any of
                    # this existed: give up immediately, no search-based
                    # recovery attempt.
                    print(f"[TQ]   Drift detected ({_tq_monitor.bad_frames} bad frames, "
                          f"score={tq:.2f}) — tracking broken, re-select target")
                    state.tracking = False
                    state.tracker  = None
                    _tq_monitor.reset()
                    cv2.putText(frame, "Drift — re-select target", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif _drift_event:
                    _reason = (f"{_tq_monitor.bad_frames} bad frames, score={tq:.2f}"
                               if not _no_aim_drift else
                               f"nothing found for {_no_aim_streak} frames (tracker score still fine)")
                    print(f"[TQ]   Drift detected ({_reason}) — searching nearby for target...")
                    # Attempt automatic recovery: search a BOUNDED region around the
                    # lost box (not the whole frame — an unbounded search risks locking
                    # onto an unrelated bright object elsewhere in view, which would
                    # steer real flight control toward the wrong thing) for the target,
                    # and if found, reinit the tracker there instead of giving up. Only
                    # runs on CONFIRMED drift — a still-valid track is never
                    # second-guessed by this (that caused a worse bug previously: a
                    # single noisy blob detection relocating a box that was fine).
                    # Close range: try cross detection first (that's what's actually
                    # reliable at this range), falling back to blob detection only
                    # if that also fails.
                    _SEARCH_EXPAND = 2.5
                    ex_w = min(mw, int(bw * _SEARCH_EXPAND))
                    ex_h = min(mh, int(bh * _SEARCH_EXPAND))
                    ex_x = max(0, min(mw - ex_w, cx - ex_w // 2))
                    ex_y = max(0, min(mh - ex_h, cy - ex_h // 2))
                    if _wt_close_range:
                        _rx, _ry, _rfound = _find_cross_centroid(frame, ex_x, ex_y, ex_w, ex_h, cx, cy)
                        if not _rfound:
                            _rx, _ry, _rfound = _refine_aim_point(frame, ex_x, ex_y, ex_w, ex_h, cx, cy)
                    else:
                        _rx, _ry, _rfound = _refine_aim_point(frame, ex_x, ex_y, ex_w, ex_h, cx, cy)
                    if _rfound:
                        nx = max(0, min(mw - bw, _rx - bw // 2))
                        ny = max(0, min(mh - bh, _ry - bh // 2))
                        xb = int(nx * sx_m2l); yb = int(ny * sy_m2l)
                        wb = max(2, int(bw * sx_m2l)); hb = max(2, int(bh * sy_m2l))
                        state.tracker = create_gts_tracker(state.bMoovingTgt)
                        state.tracker.init(lores_frame, (xb, yb, wb, hb))
                        state.last_init_source = "drift_recovery"
                        state.bbox = (nx, ny, bw, bh)
                        _last_tracker_id = id(state.tracker)
                        _tq_needs_init   = True
                        _tq_monitor.reset()
                        _log_session_id += 1
                        _log_session_source = "drift_recovery"
                        _log_prev_cx = _log_prev_cy = None
                        _no_aim_streak = 0
                        _aim_smooth = None
                        print(f"[TQ]   Recovered onto target near ({_rx}, {_ry})")
                    else:
                        print("[TQ]   Nothing found nearby — tracking broken, re-select target")
                        state.tracking = False
                        state.tracker  = None
                        _tq_monitor.reset()
                        # Skip the rest of the draw block — show lost message instead
                        cv2.putText(frame, "Drift — re-select target", (10, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Only drive the drone when quality is sufficient
                    if tq >= TrackingQualityMonitor.SCORE_UNCERTAIN:
                        _mav_x, _mav_y = pitch_norm, yaw_norm
                        _px4_pitch_err = -norm_dy * _VFOV_RAD * _PX4_RATE_GAIN
                        _px4_yaw_err   =  norm_dx * _HFOV_RAD * _PX4_RATE_GAIN

                    # Box color encodes quality level
                    if tq >= TrackingQualityMonitor.SCORE_GOOD:
                        box_color = (0, 200, 0)      # green  — good
                    elif tq >= TrackingQualityMonitor.SCORE_UNCERTAIN:
                        box_color = (0, 140, 255)    # orange — uncertain, still sending
                    else:
                        bad_left  = TrackingQualityMonitor.BAD_FRAMES_LIMIT - _tq_monitor.bad_frames
                        box_color = (0, 0, 255)      # red    — unstable, counting down

                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 2)
                    cv2.line(frame, (aim_cx - 20, aim_cy), (aim_cx + 20, aim_cy), (0, 0, 255), 1)
                    cv2.line(frame, (aim_cx, aim_cy - 20), (aim_cx, aim_cy + 20), (0, 0, 255), 1)

                _cdx = (cx - _log_prev_cx) if _log_prev_cx is not None else None
                _cdy = (cy - _log_prev_cy) if _log_prev_cy is not None else None
                _log_baseline({
                    "wall_ts": time.time(), "session_id": _log_session_id,
                    "session_source": _log_session_source, "frame_gen": _last_frame_gen,
                    "main_w": mw, "main_h": mh, "lores_w": lw, "lores_h": lh,
                    "success": 1, "cx": cx, "cy": cy, "bbox_w": bw, "bbox_h": bh,
                    "center_dx": _cdx, "center_dy": _cdy,
                    "center_dist": math.hypot(_cdx, _cdy) if _cdx is not None else None,
                    "update_ms": round(_update_ms, 3), "tq_score": round(tq, 4),
                    "bad_frames": _bad_frames_for_log, "drift_event": int(_drift_event),
                    "inst_fps": round(_est_fps, 2) if _est_fps else None,
                })
                _log_prev_cx, _log_prev_cy = cx, cy
            else:
                _tq_monitor.reset()
                cv2.putText(frame, "Tracking lost", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                _log_baseline({
                    "wall_ts": time.time(), "session_id": _log_session_id,
                    "session_source": _log_session_source, "frame_gen": _last_frame_gen,
                    "main_w": mw, "main_h": mh, "lores_w": lw, "lores_h": lh,
                    "success": 0, "cx": None, "cy": None, "bbox_w": None, "bbox_h": None,
                    "center_dx": None, "center_dy": None, "center_dist": None,
                    "update_ms": round(_update_ms, 3), "tq_score": None,
                    "bad_frames": None, "drift_event": 0,
                    "inst_fps": round(_est_fps, 2) if _est_fps else None,
                })
                _log_prev_cx = _log_prev_cy = None
        except Exception as e:
            print(f"[ERROR] Tracker update failed: {e}")
            state.tracking = False

    if AUTOPILOT == "px4":
        # PX4's OFFBOARD mode auto-exits if the setpoint stream stops even
        # briefly, so this must run every frame regardless of tracking state
        # — untracked/low-quality frames fall back to the neutral-hold
        # sentinel set above, keeping the stream alive without commanding
        # a real correction.
        _px4_thrust = 0.5 if mavlink_client._launched else 0.0
        mavlink_client.send_attitude_target(_px4_pitch_err, _px4_yaw_err, thrust=_px4_thrust)
    else:
        is_tracking = (_mav_x != 100.0)
        mavlink_client.send_vision_error(_mav_x, _mav_y, is_tracking)

    # Write to file if in record mode (timed) or live toggle recording
    if args.mode == 'record' and record_queue is not None:
        if not record_queue.full():
            record_queue.put_nowait(frame.tobytes())
        if args.duration and (time.time() - record_start_time >= args.duration):
            print("[INFO] Reached recording duration, exiting.")
            break

    if _recording and _rec_queue is not None:
        if not _rec_queue.full():
            _rec_queue.put_nowait(frame.tobytes())
        # else: queue full → skip frame rather than block the main loop

    # FPS estimate
    now = time.time()
    dt = max(1e-6, now - _prev_ts)
    _prev_ts = now
    inst_fps = 1.0 / dt
    _est_fps = _fps_alpha * _est_fps + (1.0 - _fps_alpha) * inst_fps if _est_fps > 0 else inst_fps

    # Overlay text
    _sec = int(now)
    stamp = time.strftime('%H:%M:%S', time.localtime(_sec)) + f'.{int((now - _sec) * 1000):03d}'
    overlay1 = f"{stamp}"
    overlay2 = f"MAIN {mw}x{mh} | TRACK {lw}x{lh} | {int(_est_fps)} FPS"
    cv2.putText(frame, overlay1, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, overlay2, (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    # Launched indicator
    if mavlink_client._launched:
        cv2.putText(frame, "Launched", (8, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Time since launch — same _launch_ts used by the white-target
    # auto-disable timer, so this is exactly what that's counting against.
    # Offset 100px down/left from the top-right corner: GCS-side video
    # stabilization crops/shifts the display, hiding text placed right at
    # the frame edge most of the time.
    if _launch_ts is not None:
        _t_since = f"T+{now - _launch_ts:5.1f}s"
        (_tw, _th), _ = cv2.getTextSize(_t_since, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, _t_since, (mw - _tw - 8 - 100, _th + 8 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

    # REC indicator — blinking dot + elapsed time (only when live-toggle recording is active)
    if _recording and _rec_start_time is not None:
        rec_elapsed = now - _rec_start_time
        rec_text = f"REC {int(rec_elapsed//60):02d}:{int(rec_elapsed%60):02d}"
        # Blink: show dot every other second
        if int(now) % 2 == 0:
            cv2.circle(frame, (mw - 20, 20), 8, (0, 0, 255), -1)
        cv2.putText(frame, rec_text, (mw - 130, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

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

    # Publish final frame (tagged with the raw capture's frame_gen so
    # consumers/GCS can correlate a displayed frame back to a specific
    # captured one — see _get_frame_history above)
    frame_buffer.put(frame, gen=_last_frame_gen)

    # Local window
    if SHOW_LOCAL:
        cv2.imshow("Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        state.tracking = False; state.bbox = None; state.tracker = None
        print("[INFO] Tracker reset from Pi")
    elif key == ord('l'):
        _launch_fn(not mavlink_client._launched)
    elif args.mode == 'live':
        if key == ord('x'):   _cycle_main(+1)
        elif key == ord('z'): _cycle_main(-1)
        elif key == ord('v'): _cycle_lores(+1)
        elif key == ord('c'): _cycle_lores(-1)
        elif key == ord('o'):
            if _recording:
                _stop_recording()
            else:
                _start_recording()
    elif args.mode == 'playback':
        # "Ended" overlay — shown on top of the frozen last frame
        if _playback_ended:
            cv2.putText(frame, "Playback ended", (10, mh // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "R=restart  O=open new file", (10, mh // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
            if SHOW_LOCAL:
                cv2.imshow("Tracker", frame)

        if key == ord('r') and _playback_ended:
            _restart_playback()                      # replay current file from start
        elif key == ord('o'):
            new_path = _choose_video_file()
            if new_path:
                _restart_playback(new_path)          # switch to new file
        elif key == ord('f'):
            with playback_ctrl_lock: playback_rate = min(playback_rate * 2.0, 8.0)
            print(f"[PLAYBACK] Speed {playback_rate:.1f}×")
        elif key == ord('s'):
            with playback_ctrl_lock: playback_rate = max(playback_rate / 2.0, 0.25)
            print(f"[PLAYBACK] Speed {playback_rate:.2f}×")
        elif key == ord('1'):
            with playback_ctrl_lock: playback_rate = 1.0
            print("[PLAYBACK] Speed reset to 1×")
        elif key == ord('j'):
            with playback_ctrl_lock: seek_to_msec = max(0, playback_pos_ms - 5000)
            if _playback_ended: _restart_playback()  # slider seek also restarts if ended
            print(f"[PLAYBACK] Seek −5 s")
        elif key == ord('k'):
            with playback_ctrl_lock: seek_to_msec = playback_pos_ms + 5000
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
if args.mode == 'record' and record_queue is not None:
    record_queue.put(None)
    record_thread.join()
# Finalize any live-toggle recording that was still active when we exited
_stop_recording()
if _baseline_queue is not None:
    _baseline_queue.put(None)
    _baseline_thread.join(timeout=1.0)
