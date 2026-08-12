#!/usr/bin/env python3
"""
Mahat GCS — Ground Control Station
Receives UDP video from the Pi tracker and controls it via Flask API.
The displayed video is digitally stabilized (whole-frame shake cancellation)
to make it easier to aim; this is display-only — click coordinates are
mapped back to the Pi's real, unwarped frame before being sent.

Usage:
    python gcs.py --pi 192.168.1.100
    python gcs.py          # reads gcs_ip from config.toml if present

Mouse  : click on video, release to select tracking target
         left-click on buttons → same as keyboard shortcuts

Keyboard shortcuts (work whether or not the mouse is in the window):
    R        Reset tracker
    S        Stop tracking
    L        Toggle launch
    M        Toggle Fixed / Moving target mode
    Arrows   Nudge target (5 px)
    X / Z    Cycle MAIN resolution  + / −
    V / C    Cycle TRACK resolution + / −
    F        Toggle Pi camera FPS (idle/power-save ↔ full)
    P        Toggle Pi recording
    O        Toggle local (GCS) recording
    Q        Quit

Gamepad (optional): connect a PS4/PS5 controller (USB or Bluetooth) before
launching — press the Square/rectangle button to select the target at the
current mouse position, same as releasing a mouse-drag.
"""

import argparse
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # pygame is only used for joystick input, no window needed
try:
    import pygame
except ImportError:
    pygame = None

# ── Config ────────────────────────────────────────────────────────────────────

def _load_toml():
    cfg_path = Path(__file__).parent / "config.toml"
    if cfg_path.exists():
        try:
            import tomllib
            with open(cfg_path, "rb") as f:
                cfg = tomllib.load(f)
            iface = cfg["network"]["interface"]
            return cfg["network"][iface]["bind_ip"]   # Pi's IP, not gcs_ip (that's the Mac)
        except Exception:
            pass
    return None

parser = argparse.ArgumentParser(description="Mahat GCS client")
parser.add_argument("--pi",   default=None,  help="Pi IP (overrides config.toml)")
parser.add_argument("--port", type=int, default=5000, help="Flask API port  (default 5000)")
parser.add_argument("--udp",  type=int, default=5600, help="UDP video port  (default 5600)")
args = parser.parse_args()

PI_IP    = args.pi or _load_toml() or "192.168.1.100"
FLASK    = f"http://{PI_IP}:{args.port}"
UDP_PORT = args.udp

CMD_PORT = 5601

# UDP socket for sending commands directly to the Pi (unicast)
import json as _json
_cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"[GCS] Pi={PI_IP}  video={UDP_PORT}  cmd={CMD_PORT}")

# ── GCS discovery heartbeat ───────────────────────────────────────────────────
# Send a small "hello" to the Pi every 3 s so the Pi learns our IP dynamically.
# The Pi's command listener reads the sender address from every incoming packet
# and updates its GCS_IP — no hardcoded IP needed on either side.

def _heartbeat_sender():
    import time
    msg = _json.dumps({}).encode()   # empty payload — no endpoint → Pi ignores body
    while True:
        try:
            _cmd_sock.sendto(msg, (PI_IP, CMD_PORT))
        except Exception:
            pass
        time.sleep(3)

threading.Thread(target=_heartbeat_sender, daemon=True).start()

# ── Layout constants ──────────────────────────────────────────────────────────

PANEL_W     = 210     # right-side button panel width  (px)
PANEL_MIN_H = 650     # minimum canvas height so all buttons fit
DISPLAY_W   = 1200    # video is always stretched to this width for display

# ── Shared state ──────────────────────────────────────────────────────────────

launched        = False
moving_tgt      = False
_pi_recording   = False   # Pi-side recording state (optimistic: toggled on each command)
_local_recording = False  # GCS-side recording state
_local_writer    = None   # cv2.VideoWriter when local recording is active
cam_active      = False   # Pi camera fps state: False=idle (power-save), True=full fps
_cpu_percent    = None    # Pi CPU usage %, polled from /status; None until first poll
_cpu_temp_c     = None    # Pi SoC temperature °C, polled from /status; None until first poll
_pi_tracking    = False   # Pi-side tracking state, polled from /status
_frame_gen      = None    # Pi's frame counter for the currently displayed frame; echoed by select_point()
_status    = ""
_status_ts = 0.0
_mouse_pos = [0, 0]   # updated by mouse callback; used for hover highlight
_press_on_video = False   # True from LBUTTONDOWN-on-video until release; commits select_point then
_quit         = threading.Event()  # set to break the main loop from any thread
_confirm_quit = False             # True while the "are you sure?" overlay is shown
_CONFIRM_YES  = None              # (x, y, w, h) of the Yes button in the overlay
_CONFIRM_NO   = None              # (x, y, w, h) of the No  button in the overlay

# FPS baseline for performance comparison
_fps_baseline   = None   # FPS snapshot taken when recording starts
_fps_before_rec = 0.0    # smoothed FPS just before recording started

# ── Command channel (UDP broadcast → works through AP isolation) ──────────────

def _post(endpoint, **data):
    """Send command to Pi via UDP unicast. Non-blocking, no TCP needed."""
    msg = _json.dumps({"endpoint": endpoint, **data}).encode()
    try:
        _cmd_sock.sendto(msg, (PI_IP, CMD_PORT))
    except Exception as e:
        set_status(f"CMD error: {e}")

def _get(endpoint):
    """Try Flask HTTP for read-only status; returns {} if unreachable."""
    try:
        return requests.get(f"{FLASK}/{endpoint}", timeout=1).json()
    except Exception:
        return {}

def set_status(msg, *, log=True):
    global _status, _status_ts
    _status, _status_ts = msg, time.time()
    if log:
        print(f"[GCS] {msg}")

def send_cmd(cmd):
    _post("command", cmd=cmd)
    set_status({"r": "Reset", "s": "Stop", "q": "Quit"}.get(cmd, cmd))

def quit_gcs():
    """Close the GCS window and tell the Pi to quit."""
    send_cmd('q')
    _quit.set()

def ask_quit():
    global _confirm_quit
    _confirm_quit = True

def _draw_confirm_overlay(canvas):
    """Draw a semi-transparent 'Are you sure?' dialog over the canvas."""
    global _CONFIRM_YES, _CONFIRM_NO

    ch, cw = canvas.shape[:2]

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (cw, ch), (0, 0, 0), -1)
    cv2.addWeighted(canvas, 0.5, overlay, 0.5, 0, canvas)

    dw, dh = 380, 150
    dx = (cw - dw) // 2
    dy = (ch - dh) // 2
    cv2.rectangle(canvas, (dx, dy), (dx + dw, dy + dh), (50, 50, 50), -1)
    cv2.rectangle(canvas, (dx, dy), (dx + dw, dy + dh), (150, 150, 150), 2)

    for txt, scale, oy, color in [
        ("Quit GCS?",                    0.65, 38,  (255, 255, 255)),
        ("This will also quit the Pi.",  0.46, 64,  (180, 180, 180)),
        ("Y = confirm   Esc = cancel",   0.40, 84,  (130, 130, 130)),
    ]:
        (tw, th), _ = cv2.getTextSize(txt, _FONT, scale, 1)
        cv2.putText(canvas, txt, (dx + (dw - tw) // 2, dy + oy),
                    _FONT, scale, color, 1, cv2.LINE_AA)

    bw, bh = 110, 34
    yes_x = dx + dw // 2 - bw - 10
    no_x  = dx + dw // 2 + 10
    by    = dy + dh - bh - 14

    _CONFIRM_YES = (yes_x, by, bw, bh)
    _CONFIRM_NO  = (no_x,  by, bw, bh)

    mx, my = _mouse_pos
    for (bx2, by2, bw2, bh2), lbl, col in [
        (_CONFIRM_YES, "Yes",  (40, 40, 170)),
        (_CONFIRM_NO,  "No",   (60, 60, 60)),
    ]:
        hover = bx2 <= mx < bx2 + bw2 and by2 <= my < by2 + bh2
        c = tuple(min(255, v + 45) for v in col) if hover else col
        cv2.rectangle(canvas, (bx2, by2), (bx2 + bw2, by2 + bh2), c, -1)
        cv2.rectangle(canvas, (bx2, by2), (bx2 + bw2, by2 + bh2), (110, 110, 110), 1)
        (tw, th), _ = cv2.getTextSize(lbl, _FONT, 0.50, 1)
        tx = bx2 + (bw2 - tw) // 2
        ty = by2 + (bh2 + th) // 2
        cv2.putText(canvas, lbl, (tx, ty), _FONT, 0.50, (0, 0, 0),   3, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (tx, ty), _FONT, 0.50, (240,240,240), 1, cv2.LINE_AA)

def send_launch():
    global launched
    launched = not launched
    _post("launch", state=1 if launched else 0)
    set_status("LAUNCHED" if launched else "Launch reset")

def toggle_target():
    global moving_tgt
    moving_tgt = not moving_tgt
    _post("set_target_mode", bMoovingTgt=1 if moving_tgt else 0)
    set_status(f"Mode: {'MOVING' if moving_tgt else 'FIXED'}")

def nudge(dx, dy):
    _post("nudge", dx=dx, dy=dy)
    set_status(f"Nudge ({dx:+d}, {dy:+d})", log=False)

def select_point(x, y):
    # Send normalized coords (0-1) so Pi maps correctly regardless of stream resolution
    nx = round(x / _cur_video_w, 6)
    ny = round(y / _cur_video_h, 6)
    # Echo back which frame this click was actually seen on (STABILIZATION.md,
    # "stale-click problem") — the Pi buffers recent frames and inits/replays
    # tracking against the one the operator actually clicked, instead of
    # whatever's live when this arrives. Omit entirely when unknown — the Pi
    # treats an absent frame_gen as "use the current live frame".
    extra = {"frame_gen": _frame_gen} if _frame_gen is not None else {}
    _post("select_point", nx=nx, ny=ny, **extra)
    set_status(f"Selected ({x}, {y})")

def lock_target(x, y):
    """Commit (x, y) — the current mouse position — as the tracking target.
    Triggered by the gamepad's Square/rectangle button as an alternative to
    releasing a mouse-drag."""
    global _pi_tracking
    select_point(x, y)
    _pi_tracking = True   # optimistic — confirmed/corrected by the next /status poll

def toggle_local_record(frame_w=640, frame_h=480):
    global _local_recording, _local_writer
    if _local_recording:
        if _local_writer is not None:
            _local_writer.release()
            _local_writer = None
        _local_recording = False
        set_status("Local REC stopped")
    else:
        ts    = time.strftime("%Y%m%d_%H%M%S")
        fname = f"gcs_rec_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        rec_fps = est_fps_ref[0] if est_fps_ref[0] >= 1.0 else 20.0
        _local_writer   = cv2.VideoWriter(fname, fourcc, rec_fps, (frame_w, frame_h))
        _local_recording = True
        set_status(f"Local REC → {fname}")

def toggle_fps():
    """Explicitly tell the Pi to switch camera capture between idle (power-save) and full fps."""
    global cam_active
    cam_active = not cam_active
    _post("set_fps", active=1 if cam_active else 0)
    set_status("Pi camera: FULL FPS" if cam_active else "Pi camera: IDLE (power-save)")

def _status_poller():
    """Background: poll /status every 2s to keep cam_active/_cpu_percent/_cpu_temp_c/
    _pi_tracking fresh even when nothing else is triggering a request (e.g. after Pi restarts)."""
    global cam_active, _cpu_percent, _cpu_temp_c, _pi_tracking
    while not _quit.is_set():
        data = _get("status")
        if data:
            cam_active   = data.get("active_fps", cam_active)
            _cpu_percent = data.get("cpu_percent", _cpu_percent)
            _cpu_temp_c  = data.get("cpu_temp", _cpu_temp_c)
            _pi_tracking = data.get("tracking", _pi_tracking)
        time.sleep(2.0)

threading.Thread(target=_status_poller, daemon=True).start()

def cycle_main(delta):
    _post("cycle_main", delta=delta)
    set_status(f"MAIN {'up' if delta > 0 else 'down'}")

def cycle_lores(delta):
    _post("cycle_lores", delta=delta)
    set_status(f"TRACK {'up' if delta > 0 else 'down'}")

# ── Whole-frame stabilization ──────────────────────────────────────────────────
#
# Cancels camera shake/vibration so the whole displayed picture holds still.
# Estimates the frame-to-frame global shift (phase correlation on a
# downsampled frame — cheap), accumulates it into a raw trajectory, low-pass
# filters that trajectory to get the "intended" slow motion, and shifts each
# frame by the difference (raw - smoothed) so fast jitter cancels out while
# genuine panning still comes through. Translation only (no rotation) — a
# rotation-aware version was tried and reverted after repeated real-world
# failures (runaway drift, scale creep); this simpler version is reliable.
#
# select_point() must be called with RAW-frame coordinates (matching what the
# Pi's own live frame looks like), so _to_raw_coords() undoes this shift on
# whatever pixel was clicked in the now-stabilized display.

_STAB_DOWNSCALE  = 4     # phase correlation runs at 1/this resolution, for speed
_STAB_ALPHA_MIN  = 0.01
_STAB_ALPHA_MAX  = 0.50
_STAB_ALPHA_STEP = 0.01

_stab_enabled    = True
_STAB_SLOW_ALPHA = 0.05  # how fast the "intended" trajectory adapts (lower = more shake removed)
_STAB_MAX_SHIFT  = 75    # clamp (px) so one bad correlation can't wildly warp the frame

_stab_prev_gray  = None
_stab_hann       = None
_stab_traj       = np.array([0.0, 0.0])   # cumulative raw (unfiltered) trajectory
_stab_smooth     = np.array([0.0, 0.0])   # low-pass filtered trajectory
_stab_correction = (0.0, 0.0)             # (cx, cy) applied to the currently displayed frame

def _reset_stabilizer():
    global _stab_prev_gray, _stab_traj, _stab_smooth, _stab_correction
    _stab_prev_gray  = None
    _stab_traj       = np.array([0.0, 0.0])
    _stab_smooth     = np.array([0.0, 0.0])
    _stab_correction = (0.0, 0.0)

def toggle_stabilization():
    global _stab_enabled
    _stab_enabled = not _stab_enabled
    _reset_stabilizer()   # avoid a jump from stale trajectory state when re-enabled
    set_status(f"Stabilization: {'ON' if _stab_enabled else 'OFF'}")

def cycle_stab_alpha(delta):
    """Adjust how aggressively shake is removed. Lower alpha = more smoothing
    (removes more shake, but lags more behind genuine panning)."""
    global _STAB_SLOW_ALPHA
    _STAB_SLOW_ALPHA = round(max(_STAB_ALPHA_MIN, min(_STAB_ALPHA_MAX,
                                  _STAB_SLOW_ALPHA + delta * _STAB_ALPHA_STEP)), 3)
    set_status(f"Stabilization smoothing: {_STAB_SLOW_ALPHA:.2f} "
               f"({'more shake removed' if delta < 0 else 'less lag'})")

def _stabilize(frame, is_new_frame):
    """Warp `frame` by the current shake-cancelling correction, recomputing
    that correction only when a genuinely new frame has arrived."""
    global _stab_prev_gray, _stab_hann, _stab_traj, _stab_smooth, _stab_correction
    h, w = frame.shape[:2]

    if is_new_frame:
        sw, sh = max(1, w // _STAB_DOWNSCALE), max(1, h // _STAB_DOWNSCALE)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if _stab_prev_gray is not None and _stab_prev_gray.shape == gray.shape:
            if _stab_hann is None or _stab_hann.shape != (sh, sw):
                _stab_hann = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(_stab_prev_gray, gray, _stab_hann)
            if resp > 0.05:   # low confidence (e.g. near-blank frame) → skip this sample
                dx = max(-_STAB_MAX_SHIFT, min(_STAB_MAX_SHIFT, dx * _STAB_DOWNSCALE))
                dy = max(-_STAB_MAX_SHIFT, min(_STAB_MAX_SHIFT, dy * _STAB_DOWNSCALE))
                _stab_traj += (dx, dy)

        _stab_prev_gray = gray
        _stab_smooth += _STAB_SLOW_ALPHA * (_stab_traj - _stab_smooth)
        corr = np.clip(_stab_smooth - _stab_traj, -_STAB_MAX_SHIFT, _STAB_MAX_SHIFT)
        _stab_correction = (float(corr[0]), float(corr[1]))

    cx, cy = _stab_correction
    M = np.float32([[1, 0, cx], [0, 1, cy]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def _to_raw_coords(px, py):
    """Undo the current stabilization shift — use before select_point()/lock_target()
    so the Pi (which sees its own unwarped live frame) gets the right pixel."""
    cx, cy = _stab_correction
    rx = max(0, min(_cur_video_w - 1, int(round(px - cx))))
    ry = max(0, min(_cur_video_h - 1, int(round(py - cy))))
    return rx, ry

def toggle_pi_record(cur_fps=0.0):
    """Tell the Pi to start or stop recording. Tracks state optimistically."""
    global _pi_recording, _fps_baseline, _fps_before_rec
    if _pi_recording:
        _post("toggle_record")
        _pi_recording = False
        diff = cur_fps - _fps_baseline if _fps_baseline is not None else None
        if diff is not None:
            sign  = "+" if diff >= 0 else ""
            color_hint = "▲" if diff > 0.5 else ("▼" if diff < -0.5 else "≈")
            set_status(f"Pi REC stopped  |  FPS before {_fps_baseline:.1f} → now {cur_fps:.1f}  ({sign}{diff:.1f}) {color_hint}")
        else:
            set_status("Pi REC stopped")
        _fps_baseline = None
    else:
        _fps_before_rec = cur_fps
        _fps_baseline   = cur_fps      # snapshot FPS at the moment recording starts
        _post("toggle_record")
        _pi_recording = True
        set_status(f"Pi REC started  (baseline FPS: {cur_fps:.1f})")


# ── Button widget ─────────────────────────────────────────────────────────────

_FONT = cv2.FONT_HERSHEY_SIMPLEX

class Button:
    """
    A clickable rectangle drawn on an OpenCV image.
    Both `label` and `bg` can be plain values or callables so toggle buttons
    update their text/colour automatically every frame.
    """
    def __init__(self, label, x, y, w, h, action, bg=(55, 55, 55)):
        self._label  = label   # str  or  () -> str
        self._bg     = bg      # tuple or () -> tuple
        self.x, self.y, self.w, self.h = x, y, w, h
        self.action  = action

    @property
    def label(self):
        return self._label() if callable(self._label) else self._label

    @property
    def bg(self):
        return self._bg() if callable(self._bg) else self._bg

    def hit(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img, hover=False):
        col = tuple(min(255, c + 45) for c in self.bg) if hover else self.bg
        cv2.rectangle(img, (self.x, self.y),
                      (self.x + self.w, self.y + self.h), col, -1)
        cv2.rectangle(img, (self.x, self.y),
                      (self.x + self.w, self.y + self.h), (110, 110, 110), 1)
        lbl = self.label
        scale = 0.48
        (tw, th), _ = cv2.getTextSize(lbl, _FONT, scale, 1)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        # Black outline + white text for readability on any bg
        cv2.putText(img, lbl, (tx, ty), _FONT, scale, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, lbl, (tx, ty), _FONT, scale, (240,240,240), 1, cv2.LINE_AA)


# ── Button layout ─────────────────────────────────────────────────────────────

_buttons: list[Button] = []
_cur_video_w = 0    # rebuilt whenever video width changes
est_fps_ref  = [0.0]  # [0] updated each frame; readable from button lambdas


def _build_buttons(vx: int):
    """
    Populate _buttons for a panel that starts at x=vx.
    Called once at startup (vx=640) and again if the stream resolution changes.
    """
    _buttons.clear()

    bw   = PANEL_W - 16          # button width (8 px margin each side)
    bx   = vx + 8                # button left edge
    y    = 12

    def btn(label, h, action, bg):
        _buttons.append(Button(label, bx, y, bw, h, action, bg))

    def btn2(l1, l2, h, a1, a2, bg):
        """Two equal-width buttons side by side."""
        w2 = (bw - 4) // 2
        _buttons.append(Button(l1, bx,          y, w2, h, a1, bg))
        _buttons.append(Button(l2, bx + w2 + 4, y, w2, h, a2, bg))

    # ── Main controls ─────────────────────────────────────────────────────
    btn(
        lambda: "LAUNCHED" if launched else "Launch",
        44, send_launch,
        lambda: (30, 140, 50) if launched else (30, 90, 200),
    )
    y += 52

    btn2("Reset", "Stop",  36,
         lambda: send_cmd('r'), lambda: send_cmd('s'),
         (35, 120, 35))
    y += 44

    btn("Quit", 36, ask_quit, (40, 40, 170))
    y += 52

    # ── Target mode ────────────────────────────────────────────────────────
    btn(
        lambda: f"Target: {'MOVING' if moving_tgt else 'FIXED'}",
        36, toggle_target,
        lambda: (140, 80, 20) if moving_tgt else (60, 80, 140),
    )
    y += 44

    # ── Pi Record ──────────────────────────────────────────────────────────
    btn(
        lambda: "■ Pi REC" if _pi_recording else "● Pi REC",
        36, lambda: toggle_pi_record(est_fps_ref[0]),
        lambda: (30, 30, 180) if _pi_recording else (35, 120, 35),
    )
    y += 44

    # ── Local Record ───────────────────────────────────────────────────────
    btn(
        lambda: "■ Local REC" if _local_recording else "● Local REC",
        36, lambda: toggle_local_record(_cur_video_w, _cur_video_h),
        lambda: (140, 30, 30) if _local_recording else (35, 120, 35),
    )
    y += 44

    # ── Pi camera FPS (idle/power-save ↔ full) ─────────────────────────────
    btn(
        lambda: "FPS: FULL" if cam_active else "FPS: IDLE",
        36, toggle_fps,
        lambda: (30, 140, 50) if cam_active else (90, 90, 30),
    )
    y += 44

    # ── Video stabilization ─────────────────────────────────────────────────
    btn(
        lambda: "STAB: ON" if _stab_enabled else "STAB: OFF",
        36, toggle_stabilization,
        lambda: (30, 140, 50) if _stab_enabled else (90, 90, 30),
    )
    y += 40
    btn2("STAB -", "STAB +", 32,
         lambda: cycle_stab_alpha(-1), lambda: cycle_stab_alpha(+1), (55, 55, 85))
    y += 44

    # ── D-pad ──────────────────────────────────────────────────────────────
    dw  = dh  = 46
    dpx = vx + (PANEL_W - dw * 3) // 2    # centre the 3-wide grid in panel

    _buttons.append(Button("^",  dpx + dw,      y,          dw, dh, lambda: nudge( 0, -5), (75,75,75)))
    _buttons.append(Button("<",  dpx,            y + dh,     dw, dh, lambda: nudge(-5,  0), (75,75,75)))
    _buttons.append(Button(">",  dpx + dw*2,     y + dh,     dw, dh, lambda: nudge( 5,  0), (75,75,75)))
    _buttons.append(Button("v",  dpx + dw,       y + dh*2,   dw, dh, lambda: nudge( 0,  5), (75,75,75)))
    y += dh * 3 + 16

    # ── Resolution cycling ─────────────────────────────────────────────────
    btn2("MAIN -", "MAIN +",   32,
         lambda: cycle_main(-1), lambda: cycle_main(+1), (55, 55, 85))
    y += 40
    btn2("TRACK -", "TRACK +", 32,
         lambda: cycle_lores(-1), lambda: cycle_lores(+1), (55, 55, 85))


# Build with default 640-wide video so buttons exist before stream arrives
_cur_video_w = 640
_cur_video_h = 480
_build_buttons(640)


# ── HUD overlay (drawn on video portion only) ─────────────────────────────────

def draw_hud(frame, fps):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)

    mode_col   = (60, 200, 60)  if not moving_tgt else (60, 160, 255)
    launch_col = (255,255,255)  if not launched    else (60, 160, 255)

    def txt(msg, pos, color=(240,240,240), scale=0.55):
        cv2.putText(frame, msg, pos, _FONT, scale, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, msg, pos, _FONT, scale, color,   1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(msg, _FONT, scale, 1)
        return tw

    x = 8
    GAP = 14
    x += txt(f"FPS {fps:4.1f}", (x, 25)) + GAP

    if _cpu_percent is not None:
        cpu_col = (60, 200, 60) if _cpu_percent < 50 else \
                  (60, 160, 255) if _cpu_percent < 80 else (60, 60, 220)
        x += txt(f"CPU {_cpu_percent:3.0f}%", (x, 25), color=cpu_col) + GAP

    if _cpu_temp_c is not None:
        # Pi SoC starts soft-throttling around ~80C, hard throttle ~85C
        temp_col = (60, 200, 60) if _cpu_temp_c < 60 else \
                   (60, 160, 255) if _cpu_temp_c < 75 else (60, 60, 220)
        x += txt(f"{_cpu_temp_c:4.1f}C", (x, 25), color=temp_col) + GAP

    x += txt("FULL" if cam_active else "IDLE", (x, 25),
             color=(60, 200, 60) if cam_active else (150, 150, 150)) + GAP

    x += txt(f"{'MOVING' if moving_tgt else 'FIXED'}", (x, 25), color=mode_col) + GAP
    txt("LAUNCHED" if launched else "READY", (x, 25), color=launch_col)

    # REC indicators — blinking every second
    rec_x = w - 16
    if _local_recording:
        dot_col = (40, 220, 40) if int(time.time()) % 2 == 0 else (100, 255, 100)
        cv2.circle(frame, (rec_x, 18), 7, dot_col, -1)
        txt("Local REC", (rec_x - 100, 25), color=(100, 255, 100))
        rec_x -= 120

    if _pi_recording:
        dot_col = (40, 40, 220) if int(time.time()) % 2 == 0 else (100, 100, 255)
        cv2.circle(frame, (rec_x, 18), 7, dot_col, -1)
        if _fps_baseline is not None:
            delta = fps - _fps_baseline
            sign  = "+" if delta >= 0 else ""
            delta_col = (80, 200, 80) if delta > -0.5 else (80, 80, 220)
            txt(f"Pi REC  ({sign}{delta:.1f})", (rec_x - 139, 25), color=delta_col)
        else:
            txt("Pi REC", (rec_x - 64, 25), color=(100, 100, 255))

    # Status bar (bottom, fades after 4 s)
    age = time.time() - _status_ts
    if _status and age < 4.0:
        fade = min(1.0, (4.0 - age) / 0.5)
        bar  = frame.copy()
        cv2.rectangle(bar, (0, h - 28), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)
        col = tuple(int(c * fade) for c in (100, 255, 100))
        txt(_status, (8, h - 9), color=col)


# ── Waiting screen ────────────────────────────────────────────────────────────

def _waiting_frame(w=640, h=480):
    img = np.zeros((h, w, 3), np.uint8)
    def txt(msg, pos, scale=0.6, color=(200,200,200)):
        cv2.putText(img, msg, pos, _FONT, scale, (0,0,0),   3, cv2.LINE_AA)
        cv2.putText(img, msg, pos, _FONT, scale, color,     1, cv2.LINE_AA)
    txt(f"Waiting for UDP stream on port {UDP_PORT}",
        (max(8, w//2 - 220), h//2 - 14))
    txt(f"Pi: {PI_IP}",
        (max(8, w//2 - 60),  h//2 + 18), scale=0.5, color=(140,140,140))
    return img


# ── Arrow-key detection (cross-platform) ─────────────────────────────────────

_ARROW = {
    65362:(0,-1), 65364:(0,1), 65361:(-1,0), 65363:(1,0),   # Linux X11
    63232:(0,-1), 63233:(0,1), 63234:(-1,0), 63235:(1,0),   # macOS
    2490368:(0,-1), 2621440:(0,1), 2424832:(-1,0), 2555904:(1,0),  # Windows
    82:(0,-1), 84:(0,1), 81:(-1,0), 83:(1,0),               # Linux fallback
}


# ── Gamepad (optional) — Square/rectangle button locks the target ─────────────
#
# Button index is a best guess (raw HID face-button order on a PS4 controller
# over Bluetooth) — unverified; adjust _LOCK_BUTTON if the wrong button fires.

_LOCK_BUTTON = 2   # Square/rectangle face button

class _Gamepad:
    def __init__(self):
        self._joy = None
        self._prev_lock = False
        if pygame is None:
            return
        try:
            pygame.display.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self._joy = pygame.joystick.Joystick(0)
                self._joy.init()
                print(f"[Gamepad] connected: {self._joy.get_name()}")
        except Exception as e:
            print(f"[Gamepad] unavailable: {e}")

    def lock_pressed(self):
        """True on the frame the Square/rectangle button transitions to pressed
        (edge-triggered — holding it down doesn't fire repeatedly)."""
        if self._joy is None:
            return False
        pygame.event.pump()
        cur = self._joy.get_numbuttons() > _LOCK_BUTTON and self._joy.get_button(_LOCK_BUTTON)
        edge = cur and not self._prev_lock
        self._prev_lock = cur
        return edge


# ── Live capture — background reader thread ───────────────────────────────────
#
# Problem: cv2.VideoCapture.read() returns frames in decode order from an internal
# queue. When the main loop is busy (drawing, key handling), that queue grows and
# read() starts returning frames from seconds ago — causing the delay you saw.
#
# Fix: a daemon thread that drains the queue as fast as the decoder produces frames
# and only ever keeps the most recent one. The main loop always gets "now".
#
# The Pi sends JPEG-encoded frames as individual UDP datagrams (broadcast).
# Each datagram = one complete JPEG image — no stream reassembly needed.

_JPEG_SOI = b"\xff\xd8"   # JPEG Start-Of-Image marker — always the first 2 bytes of a JPEG

def _split_frame_gen(data: bytes):
    """Split a video datagram into (frame_gen_or_None, jpeg_bytes).

    Wire format (see STABILIZATION.md, "stale-click problem"): the Pi may
    prepend a 4-byte big-endian frame_gen counter before the JPEG bytes, so
    the GCS can echo it back with select_point() and the Pi can look up the
    exact frame that was clicked on, instead of using whatever's live at
    request time. Detected via the JPEG SOI marker so this stays backward
    compatible with a Pi that isn't sending the header yet.
    """
    if data[:2] == _JPEG_SOI:
        return None, data            # legacy: no header, whole payload is the JPEG
    if len(data) > 4 and data[4:6] == _JPEG_SOI:
        return struct.unpack(">I", data[:4])[0], data[4:]
    return None, data                # unrecognized — best-effort fallback

class _LiveCapture:
    def __init__(self, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)  # 1 MB
        self._sock.bind(('', port))
        self._sock.settimeout(1.0)
        self._frame     = None
        self._ok        = False
        self._frame_id  = 0
        self._frame_gen = None   # Pi-side frame counter, echoed back with select_point()
        self._lock      = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        """Receive JPEG datagrams and decode them; marks _ok=False on timeout."""
        while True:
            try:
                data, _ = self._sock.recvfrom(1 << 16)  # 65536 bytes max UDP payload
                frame_gen, jpeg = _split_frame_gen(data)
                arr   = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._frame     = frame
                        self._ok        = True
                        self._frame_id += 1
                        self._frame_gen = frame_gen
            except socket.timeout:
                with self._lock:
                    self._ok = False   # no packet for 1 s → show waiting screen
            except Exception as e:
                print(f"[UDP] recv: {e}")

    def read(self):
        """Return (ok, frame_copy, frame_id, frame_gen).  Never blocks more than the lock."""
        with self._lock:
            if self._frame is None:
                return False, None, 0, None
            return self._ok, self._frame.copy(), self._frame_id, self._frame_gen



# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    global launched, _cur_video_w, _cur_video_h, est_fps_ref, _confirm_quit, _frame_gen

    data     = _get("status")
    launched = data.get("launched", False)
    if data:
        set_status(f"Connected to {PI_IP}")
    else:
        set_status(f"Commands via UDP broadcast — waiting for video…")

    cap     = _LiveCapture(UDP_PORT)
    gamepad = _Gamepad()

    cv2.namedWindow("Mahat GCS", cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, _):
        global _confirm_quit, _press_on_video
        _mouse_pos[0], _mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            if _confirm_quit:
                if _CONFIRM_YES and _CONFIRM_YES[0] <= x < _CONFIRM_YES[0] + _CONFIRM_YES[2] \
                                and _CONFIRM_YES[1] <= y < _CONFIRM_YES[1] + _CONFIRM_YES[3]:
                    _confirm_quit = False
                    quit_gcs()
                elif _CONFIRM_NO and _CONFIRM_NO[0] <= x < _CONFIRM_NO[0] + _CONFIRM_NO[2] \
                                 and _CONFIRM_NO[1] <= y < _CONFIRM_NO[1] + _CONFIRM_NO[3]:
                    _confirm_quit = False
            elif x < _cur_video_w:
                _press_on_video = True      # commit on release, not on press
            else:
                for btn in _buttons:        # click on panel → button action
                    if btn.hit(x, y):
                        btn.action()
                        break
        elif event == cv2.EVENT_LBUTTONUP:
            if _press_on_video:
                _press_on_video = False
                fx, fy = _to_raw_coords(x, y)   # undo display-only stabilization shift
                select_point(fx, fy)            # release → track target

    cv2.setMouseCallback("Mahat GCS", on_mouse)

    last_frame_ts   = None
    last_frame_id   = 0
    last_rec_id     = 0
    est_fps         = 0.0
    FPS_A           = 0.9

    while not _quit.is_set():
        ok, frame, frame_id, frame_gen = cap.read()

        if not ok or frame is None:
            frame = _waiting_frame(_cur_video_w, _cur_video_h)
        else:
            # Stretch to DISPLAY_W regardless of stream resolution
            # so the window stays the same size even when stream is downscaled
            fh, fw = frame.shape[:2]
            if fw != DISPLAY_W:
                dh = int(fh * DISPLAY_W / fw)
                frame = cv2.resize(frame, (DISPLAY_W, dh), interpolation=cv2.INTER_LINEAR)

            is_new_frame = frame_id != last_frame_id

            # FPS — only count genuinely new UDP frames, not repeated buffer reads
            if is_new_frame:
                now = time.time()
                if last_frame_ts is not None:
                    inst    = 1.0 / max(1e-6, now - last_frame_ts)
                    est_fps = FPS_A * est_fps + (1 - FPS_A) * inst if est_fps else inst
                last_frame_ts = now
                last_frame_id = frame_id
                _frame_gen    = frame_gen

            if _stab_enabled:
                frame = _stabilize(frame, is_new_frame)

        h, w = frame.shape[:2]

        # Rebuild button layout if display width changed
        if w != _cur_video_w:
            _cur_video_w = w
            _cur_video_h = h
            _build_buttons(w)

        est_fps_ref[0] = est_fps   # share live FPS with button lambdas

        draw_hud(frame, est_fps)

        # ── Gamepad lock — Square/rectangle button selects at mouse position ─
        if gamepad.lock_pressed():
            fx, fy = _to_raw_coords(min(w - 1, _mouse_pos[0]), min(h - 1, _mouse_pos[1]))
            lock_target(fx, fy)

        # Write to local recorder (video + HUD, no panel) — only on genuinely
        # new frames, so recorded playback speed matches real time instead of
        # being stretched by the render loop re-writing the same cached frame.
        if _local_recording and _local_writer is not None and frame_id != last_rec_id:
            last_rec_id = frame_id
            fh, fw = frame.shape[:2]
            rec_frame = frame if (fw, fh) == (_cur_video_w, _cur_video_h) else \
                        cv2.resize(frame, (_cur_video_w, _cur_video_h))
            _local_writer.write(rec_frame)

        # ── Composite canvas: video left + button panel right ──────────────
        canvas_h = max(h, PANEL_MIN_H)
        canvas   = np.zeros((canvas_h, w + PANEL_W, 3), np.uint8)
        canvas[:h, :w] = frame

        # Panel background + separator line
        cv2.rectangle(canvas, (w, 0), (w + PANEL_W, canvas_h), (30, 30, 30), -1)
        cv2.line(canvas, (w, 0), (w, canvas_h), (70, 70, 70), 1)

        # Buttons (with hover highlight)
        mx, my = _mouse_pos
        for btn in _buttons:
            btn.draw(canvas, hover=btn.hit(mx, my))

        if _confirm_quit:
            _draw_confirm_overlay(canvas)

        cv2.imshow("Mahat GCS", canvas)

        # ── Key handling ───────────────────────────────────────────────────
        key = cv2.waitKeyEx(1)
        if key == -1:
            continue

        if _confirm_quit:
            k = key & 0xFF
            if k in (ord('y'), ord('Y'), 13):   # Y or Enter → confirm
                _confirm_quit = False
                quit_gcs()
            else:                               # anything else → cancel
                _confirm_quit = False
            continue

        direction = _ARROW.get(key)
        if direction:
            dx, dy = direction
            nudge(dx * 5, dy * 5)
            continue

        k = key & 0xFF
        if   k in (ord('q'), ord('Q')): ask_quit()
        elif k in (ord('r'), ord('R')): send_cmd('r')
        elif k in (ord('s'), ord('S')): send_cmd('s')
        elif k in (ord('l'), ord('L')): send_launch()
        elif k in (ord('m'), ord('M')): toggle_target()
        elif k in (ord('x'), ord('X')): cycle_main(+1)
        elif k in (ord('z'), ord('Z')): cycle_main(-1)
        elif k in (ord('v'), ord('V')): cycle_lores(+1)
        elif k in (ord('c'), ord('C')): cycle_lores(-1)
        elif k in (ord('f'), ord('F')): toggle_fps()
        elif k in (ord('p'), ord('P')): toggle_pi_record(est_fps)
        elif k in (ord('o'), ord('O')): toggle_local_record(_cur_video_w, _cur_video_h)

        if _quit.is_set():
            break

    cv2.destroyAllWindows()
    print("[GCS] Bye.")


if __name__ == "__main__":
    main()
