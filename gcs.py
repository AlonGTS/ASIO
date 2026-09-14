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
         (toggle ZOOM in the panel to show a magnifier while dragging)
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
import math
import os
import signal
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

def _load_video_mode():
    """Which capture class to use — must match tracker-so.py's video_mode."""
    cfg_path = Path(__file__).parent / "config.toml"
    if cfg_path.exists():
        try:
            import tomllib
            with open(cfg_path, "rb") as f:
                cfg = tomllib.load(f)
            return cfg["network"].get("video_mode", "jpeg_udp")
        except Exception:
            pass
    return "jpeg_udp"

def _load_flask_port():
    """Must match tracker-so.py's own [network] flask_port — a hardcoded
    default here silently drifts from whatever the Pi is actually bound to
    (e.g. config.toml sets 5050 to dodge macOS AirPlay squatting the default
    5000 for local testing), leaving /status calls hitting a dead port with
    no visible error — CPU/temp and everything else in the panel just goes
    blank with no explanation."""
    cfg_path = Path(__file__).parent / "config.toml"
    if cfg_path.exists():
        try:
            import tomllib
            with open(cfg_path, "rb") as f:
                cfg = tomllib.load(f)
            return cfg["network"].get("flask_port", 5000)
        except Exception:
            pass
    return 5000

parser = argparse.ArgumentParser(description="Mahat GCS client")
parser.add_argument("--pi",   default=None,  help="Pi IP (overrides config.toml)")
parser.add_argument("--port", type=int, default=None, help="Flask API port (overrides config.toml, default 5000)")
parser.add_argument("--udp",  type=int, default=5600, help="UDP video port  (default 5600)")
parser.add_argument("--file", default=None,
                     help="Play a local video file instead of connecting to a Pi "
                          "(offline testing — e.g. one of the gcs_rec_*.mp4 recordings). "
                          "Loops when it reaches the end. All Pi commands become no-ops.")
args = parser.parse_args()

PI_IP    = args.pi or _load_toml() or "192.168.1.100"
FLASK    = f"http://{PI_IP}:{args.port if args.port is not None else _load_flask_port()}"
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
PANEL_MIN_H = 914     # minimum canvas height so all buttons fit (was 870 — +44 for the HIRES ZOOM button)

def _screen_display_width(panel_w, fallback=1200):
    """Video display width sized to fill as much of the screen as possible
    at startup — cv2's own window (WINDOW_AUTOSIZE) always matches the
    rendered canvas size exactly, so picking a big DISPLAY_W here is what
    makes the window open large instead of needing a manual resize.

    Bounded by BOTH screen dimensions, not just width: the video's height is
    derived from DISPLAY_W via the stream's own aspect ratio (gcs.py's main
    loop resizes to (DISPLAY_W, dh)), which isn't known yet this early (no
    frame has arrived) — sizing by width alone risks a window taller than
    the actual screen for a narrower/taller stream. Assumes a conservative
    worst-case aspect ratio matching this project's tallest configured
    camera mode (4:3, config.toml's main_sizes) so the window fits
    vertically regardless of which resolution ends up streaming.

    Falls back to the old fixed width if the screen size can't be read
    (e.g. no display attached)."""
    try:
        import tkinter as _tk
        _root = _tk.Tk()
        _root.withdraw()
        sw, sh = _root.winfo_screenwidth(), _root.winfo_screenheight()
        _root.destroy()
        # Margin for OS chrome (menu bar/dock/title bar).
        avail_w = sw - panel_w - 60
        avail_h = sh - 100
        _WORST_CASE_ASPECT_H_OVER_W = 0.76   # 4:3 (~0.75), this project's tallest configured mode
        width_from_h = avail_h / _WORST_CASE_ASPECT_H_OVER_W
        return max(640, min(avail_w, int(width_from_h)))
    except Exception:
        return fallback

DISPLAY_W   = _screen_display_width(PANEL_W)   # video is stretched to this width for display

# NOTE: tried making this reactive to live window resize/maximize via
# cv2.getWindowImageRect() polled every frame — reverted. On this OpenCV/
# macOS (Cocoa, non-Qt) build, that call reflects the highgui content
# view's own size, not the outer OS window frame, and the content view does
# not resize when the user drags/maximizes the window. Worse, feeding that
# stale value back into computing the next frame's target size created a
# feedback loop that locked onto a SMALLER size than this fixed startup
# value, regardless of the actual window size. A fixed, generously-computed-
# at-startup width is the reliable option on this backend.

# ── Shared state ──────────────────────────────────────────────────────────────

launched        = False
moving_tgt      = False
_pi_recording   = False   # Pi-side recording state (optimistic: toggled on each command)
_local_recording = False  # GCS-side recording state
_local_writer    = None   # cv2.VideoWriter when local recording is active
cam_active      = False   # Pi camera fps state: False=idle (power-save), True=full fps
white_target_enabled = False   # Pi-side white-target aim refinement/recovery on/off
aim_phase       = "off"   # "off" | "blob" | "cross" — which detector is currently active, polled from /status
_cpu_percent    = None    # Pi CPU usage %, polled from /status; None until first poll
_cpu_temp_c     = None    # Pi SoC temperature °C, polled from /status; None until first poll
_pi_tracking    = False   # Pi-side tracking state, polled from /status
_frame_gen      = None    # Pi's frame counter for the currently displayed frame; echoed by select_point()
_status    = ""
_status_ts = 0.0
_mouse_pos = [0, 0]   # updated by mouse callback; used for hover highlight
_press_on_video = False   # True from LBUTTONDOWN-on-video until release; commits select_point then
_press_start_ts = 0.0     # time.time() when the press started; gates the zoom loupe's appearance
_drag_start = [0, 0]      # display-coord position of the LBUTTONDOWN that started the current press
DRAG_MIN_PX = 10          # release within this of the press start = a click; further = a dragged area
_quit         = threading.Event()  # set to break the main loop from any thread

def _handle_term_signal(signum, frame):
    """SIGTERM/SIGINT (kill, Ctrl-C) → graceful shutdown via the normal
    _quit path, so an in-progress local recording still gets its writer
    released (finalizes the MP4's index) instead of being left corrupt.
    Doesn't help against SIGKILL (kill -9) — nothing in-process can."""
    _quit.set()

signal.signal(signal.SIGTERM, _handle_term_signal)
signal.signal(signal.SIGINT, _handle_term_signal)
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
    if cmd == 'r':
        _vt_clear()   # no active target anymore

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
    _vt_reset(x, y)

def select_area(x0, y0, x1, y1):
    """Init tracking on a manually-dragged rectangle instead of a fixed-size
    box around a single clicked point — much easier to land on a small/dim
    target when the video itself is shaking too much to click it precisely.
    (x0,y0)-(x1,y1) in raw-frame coords, any drag direction (corners get
    sorted below, so top-left-to-bottom-right isn't required)."""
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    nx0 = round(x0 / _cur_video_w, 6)
    ny0 = round(y0 / _cur_video_h, 6)
    nx1 = round(x1 / _cur_video_w, 6)
    ny1 = round(y1 / _cur_video_h, 6)
    extra = {"frame_gen": _frame_gen} if _frame_gen is not None else {}
    _post("select_point", nx0=nx0, ny0=ny0, nx1=nx1, ny1=ny1, **extra)
    set_status(f"Selected area ({x0},{y0})-({x1},{y1})")
    _vt_reset((x0 + x1) // 2, (y0 + y1) // 2)

def lock_target(x, y):
    """Commit (x, y) — the current mouse position — as the tracking target.
    Triggered by the gamepad's Square/rectangle button as an alternative to
    releasing a mouse-drag."""
    global _pi_tracking
    select_point(x, y)
    _pi_tracking = True   # optimistic — confirmed/corrected by the next /status poll

# ── Feature tracking — detect, track, and expose their shared motion ───────
#
# Detects a set of "good" corners (goodFeaturesToTrack), tracks them frame-
# to-frame via optical flow, and re-seeds a fresh set whenever too few
# survive. Dots stay glued to the same physical features as the camera pans
# — the FEATURES toggle draws them so that can be visually confirmed.
#
# Also the single source of "how did the background move this frame" for
# virtual-target propagation (see _vt_update below): every frame, alongside
# tracking, it computes the median x/y displacement of whichever points
# survived (robust to the odd bad point without needing a full RANSAC/affine
# fit — points already excluded from contaminated zones before this, so a
# plain median of what's left is enough). One shared computation for both
# consumers instead of a second, separate optical-flow pass — tracking runs
# whenever either consumer needs it, independent of whether the dots
# themselves are being drawn.

_feat_enabled     = False
_feat_points      = None   # Nx1x2 float32 tracked points, or None
_feat_prev_gray   = None
_feat_last_delta  = None   # (median_dx, median_dy) this frame's background translation, or None
_feat_last_scale  = None   # this frame's background scale factor (>1 = zoomed in), or None
_feat_last_angle  = None   # this frame's background rotation, degrees (+ = counterclockwise), or None
_feat_last_ts     = 0.0    # wall-clock time of the last processed frame, for gap detection
_FEAT_MIN         = 40     # re-seed a fresh set once fewer survivors than this
_FEAT_MAX         = 200
_FEAT_MIN_DELTA     = 8    # need at least this many surviving points to trust their median motion
_FEAT_MIN_BASELINE_PX = 20 # ignore a point-pair this close together when estimating scale/angle —
                            # a small denominator turns ordinary tracking noise into a wild ratio
_FEAT_SCALE_CLAMP   = (0.85, 1.15)   # reject a single frame implying more than +/-15% zoom —
                                      # generous for a genuine fast approach, tight enough to
                                      # catch estimation noise (a bad scale/angle reading here
                                      # happens on every approach/retreat or camera roll, not
                                      # occasionally, so it matters more than it would elsewhere)
_FEAT_ANGLE_CLAMP   = 3.0  # reject a single frame implying more than +/-3deg of rotation (degrees)
_FEAT_GAP_RESET_S = 1.0    # don't compute a delta across a long interruption — resync instead

# OpenCV's calcOpticalFlowPyrLK defaults (21x21 window, 3 pyramid levels) only
# reliably track small frame-to-frame motion — too little for a fast whip-pan
# (observed live: the virtual target "misses" a fast out-of-frame move).
# Widening the window/pyramid alone isn't enough on its own, though — past a
# point it just makes points converge to the wrong (but similar-looking)
# nearby match while still reporting "success" (see _LK_MAX_ERR below for how
# that's caught). Widened anyway since it measurably extends the range where
# a *correct* match is even reachable in the first place.
_LK_PARAMS = dict(winSize=(41, 41), maxLevel=4,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
_LK_MAX_ERR = 10   # reject a "successful" point whose LK match residual is still this high —
                   # see _feat_update() for the measurement behind this number

def toggle_features():
    global _feat_enabled, _feat_points, _feat_prev_gray
    _feat_enabled = not _feat_enabled
    _feat_points = None
    _feat_prev_gray = None
    set_status(f"Feature tracking: {'ON' if _feat_enabled else 'OFF'}")

_BOX_COLORS_BGR = [(0, 200, 0), (0, 140, 255), (0, 0, 255)]   # tracker-so.py's good/uncertain/unstable box colors
_BOX_COLOR_TOL  = 40    # per-channel tolerance for JPEG/H264 compression blur around the exact color

def _box_color_mask(frame_bgr):
    """Raw (undilated) mask of pixels matching one of the Pi's tracking-box
    colors — the box border plus its small center crosshair, both drawn in
    the same color. Used by the exclusion mask below to keep the box's own
    pixels out of the background feature tracking it moves independently of."""
    box_hit = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    for b, g, r in _BOX_COLORS_BGR:
        lo = (max(0, b - _BOX_COLOR_TOL), max(0, g - _BOX_COLOR_TOL), max(0, r - _BOX_COLOR_TOL))
        hi = (min(255, b + _BOX_COLOR_TOL), min(255, g + _BOX_COLOR_TOL), min(255, r + _BOX_COLOR_TOL))
        box_hit |= cv2.inRange(frame_bgr, lo, hi)
    return box_hit

def _text_zone_mask(shape):
    """255 everywhere except the Pi's baked-in status text zones (top
    status/timestamp block — including the red "Launched" line, which
    otherwise color-matches the box's own "unstable" red — and the
    bottom-left LAUNCHED badge). Screen-locked, so a fixed position works."""
    h, w = shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[:int(0.22 * h), :] = 0                 # top status/timestamp/"Launched" block
    mask[int(0.92 * h):, :int(0.15 * w)] = 0    # bottom-left "LAUNCHED" badge
    return mask

def _feat_exclude_mask(frame_bgr):
    """Combined exclusion mask: the Pi's baked-in status text zones, plus
    any pixel close to one of its tracking-box colors (including the small
    center crosshair). The box tracks the TARGET, not the background, and
    moves — so unlike the text it can't be excluded by a fixed screen
    position, only by its distinctive color."""
    mask = _text_zone_mask(frame_bgr.shape[:2])
    box_hit = _box_color_mask(frame_bgr)
    if box_hit.any():
        box_hit = cv2.dilate(box_hit, np.ones((9, 9), np.uint8))   # cover the anti-aliased/compressed halo
        mask[box_hit > 0] = 0
    return mask

def _feat_update(frame, is_new_frame):
    """Track existing points forward one frame, dropping any optical flow
    lost or that drifted into an excluded zone (baked-in text, or the Pi's
    tracking-box color); re-seed a fresh set if too few survive. Also
    updates _feat_last_delta/_feat_last_scale/_feat_last_angle — this
    frame's background motion, for _vt_update() to consume. `frame` must be
    raw (pre-stabilization-warp) coordinates. Runs whenever the FEATURES
    diagnostic is on OR a virtual target is selected — either one needs
    this same tracking."""
    global _feat_points, _feat_prev_gray, _feat_last_delta, _feat_last_scale, _feat_last_angle, _feat_last_ts
    if not (_feat_enabled or _virtual_target is not None) or not is_new_frame:
        return

    now = time.time()
    if _feat_last_ts and (now - _feat_last_ts) > _FEAT_GAP_RESET_S:
        _feat_prev_gray = None   # long interruption — resync instead of comparing across the gap
    _feat_last_ts = now

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    mask = _feat_exclude_mask(frame)

    _feat_last_delta = None
    _feat_last_scale = None
    _feat_last_angle = None
    if (_feat_prev_gray is not None and _feat_points is not None
            and len(_feat_points) > 0 and _feat_prev_gray.shape == gray.shape):
        old_pts = _feat_points
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(_feat_prev_gray, gray, old_pts, None, **_LK_PARAMS)
        status = status.reshape(-1).astype(bool)
        # status alone means "converged to *something*", not "converged to the
        # right thing" — on a fast pan, points can lock onto the wrong match
        # (aliasing against similar-looking nearby texture) while still
        # reporting success. Measured on a real frame: at a 150px synthetic
        # jump, ~90% of "successful" points had actually converged to the
        # wrong spot, silently corrupting the median with garbage. err (LK's
        # own match residual) reliably separates the two — err<10 gave the
        # exact correct median through 150px, and safely fell below
        # _FEAT_MIN_DELTA (skip the frame) beyond that, rather than
        # confidently returning a wrong answer.
        good = status & (err.reshape(-1) < _LK_MAX_ERR)
        old_pts, new_pts = old_pts[good], new_pts[good]
        if len(new_pts) > 0:
            pts2 = new_pts.reshape(-1, 2)
            xi = np.clip(pts2[:, 0].astype(int), 0, w - 1)
            yi = np.clip(pts2[:, 1].astype(int), 0, h - 1)
            keep = mask[yi, xi] > 0
            old_pts, new_pts = old_pts[keep], new_pts[keep]
        if len(new_pts) >= _FEAT_MIN_DELTA:
            old_xy = old_pts.reshape(-1, 2)
            new_xy = new_pts.reshape(-1, 2)
            # Scale and rotation, estimated the same "smart average" way,
            # from the SAME point pairs: how did each pair's connecting
            # vector change, new frame vs old. Both are pivot-independent —
            # neither needs a "center" chosen up front — which is exactly
            # why this avoids the old rotation-amplification bug: that one
            # came from a single noisy RANSAC-fit angle applied via a matrix
            # pivoted at the frame's (0,0) corner, not from rotation itself
            # being inherently unsafe. A robust median over every pair, each
            # measured relative to the OTHER tracked points rather than an
            # arbitrary corner, doesn't have that failure mode — verified via
            # a real recorded session where an uncompensated real camera
            # rotation was steadily walking the marker away from a
            # confirmed-fixed target even while fully tracked (translation-
            # and-scale-only couldn't represent it; this can).
            #   scale_ij = dist(new_i,new_j) / dist(old_i,old_j)
            #   angle_ij = angle(new_j-new_i) - angle(old_j-old_i)
            # Both clamped hard per frame — scale because it matters on
            # every approach/retreat, not occasionally, so an unclamped bad
            # reading would be a frequent problem; angle because that's
            # exactly the failure mode being reintroduced here, just via a
            # more robust estimator this time, not an excuse to skip the
            # safety margin entirely.
            n = len(old_xy)
            iu = np.triu_indices(n, k=1)
            vec_old = old_xy[iu[1]] - old_xy[iu[0]]
            vec_new = new_xy[iu[1]] - new_xy[iu[0]]
            d_old = np.linalg.norm(vec_old, axis=1)
            d_new = np.linalg.norm(vec_new, axis=1)
            valid = d_old > _FEAT_MIN_BASELINE_PX   # a near-coincident pair's ratio/angle is pure noise
            enough = valid.sum() >= _FEAT_MIN_DELTA

            scale = float(np.median(d_new[valid] / d_old[valid])) if enough else 1.0
            scale = max(_FEAT_SCALE_CLAMP[0], min(_FEAT_SCALE_CLAMP[1], scale))

            if enough:
                ang_old = np.degrees(np.arctan2(vec_old[valid, 1], vec_old[valid, 0]))
                ang_new = np.degrees(np.arctan2(vec_new[valid, 1], vec_new[valid, 0]))
                dangle = float(np.median(_wrap_deg(ang_new - ang_old)))
            else:
                dangle = 0.0
            dangle = max(-_FEAT_ANGLE_CLAMP, min(_FEAT_ANGLE_CLAMP, dangle))

            a = np.radians(dangle)
            c, s = np.cos(a), np.sin(a)
            R = scale * np.array([[c, -s], [s, c]])   # 2x2 similarity (scale + rotation) linear part

            translation = new_xy - old_xy @ R.T
            _feat_last_delta = (float(np.median(translation[:, 0])), float(np.median(translation[:, 1])))
            _feat_last_scale = scale
            _feat_last_angle = dangle
        _feat_points = new_pts
    else:
        _feat_points = None

    if _feat_points is None or len(_feat_points) < _FEAT_MIN:
        detected = cv2.goodFeaturesToTrack(gray, maxCorners=_FEAT_MAX, qualityLevel=0.01,
                                            minDistance=8, blockSize=7, mask=mask)
        _feat_points = detected if detected is not None else np.empty((0, 1, 2), dtype=np.float32)

    _feat_prev_gray = gray

def draw_feature_points(frame):
    if not _feat_enabled or _feat_points is None:
        return
    for p in _feat_points.reshape(-1, 2):
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)

# ── Shared global-motion estimator ─────────────────────────────────────────
#
# One background-motion estimate, reused by stabilization (and, later,
# anything like virtual-target propagation) instead of duplicating the same
# optical-flow work. Detects fresh features each call (stateless — no
# persistent point identity to manage), tracks them, and fits a similarity
# transform via RANSAC. Returns only rotation+translation: scale is fit
# internally by estimateAffinePartial2D but deliberately discarded here and
# never returned, since accumulating a fitted scale over many frames is what
# caused runaway shrinkage in an earlier version of this code — the caller
# gets no way to (accidentally) accumulate it.

_GM_MIN_FEATURES = 12

def _estimate_global_motion(prev_gray, gray, mask=None):
    """Return (dx, dy, dangle_deg, n_features, n_inliers), or None if too
    few reliable points to trust the estimate."""
    pts_prev = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01,
                                        minDistance=8, blockSize=7, mask=mask)
    if pts_prev is None or len(pts_prev) < _GM_MIN_FEATURES:
        return None
    n_features = len(pts_prev)

    pts_cur, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev, None)
    status = status.reshape(-1).astype(bool)
    good_prev, good_cur = pts_prev[status], pts_cur[status]
    if len(good_prev) < _GM_MIN_FEATURES:
        return None

    M, inliers = cv2.estimateAffinePartial2D(good_prev, good_cur, method=cv2.RANSAC)
    if M is None or inliers is None:
        return None
    n_inliers = int(inliers.sum())
    if n_inliers < _GM_MIN_FEATURES:
        return None
    if not np.all(np.isfinite(M)):
        return None

    dx, dy = float(M[0, 2]), float(M[1, 2])
    dangle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return dx, dy, dangle, n_features, n_inliers

# ── Affine composition helpers (rotation+translation, scale always 1) ─────

def _compose(M_second, M_first):
    """2x3 affine composition: apply M_first, then M_second."""
    A1 = np.vstack([M_first, [0, 0, 1]])
    A2 = np.vstack([M_second, [0, 0, 1]])
    return (A2 @ A1)[:2].astype(np.float32)

def _wrap_deg(a):
    """Wrap a degree value into (-180, 180] — angle is a circular quantity,
    treating it as a plain linear number breaks at the ±180° discontinuity
    (this exact bug caused a runaway correction in an earlier version)."""
    return (a + 180.0) % 360.0 - 180.0

def _decompose_xya(M):
    """(x, y, angle_deg) from a similarity matrix — self-consistent with
    _recompose_xya(); ignores/discards any scale in M."""
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    return float(M[0, 2]), float(M[1, 2]), float(angle)

def _recompose_xya(x, y, angle_deg):
    """Build a rotation+translation matrix with scale forced to exactly 1."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.float32([[c, -s, x], [s, c, y]])

# ── Virtual target — estimates where the selected point currently is even
# when it's outside the frame ──────────────────────────────────────────────
#
# GCS-only, no Pi telemetry needed. Doesn't try to re-detect the target
# itself — once it's gone, there's nothing left to match against. Instead:
# the same feature points FEATURES already tracks are anchored to the
# background (goodFeaturesToTrack + optical flow, excluded from the Pi's
# text/tracking-box zones), so however THEY moved this frame is exactly how
# a background-anchored point should move too. _feat_update() computes that
# shared median motion once per frame (_feat_last_delta); this just applies
# it directly to the point — every frame, the same way, whether the target
# is on-screen or not. No separate estimation pass, no rotation (rotation's
# error grows with a point's distance from the frame origin and was a
# measured, real source of drift — see STABILIZATION.md), no special-casing
# "in view" vs "out of view".
#
# The propagated point is allowed to go negative or past the frame edge —
# that's the whole idea: "250px left of frame" is still a valid, useful
# estimate, not an error, and lets us know where to look — and re-lock —
# once the camera swings back.
#
# Reset to the freshly clicked point on every select_point() call, so drift
# only accumulates over however long the point has actually been
# unconfirmed, not over the whole session.

_vt_enabled     = False   # opt-in — still being tuned; select_point() won't anchor a target,
                          # and _feat_update()/draw_virtual_target() won't run for it, while off
_virtual_target = None   # (x, y) in raw-frame px, or None if nothing selected
_vt_last_log_ts = 0.0     # throttles the [VT] debug print
_vt_miss_count  = 0       # consecutive frames with no valid delta/scale to propagate by
_VT_MAX_MISSES  = 5       # clear the target after this many in a row — mirrors the Pi's own
                          # "Drift — re-select target" recovery: past this point the last known
                          # position was measured against a scene that (per _FEAT_GAP_RESET_S's
                          # time-based reset, or a genuine hard content cut a time-based check
                          # can't catch) may no longer have anything to do with the current one,
                          # so continuing to show it is actively misleading, not just stale

def toggle_virtual_target():
    global _vt_enabled
    _vt_enabled = not _vt_enabled
    _vt_clear()   # stale/no target either way once the mode changes
    set_status(f"Virtual target: {'ON' if _vt_enabled else 'OFF'}")

def _vt_reset(x, y):
    global _virtual_target, _vt_miss_count
    if not _vt_enabled:
        return
    _virtual_target = (float(x), float(y))
    _vt_miss_count = 0

def _vt_clear():
    global _virtual_target, _vt_miss_count
    _virtual_target = None
    _vt_miss_count = 0

def _vt_update(frame, is_new_frame):
    """Propagate _virtual_target by this frame's shared background-feature
    motion (_feat_last_delta/_feat_last_scale/_feat_last_angle, computed in
    _feat_update() — call that first). `frame` must be raw
    (pre-stabilization-warp) coordinates — the same space select_point()
    and _to_raw_coords() use."""
    global _virtual_target, _vt_last_log_ts, _vt_miss_count
    if _virtual_target is None or not is_new_frame:
        return
    if _feat_last_delta is None or _feat_last_scale is None or _feat_last_angle is None:
        _vt_miss_count += 1
        if _vt_miss_count >= _VT_MAX_MISSES:
            print(f"[VT] lost tracking for {_vt_miss_count} consecutive frames — "
                  f"clearing (re-select target)")
            _virtual_target = None
        return
    _vt_miss_count = 0
    dx, dy = _feat_last_delta
    scale = _feat_last_scale
    angle = _feat_last_angle
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    x, y = _virtual_target
    nx = scale * (c * x - s * y) + dx
    ny = scale * (s * x + c * y) + dy
    if not (math.isfinite(nx) and math.isfinite(ny)):
        return
    now = time.time()
    if now - _vt_last_log_ts > 0.5:
        print(f"[VT] delta=({dx:+5.1f},{dy:+5.1f}) scale={scale:.3f} angle={angle:+5.2f}  "
              f"point ({x:7.1f},{y:7.1f}) -> ({nx:7.1f},{ny:7.1f})")
        _vt_last_log_ts = now
    _virtual_target = (nx, ny)

def _vt_to_display(x, y):
    """Raw-frame point -> current display-frame point — applies the full
    stabilization transform forward (the inverse of _to_raw_coords())."""
    if _stab_M is None:
        return x, y
    return (_stab_M[0, 0] * x + _stab_M[0, 1] * y + _stab_M[0, 2],
            _stab_M[1, 0] * x + _stab_M[1, 1] * y + _stab_M[1, 2])

def draw_virtual_target(frame):
    """Marker at the virtual target if it's on-screen; otherwise an arrow at
    the frame edge pointing toward it, with its off-screen distance."""
    if _virtual_target is None:
        return
    h, w = frame.shape[:2]
    dx, dy = _vt_to_display(*_virtual_target)
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return
    color = (0, 220, 220)

    if 0 <= dx < w and 0 <= dy < h:
        p = (int(dx), int(dy))
        cv2.drawMarker(frame, p, color, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        cv2.circle(frame, p, 14, color, 2, cv2.LINE_AA)
        return

    margin = 30
    cx, cy = w / 2.0, h / 2.0
    vx, vy = dx - cx, dy - cy
    if vx == 0 and vy == 0:
        return
    t_candidates = []
    if vx > 0: t_candidates.append((w - margin - cx) / vx)
    elif vx < 0: t_candidates.append((margin - cx) / vx)
    if vy > 0: t_candidates.append((h - margin - cy) / vy)
    elif vy < 0: t_candidates.append((margin - cy) / vy)
    t = min(t for t in t_candidates if t > 0)
    ex, ey = cx + vx * t, cy + vy * t
    angle = math.atan2(vy, vx)

    size = 15
    tip   = (ex + size * math.cos(angle),        ey + size * math.sin(angle))
    base1 = (ex - size * math.cos(angle - 0.5),  ey - size * math.sin(angle - 0.5))
    base2 = (ex - size * math.cos(angle + 0.5),  ey - size * math.sin(angle + 0.5))
    pts = np.array([tip, base1, base2], dtype=np.int32)
    cv2.fillConvexPoly(frame, pts, color, cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)

    dist_px = math.hypot(dx - cx, dy - cy)
    label = f"{dist_px:.0f}px"
    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
    lx, ly = int(ex - tw / 2), int(ey + (th + 20 if math.sin(angle) < 0 else -14))
    cv2.putText(frame, label, (lx, ly), _FONT, 0.45, (0, 0, 0),   3, cv2.LINE_AA)
    cv2.putText(frame, label, (lx, ly), _FONT, 0.45, color,       1, cv2.LINE_AA)

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

def toggle_white_target():
    """Explicitly tell the Pi to turn white-target aim refinement/recovery on
    or off. Off = tracking behaves exactly as plain bbox-center tracking."""
    global white_target_enabled
    white_target_enabled = not white_target_enabled
    _post("set_white_target", enabled=1 if white_target_enabled else 0)
    set_status("White target: ON" if white_target_enabled else "White target: OFF")

def _status_poller():
    """Background: poll /status every 2s to keep cam_active/_cpu_percent/_cpu_temp_c/
    _pi_tracking/white_target_enabled/aim_phase fresh even when nothing else is
    triggering a request (e.g. after Pi restarts)."""
    global cam_active, _cpu_percent, _cpu_temp_c, _pi_tracking, white_target_enabled, aim_phase
    while not _quit.is_set():
        data = _get("status")
        if data:
            cam_active   = data.get("active_fps", cam_active)
            _cpu_percent = data.get("cpu_percent", _cpu_percent)
            _cpu_temp_c  = data.get("cpu_temp", _cpu_temp_c)
            _pi_tracking = data.get("tracking", _pi_tracking)
            white_target_enabled = data.get("white_target", white_target_enabled)
            aim_phase    = data.get("aim_phase", aim_phase)
        time.sleep(2.0)

threading.Thread(target=_status_poller, daemon=True).start()

def cycle_main(delta):
    _post("cycle_main", delta=delta)
    set_status(f"MAIN {'up' if delta > 0 else 'down'}")

def cycle_lores(delta):
    _post("cycle_lores", delta=delta)
    set_status(f"TRACK {'up' if delta > 0 else 'down'}")

# ── Video transport switch (jpeg_udp <-> h264_udp, live) ───────────────────────
#
# cap/_gcs_video_mode are module-level (not local to main()) so this can
# reassign the active capture object from a button click mid-session. The
# capture classes (_LiveCapture/_H264LiveCapture) are defined later in this
# file — fine, since this function's body isn't evaluated until it's
# actually called, well after the whole module has loaded.

cap = None               # current capture object — set by main(), swapped by toggle_video_mode()
_gcs_video_mode = None   # "jpeg_udp" or "h264_udp", whichever `cap` currently is; None in --file
                          # mode, or if starting mode is "webrtc" (not switchable — see tracker-so.py)

def toggle_video_mode():
    """Live-switch the video transport (jpeg_udp <-> h264_udp) without
    restarting either side. Tells the Pi to make the same switch, then
    rebuilds the local capture object on the same UDP port. A few seconds
    of no video during the swap is expected on both ends — this exists for
    exactly the case where the current transport (typically h264_udp) is
    struggling under poor comms and dropping to the simpler, loss-tolerant
    jpeg_udp (independent per-frame JPEGs — one lost packet loses one
    frame, not everything until the next keyframe) is worth a brief gap."""
    global cap, _gcs_video_mode
    if _gcs_video_mode is None:
        set_status("Video mode switch not available in this mode")
        return
    new_mode = "jpeg_udp" if _gcs_video_mode == "h264_udp" else "h264_udp"
    set_status(f"Switching video -> {new_mode}…")
    _post("set_video_mode", mode=new_mode)
    cap.close()
    cap = _H264LiveCapture(UDP_PORT) if new_mode == "h264_udp" else _LiveCapture(UDP_PORT)
    _gcs_video_mode = new_mode

# ── Whole-frame stabilization (feature-based) ──────────────────────────────────
#
# Cancels camera shake/vibration — including rotation, not just translation —
# so the whole displayed picture holds still. Uses _estimate_global_motion()
# (features + optical flow + RANSAC, same foundation as the FEATURES
# diagnostic, with the same text/tracking-box exclusion mask) instead of the
# old phaseCorrelate approach, so a genuine tilt/rotation event gets tracked
# properly instead of only being approximated as a shift.
#
# Maintains a cumulative RAW trajectory (x, y, angle) — scale is never part
# of it; each per-frame increment is rebuilt via _recompose_xya() with scale
# forced to 1 before being composed in, so nothing can accumulate scale
# drift the way an earlier version of this code did. The trajectory is
# low-pass filtered (_STAB_SLOW_ALPHA) to separate "intended" slow motion
# from fast jitter — the angle component of that filter update wraps
# through _wrap_deg() at every step, fixing the ±180° discontinuity bug
# that caused a runaway correction in that earlier version.
#
# select_point() must be called with RAW-frame coordinates (matching what the
# Pi's own live frame looks like), so _to_raw_coords() undoes the full
# stabilization transform (not just an x/y shift) on whatever was clicked.

_STAB_ALPHA_MIN  = 0.01
_STAB_ALPHA_MAX  = 0.50
_STAB_ALPHA_STEP = 0.01

_stab_enabled    = True
_STAB_SLOW_ALPHA = 0.05  # how fast the "intended" trajectory adapts (lower = more shake removed)
_STAB_MAX_SHIFT  = 75    # clamp (px) — reject a per-frame estimate implying a bigger jump than this
_STAB_MAX_ANGLE  = 6.0   # clamp (degrees) — same, for rotation
_STAB_GAP_RESET_S = 1.0  # if this long since the last successful update, don't estimate motion
                          # across the gap — treat the next frame as a fresh reference instead

# Adaptive smoothing: a single fixed alpha can't be both "strong for tiny
# jitter" and "quick to catch up on a genuine big pan" — a low alpha strong
# enough to remove shake also makes the smoothed trajectory lag far behind
# during a real fast pan, so the correction (raw - smoothed) grows large and
# warps a big strip of BORDER_REPLICATE padding into view. Instead, blend
# toward a faster alpha as the per-frame motion itself gets larger — small
# jitter still gets the full _STAB_SLOW_ALPHA smoothing, but a big frame-to-
# frame jump (a real pan/tilt, not shake) lets the trajectory adapt quickly
# so the correction — and the border it reveals — stays small.
_STAB_FAST_ALPHA     = 0.45   # ceiling for how responsive it gets during a big move
_STAB_BIG_MOVE_PX    = 18     # per-frame translation magnitude that's fully "big move"
_STAB_BIG_MOVE_ANGLE = 2.5    # per-frame rotation magnitude (degrees) that's fully "big move"

_stab_prev_gray   = None
_stab_last_ts     = 0.0   # wall-clock time of the last processed new frame, for gap detection
_stab_cum         = np.float32([[1, 0, 0], [0, 1, 0]])   # cumulative raw transform: reference -> now
_stab_smooth_xya  = np.array([0.0, 0.0, 0.0])             # low-pass filtered [x, y, angle_deg]
_stab_M           = None   # the actual transform last applied to the displayed frame — for inversion
_stab_last_log_ts = 0.0    # throttles the [STAB] debug print

def _reset_stabilizer():
    global _stab_prev_gray, _stab_last_ts, _stab_cum, _stab_smooth_xya, _stab_M
    _stab_prev_gray  = None
    _stab_last_ts    = 0.0
    _stab_cum        = np.float32([[1, 0, 0], [0, 1, 0]])
    _stab_smooth_xya = np.array([0.0, 0.0, 0.0])
    _stab_M          = None

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

# ── Zoom loupe (optional) — press + drag on video to aim precisely ────────────
#
# While enabled, pressing on the video shows a magnifier centered exactly on
# the cursor (never offset — an offset loupe was tried before and found
# confusing). Position never drifts from the real cursor; only the pixel
# content is smoothed across recent frames (a short motion-blur) to damp
# rapid shake without ever decoupling from the mouse. Off by default since
# an earlier always-on version didn't work well in practice — this makes it
# opt-in per session.

_zoom_enabled = False

_ZOOM_SRC       = 150   # px cropped from the frame around the cursor
_ZOOM_OUT       = 220   # loupe window size on screen
_ZOOM_HOLD_S    = 0.5   # seconds the button must be held before the loupe appears —
                         # a quick click just selects the cursor point directly

_LOUPE_BLEND_ALPHA = 0.35   # weight of the newest frame; lower = smoother/more lag

_loupe_blend = None   # float32 running average of the source crop, while dragging

def toggle_zoom():
    global _zoom_enabled
    _zoom_enabled = not _zoom_enabled
    _reset_loupe_blend()
    set_status(f"Zoom loupe: {'ON' if _zoom_enabled else 'OFF'}")

def _smoothed_crop(frame, mx, my):
    """Return (crop, x0, y0): an exponentially-smoothed crop of the source
    frame, centered on (mx, my)."""
    global _loupe_blend
    h, w = frame.shape[:2]
    half = _ZOOM_SRC // 2
    x0 = max(0, min(w - _ZOOM_SRC, mx - half))
    y0 = max(0, min(h - _ZOOM_SRC, my - half))
    crop = frame[y0:y0 + _ZOOM_SRC, x0:x0 + _ZOOM_SRC].astype(np.float32)

    if _loupe_blend is None or _loupe_blend.shape != crop.shape:
        _loupe_blend = crop
    else:
        _loupe_blend = _LOUPE_BLEND_ALPHA * crop + (1 - _LOUPE_BLEND_ALPHA) * _loupe_blend

    return np.clip(_loupe_blend, 0, 255).astype(np.uint8), x0, y0

def _reset_loupe_blend():
    global _loupe_blend
    _loupe_blend = None

def draw_zoom_loupe(frame, mx, my, crop, x0, y0):
    """Picture-in-picture magnifier around (mx, my), with a crosshair marking
    the exact point that will be selected when the mouse button is released."""
    h, w = frame.shape[:2]
    if w < _ZOOM_SRC or h < _ZOOM_SRC:
        return
    out = min(_ZOOM_OUT, w, h)   # shrink to fit if the frame is smaller than the loupe (e.g. 16:9)
    zoom = cv2.resize(crop, (out, out), interpolation=cv2.INTER_NEAREST)

    cx = int((mx - x0) * out / _ZOOM_SRC)
    cy = int((my - y0) * out / _ZOOM_SRC)
    cv2.line(zoom, (cx - 12, cy), (cx + 12, cy), (50, 50, 230), 1, cv2.LINE_AA)
    cv2.line(zoom, (cx, cy - 12), (cx, cy + 12), (50, 50, 230), 1, cv2.LINE_AA)
    cv2.rectangle(zoom, (0, 0), (out - 1, out - 1), (255, 255, 255), 2)

    # Shifted up relative to the cursor so the OS pointer icon (hotspot at its
    # tip, graphic extending down-right from there) doesn't sit on top of the
    # crosshair — the aim point ends up lower in the box, not dead center.
    y_bias = out // 4
    lx = max(0, min(w - out, mx - out // 2))
    ly = max(0, min(h - out, my - out // 2 - y_bias))

    frame[ly:ly + out, lx:lx + out] = zoom

# ── Hi-res pre-launch zoom (optional) — long-press requests a full-native-
# sensor-resolution crop from the Pi instead of a digital upscale, so real
# detail is visible around the cursor rather than just enlarged pixels.
# Takes over the long-press gesture from the plain zoom loupe while
# enabled. Only usable pre-launch: the capture briefly stutters the live
# video (a genuine camera reconfigure on the Pi), which is only acceptable
# when nothing flight-critical is happening — enforced both here (so the
# button/gesture give instant feedback) and on the Pi side (authoritative).

_hires_zoom_enabled = False
_hires_pending      = False   # a fetch is currently in flight
_hires_crop_img     = None    # decoded numpy array of the last fetched crop, or None
_hires_request_id   = 0       # bumped per request; lets a stale in-flight response be ignored

def toggle_hires_zoom():
    global _hires_zoom_enabled
    if launched:
        set_status("Hires zoom: not available after launch")
        return
    _hires_zoom_enabled = not _hires_zoom_enabled
    set_status(f"Hires zoom: {'ON' if _hires_zoom_enabled else 'OFF'}")

def _fetch_hires_crop(nx, ny, req_id):
    global _hires_crop_img, _hires_pending
    try:
        r = requests.get(f"{FLASK}/hires_crop", params={"nx": nx, "ny": ny}, timeout=8)
        if req_id != _hires_request_id:
            return   # superseded by a new press, or this one already released
        if r.status_code == 200:
            arr = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if req_id == _hires_request_id and img is not None:
                _hires_crop_img = img
                fname = f"hires_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                try:
                    with open(fname, "wb") as f:
                        f.write(r.content)   # raw bytes from the Pi — no re-encode
                    set_status(f"Hires crop saved → {fname}", log=False)
                except OSError as e:
                    set_status(f"Hires crop save failed: {e}", log=False)
        else:
            set_status(f"Hires zoom: {r.text}", log=False)
    except Exception as e:
        if req_id == _hires_request_id:
            set_status(f"Hires zoom failed: {e}", log=False)
    finally:
        if req_id == _hires_request_id:
            _hires_pending = False

def draw_hires_loupe(frame, mx, my, crop_img):
    """Picture-in-picture magnifier showing a real hi-res capture instead of
    a digital crop — always centered exactly on the requested point (the Pi
    crops it that way already), so the crosshair goes at the image's own
    center rather than needing a source-offset like the digital loupe."""
    h, w = frame.shape[:2]
    out = min(_ZOOM_OUT, w, h)
    zoom = cv2.resize(crop_img, (out, out), interpolation=cv2.INTER_AREA)

    cx = cy = out // 2
    cv2.line(zoom, (cx - 12, cy), (cx + 12, cy), (50, 50, 230), 1, cv2.LINE_AA)
    cv2.line(zoom, (cx, cy - 12), (cx, cy + 12), (50, 50, 230), 1, cv2.LINE_AA)
    cv2.rectangle(zoom, (0, 0), (out - 1, out - 1), (255, 255, 255), 2)

    y_bias = out // 4
    lx = max(0, min(w - out, mx - out // 2))
    ly = max(0, min(h - out, my - out // 2 - y_bias))
    frame[ly:ly + out, lx:lx + out] = zoom

def _stabilize(frame, is_new_frame):
    """Warp `frame` by the current shake-cancelling correction, recomputing
    that correction only when a genuinely new frame has arrived. `frame`
    must be raw (undistorted by any previous warp) — this is always called
    before any stabilization warp is applied, and the reference frame it
    keeps for the next comparison is taken from this same raw input, never
    from the warped output."""
    global _stab_prev_gray, _stab_last_ts, _stab_cum, _stab_smooth_xya, _stab_M, _stab_last_log_ts
    h, w = frame.shape[:2]

    if is_new_frame:
        now = time.time()
        if _stab_last_ts and (now - _stab_last_ts) > _STAB_GAP_RESET_S:
            # Long interruption (e.g. dropped/late frames) — don't estimate
            # motion across that gap, just resync on the next frame.
            _stab_prev_gray = None
        _stab_last_ts = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        n_feat = n_inliers = 0
        dx = dy = dangle = 0.0

        if _stab_prev_gray is not None and _stab_prev_gray.shape == gray.shape:
            mask = _feat_exclude_mask(frame)   # skip Pi status text + tracking-box pixels
            result = _estimate_global_motion(_stab_prev_gray, gray, mask=mask)
            if result is not None:
                dx, dy, dangle, n_feat, n_inliers = result
                if abs(dx) <= _STAB_MAX_SHIFT and abs(dy) <= _STAB_MAX_SHIFT and abs(dangle) <= _STAB_MAX_ANGLE:
                    M_frame = _recompose_xya(dx, dy, dangle)   # scale forced to 1, never accumulated
                    _stab_cum = _compose(M_frame, _stab_cum)
                # else: unreliable-looking jump — keep the previous _stab_cum unchanged

        _stab_prev_gray = gray

        # Blend toward a faster alpha when THIS frame's own motion looks like
        # a real pan/tilt rather than shake — keeps small jitter smoothed as
        # strongly as configured, while a genuine big move gets caught up to
        # quickly instead of dragging a large, slowly-shrinking correction
        # (and the border padding it reveals) behind it.
        move_frac = max(min(1.0, math.hypot(dx, dy) / _STAB_BIG_MOVE_PX),
                         min(1.0, abs(dangle) / _STAB_BIG_MOVE_ANGLE))
        alpha = _STAB_SLOW_ALPHA + (_STAB_FAST_ALPHA - _STAB_SLOW_ALPHA) * move_frac

        x, y, a = _decompose_xya(_stab_cum)
        diff = np.array([x, y, a]) - _stab_smooth_xya
        diff[2] = _wrap_deg(diff[2])   # angle is circular — a raw difference can spike near ±180°
        _stab_smooth_xya += alpha * diff
        _stab_smooth_xya[2] = _wrap_deg(_stab_smooth_xya[2])
        M_smooth_cum = _recompose_xya(*_stab_smooth_xya)
        # Where the smoothed/intended path would be, minus where the shaky raw
        # path actually is → the correction to apply to the current raw frame.
        M_candidate = _compose(M_smooth_cum, cv2.invertAffineTransform(_stab_cum))

        # Safety net: a correctly-behaving correction should never need to be
        # huge. If it is, something upstream went wrong — reset rather than
        # apply a runaway warp that could blank/distort the display.
        ccx, ccy, cca = _decompose_xya(M_candidate)
        if abs(ccx) > 4 * _STAB_MAX_SHIFT or abs(ccy) > 4 * _STAB_MAX_SHIFT or abs(cca) > 4 * _STAB_MAX_ANGLE:
            _reset_stabilizer()
            M_candidate = np.float32([[1, 0, 0], [0, 1, 0]])
            ccx = ccy = cca = 0.0

        _stab_M = M_candidate

        now2 = time.time()
        if now2 - _stab_last_log_ts > 0.5:
            print(f"[STAB] feat={n_feat:3d} inliers={n_inliers:3d} alpha={alpha:.2f}  "
                  f"dx={dx:+6.1f} dy={dy:+6.1f} dangle={dangle:+5.2f}  "
                  f"-> correction=({ccx:+6.1f},{ccy:+6.1f},{cca:+5.2f}°)")
            _stab_last_log_ts = now2

    if _stab_M is None:
        return frame
    return cv2.warpAffine(frame, _stab_M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def _to_raw_coords(px, py):
    """Undo the current stabilization transform — use before select_point()/
    lock_target() so the Pi (which sees its own unwarped live frame) gets the
    right pixel, including when rotation is part of the correction."""
    if _stab_M is None:
        rx, ry = px, py
    else:
        Minv = cv2.invertAffineTransform(_stab_M)
        rx = Minv[0, 0] * px + Minv[0, 1] * py + Minv[0, 2]
        ry = Minv[1, 0] * px + Minv[1, 1] * py + Minv[1, 2]
    rx = max(0, min(_cur_video_w - 1, int(round(rx))))
    ry = max(0, min(_cur_video_h - 1, int(round(ry))))
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

    # ── White target mode ────────────────────────────────────────────────────
    btn(
        lambda: f"WHITE TARGET: {aim_phase.upper()}" if white_target_enabled else "WHITE TARGET: OFF",
        36, toggle_white_target,
        lambda: (30, 140, 50) if white_target_enabled else (90, 90, 30),
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

    # ── Zoom loupe (press + drag on video to aim) ───────────────────────────
    btn(
        lambda: "ZOOM: ON" if _zoom_enabled else "ZOOM: OFF",
        36, toggle_zoom,
        lambda: (30, 140, 50) if _zoom_enabled else (90, 90, 30),
    )
    y += 44

    # ── Hi-res pre-launch zoom (long-press on video, real sensor detail) ────
    btn(
        lambda: "HIRES ZOOM: LOCKED" if launched else
                ("HIRES ZOOM: ON" if _hires_zoom_enabled else "HIRES ZOOM: OFF"),
        36, toggle_hires_zoom,
        lambda: (90, 40, 40) if launched else
                ((30, 140, 50) if _hires_zoom_enabled else (90, 90, 30)),
    )
    y += 44

    # ── Feature tracking diagnostic (green dots) ────────────────────────────
    btn(
        lambda: "FEATURES: ON" if _feat_enabled else "FEATURES: OFF",
        36, toggle_features,
        lambda: (30, 140, 50) if _feat_enabled else (90, 90, 30),
    )
    y += 44

    # ── Virtual target (yellow marker — estimates position out of FOV) ──────
    btn(
        lambda: "VT: ON" if _vt_enabled else "VT: OFF",
        36, toggle_virtual_target,
        lambda: (30, 140, 50) if _vt_enabled else (90, 90, 30),
    )
    y += 44

    # ── Video transport (jpeg_udp <-> h264_udp, live) ────────────────────────
    btn(
        lambda: f"VIDEO: {_gcs_video_mode.replace('_udp', '').upper()}" if _gcs_video_mode else "VIDEO: n/a",
        36, toggle_video_mode,
        lambda: (55, 55, 85) if _gcs_video_mode else (60, 60, 60),
    )
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
        self._closed    = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        """Receive JPEG datagrams and decode them; marks _ok=False on timeout."""
        while not self._closed:
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
            except OSError:
                break   # socket closed via close() — exit cleanly, not an error
            except Exception as e:
                print(f"[UDP] recv: {e}")

    def read(self):
        """Return (ok, frame_copy, frame_id, frame_gen).  Never blocks more than the lock."""
        with self._lock:
            if self._frame is None:
                return False, None, 0, None
            return self._ok, self._frame.copy(), self._frame_id, self._frame_gen

    def close(self):
        """Stop the reader thread and release the socket — call before
        switching to a different capture class on the same UDP port."""
        self._closed = True
        try:
            self._sock.close()
        except Exception:
            pass


# ── H.264/RTP live capture (video_mode = "h264_udp") ───────────────────────────
#
# Same read() contract as _LiveCapture — (ok, frame, frame_id, frame_gen) —
# so main() only needs to pick which class to instantiate; everything else
# (HUD, stabilization, zoom loupe, recording, select_point's frame_gen echo)
# is unaware of which transport is behind cap.
#
# Deliberately no jitter buffer / reordering: packets are handled strictly
# in arrival order, and any sequence-number gap (including the arrival
# after a Wi-Fi drop, since tracker-so.py's sender keeps sendto()-ing
# through an outage) immediately discards the in-progress frame and any
# partial fragment reassembly, then waits for the next keyframe before
# decoding again — real-time video over recovering every frame, and no
# handshake/restart needed on either side to resync.

_RTP_EXT_ID = 1   # matches h264_udp_server.py — one-byte-header extension carrying frame_gen
_FU_A_TYPE  = 28

def _parse_rtp(data: bytes):
    """Return (seq, marker, payload, frame_gen) or None if unparseable."""
    if len(data) < 12:
        return None
    b0, b1 = data[0], data[1]
    if (b0 >> 6) != 2:          # RTP version must be 2
        return None
    x_bit  = bool(b0 & 0x10)
    marker = bool(b1 & 0x80)
    seq    = struct.unpack('>H', data[2:4])[0]
    off    = 12
    frame_gen = None
    if x_bit:
        if len(data) < off + 4:
            return None
        ext_len_words = struct.unpack('>H', data[off + 2:off + 4])[0]
        ext_total = 4 + ext_len_words * 4
        if len(data) < off + ext_total:
            return None
        block = data[off + 4:off + ext_total]
        p = 0
        while p < len(block):
            b = block[p]
            if b == 0:            # padding byte
                p += 1
                continue
            eid, elen = b >> 4, (b & 0x0F) + 1
            edata = block[p + 1:p + 1 + elen]
            if eid == _RTP_EXT_ID and elen == 4 and len(edata) == 4:
                frame_gen = struct.unpack('>I', edata)[0]
            p += 1 + elen
        off += ext_total
    return seq, marker, data[off:], frame_gen


class _H264LiveCapture:
    def __init__(self, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.bind(('', port))
        self._sock.settimeout(1.0)

        self._frame     = None
        self._ok        = False
        self._frame_id  = 0
        self._frame_gen = None
        self._lock      = threading.Lock()

        self._decoder   = None
        self._need_idr  = True
        self._last_seq  = None
        self._au_nals   = []
        self._au_gen    = None
        self._fu_buf    = None
        self._fu_type   = None
        self._closed    = False

        threading.Thread(target=self._reader, daemon=True).start()

    def _new_decoder(self):
        import av
        self._decoder = av.CodecContext.create('h264', 'r')

    def _reset_for_loss(self, reason):
        print(f"[H264] {reason} — waiting for next keyframe")
        self._need_idr = True
        self._au_nals   = []
        self._au_gen    = None
        self._fu_buf    = None
        self._new_decoder()   # drop any stale reference frames from before the gap

    def _reader(self):
        self._new_decoder()
        while not self._closed:
            try:
                data, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                with self._lock:
                    self._ok = False
                continue
            except OSError:
                break   # socket closed via close() — exit cleanly, not an error
            except Exception as e:
                print(f"[H264] recv: {e}")
                continue

            parsed = _parse_rtp(data)
            if parsed is None:
                continue
            seq, marker, payload, frame_gen = parsed
            if not payload:
                continue

            if self._last_seq is not None and seq != (self._last_seq + 1) & 0xFFFF:
                self._reset_for_loss(f"packet loss (seq {self._last_seq}→{seq})")
            self._last_seq = seq

            nal_type = payload[0] & 0x1F
            if nal_type == _FU_A_TYPE:
                if len(payload) < 2:
                    continue
                fu_header = payload[1]
                start, end = bool(fu_header & 0x80), bool(fu_header & 0x40)
                orig_type  = fu_header & 0x1F
                fnri       = payload[0] & 0xE0
                chunk      = payload[2:]
                if start:
                    self._fu_buf  = bytearray([fnri | orig_type]) + chunk
                elif self._fu_buf is not None:
                    self._fu_buf += chunk
                if end and self._fu_buf is not None:
                    self._au_nals.append(bytes(self._fu_buf))
                    self._fu_buf = None
            else:
                self._au_nals.append(payload)

            if frame_gen is not None:
                self._au_gen = frame_gen

            if marker:
                self._handle_access_unit(self._au_nals, self._au_gen)
                self._au_nals = []
                self._au_gen  = None

    def _handle_access_unit(self, nals, frame_gen):
        if not nals:
            return
        has_idr = any((n[0] & 0x1F) == 5 for n in nals)
        if self._need_idr and not has_idr:
            return   # still resyncing — drop inter-frame AUs until the next keyframe
        if has_idr:
            self._need_idr = False

        import av
        bitstream = b"".join(b"\x00\x00\x00\x01" + n for n in nals)
        try:
            frames = self._decoder.decode(av.Packet(bitstream))
        except Exception as e:
            self._reset_for_loss(f"decode error ({e})")
            return

        for vf in frames:
            img = vf.to_ndarray(format='bgr24')
            with self._lock:
                self._frame     = img
                self._ok        = True
                self._frame_id += 1
                self._frame_gen = frame_gen

    def read(self):
        """Return (ok, frame_copy, frame_id, frame_gen). Same contract as _LiveCapture."""
        with self._lock:
            if self._frame is None:
                return False, None, 0, None
            return self._ok, self._frame.copy(), self._frame_id, self._frame_gen

    def close(self):
        """Stop the reader thread and release the socket — call before
        switching to a different capture class on the same UDP port."""
        self._closed = True
        try:
            self._sock.close()
        except Exception:
            pass


# ── Local file capture (--file, no Pi needed) ───────────────────────────────
#
# Same read() contract as _LiveCapture/_H264LiveCapture — (ok, frame,
# frame_id, frame_gen) — so every downstream consumer (HUD, stabilization,
# feature tracking, virtual target, zoom loupe, recording) works unmodified.
# frame_gen is always None: select_point()'s stale-click replay is a Pi-side
# feature with nothing to replay against here, and _post's absent-frame_gen
# fallback already handles that. Loops back to frame 0 at end of file so a
# short recording is still useful for an extended test session.

class _FileCapture:
    def __init__(self, path):
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise SystemExit(f"Could not open video file: {path}")
        src_fps   = self._cap.get(cv2.CAP_PROP_FPS)
        self._delay = 1.0 / src_fps if src_fps and src_fps > 1 else 1.0 / 30.0

        self._frame    = None
        self._ok       = False
        self._frame_id = 0
        self._lock     = threading.Lock()

        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while True:
            ok, frame = self._cap.read()
            if not ok:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop
                continue
            with self._lock:
                self._frame     = frame
                self._ok        = True
                self._frame_id += 1
            time.sleep(self._delay)

    def read(self):
        """Return (ok, frame_copy, frame_id, frame_gen). Same contract as _LiveCapture."""
        with self._lock:
            if self._frame is None:
                return False, None, 0, None
            return self._ok, self._frame.copy(), self._frame_id, None


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    global launched, _cur_video_w, _cur_video_h, est_fps_ref, _confirm_quit, _frame_gen
    global cap, _gcs_video_mode
    global _local_recording, _local_writer
    global _hires_pending, _hires_crop_img, _hires_request_id

    if args.file:
        cap = _FileCapture(args.file)
        set_status(f"Offline: playing {args.file}")
        print(f"[GCS] video_mode=file ({args.file}) — no Pi connection, commands are no-ops")
    else:
        data     = _get("status")
        launched = data.get("launched", False)
        if data:
            set_status(f"Connected to {PI_IP}")
        else:
            set_status(f"Commands via UDP broadcast — waiting for video…")

        video_mode = _load_video_mode()
        cap = _H264LiveCapture(UDP_PORT) if video_mode == "h264_udp" else _LiveCapture(UDP_PORT)
        if video_mode in ("jpeg_udp", "h264_udp"):
            _gcs_video_mode = video_mode
        print(f"[GCS] video_mode={video_mode}")
    gamepad = _Gamepad()

    cv2.namedWindow("Mahat GCS", cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, _):
        global _confirm_quit, _press_on_video, _press_start_ts, _hires_crop_img
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
                _press_start_ts = time.time()
                _drag_start[0], _drag_start[1] = x, y
                _hires_crop_img = None      # new press — drop any stale hires result
            else:
                for btn in _buttons:        # click on panel → button action
                    if btn.hit(x, y):
                        btn.action()
                        break
        elif event == cv2.EVENT_LBUTTONUP:
            if _press_on_video:
                _press_on_video = False
                # A small release-vs-press movement is a plain click (fixed-
                # size box at the point, as before); dragging further draws
                # a rectangle instead — much easier to land on a target when
                # the video itself is shaking too much to click it precisely.
                # Only offered in white-target mode — that's the only case
                # this was built for (searching a white blob within a marked
                # area); a plain click always behaves exactly as before it.
                if not white_target_enabled or (
                        abs(x - _drag_start[0]) < DRAG_MIN_PX
                        and abs(y - _drag_start[1]) < DRAG_MIN_PX):
                    fx, fy = _to_raw_coords(x, y)   # undo display-only stabilization shift
                    select_point(fx, fy)
                else:
                    fx0, fy0 = _to_raw_coords(_drag_start[0], _drag_start[1])
                    fx1, fy1 = _to_raw_coords(x, y)
                    select_area(fx0, fy0, fx1, fy1)
                _reset_loupe_blend()

    cv2.setMouseCallback("Mahat GCS", on_mouse)

    last_frame_ts   = None
    last_frame_id   = 0
    last_rec_id     = 0
    est_fps         = 0.0
    FPS_A           = 0.9

    while not _quit.is_set():
        ok, frame, frame_id, frame_gen = cap.read()
        unstab_frame = None   # set below only for a real (non-placeholder) frame while recording

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

            _feat_update(frame, is_new_frame)   # raw-frame coords — must run before the stabilization warp,
                                                 # and before _vt_update, which consumes its _feat_last_delta
            _vt_update(frame, is_new_frame)
            draw_feature_points(frame)          # drawn here (pre-warp) so dots move WITH the content, no
                                                 # separate raw->display coordinate conversion needed yet

            # Snapshot BEFORE the stabilization warp, for local recording —
            # stabilization is a display-only aid (it doesn't touch the real
            # tracker/MAVLink pipeline either); a recording made from it
            # would bake in the crop/warp and any stabilizer artifacts
            # permanently, unlike the live view where toggling STAB back off
            # immediately shows the raw feed again. Must be an actual copy,
            # not just the same reference — _stabilize() returns a NEW array
            # from cv2.warpAffine when it applies a correction, but the SAME
            # array unchanged when it has no correction yet, so relying on
            # the reference alone would inconsistently alias `frame` (and
            # then pick up the HUD drawn on the display copy below, or not,
            # depending on stabilizer state).
            unstab_frame = frame.copy() if _local_recording else None

            if _stab_enabled:
                frame = _stabilize(frame, is_new_frame)

        h, w = frame.shape[:2]

        # Rebuild button layout if display size changed. Width alone isn't
        # enough to detect this: the resize above always forces width to
        # DISPLAY_W regardless of the stream's aspect ratio, so switching
        # the Pi's MAIN resolution between 4:3 and 16:9 changes the real
        # displayed height without changing w — leaving _cur_video_h stuck
        # at a stale value and silently breaking ny normalization (worse the
        # further from y=0, since the wrong denominator's effect grows with
        # distance from the origin — this was the "top correct, bottom
        # wrong" click bug).
        if w != _cur_video_w or h != _cur_video_h:
            _cur_video_w = w
            _cur_video_h = h
            _build_buttons(w)
            _reset_stabilizer()   # stale trajectory/reference frame would no longer match this size

        est_fps_ref[0] = est_fps   # share live FPS with button lambdas

        draw_hud(frame, est_fps)
        draw_virtual_target(frame)
        if unstab_frame is not None:
            draw_hud(unstab_frame, est_fps)
            # VT marker deliberately NOT drawn here — draw_virtual_target()
            # positions it via _vt_to_display(), which applies the
            # stabilization transform forward; on the unwarped frame that
            # would place the marker at the wrong spot rather than just
            # omitting it.

        # ── Drag-to-select-area preview — once the press has moved past the
        # click/drag threshold, show the rectangle being drawn instead of the
        # zoom loupe (the loupe is for landing a precise single point; once
        # the operator is clearly dragging an area, that precision isn't the
        # goal any more) ──────────────────────────────────────────────────
        dragging_area = (white_target_enabled and _press_on_video
                          and (abs(_mouse_pos[0] - _drag_start[0]) >= DRAG_MIN_PX
                               or abs(_mouse_pos[1] - _drag_start[1]) >= DRAG_MIN_PX))
        if dragging_area:
            mx, my = _mouse_pos
            rx0, rx1 = sorted((_drag_start[0], min(mx, w - 1)))
            ry0, ry1 = sorted((_drag_start[1], min(my, h - 1)))
            cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (0, 255, 255), 1, cv2.LINE_AA)

        # ── Hires zoom — takes over the long-press gesture from the plain
        # zoom loupe while enabled (pre-launch only; the button/gesture
        # already refuse post-launch, this is just the render side).
        # One fetch per press: triggered the moment the hold threshold is
        # crossed, shown once it arrives, held steady until release starts
        # a fresh press (see on_mouse's LBUTTONDOWN reset of _hires_crop_img).
        hires_active = (_hires_zoom_enabled and not launched
                        and _press_on_video and not dragging_area
                        and (time.time() - _press_start_ts) >= _ZOOM_HOLD_S)
        if hires_active:
            mx, my = _mouse_pos
            if mx < w:
                if _hires_crop_img is None and not _hires_pending:
                    _hires_pending = True
                    _hires_request_id += 1
                    fx, fy = _to_raw_coords(min(w - 1, mx), min(h - 1, my))
                    nx_req = fx / _cur_video_w
                    ny_req = fy / _cur_video_h
                    threading.Thread(target=_fetch_hires_crop,
                                      args=(nx_req, ny_req, _hires_request_id),
                                      daemon=True).start()
                if _hires_crop_img is not None:
                    draw_hires_loupe(frame, min(w - 1, mx), min(h - 1, my), _hires_crop_img)
                else:
                    cv2.putText(frame, "High res capturing",
                                (min(w - 220, mx + 15), max(20, my - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 230), 1, cv2.LINE_AA)

        # ── Zoom loupe — appears only after a long press (a quick click just
        # selects the cursor point directly) ─────────────────────────────────
        elif (_zoom_enabled and _press_on_video and not dragging_area
                and (time.time() - _press_start_ts) >= _ZOOM_HOLD_S):
            mx, my = _mouse_pos
            if mx < w:
                mx_c, my_c = min(w - 1, mx), min(h - 1, my)
                crop, x0, y0 = _smoothed_crop(frame, mx_c, my_c)
                draw_zoom_loupe(frame, mx_c, my_c, crop, x0, y0)

        # ── Gamepad lock — Square/rectangle button selects at mouse position ─
        if gamepad.lock_pressed():
            fx, fy = _to_raw_coords(min(w - 1, _mouse_pos[0]), min(h - 1, _mouse_pos[1]))
            lock_target(fx, fy)

        # Write to local recorder (video + HUD, no panel) — only on genuinely
        # new frames, so recorded playback speed matches real time instead of
        # being stretched by the render loop re-writing the same cached frame.
        # Recorded from unstab_frame (pre-stabilization-warp) when available,
        # so the recording is unaffected by STAB even while it's on for the
        # live view — falls back to the placeholder frame when there's no
        # real video yet (unstab_frame is only set for a genuine capture).
        if _local_recording and _local_writer is not None and frame_id != last_rec_id:
            last_rec_id = frame_id
            rec_source = unstab_frame if unstab_frame is not None else frame
            fh, fw = rec_source.shape[:2]
            rec_frame = rec_source if (fw, fh) == (_cur_video_w, _cur_video_h) else \
                        cv2.resize(rec_source, (_cur_video_w, _cur_video_h))
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

    # Finalize any in-progress local recording — without this, quitting
    # (window close, Q key, or an external kill/Ctrl-C now caught above)
    # while still recording leaves the MP4 without its index, permanently
    # unplayable rather than just missing its last few seconds.
    if _local_writer is not None:
        _local_writer.release()
        _local_writer = None
        _local_recording = False
        print("[GCS] Local recording finalized on exit.")

    cv2.destroyAllWindows()
    print("[GCS] Bye.")


if __name__ == "__main__":
    main()
