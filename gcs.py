#!/usr/bin/env python3
"""
Mahat GCS — Ground Control Station
Receives UDP video from the Pi tracker and controls it via Flask API.

Usage:
    python gcs.py --pi 192.168.1.100
    python gcs.py          # reads gcs_ip from config.toml if present

Mouse  : left-click on video → select tracking target
         left-click on buttons → same as keyboard shortcuts

Keyboard shortcuts (work whether or not the mouse is in the window):
    R        Reset tracker
    S        Stop tracking
    L        Toggle launch
    M        Toggle Fixed / Moving target mode
    Arrows   Nudge target (5 px)
    X / Z    Cycle MAIN resolution  + / −
    V / C    Cycle TRACK resolution + / −
    P        Toggle local recording
    Q        Quit
"""

import argparse
import os
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

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
PANEL_MIN_H = 560     # minimum canvas height so all buttons fit
DISPLAY_W   = 640     # video is always stretched to this width for display

# ── Shared state ──────────────────────────────────────────────────────────────

launched      = False
moving_tgt    = False
_pi_recording = False   # Pi-side recording state (optimistic: toggled on each command)
_status    = ""
_status_ts = 0.0
_mouse_pos = [0, 0]   # updated by mouse callback; used for hover highlight
_quit      = threading.Event()  # set to break the main loop from any thread

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
    _post("select_point", nx=nx, ny=ny)
    set_status(f"Selected ({x}, {y})")

def cycle_main(delta):
    _post("cycle_main", delta=delta)
    set_status(f"MAIN {'up' if delta > 0 else 'down'}")

def cycle_lores(delta):
    _post("cycle_lores", delta=delta)
    set_status(f"TRACK {'up' if delta > 0 else 'down'}")

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

    btn("Quit", 36, quit_gcs, (40, 40, 170))
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
    y += 52

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

    txt(f"FPS {fps:4.1f}",                            (8,   25))
    txt(f"{'MOVING' if moving_tgt else 'FIXED'}",     (130, 25), color=mode_col)
    txt("LAUNCHED" if launched else "READY",           (255, 25), color=launch_col)

    # Pi REC indicator — blinking every second, plus FPS delta from baseline
    if _pi_recording:
        # Blink: alternate dot colour each second
        dot_col = (40, 40, 220) if int(time.time()) % 2 == 0 else (100, 100, 255)
        cv2.circle(frame, (w - 16, 18), 7, dot_col, -1)
        if _fps_baseline is not None:
            delta = fps - _fps_baseline
            sign  = "+" if delta >= 0 else ""
            delta_col = (80, 200, 80) if delta > -0.5 else (80, 80, 220)
            txt(f"Pi REC  ({sign}{delta:.1f})", (w - 155, 25), color=delta_col)
        else:
            txt("Pi REC", (w - 80, 25), color=(100, 100, 255))

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

class _LiveCapture:
    def __init__(self, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)  # 1 MB
        self._sock.bind(('', port))
        self._sock.settimeout(1.0)
        self._frame = None
        self._ok    = False
        self._lock  = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        """Receive JPEG datagrams and decode them; marks _ok=False on timeout."""
        while True:
            try:
                data, _ = self._sock.recvfrom(1 << 16)  # 65536 bytes max UDP payload
                arr   = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._ok    = True
            except socket.timeout:
                with self._lock:
                    self._ok = False   # no packet for 1 s → show waiting screen
            except Exception as e:
                print(f"[UDP] recv: {e}")

    def read(self):
        """Return (ok, frame_copy).  Never blocks more than the lock."""
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    global launched, _cur_video_w, _cur_video_h, est_fps_ref

    data     = _get("status")
    launched = data.get("launched", False)
    if data:
        set_status(f"Connected to {PI_IP}")
    else:
        set_status(f"Commands via UDP broadcast — waiting for video…")

    cap = _LiveCapture(UDP_PORT)

    cv2.namedWindow("Mahat GCS", cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, _):
        _mouse_pos[0], _mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < _cur_video_w:
                select_point(x, y)          # click on video → track target
            else:
                for btn in _buttons:        # click on panel → button action
                    if btn.hit(x, y):
                        btn.action()
                        break

    cv2.setMouseCallback("Mahat GCS", on_mouse)

    prev_ts  = time.time()
    est_fps  = 0.0
    FPS_A    = 0.9

    while not _quit.is_set():
        ok, frame = cap.read()

        if not ok or frame is None:
            frame = _waiting_frame(_cur_video_w, _cur_video_h)
        else:
            # Stretch to DISPLAY_W regardless of stream resolution
            # so the window stays the same size even when stream is downscaled
            fh, fw = frame.shape[:2]
            if fw != DISPLAY_W:
                dh = int(fh * DISPLAY_W / fw)
                frame = cv2.resize(frame, (DISPLAY_W, dh), interpolation=cv2.INTER_LINEAR)

        h, w = frame.shape[:2]

        # Rebuild button layout if display width changed
        if w != _cur_video_w:
            _cur_video_w = w
            _cur_video_h = h
            _build_buttons(w)

        # FPS
        now      = time.time()
        dt       = max(1e-6, now - prev_ts)
        prev_ts  = now
        inst     = 1.0 / dt
        est_fps  = FPS_A * est_fps + (1 - FPS_A) * inst if est_fps else inst

        est_fps_ref[0] = est_fps   # share live FPS with button lambdas

        draw_hud(frame, est_fps)

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

        cv2.imshow("Mahat GCS", canvas)

        # ── Key handling ───────────────────────────────────────────────────
        key = cv2.waitKeyEx(1)
        if key == -1:
            continue

        direction = _ARROW.get(key)
        if direction:
            dx, dy = direction
            nudge(dx * 5, dy * 5)
            continue

        k = key & 0xFF
        if   k in (ord('q'), ord('Q')): quit_gcs()
        elif k in (ord('r'), ord('R')): send_cmd('r')
        elif k in (ord('s'), ord('S')): send_cmd('s')
        elif k in (ord('l'), ord('L')): send_launch()
        elif k in (ord('m'), ord('M')): toggle_target()
        elif k in (ord('x'), ord('X')): cycle_main(+1)
        elif k in (ord('z'), ord('Z')): cycle_main(-1)
        elif k in (ord('v'), ord('V')): cycle_lores(+1)
        elif k in (ord('c'), ord('C')): cycle_lores(-1)
        elif k in (ord('p'), ord('P')): toggle_pi_record(est_fps)

        if _quit.is_set():
            break

    cv2.destroyAllWindows()
    print("[GCS] Bye.")


if __name__ == "__main__":
    main()
