#!/usr/bin/env python3
"""
Mahat GCS — Ground Control Station
Receives UDP video from the Pi tracker and controls it via Flask API.

Usage:
    python gcs.py --pi 192.168.1.100
    python gcs.py          # reads gcs_ip from config.toml if present

Keyboard shortcuts:
    R        Reset tracker
    S        Stop tracking
    L        Toggle launch
    M        Toggle Fixed / Moving target mode
    Arrows   Nudge target (5 px)
    X / Z    Cycle MAIN resolution  ▲ / ▼
    V / C    Cycle TRACK resolution ▲ / ▼
    P        Toggle local recording (saves .mp4 next to this file)
    Q        Quit

Mouse:
    Left-click on video → select tracking target
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

# ── Config ───────────────────────────────────────────────────────────────────

def _load_toml():
    """Read Pi IP from config.toml if it lives next to gcs.py."""
    cfg_path = Path(__file__).parent / "config.toml"
    if cfg_path.exists():
        try:
            import tomllib
            with open(cfg_path, "rb") as f:
                cfg = tomllib.load(f)
            iface = cfg["network"]["interface"]
            return cfg["network"][iface]["gcs_ip"]
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

print(f"[GCS] Pi={PI_IP}  Flask={FLASK}  UDP port={UDP_PORT}")

# ── Shared state ──────────────────────────────────────────────────────────────

launched   = False
moving_tgt = False
_status    = ""
_status_ts = 0.0


def set_status(msg, *, log=True):
    global _status, _status_ts
    _status    = msg
    _status_ts = time.time()
    if log:
        print(f"[GCS] {msg}")


# ── Flask API helpers ─────────────────────────────────────────────────────────

def _post(endpoint, **data):
    """Fire-and-forget POST — never blocks the video loop."""
    def _go():
        try:
            r = requests.post(f"{FLASK}/{endpoint}", data=data, timeout=2)
            if not r.ok:
                set_status(f"API {r.status_code}: /{endpoint}")
        except requests.exceptions.ConnectionError:
            set_status(f"Pi not reachable ({PI_IP})")
        except Exception as e:
            set_status(f"API error: {e}")
    threading.Thread(target=_go, daemon=True).start()


def _get(endpoint):
    try:
        return requests.get(f"{FLASK}/{endpoint}", timeout=2).json()
    except Exception:
        return {}


def send_cmd(cmd):
    _post("command", cmd=cmd)
    set_status({"r": "Reset", "s": "Stop", "q": "Quit"}.get(cmd, cmd))


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
    _post("select_point", x=x, y=y)
    set_status(f"Selected ({x}, {y})")


def cycle_main(delta):
    _post("cycle_main", delta=delta)
    set_status(f"MAIN {'▲' if delta > 0 else '▼'}")


def cycle_lores(delta):
    _post("cycle_lores", delta=delta)
    set_status(f"TRACK {'▲' if delta > 0 else '▼'}")


# ── HUD drawing ───────────────────────────────────────────────────────────────

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_GREEN = (100, 255, 100)
_AMBER = (60,  180, 255)   # BGR — looks orange on screen
_GREY  = (160, 160, 160)
_RED   = (60,   60, 220)
_BLACK = (0,   0,   0)

HINTS = [
    "Click  select target",
    "R      reset tracker",
    "S      stop tracking",
    "L      launch toggle",
    "M      fixed/moving",
    "Arrows nudge (5 px)",
    "X/Z    MAIN res +/-",
    "V/C    TRACK res +/-",
    "P      record  (local)",
    "Q      quit",
]


def _txt(img, msg, pos, scale=0.55, color=_WHITE, thick=1):
    """Draw text with black outline for readability on any background."""
    cv2.putText(img, msg, pos, _FONT, scale, _BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, msg, pos, _FONT, scale, color,  thick,     cv2.LINE_AA)


def draw_hud(frame, fps, recording):
    h, w = frame.shape[:2]

    # ── top status bar ────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)

    mode_col   = _AMBER if moving_tgt else _GREEN
    launch_col = _AMBER if launched   else _WHITE
    rec_col    = _RED

    _txt(frame, f"FPS {fps:4.1f}",                     (8,   24))
    _txt(frame, f"Mode: {'MOVING' if moving_tgt else 'FIXED'}", (120, 24), color=mode_col)
    _txt(frame, "LAUNCHED" if launched else "READY",   (300, 24), color=launch_col)
    if recording:
        _txt(frame, "REC", (w - 55, 24), color=rec_col)

    # ── hint panel (right side, semi-transparent) ─────────────────────────
    pw = 190
    px = w - pw - 4
    py = 42
    row_h = 17
    panel_h = len(HINTS) * row_h + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (w, py + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, hint in enumerate(HINTS):
        _txt(frame, hint, (px + 4, py + 14 + i * row_h), scale=0.38, color=_GREY)

    # ── status bar (bottom, fades after 4 s) ─────────────────────────────
    age = time.time() - _status_ts
    if _status and age < 4.0:
        fade  = min(1.0, (4.0 - age) / 0.5)
        bar   = frame.copy()
        cv2.rectangle(bar, (0, h - 28), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)
        col = tuple(int(c * fade) for c in _GREEN)
        _txt(frame, _status, (8, h - 9), color=col)


# ── Waiting screen ────────────────────────────────────────────────────────────

def _waiting_frame(w=640, h=480):
    img = np.zeros((h, w, 3), np.uint8)
    _txt(img, f"Waiting for UDP stream on port {UDP_PORT}…",
         (max(8, w // 2 - 230), h // 2 - 12), scale=0.6)
    _txt(img, f"Pi: {PI_IP}   Flask: {FLASK}",
         (max(8, w // 2 - 180), h // 2 + 18), scale=0.5, color=_GREY)
    return img


# ── Arrow-key detection (cross-platform) ─────────────────────────────────────

# waitKeyEx returns different codes on Linux / macOS / Windows
_ARROW_MAP = {
    # (up,   down,  left,  right)
    65362: (0, -1), 65364: (0, 1), 65361: (-1, 0), 65363: (1, 0),  # Linux X11
    63232: (0, -1), 63233: (0, 1), 63234: (-1, 0), 63235: (1, 0),  # macOS
    2490368: (0,-1), 2621440:(0,1), 2424832:(-1,0), 2555904:(1,0), # Windows
    82:    (0, -1), 84:    (0, 1), 81:    (-1, 0), 83:    (1, 0),  # Linux waitKey fallback
}


def _arrow_dir(key):
    return _ARROW_MAP.get(key)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    global launched

    # Sync launch state from Pi at startup
    data   = _get("status")
    launched = data.get("launched", False)
    set_status(f"Connected to Pi ({PI_IP})" if data else f"Pi not reachable — {PI_IP}")

    # Open UDP stream
    # WINDOW_AUTOSIZE keeps 1-to-1 pixel mapping so click coords need no scaling
    cap = cv2.VideoCapture(f"udp://@:{UDP_PORT}", cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # discard buffered frames → low latency

    cv2.namedWindow("Mahat GCS", cv2.WINDOW_AUTOSIZE)

    # Mouse → select tracking point
    # WINDOW_AUTOSIZE: (x, y) from callback == frame pixel coords, no scaling needed
    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            select_point(x, y)

    cv2.setMouseCallback("Mahat GCS", on_mouse)

    # Local recording
    writer    = None
    recording = False

    # FPS meter
    prev_ts   = time.time()
    est_fps   = 0.0
    FPS_ALPHA = 0.9

    last_w = last_h = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            cv2.imshow("Mahat GCS", _waiting_frame(last_w or 640, last_h or 480))
            cap.open(f"udp://@:{UDP_PORT}")   # keep retrying
            key = cv2.waitKeyEx(500)
            if key != -1 and (key & 0xFF) in (ord('q'), ord('Q')):
                break
            continue

        last_h, last_w = frame.shape[:2]

        # FPS
        now     = time.time()
        dt      = max(1e-6, now - prev_ts)
        prev_ts = now
        inst    = 1.0 / dt
        est_fps = FPS_ALPHA * est_fps + (1 - FPS_ALPHA) * inst if est_fps else inst

        draw_hud(frame, est_fps, recording)

        if recording and writer is not None:
            writer.write(frame)

        cv2.imshow("Mahat GCS", frame)

        key = cv2.waitKeyEx(1)
        if key == -1:
            continue

        # Arrow keys
        direction = _arrow_dir(key)
        if direction:
            dx, dy = direction
            nudge(dx * 5, dy * 5)
            continue

        k = key & 0xFF

        if   k in (ord('q'), ord('Q')): send_cmd('q'); break
        elif k in (ord('r'), ord('R')): send_cmd('r')
        elif k in (ord('s'), ord('S')): send_cmd('s')
        elif k in (ord('l'), ord('L')): send_launch()
        elif k in (ord('m'), ord('M')): toggle_target()
        elif k in (ord('x'), ord('X')): cycle_main(+1)
        elif k in (ord('z'), ord('Z')): cycle_main(-1)
        elif k in (ord('v'), ord('V')): cycle_lores(+1)
        elif k in (ord('c'), ord('C')): cycle_lores(-1)
        elif k in (ord('p'), ord('P')):
            if not recording:
                ts     = time.strftime('%Y%m%d_%H%M%S')
                fname  = f"gcs_{ts}.mp4"
                h, w   = last_h, last_w
                writer = cv2.VideoWriter(
                    fname,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    25, (w, h)
                )
                recording = True
                set_status(f"Recording → {fname}")
            else:
                if writer is not None:
                    writer.release()
                    writer = None
                recording = False
                set_status("Recording saved")

    # ── Cleanup ────────────────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[GCS] Bye.")


if __name__ == "__main__":
    main()
