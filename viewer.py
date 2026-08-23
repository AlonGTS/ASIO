#!/usr/bin/env python3
"""
Mahat GCS Viewer — read-only video, no controls.

For a second (third, ...) computer watching alongside the one real
controller GCS (gcs.py). The Pi sends video once, to a UDP multicast
group (see config.toml's video_multicast_group) — this just joins that
group and displays whatever arrives. No connection to the Pi at all:
nothing is ever sent, so there's no way for this script to issue a
command even by accident.

Usage:
    python viewer.py                 # reads video_mode/group from config.toml
    python viewer.py --udp 5600

Keyboard: Q to close this window. Does not affect the Pi or the controller
GCS in any way — this window is the only thing that closes.
"""
import argparse
import os
import time
from pathlib import Path

import cv2

from video_capture import LiveCapture, H264LiveCapture, DEFAULT_MULTICAST_GROUP

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# ── Config (same lookup gcs.py uses, kept tiny/self-contained on purpose —
# this file has no other reason to depend on gcs.py, which has heavy
# import-time side effects of its own: argparse over sys.argv, opening
# sockets, starting threads) ────────────────────────────────────────────────

def _load_video_mode():
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

def _load_multicast_group():
    cfg_path = Path(__file__).parent / "config.toml"
    if cfg_path.exists():
        try:
            import tomllib
            with open(cfg_path, "rb") as f:
                cfg = tomllib.load(f)
            return cfg["network"].get("video_multicast_group", DEFAULT_MULTICAST_GROUP)
        except Exception:
            pass
    return DEFAULT_MULTICAST_GROUP

parser = argparse.ArgumentParser(description="Mahat GCS Viewer (read-only)")
parser.add_argument("--udp",   type=int, default=5600, help="UDP video port  (default 5600)")
parser.add_argument("--group", default=None, help="Multicast group (overrides config.toml)")
args = parser.parse_args()

MCAST_GROUP = args.group or _load_multicast_group()
print(f"[Viewer] group={MCAST_GROUP}:{args.udp}")

# ── Main loop ────────────────────────────────────────────────────────────────

DISPLAY_W = 1000
_FONT = cv2.FONT_HERSHEY_SIMPLEX

def _waiting_frame(w, h):
    import numpy as np
    frame = np.zeros((h, w, 3), dtype="uint8")
    cv2.putText(frame, "Waiting for video…", (24, h // 2), _FONT, 0.8, (150, 150, 150), 2, cv2.LINE_AA)
    return frame

def main():
    video_mode = _load_video_mode()
    cap = H264LiveCapture(args.udp, MCAST_GROUP) if video_mode == "h264_udp" else LiveCapture(args.udp, MCAST_GROUP)
    print(f"[Viewer] video_mode={video_mode}")

    cv2.namedWindow("Mahat GCS Viewer", cv2.WINDOW_AUTOSIZE)

    est_fps, last_ts, last_id = 0.0, None, 0
    FPS_A = 0.9

    while True:
        ok, frame, frame_id, _frame_gen = cap.read()

        if not ok or frame is None:
            frame = _waiting_frame(DISPLAY_W, int(DISPLAY_W * 3 / 4))
        else:
            if frame_id != last_id:
                now = time.time()
                if last_ts is not None:
                    inst = 1.0 / max(1e-6, now - last_ts)
                    est_fps = FPS_A * est_fps + (1 - FPS_A) * inst if est_fps else inst
                last_ts, last_id = now, frame_id

            fh, fw = frame.shape[:2]
            if fw != DISPLAY_W:
                dh = int(fh * DISPLAY_W / fw)
                frame = cv2.resize(frame, (DISPLAY_W, dh), interpolation=cv2.INTER_LINEAR)

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 30), (20, 20, 20), -1)
        cv2.putText(frame, f"VIEWER (read-only)   FPS {est_fps:4.1f}", (8, 21),
                    _FONT, 0.55, (0, 165, 255), 1, cv2.LINE_AA)

        cv2.imshow("Mahat GCS Viewer", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
