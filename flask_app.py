#!/usr/bin/env python3
"""
Flask control API for tracker.py.

Usage:
    import flask_app
    app = flask_app.create_app(state, create_csrt_tracker)
    # then run app in a thread
"""
import cv2
from flask import Flask, request
from flask_cors import CORS


def create_app(state, create_tracker_fn, cycle_main_fn=None, cycle_lores_fn=None):
    """
    Build and return the Flask app with all control routes bound to `state`.
    state is a SimpleNamespace with: command_from_remote, bbox, tracking,
    tracker, current_frame, bMoovingTgt, lores_size.
    create_tracker_fn(moving: bool) -> cv2 tracker
    """
    app = Flask(__name__)
    CORS(app)

    @app.route('/command', methods=['POST'])
    def command():
        """Accept 'r' (reset), 's' (stop), 'q' (quit) from the web UI."""
        cmd = request.form.get("cmd")
        if cmd in ['r', 's', 'q']:
            state.command_from_remote = cmd
            return "OK", 200
        return "Invalid", 400

    @app.route('/select_point', methods=['POST'])
    def select_point():
        """
        Initialize tracker around a clicked point.
        Prefer normalized coords (nx, ny in [0..1]); fall back to absolute (x, y).
        """
        try:
            with state.frame_lock:
                if state.current_frame is None:
                    return "No frame", 400
                frame_snap = state.current_frame.copy()

            mh, mw = frame_snap.shape[:2]
            nx = request.form.get("nx")
            ny = request.form.get("ny")
            if nx is not None and ny is not None:
                nx = max(0.0, min(1.0, float(nx)))
                ny = max(0.0, min(1.0, float(ny)))
                x = int(round(nx * (mw - 1)))
                y = int(round(ny * (mh - 1)))
            else:
                x = max(0, min(mw - 1, int(request.form.get("x"))))
                y = max(0, min(mh - 1, int(request.form.get("y"))))

            w, h = (30, 30) if state.bMoovingTgt else (80, 80)
            x0 = max(0, min(mw - w, x - w // 2))
            y0 = max(0, min(mh - h, y - h // 2))

            lw, lh = state.lores_size
            sx = lw / mw; sy = lh / mh
            xb = int(x0 * sx); yb = int(y0 * sy)
            wb = max(2, int(w * sx)); hb = max(2, int(h * sy))

            lores_frame = cv2.resize(frame_snap, (lw, lh),
                                     interpolation=cv2.INTER_LINEAR)
            state.tracking = False
            state.bbox = None
            state.tracker = None

            t = create_tracker_fn(state.bMoovingTgt)
            t.init(lores_frame, (xb, yb, wb, hb))
            state.tracker = t
            state.bbox = (x0, y0, w, h)
            state.tracking = True

            print(f"[INFO] Tracker init @ MAIN({mw}x{mh}) from "
                  f"{'normalized' if request.form.get('nx') is not None else 'absolute'} "
                  f"click → bbox {state.bbox} | LORES {lw}x{lh}")
            return "OK", 200
        except Exception as e:
            print(f"[ERROR] select_point: {e}")
            return f"Error: {e}", 400

    @app.route('/nudge', methods=['POST'])
    def nudge():
        """
        Shift bbox by (dx, dy) in MAIN coords and reinitialize tracker.
        If no bbox yet, starts one at the frame center.
        """
        try:
            dx = int(request.form.get("dx", 0))
            dy = int(request.form.get("dy", 0))
            with state.frame_lock:
                if state.current_frame is None:
                    return "No frame", 400
                frame_snap = state.current_frame.copy()

            mh, mw = frame_snap.shape[:2]
            lw, lh = state.lores_size
            sx = lw / mw; sy = lh / mh

            if state.bbox is None:
                bw = bh = 60
                x = max(0, mw // 2 - bw // 2)
                y = max(0, mh // 2 - bh // 2)
            else:
                x, y, bw, bh = map(int, state.bbox)

            x = max(0, min(mw - bw, x + dx))
            y = max(0, min(mh - bh, y + dy))
            bbox_main = (x, y, bw, bh)

            lores_frame = cv2.resize(frame_snap, (lw, lh),
                                     interpolation=cv2.INTER_LINEAR)
            xb = int(x * sx); yb = int(y * sy)
            wb = max(2, int(bw * sx)); hb = max(2, int(bh * sy))

            t = create_tracker_fn(state.bMoovingTgt)
            t.init(lores_frame, (xb, yb, wb, hb))
            state.tracker = t
            state.bbox = bbox_main
            state.tracking = True
            print(f"[INFO] Nudged bbox MAIN→{bbox_main} (LORES {lw}x{lh})")
            return "OK", 200
        except Exception as e:
            print(f"[ERROR] Nudge failed: {e}")
            return f"Error: {e}", 400

    @app.route('/set_target_mode', methods=['POST'])
    def set_target_mode():
        """Toggle fixed/moving target mode (affects CSRT params on next init)."""
        val = request.form.get("bMoovingTgt", "0")
        state.bMoovingTgt = (val == "1")
        print(f"[INFO] Target mode set to: {'MOVING' if state.bMoovingTgt else 'FIXED'}")
        return "OK", 200

    @app.route('/cycle_main', methods=['POST'])
    def cycle_main():
        """Cycle MAIN capture resolution up (+1) or down (-1). Live mode only."""
        if cycle_main_fn is None:
            return "Not available", 400
        try:
            delta = int(request.form.get("delta", 1))
            cycle_main_fn(delta)
            return "OK", 200
        except Exception as e:
            return f"Error: {e}", 400

    @app.route('/cycle_lores', methods=['POST'])
    def cycle_lores():
        """Cycle LORES tracking resolution up (+1) or down (-1)."""
        if cycle_lores_fn is None:
            return "Not available", 400
        try:
            delta = int(request.form.get("delta", 1))
            cycle_lores_fn(delta)
            return "OK", 200
        except Exception as e:
            return f"Error: {e}", 400

    return app
