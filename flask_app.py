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

# Selection-box size, scaled from a fixed pixel size at the 640x480
# baseline (moving 22x22, fixed 58x58, nudge-default 43x43 — ~28% smaller
# than the original 30/80/60) so the box covers roughly the same
# real-world target size no matter which MAIN capture resolution is
# active. The box is always square: it's scaled by a single factor
# (geometric mean of the width and height ratios to the baseline) rather
# than independent width/height fractions, so switching to a non-4:3
# resolution (e.g. 1280x720) can't skew it into a rectangle.
BOX_BASE_MOVING        = 22
BOX_BASE_FIXED         = 58
BOX_BASE_NUDGE_DEFAULT = 43


def _scaled_square(base, mw, mh):
    scale = ((mw / 640) * (mh / 480)) ** 0.5
    side = max(2, round(base * scale))
    return side, side


def box_size(mw, mh, moving):
    """Selection-box (w, h) in MAIN pixels, scaled to the mw x mh frame. Always square."""
    base = BOX_BASE_MOVING if moving else BOX_BASE_FIXED
    return _scaled_square(base, mw, mh)


def create_app(state, create_tracker_fn, cycle_main_fn=None, cycle_lores_fn=None, launch_fn=None, get_launch_state_fn=None, toggle_record_fn=None, get_record_state_fn=None, set_fps_fn=None, get_fps_state_fn=None, get_cpu_fn=None, get_cpu_temp_fn=None, get_frame_history_fn=None, set_video_mode_fn=None, get_video_mode_fn=None, set_white_target_fn=None, get_white_target_fn=None, get_aim_phase_fn=None):
    """
    Build and return the Flask app with all control routes bound to `state`.
    state is a SimpleNamespace with: command_from_remote, bbox, tracking,
    tracker, current_frame, bMoovingTgt, lores_size.
    create_tracker_fn(moving: bool) -> cv2 tracker
    get_frame_history_fn(min_gen) -> [(gen, frame), ...] oldest-first for
    min_gen onward, or None if min_gen already aged out of the buffer. Lets
    /select_point initialize against the frame the operator actually clicked
    on (echoed back as frame_gen) and replay forward to the live frame,
    instead of a stale click landing on whatever is live when the request
    is processed. See STABILIZATION.md.
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

        If the request includes frame_gen (the id of the frame the GCS
        operator actually saw — echoed back from the frame_gen embedded in
        the video stream, see tracker-so.py's UDP stream worker), initialize
        against that historical frame instead of whatever is live now, then
        replay the tracker forward through the buffered frames captured
        since then so it arrives at the live frame already converged,
        instead of jumping across the round-trip gap in one step. Falls back
        to plain live-frame init if frame_gen is absent, already aged out of
        the buffer, or the replay loses the target partway through.
        """
        try:
            def _live_frame_snap():
                with state.frame_lock:
                    if state.current_frame is None:
                        return None
                    return state.current_frame.copy()

            requested_gen = request.form.get("frame_gen")
            replay_frames = None
            if requested_gen is not None and get_frame_history_fn is not None:
                replay_frames = get_frame_history_fn(int(requested_gen))
                if replay_frames is None:
                    print(f"[INFO] frame_gen={requested_gen} already aged out of "
                          f"history — falling back to live-frame init")

            frame_snap = replay_frames[0][1] if replay_frames else _live_frame_snap()
            if frame_snap is None:
                return "No frame", 400

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

            w, h = box_size(mw, mh, state.bMoovingTgt)
            x0 = max(0, min(mw - w, x - w // 2))
            y0 = max(0, min(mh - h, y - h // 2))

            lw, lh = state.lores_size
            sx = lw / mw; sy = lh / mh
            xb = int(x0 * sx); yb = int(y0 * sy)
            wb = max(2, int(w * sx)); hb = max(2, int(h * sy))

            lores_frame = cv2.resize(frame_snap, (lw, lh), interpolation=cv2.INTER_LINEAR)
            t = create_tracker_fn(state.bMoovingTgt)
            t.init(lores_frame, (xb, yb, wb, hb))
            bbox_main = (x0, y0, w, h)

            # Replay forward through the frames captured after the clicked
            # one — same resize/update the live loop does every iteration —
            # so the tracker catches the actual intervening motion instead
            # of guessing across the whole gap in a single jump.
            replayed = 0
            if replay_frames and len(replay_frames) > 1:
                for _, hist_frame in replay_frames[1:]:
                    hist_lores = cv2.resize(hist_frame, (lw, lh), interpolation=cv2.INTER_NEAREST)
                    success, bbox_lo = t.update(hist_lores)
                    if not success:
                        print(f"[INFO] Replay lost target after {replayed} buffered frame(s) "
                              f"(clicked frame_gen={requested_gen}) — falling back to live-frame init")
                        frame_snap = _live_frame_snap()
                        if frame_snap is None:
                            return "No frame", 400
                        mh, mw = frame_snap.shape[:2]
                        sx = lw / mw; sy = lh / mh
                        x0 = max(0, min(mw - w, x - w // 2))
                        y0 = max(0, min(mh - h, y - h // 2))
                        xb = int(x0 * sx); yb = int(y0 * sy)
                        wb = max(2, int(w * sx)); hb = max(2, int(h * sy))
                        lores_frame = cv2.resize(frame_snap, (lw, lh), interpolation=cv2.INTER_LINEAR)
                        t = create_tracker_fn(state.bMoovingTgt)
                        t.init(lores_frame, (xb, yb, wb, hb))
                        bbox_main = (x0, y0, w, h)
                        replay_frames = None
                        break
                    xl, yl, wl, hl = map(int, bbox_lo)
                    bbox_main = (int(xl / sx), int(yl / sy),
                                 max(2, int(wl / sx)), max(2, int(hl / sy)))
                    replayed += 1

            state.tracking = False
            state.bbox = None
            state.tracker = None

            state.last_init_source = "click"
            state.tracker = t
            state.bbox = bbox_main
            state.tracking = True

            if replay_frames:
                print(f"[INFO] Tracker init @ MAIN({mw}x{mh}) from frame_gen={requested_gen}, "
                      f"replayed {replayed} frame(s) → bbox {state.bbox} | LORES {lw}x{lh}")
            else:
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
                bw, bh = _scaled_square(BOX_BASE_NUDGE_DEFAULT, mw, mh)
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
            state.last_init_source = "nudge"
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

    @app.route('/launch', methods=['POST'])
    def launch():
        """Set launch state explicitly: state=1 to launch, state=0 to reset."""
        if launch_fn is None:
            return "Not available", 400
        state_val = request.form.get("state")
        launch_fn(state_val == '1' if state_val is not None else None)
        return "OK", 200

    @app.route('/toggle_record', methods=['POST'])
    def toggle_record():
        """Toggle Pi-side recording on or off."""
        if toggle_record_fn is None:
            return "Not available in this mode", 400
        try:
            toggle_record_fn()
            recording = get_record_state_fn() if get_record_state_fn else None
            return {"recording": recording}, 200
        except Exception as e:
            return f"Error: {e}", 400

    @app.route('/set_fps', methods=['POST'])
    def set_fps():
        """Explicit GCS toggle: state=1 → full fps (active), state=0 → idle (power-save)."""
        if set_fps_fn is None:
            return "Not available in this mode", 400
        try:
            state_val = request.form.get("active")
            set_fps_fn(state_val == '1' if state_val is not None else True)
            return "OK", 200
        except Exception as e:
            return f"Error: {e}", 400

    @app.route('/set_white_target', methods=['POST'])
    def set_white_target():
        """Explicit GCS toggle: state=1 → white-target aim refinement/recovery
        on, state=0 → plain bbox-center tracking (as before that feature)."""
        if set_white_target_fn is None:
            return "Not available in this mode", 400
        try:
            state_val = request.form.get("enabled")
            set_white_target_fn(state_val == '1' if state_val is not None else True)
            return "OK", 200
        except Exception as e:
            return f"Error: {e}", 400

    @app.route('/set_video_mode', methods=['POST'])
    def set_video_mode():
        """Live-switch the video transport: mode='jpeg_udp' or 'h264_udp'.
        Not available if the Pi started in 'webrtc' mode, or a mode the
        underlying tracker-so.py doesn't recognize — see set_video_mode_fn."""
        if set_video_mode_fn is None:
            return "Not available in this mode", 400
        try:
            mode = request.form.get("mode")
            ok = set_video_mode_fn(mode)
            return ("OK", 200) if ok else (f"Could not switch to {mode!r}", 400)
        except Exception as e:
            return f"Error: {e}", 400

    @app.route('/status', methods=['GET'])
    def status():
        """Return current server-side state for UI initialization."""
        from flask import jsonify
        launched  = get_launch_state_fn()  if get_launch_state_fn  else False
        recording = get_record_state_fn()  if get_record_state_fn  else False
        active_fps  = get_fps_state_fn()   if get_fps_state_fn     else False
        cpu_percent = get_cpu_fn()         if get_cpu_fn           else None
        cpu_temp    = get_cpu_temp_fn()    if get_cpu_temp_fn      else None
        video_mode  = get_video_mode_fn()  if get_video_mode_fn    else None
        white_target = get_white_target_fn() if get_white_target_fn else False
        aim_phase   = get_aim_phase_fn()   if get_aim_phase_fn    else "off"
        return jsonify({
            "launched": launched, "recording": recording,
            "active_fps": active_fps, "cpu_percent": cpu_percent,
            "cpu_temp": cpu_temp, "video_mode": video_mode,
            "white_target": white_target, "aim_phase": aim_phase,
        })

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
