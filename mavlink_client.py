#!/usr/bin/env python3
"""
MAVLink client for tracker-so.py — routes through MAVProxy.

Start MAVProxy on the RPi before running tracker-so.py:
    mavproxy.py --master=/dev/ttyACM0 --baud=115200 \
                --out=udpout:127.0.0.1:14551 \
                --out=udp:<GCS_IP>:14550

Usage:
    import mavlink_client
    mavlink_client.set_autopilot("custom")    # or "px4" — call before connect()
    mavlink_client.connect()                  # call once at startup
    mavlink_client.send_vision_error(p, y, is_tracking)  # "custom" mode, every frame
    mavlink_client.send_attitude_target(pitch_err, yaw_err, thrust=t)  # "px4" mode, every frame

Two control schemes, selected by set_autopilot():
  "custom" (default) — our own Simulink-generated flight-control app running
      on the Pixhawk (PX4-based, see MAHAT.cpp). It owns all attitude control;
      we just feed it DEBUG_VECT "vision_err" + NAMED_VALUE_FLOAT "launch".
  "px4" — drive the stock PX4 flight stack directly: OFFBOARD mode entered
      via MAV_CMD_DO_SET_MODE, attitude/rate control via SET_ATTITUDE_TARGET,
      arm/disarm via MAV_CMD_COMPONENT_ARM_DISARM.
"""
import math
import os
import sys
import time
import struct
import threading
import subprocess
import atexit
import shutil

_connection    = None
_enabled       = False
_ser           = None
_launched      = False
_mavproxy_proc = None
DEBUG           = False  # set True to print pitch/yaw values every frame
SHOW_TELEMETRY  = False  # set True to print incoming ATTITUDE in the console

# "custom" or "px4" — set via set_autopilot() before connect(). Selects how
# connect() negotiates with the FC and which send_* function tracker-so.py
# should be driving every frame.
_autopilot = "custom"

# PX4 custom_mode packs main_mode into bits 16-23 (sub_mode in 24-31).
# Values from PX4's mavlink/mavlink_main.h custom mode enum.
_PX4_MAIN_MODE_MANUAL   = 1
_PX4_MAIN_MODE_OFFBOARD = 6
_PX4_MAIN_MODE_NAMES = {
    1: "MANUAL", 2: "ALTCTL", 3: "POSCTL", 4: "AUTO", 5: "ACRO",
    6: "OFFBOARD", 7: "STABILIZED", 8: "RATTITUDE",
}

# send_attitude_target() clamp: bounds the body rate ever commanded on any
# axis (rad/s — camera-error angles are reinterpreted directly as rates, see
# that function's docstring).
MAX_ANGLE = math.radians(25)

# PX4 params this rig always wants fixed to a specific value on every connect
# (px4 mode only). Blind writes, not read-modify-write: MAVProxy does its own
# full param bulk fetch right after connecting, which floods/delays
# PARAM_VALUE replies enough that waiting on a read reliably timed out on the
# bench. Confirmation is logged asynchronously via PARAM_VALUE in
# _telemetry_reader instead of blocking connect() on a reply.
_PX4_FIXED_PARAMS = {
    # Bitmask bit 2 = exempt Offboard from the RC-loss failsafe. Without
    # this, PX4 treats "no RC ever received" as RC-lost from boot and
    # silently reverts/blocks OFFBOARD entry.
    'COM_RCL_EXCEPT': 4,
}


def set_autopilot(kind: str):
    """Select "custom" or "px4" connect/control behavior. Call before connect()."""
    global _autopilot
    kind = kind.lower()
    if kind not in ("custom", "px4"):
        raise ValueError(f"unknown autopilot kind: {kind!r} (expected 'custom' or 'px4')")
    _autopilot = kind


# ---------------------------------------------------------------------------
# Launch state
# ---------------------------------------------------------------------------

_launch_lock = threading.Lock()
_last_launch_change = 0.0

def set_launch(value: bool):
    global _launched, _last_launch_change
    with _launch_lock:           # Flask is threaded — lock prevents two threads
        if _launched == value:   # racing past the debounce simultaneously
            return
        now = time.time()
        if now - _last_launch_change < 0.3:
            print(f"[Launch] debounced rapid toggle to {value}")
            return
        _last_launch_change = now
        _launched = value
    print(f"[Launch] {'LAUNCHED' if value else 'RESET'}")


# ---------------------------------------------------------------------------
# Send (synchronous — called directly from main loop, no queue)
# ---------------------------------------------------------------------------

def send_vision_error(pitch_err, yaw_err, is_tracking=False):
    """Send MAVLink debug messages synchronously from the main loop.
    Both is_tracking and _launched are read at the same instant, eliminating
    the race condition that existed when a sender thread read _launched later.
    """
    if is_tracking:
        x, y, z = float(pitch_err), float(yaw_err), 1.0
    else:
        x, y, z = 0.0, 0.0, 0.0

    launch_val = 1.0 if _launched else -1.0

    if _enabled:
        try:
            _connection.mav.debug_vect_send(
                b"vision_err",
                int(time.time() * 1e6),
                x, y, z
            )
            _connection.mav.named_value_float_send(
                int(time.time() * 1000) & 0xFFFFFFFF,
                b"launch",
                launch_val
            )
            if DEBUG:
                print(f"[MAVLink] x={x:.4f}, y={y:.4f}, z={z:.4f}, launch={launch_val:.0f}")
        except Exception as e:
            print(f"[MAVLink] send failed: {e}")
    if _ser is not None:
        packet = struct.pack('<BBff', 0xAA, 0x55, x, y)
        try:
            _ser.write(packet)
            if DEBUG:
                print(f"[Serial] x={x:.4f}, y={y:.4f}")
        except Exception as e:
            print(f"[Serial] Send failed: {e}")


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------

def _find_mavproxy(override=None):
    """Locate mavproxy.py without hardcoding a venv name — this project runs
    on several RPis whose venvs aren't all named/laid out the same way, and a
    hardcoded absolute path only ever matches one of them. Preference order:
    1. explicit override (e.g. config.toml's [mavlink] mavproxy_path)
    2. same venv as the running interpreter (sys.executable's bin/ dir) —
       correct as long as mavproxy is installed alongside tracker-so.py's own
       deps, which is how every Pi in the fleet is actually set up
    3. PATH lookup, for a bare/system install with no venv at all
    """
    if override:
        return override
    candidate = os.path.join(os.path.dirname(sys.executable), "mavproxy.py")
    if os.path.exists(candidate):
        return candidate
    found = shutil.which("mavproxy.py")
    if found:
        return found
    print(f"[MAVProxy] WARNING: mavproxy.py not found next to {sys.executable} "
          f"or on PATH — falling back to {candidate}")
    return candidate


def start_mavproxy(pixhawk_port="/dev/ttyACM0", pixhawk_baud=115200,
                   gcs_port=14550, local_port=14551,
                   extra_outputs=None, mavproxy_path=None):
    """
    Launch MAVProxy as a background subprocess.
    Automatically killed when the Python process exits.

    extra_outputs: list of IP strings that each get a dedicated unicast
                   --out=udpout:<ip>:<gcs_port> added to the MAVProxy command.
    mavproxy_path: explicit path to mavproxy.py; auto-detected via
                   _find_mavproxy() if omitted.
    """
    global _mavproxy_proc
    cmd = [
        _find_mavproxy(mavproxy_path),
        f"--master={pixhawk_port}",
        f"--baud={pixhawk_baud}",
        f"--out=udpout:127.0.0.1:{local_port}",
    ]
    for ip in (extra_outputs or []):
        cmd.append(f"--out=udpout:{ip}:{gcs_port}")
        print(f"[MAVProxy] Extra unicast output → {ip}:{gcs_port}")
    cmd.append("--daemon")
    print(f"[MAVProxy] Starting: {' '.join(cmd)}")
    _mavproxy_proc = subprocess.Popen(cmd)
    atexit.register(_stop_mavproxy)
    time.sleep(2)  # give MAVProxy time to connect to Pixhawk


def _stop_mavproxy():
    if _mavproxy_proc and _mavproxy_proc.poll() is None:
        print("[MAVProxy] Stopping...")
        _mavproxy_proc.terminate()
        try:
            _mavproxy_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print("[MAVProxy] Force-killing...")
            _mavproxy_proc.kill()


def _lock_onto_autopilot(conn, timeout=5.0):
    """wait_heartbeat() alone does NOT set target_system/target_component —
    that's a common pymavlink footgun. target_system auto-populates only as
    a side effect (pymavlink locks onto the srcSystem of the *first* HEARTBEAT
    that looks vehicle-like), and only if that first HEARTBEAT wasn't, say,
    MAVProxy's own GCS-type heartbeat on the same link. target_component is
    NEVER auto-populated by pymavlink — it silently stays 0 forever unless we
    set it explicitly. Loop until we see a HEARTBEAT specifically from the
    autopilot component (compid 1), then set target_system/target_component
    from it. Needed for px4 mode's PARAM_SET/arm/mode commands to reliably
    reach the FC rather than component 0."""
    from pymavlink import mavutil
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = conn.recv_match(type='HEARTBEAT', blocking=True, timeout=deadline - time.time())
        if msg and msg.get_srcComponent() == mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1:
            conn.target_system = msg.get_srcSystem()
            conn.target_component = msg.get_srcComponent()
            return True
    return False


def _apply_px4_fixed_params():
    from pymavlink import mavutil
    for name, value in _PX4_FIXED_PARAMS.items():
        packed = struct.unpack('<f', struct.pack('<i', value))[0]
        _connection.mav.param_set_send(
            _connection.target_system, _connection.target_component,
            name.encode(), packed, mavutil.mavlink.MAV_PARAM_TYPE_INT32
        )
        print(f"[MAVLink] {name} -> {value} (sent, see async confirmation)")


def _request_attitude_stream(rate_hz=20):
    """Ask the FC to push ATTITUDE at rate_hz. send_attitude_target() composes
    onto the latest ATTITUDE reading, so the default (~2-4 Hz) stream rate is
    too stale for a per-frame control loop — this speeds it up explicitly."""
    from pymavlink import mavutil
    _connection.mav.command_long_send(
        _connection.target_system,
        _connection.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
        int(1e6 / rate_hz),  # microseconds between messages
        0, 0, 0, 0, 0
    )


def connect(url="udpin:0.0.0.0:14551", fallback_url=None):
    """
    Connect to MAVProxy via UDP and start the telemetry reader thread.
    MAVProxy must be running with --out=udpout:127.0.0.1:14551.
    If the primary connection gets no heartbeat (e.g. no USB), falls back to
    fallback_url (e.g. udpout:GCS_IP:14550) so debug_vect still reaches the GCS.

    In "px4" mode (see set_autopilot()), also locks onto the autopilot
    component, applies _PX4_FIXED_PARAMS, requests a fast ATTITUDE stream,
    and switches into OFFBOARD (without arming) so the FC is ready well
    before Launch is pressed.
    """
    global _connection, _enabled
    from pymavlink import mavutil
    try:
        _connection = mavutil.mavlink_connection(url)
        if _autopilot == "px4":
            if not _lock_onto_autopilot(_connection, timeout=15.0):
                raise RuntimeError("no HEARTBEAT from the autopilot component (compid 1)")
            _apply_px4_fixed_params()
            _enabled = True
            _request_attitude_stream()
            set_guided_mode()
        else:
            _connection.wait_heartbeat(timeout=5)
            _enabled = True
        print(f"[MAVLink] Connected via MAVProxy ({url}), heartbeat received.")
    except Exception as e:
        print(f"[WARNING] MAVLink primary connection failed: {e}")
        if fallback_url:
            try:
                _connection = mavutil.mavlink_connection(fallback_url)
                _enabled = True
                if _autopilot == "px4":
                    _request_attitude_stream()
                print(f"[MAVLink] Fallback connected ({fallback_url}), sending debug_vect to GCS directly.")
            except Exception as e2:
                print(f"[WARNING] MAVLink fallback also failed: {e2}")
                _connection = None
                _enabled    = False
        else:
            _connection = None
            _enabled    = False


def connect_serial(port="/dev/serial0", baud=57600):
    """Open a raw serial port for sending pitch/yaw packets (non-MAVLink)."""
    global _ser
    try:
        import serial
        _ser = serial.Serial(
            port=port, baudrate=baud,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE, timeout=1,
            rtscts=False, dsrdtr=False, xonxoff=False,
        )
        print(f"[Serial] Connected to {port} at {baud} baud.")
    except Exception as e:
        print(f"[WARNING] Serial not connected: {e}")
        _ser = None


# ---------------------------------------------------------------------------
# Telemetry reader thread
# ---------------------------------------------------------------------------

_last_hb_armed = None
_last_hb_mode  = None


def _telemetry_reader():
    from pymavlink import mavutil
    mav_result_names = mavutil.mavlink.enums['MAV_RESULT']
    while True:
        if not _enabled or _connection is None:
            time.sleep(0.5)
            continue
        try:
            msg = _connection.recv_match(type=['ATTITUDE', 'STATUSTEXT', 'COMMAND_ACK', 'HEARTBEAT', 'PARAM_VALUE'],
                                          blocking=True, timeout=1.0)
            if msg is None:
                continue
            mtype = msg.get_type()
            if mtype == 'ATTITUDE':
                if SHOW_TELEMETRY:
                    print(f"[Telem] roll={math.degrees(msg.roll):+.1f}°  "
                          f"pitch={math.degrees(msg.pitch):+.1f}°  "
                          f"yaw={math.degrees(msg.yaw):+.1f}°")
            elif mtype == 'PARAM_VALUE':
                name = msg.param_id.rstrip('\x00')
                if name in _PX4_FIXED_PARAMS:
                    value = struct.unpack('<i', struct.pack('<f', msg.param_value))[0]
                    print(f"[FC] PARAM_VALUE {name} = {value}")
            elif mtype == 'STATUSTEXT':
                print(f"[FC] {msg.text.strip()}")
            elif mtype == 'COMMAND_ACK':
                result = mav_result_names.get(msg.result)
                result_name = result.name if result else msg.result
                cmd = mavutil.mavlink.enums['MAV_CMD'].get(msg.command)
                cmd_name = cmd.name if cmd else msg.command
                print(f"[FC] COMMAND_ACK {cmd_name} -> {result_name}")
            elif mtype == 'HEARTBEAT':
                # Ground truth for the actual current mode — DO_SET_MODE's
                # COMMAND_ACK only means "command parsed", not "transition
                # accepted"; PX4 can silently reject the state change after
                # ACKing receipt. Only print on change to avoid flooding
                # (HEARTBEAT streams at ~1Hz regardless).
                if _connection is not None and msg.get_srcSystem() == _connection.target_system:
                    global _last_hb_armed, _last_hb_mode
                    armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    if _autopilot == "px4":
                        main_mode = (msg.custom_mode >> 16) & 0xFF
                        mode_name = _PX4_MAIN_MODE_NAMES.get(main_mode, f"main_mode={main_mode}")
                    else:
                        mode_name = f"custom_mode={msg.custom_mode}"
                    if (armed, mode_name) != (_last_hb_armed, _last_hb_mode):
                        print(f"[FC] HEARTBEAT armed={armed} mode={mode_name}")
                        _last_hb_armed, _last_hb_mode = armed, mode_name
        except Exception as e:
            print(f"[Telem] read error: {e}")
            time.sleep(0.5)

_telem_thread = threading.Thread(target=_telemetry_reader, daemon=True)
_telem_thread.start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_attitude(pitch, yaw):
    """Push pitch/yaw commands via MAVLink RC override (units: radians)."""
    if not _enabled:
        print(f"[DEBUG] Pitch: {math.degrees(pitch):.2f}deg, Yaw: {math.degrees(yaw):.2f}deg")
        return
    try:
        pitch_pwm = int(1500 + pitch * 500)
        yaw_pwm   = int(1500 + yaw   * 500)
        print(f"[MAVLink] Sent pitch PWM: {pitch_pwm}, yaw PWM: {yaw_pwm}")
    except Exception as e:
        print(f"[MAVLink] Failed to send attitude: {e}")


# ---------------------------------------------------------------------------
# PX4 mode: OFFBOARD / arm / disarm / attitude-rate control
# ---------------------------------------------------------------------------

def _is_armed():
    """Return True if the FC heartbeat shows armed. Filters for FC sysid, not MAVProxy."""
    from pymavlink import mavutil
    deadline = time.time() + 3.0
    while time.time() < deadline:
        try:
            hb = _connection.recv_match(type='HEARTBEAT', blocking=True, timeout=1.0)
            if hb and hb.get_srcSystem() == _connection.target_system:
                return bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        except Exception:
            pass
    return False


def set_guided_mode():
    """Switch into PX4 OFFBOARD mode, without arming. Called automatically at
    the end of connect() so the FC is already in the right mode well before
    Launch is pressed — arm() then only has to arm. Relies on the main loop
    streaming a neutral-hold attitude target continuously from connect time
    onward (not gated on "launched"), since PX4 exits OFFBOARD if the
    setpoint stream stops even briefly. px4 mode only."""
    if _autopilot != "px4":
        print("[MAVLink] set_guided_mode() only applies to autopilot='px4'")
        return
    if not _enabled:
        print("[MAVLink] Not connected — skipping mode switch")
        return
    from pymavlink import mavutil
    with _launch_lock:
        try:
            # PX4 rejects the switch into OFFBOARD unless a setpoint stream is
            # already flowing — prime it with a few no-op attitude targets
            # before requesting the mode change.
            for _ in range(10):
                send_attitude_target(0.0, 0.0, thrust=0.0)
                time.sleep(0.05)
            base_mode = (mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
                         | mavutil.mavlink.MAV_MODE_FLAG_AUTO_ENABLED
                         | mavutil.mavlink.MAV_MODE_FLAG_STABILIZE_ENABLED
                         | mavutil.mavlink.MAV_MODE_FLAG_GUIDED_ENABLED)
            # NOTE: no <<16 shift here. That packed encoding (main_mode in
            # bits 16-23) is only for the 32-bit custom_mode field of
            # HEARTBEAT / the legacy SET_MODE message. MAV_CMD_DO_SET_MODE
            # sent via COMMAND_LONG is different: PX4's mavlink_receiver
            # forwards param2 straight through, and Commander reads it as
            # (uint8_t)param2 — the raw main_mode number, unshifted. A
            # shifted value here truncates to 0 on the FC side and matches
            # no valid mode, so the switch silently never takes effect.
            custom_mode = _PX4_MAIN_MODE_OFFBOARD
            # A single DO_SET_MODE can land while the link is congested
            # (e.g. MAVProxy's post-connect param bulk fetch) right when
            # PX4 checks setpoint recency, and get silently ignored with
            # no STATUSTEXT. Retry a few times, interleaved with attitude
            # targets, so the setpoint stream stays fresh across attempts.
            for attempt in range(5):
                _connection.mav.command_long_send(
                    _connection.target_system,
                    _connection.target_component,
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                    0,
                    base_mode,
                    custom_mode,
                    0, 0, 0, 0, 0
                )
                print(f"[MAVLink] OFFBOARD mode command sent (attempt {attempt + 1}/5)")
                send_attitude_target(0.0, 0.0, thrust=0.0)
                time.sleep(0.2)
        except Exception as e:
            print(f"[MAVLink] set_guided_mode error: {e}")


def arm():
    """Arm the FC. Assumes OFFBOARD mode was already set by set_guided_mode()
    (called automatically at connect time) — this only arms, so Launch
    doesn't have to wait on the mode-switch retries. px4 mode only."""
    if _autopilot != "px4":
        print("[MAVLink] arm() only applies to autopilot='px4' — use set_launch() for custom mode")
        return
    if not _enabled:
        print("[MAVLink] Not connected — skipping arm")
        return
    from pymavlink import mavutil
    with _launch_lock:
        try:
            _connection.mav.command_long_send(
                _connection.target_system,
                _connection.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,      # arm
                21196,  # force
                0, 0, 0, 0, 0
            )
            print("[MAVLink] ARM command sent")
            global _launched
            _launched = True
        except Exception as e:
            print(f"[MAVLink] arm error: {e}")


def disarm():
    """Disarm the FC. px4 mode only."""
    if _autopilot != "px4":
        print("[MAVLink] disarm() only applies to autopilot='px4' — use set_launch() for custom mode")
        return
    if not _enabled:
        return
    from pymavlink import mavutil
    # Flask is threaded — set_guided_mode()/arm() and disarm() share this lock
    # so an overlapping call (e.g. a stale UI auto-disarm racing a real
    # launch click) can't interleave mode/arm commands with this one.
    with _launch_lock:
        try:
            armed = _is_armed()
            print(f"[MAVLink] FC is {'ARMED' if armed else 'DISARMED'} — {'sending disarm' if armed else 'nothing to do'}")
            global _launched
            if not armed:
                _launched = False
                return
            # PX4 MANUAL: main_mode=1, sub_mode=0.
            base_mode = (mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
                         | mavutil.mavlink.MAV_MODE_FLAG_STABILIZE_ENABLED
                         | mavutil.mavlink.MAV_MODE_FLAG_MANUAL_INPUT_ENABLED)
            custom_mode = _PX4_MAIN_MODE_MANUAL
            _connection.mav.command_long_send(
                _connection.target_system,
                _connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                base_mode,
                custom_mode,
                0, 0, 0, 0, 0
            )
            time.sleep(0.5)
            _connection.mav.command_long_send(
                _connection.target_system,
                _connection.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0,      # disarm
                21196,  # force
                0, 0, 0, 0, 0
            )
            time.sleep(0.5)
            still_armed = _is_armed()
            print(f"[MAVLink] DISARM {'succeeded' if not still_armed else 'FAILED — FC still armed'}")
            _launched = False
        except Exception as e:
            print(f"[MAVLink] disarm error: {e}")


def send_attitude_target(pitch_err, yaw_err, roll_err=0.0, thrust=0.5):
    """Send SET_ATTITUDE_TARGET every frame. pitch_err/yaw_err/roll_err are
    BODY-FRAME camera tracking errors (radians), clamped to MAX_ANGLE and
    used directly as body RATE setpoints (rad/s) — a P-gain-of-1 open-loop
    mapping, not an absolute attitude target. px4 mode only — for "custom"
    mode use send_vision_error() instead.

    roll_err defaults to 0.0 (tracker-so.py never computes a nonzero one),
    which commands zero roll RATE — i.e. "stop rotating", not "return to a
    level/zero roll angle". This airframe launches vertically and has no
    meaningful "level" attitude, so a self-level controller driving roll
    toward 0 would fight whatever attitude the vehicle is actually in (and,
    since it would read the FC's Euler roll, breaks down entirely near
    pitch=90° from gimbal lock). Pure rate damping avoids both problems by
    never reading current attitude at all.

    Runs PX4 in full RATE mode: type_mask sets
    ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE, so the quaternion (q) is
    unused (sent as an identity placeholder) and all three axes are driven
    as explicit body rate setpoints. This bypasses FW_ATT_CONTROL's angle
    loop (and its turn-coordination yaw logic) entirely, going straight to
    the rate controller / control allocation for roll, pitch AND yaw.

    Call at ~10 Hz or faster."""
    if _autopilot != "px4":
        return
    if not _enabled:
        if DEBUG:
            print(f"[DEBUG] pitch_err={math.degrees(pitch_err):.2f}° yaw_err={math.degrees(yaw_err):.2f}°")
        return
    roll  = max(-MAX_ANGLE, min(MAX_ANGLE, roll_err))
    pitch = max(-MAX_ANGLE, min(MAX_ANGLE, pitch_err))
    yaw   = max(-MAX_ANGLE, min(MAX_ANGLE, yaw_err))
    from pymavlink.quaternion import QuaternionBase
    try:
        q = QuaternionBase([0.0, 0.0, 0.0])  # identity; unused (ATTITUDE_IGNORE bit set)
        # NOTE: bit 6 (0b01000000=64) is ATTITUDE_TARGET_TYPEMASK_THRUST_IGNORE,
        # NOT ignore-attitude. ignore_attitude is bit 7 (0b10000000=128).
        type_mask = 0b10000000  # ignore attitude only -> use body rates + thrust
        _connection.mav.set_attitude_target_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            _connection.target_system,
            _connection.target_component,
            type_mask,
            q,
            roll, pitch, yaw,
            thrust
        )
        if DEBUG:
            print(f"[MAVLink] SET_ATTITUDE_TARGET (rate mode) "
                  f"sent_rates=(roll={math.degrees(roll):+.2f}°/s,"
                  f"pitch={math.degrees(pitch):+.2f}°/s,"
                  f"yaw={math.degrees(yaw):+.2f}°/s) thrust={thrust:.2f}")
    except Exception as e:
        print(f"[MAVLink] set_attitude_target failed: {e}")
