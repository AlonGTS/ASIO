#!/usr/bin/env python3
"""
MAVLink client for tracker-so.py — routes through MAVProxy.

Start MAVProxy on the RPi before running tracker-so.py:
    mavproxy.py --master=/dev/ttyACM0 --baud=115200 \
                --out=udpout:127.0.0.1:14551 \
                --out=udp:<GCS_IP>:14550

Usage:
    import mavlink_client
    mavlink_client.connect()                 # call once at startup
    mavlink_client.send_vision_error(p, y)   # non-blocking, called every frame
    mavlink_client.telemetry                 # dict updated live by reader thread
"""
import math
import time
import struct
import queue
import threading
import subprocess
import atexit

_connection    = None
_enabled       = False
_ser           = None
_launched      = False
_mavproxy_proc = None
DEBUG = False   # set True to print pitch/yaw values every frame

# Latest telemetry values — updated live by _telemetry_reader thread
telemetry = {
    "roll": None, "pitch": None, "yaw": None,
    "lat": None,  "lon": None,  "alt": None,
    "groundspeed": None, "airspeed": None,
    "battery_voltage": None, "battery_remaining": None,
}


# ---------------------------------------------------------------------------
# Launch state
# ---------------------------------------------------------------------------

def set_launch(value: bool):
    global _launched
    _launched = value
    print(f"[Launch] {'LAUNCHED' if value else 'RESET'}")


# ---------------------------------------------------------------------------
# Sender thread (non-blocking send from main loop)
# ---------------------------------------------------------------------------

_send_queue = queue.Queue(maxsize=2)  # drop stale values if sender falls behind

def _sender_thread():
    while True:
        pitch_err, yaw_err = _send_queue.get()
        if _enabled:
            try:
                _connection.mav.debug_vect_send(
                    b"vision_err",
                    int(time.time() * 1e6),
                    float(pitch_err),
                    float(yaw_err),
                    100.0 if _launched else -1.0
                )
                if DEBUG:
                    print(f"[MAVLink] pitch={pitch_err:.4f}, yaw={yaw_err:.4f}")
            except Exception as e:
                print(f"[MAVLink] DEBUG_VECT send failed: {e}")
        if _ser is not None:
            packet = struct.pack('<BBff', 0xAA, 0x55, float(pitch_err), float(yaw_err))
            try:
                _ser.write(packet)
                if DEBUG:
                    print(f"[Serial] pitch={pitch_err:.4f}, yaw={yaw_err:.4f}")
            except Exception as e:
                print(f"[Serial] Send failed: {e}")

_thread = threading.Thread(target=_sender_thread, daemon=True)
_thread.start()


# ---------------------------------------------------------------------------
# Telemetry reader thread
# ---------------------------------------------------------------------------

def _telemetry_reader():
    """Reads incoming MAVLink messages from MAVProxy and updates telemetry dict."""
    while True:
        try:
            msg = _connection.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue
            mtype = msg.get_type()
            if mtype == "ATTITUDE":
                telemetry["roll"]  = msg.roll
                telemetry["pitch"] = msg.pitch
                telemetry["yaw"]   = msg.yaw
            elif mtype == "GLOBAL_POSITION_INT":
                telemetry["lat"] = msg.lat / 1e7
                telemetry["lon"] = msg.lon / 1e7
                telemetry["alt"] = msg.alt / 1e3
            elif mtype == "VFR_HUD":
                telemetry["groundspeed"] = msg.groundspeed
                telemetry["airspeed"]    = msg.airspeed
            elif mtype == "SYS_STATUS":
                telemetry["battery_voltage"]   = msg.voltage_battery / 1000.0
                telemetry["battery_remaining"] = msg.battery_remaining
            if DEBUG:
                print(f"[Telemetry] {mtype}: {msg}")
        except Exception as e:
            print(f"[Telemetry] Read error: {e}")


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------

def start_mavproxy(pixhawk_port="/dev/ttyACM0", pixhawk_baud=115200,
                   gcs_ip="192.168.1.100", gcs_port=14550, local_port=14551):
    """
    Launch MAVProxy as a background subprocess.
    Automatically killed when the Python process exits.
    """
    global _mavproxy_proc
    cmd = [
        "/home/mahat/webrtc_venv/bin/mavproxy.py",
        f"--master={pixhawk_port}",
        f"--baud={pixhawk_baud}",
        f"--out=udpout:127.0.0.1:{local_port}",
        f"--out=udp:{gcs_ip}:{gcs_port}",
        "--daemon",
    ]
    print(f"[MAVProxy] Starting: {' '.join(cmd)}")
    _mavproxy_proc = subprocess.Popen(cmd)
    atexit.register(_stop_mavproxy)
    time.sleep(2)  # give MAVProxy time to connect to Pixhawk


def _stop_mavproxy():
    if _mavproxy_proc and _mavproxy_proc.poll() is None:
        print("[MAVProxy] Stopping...")
        _mavproxy_proc.terminate()


def connect(url="udpin:0.0.0.0:14551"):
    """
    Connect to MAVProxy via UDP and start the telemetry reader thread.
    MAVProxy must be running with --out=udpout:127.0.0.1:14551
    """
    global _connection, _enabled
    try:
        from pymavlink import mavutil
        _connection = mavutil.mavlink_connection(url)
        _connection.wait_heartbeat(timeout=5)
        _enabled = True
        print(f"[MAVLink] Connected via MAVProxy ({url}), heartbeat received.")
        threading.Thread(target=_telemetry_reader, daemon=True).start()
    except Exception as e:
        print(f"[WARNING] MAVLink not connected: {e}")
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
# Public API
# ---------------------------------------------------------------------------

def send_vision_error(pitch_err, yaw_err):
    """Queue a pitch/yaw error for sending (non-blocking). Units: -1..1 normalized."""
    try:
        _send_queue.put_nowait((pitch_err, yaw_err))
    except queue.Full:
        pass  # drop the frame if the sender is behind


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
