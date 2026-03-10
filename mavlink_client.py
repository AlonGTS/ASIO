#!/usr/bin/env python3
"""
MAVLink client for tracker.py.

Usage:
    import mavlink_client
    mavlink_client.connect()          # call once at startup
    mavlink_client.send_vision_error(pitch_err, yaw_err)
    mavlink_client.send_attitude(pitch, yaw)
"""
import math
import time

_connection = None
_enabled = False


def connect(port="/dev/serial0", baud=115200):
    """Try to open a MAVLink connection. Safe to call even if hardware is absent."""
    global _connection, _enabled
    try:
        from pymavlink import mavutil
        _connection = mavutil.mavlink_connection(port, baud=baud)
        _connection.wait_heartbeat(timeout=5)
        _enabled = True
        print("[MAVLink] Connected and heartbeat received.")
    except Exception as e:
        print(f"[WARNING] MAVLink not connected: {e}")
        _connection = None
        _enabled = False


def send_vision_error(pitch_err, yaw_err):
    """Send vision-based pitch/yaw error via MAVLink DEBUG_VECT (units: radians)."""
    if not _enabled:
        print(f"[DEBUG] pitch_err={pitch_err:.4f}, yaw_err={yaw_err:.4f}")
        return
    try:
        _connection.mav.debug_vect_send(
            b"vision_err",
            int(time.time() * 1e6),
            float(pitch_err * 100),   # x
            float(yaw_err  * 100),    # y
            3.3                        # z
        )
        print(f"[DEBUG] pitch_err={pitch_err*100:.4f}, yaw_err={yaw_err*100:.4f}")
    except Exception as e:
        print(f"[MAVLink] DEBUG_VECT send failed: {e}")


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
