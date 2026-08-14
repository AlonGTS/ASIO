#!/usr/bin/env python3
"""
H.264-over-UDP video server for tracker-so.py (config.toml video_mode = "h264_udp").

Connectionless by design: no handshake, no ACK/NACK, no retransmission. The
sender just keeps pushing RTP packets at whatever GCS_IP it currently knows;
a receiver can start decoding mid-stream at the next keyframe with no
coordination needed from this side. See gcs.py's _H264LiveCapture for the
matching receiver.

Usage from tracker-so.py:
    import h264_udp_server
    h264_udp_server.start(frame_buffer, gcs_ip_getter=lambda: GCS_IP, ...)
"""
import os
import socket
import struct
import threading
import time
from fractions import Fraction

import av
import cv2

# RFC 5285 one-byte-header RTP extension carrying the Pi's frame_gen counter
# alongside the H.264 payload — the RTP-native equivalent of jpeg_udp's
# 4-byte frame_gen prefix (see STABILIZATION.md, "stale-click problem").
_RTP_EXT_ID = 1
_FU_A_TYPE  = 28   # RFC 6184 fragmentation-unit NAL type


def _even(v: int) -> int:
    return v - (v % 2)


def _iter_annexb_nals(data: bytes):
    """Split an Annex-B bitstream (from av.Packet.to_bytes()) into individual
    NAL units, stripping start codes. Correctly distinguishes 3-byte (00 00
    01) from 4-byte (00 00 00 01) start codes so the extra leading zero of a
    4-byte code isn't left dangling on the previous NAL."""
    occurrences = []   # (core_start_pos, is_four_byte)
    i = 0
    while True:
        i = data.find(b"\x00\x00\x01", i)
        if i == -1:
            break
        occurrences.append((i, i > 0 and data[i - 1] == 0))
        i += 3
    n = len(data)
    for idx, (pos, four) in enumerate(occurrences):
        nal_start = pos + 3
        if idx + 1 < len(occurrences):
            next_pos, next_four = occurrences[idx + 1]
            nal_end = next_pos - (1 if next_four else 0)
        else:
            nal_end = n
        if nal_end > nal_start:
            yield data[nal_start:nal_end]


def _open_encoder(width, height, fps, gop, bitrate_kbps):
    """Try the Pi's hardware H.264 encoder (V4L2 M2M) first; fall back to
    libx264 (software) only if the hardware path is unavailable — Pi 5 has
    no hardware H.264 *encoder* block (only decode), Pi 4/CM4 do."""
    width, height = _even(width), _even(height)
    candidates = (
        ('h264_v4l2m2m', 'HW', {}),
        ('libx264', 'SW', {'preset': 'ultrafast', 'tune': 'zerolatency',
                            'x264-params': 'repeat-headers=1'}),
    )
    last_err = None
    for name, tag, opts in candidates:
        try:
            ctx = av.CodecContext.create(name, 'w')
            ctx.width, ctx.height = width, height
            ctx.pix_fmt = 'yuv420p'
            ctx.framerate = Fraction(int(fps), 1)
            ctx.time_base = Fraction(1, int(fps))
            ctx.gop_size = max(1, gop)
            ctx.bit_rate = max(1, bitrate_kbps) * 1000
            if opts:
                ctx.options = opts
            ctx.open()
            print(f"[H264] encoder: {name} ({tag})  {width}x{height} "
                  f"gop={gop}f  {bitrate_kbps}kbps")
            return ctx
        except Exception as e:
            last_err = e
            print(f"[H264] {name} unavailable ({e}) — trying next…")
    raise RuntimeError(f"No usable H264 encoder (hardware or software): {last_err}")


def _pack_rtp_header(seq: int, ts: int, marker: bool, ssrc: int, frame_gen: int) -> bytes:
    b0 = (2 << 6) | (1 << 4)                    # V=2, P=0, X=1 (extension present), CC=0
    b1 = (0x80 if marker else 0x00) | 96        # M bit, PT=96 (dynamic)
    header = struct.pack('>BBHII', b0, b1, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc)

    # One-byte-header extension element: id=1, len=4 bytes of frame_gen,
    # padded to a 4-byte-aligned block (RFC 5285).
    elem = bytes([(_RTP_EXT_ID << 4) | 3]) + struct.pack('>I', frame_gen & 0xFFFFFFFF)
    elem += b'\x00' * ((-len(elem)) % 4)
    ext = struct.pack('>HH', 0xBEDE, len(elem) // 4) + elem
    return header + ext


def _send_rtp(sock, addr, payload, seq_box, ts, marker, ssrc, frame_gen):
    seq = seq_box[0]
    seq_box[0] = (seq + 1) & 0xFFFF
    packet = _pack_rtp_header(seq, ts, marker, ssrc, frame_gen) + payload
    try:
        sock.sendto(packet, addr)
    except Exception as e:
        print(f"[H264] send error: {e}")


def _send_nal(sock, addr, nal, frame_gen, is_last_nal, ts, ssrc, seq_box, max_payload):
    """Send one NAL unit as a single RTP packet, or as RFC 6184 FU-A
    fragments if it's larger than max_payload (keeps every UDP datagram
    MTU-safe — no IP fragmentation)."""
    if len(nal) <= max_payload:
        _send_rtp(sock, addr, nal, seq_box, ts, is_last_nal, ssrc, frame_gen)
        return

    fnri = nal[0] & 0xE0
    nal_type = nal[0] & 0x1F
    body = nal[1:]
    chunk_size = max(1, max_payload - 2)   # 2 bytes for the FU indicator+header
    offset = 0
    first = True
    while offset < len(body):
        chunk = body[offset:offset + chunk_size]
        offset += len(chunk)
        last = offset >= len(body)
        fu_indicator = fnri | _FU_A_TYPE
        fu_header = (0x80 if first else 0x00) | (0x40 if last else 0x00) | nal_type
        payload = bytes([fu_indicator, fu_header]) + chunk
        _send_rtp(sock, addr, payload, seq_box, ts, last and is_last_nal, ssrc, frame_gen)
        first = False


def _h264_stream_worker(frame_buffer, gcs_ip_getter, port, stream_width, stream_fps,
                         bitrate_kbps, gop_seconds, rtp_payload):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)

    ssrc = struct.unpack('>I', os.urandom(4))[0]
    seq_box = [struct.unpack('>H', os.urandom(2))[0]]

    stream_interval = (1.0 / stream_fps) if stream_fps > 0 else 0.0
    # GOP is computed from the rate frames actually reach the encoder at
    # (the gcs_stream_fps throttle), not the camera's active_fps — those are
    # only the same when stream_fps is uncapped.
    feed_fps = stream_fps if stream_fps > 0 else 30
    gop = max(1, round(gop_seconds * feed_fps))

    encoder = None
    enc_w = enc_h = 0
    pts = 0
    last_gen = -1
    last_send_ts = 0.0
    t0, sent = time.time(), 0

    print(f"[H264] stream worker ready — waiting for GCS to announce "
          f"(cap {stream_fps if stream_fps > 0 else 'uncapped'} fps, "
          f"gop={gop}f/{gop_seconds}s)")

    while True:
        frame, gen = frame_buffer.get(last_gen=last_gen, timeout=0.1)
        if frame is None:
            continue
        last_gen = gen

        gcs_ip = gcs_ip_getter()
        if gcs_ip is None:
            continue

        now_gate = time.time()
        if stream_interval > 0 and (now_gate - last_send_ts) < stream_interval:
            continue
        last_send_ts = now_gate

        sent += 1
        now = time.time()
        if now - t0 >= 5.0:
            print(f"[H264] {sent / (now - t0):.1f} fps  ({sent} frames in {now-t0:.1f}s)  → {gcs_ip}")
            t0, sent = now, 0

        h_f, w_f = frame.shape[:2]
        target_w = _even(stream_width if stream_width > 0 else w_f)
        target_h = _even(int(h_f * target_w / w_f))

        if encoder is None or (target_w, target_h) != (enc_w, enc_h):
            try:
                encoder = _open_encoder(target_w, target_h, feed_fps, gop, bitrate_kbps)
            except Exception as e:
                print(f"[H264] {e}")
                continue
            enc_w, enc_h = target_w, target_h
            pts = 0

        frame_r = frame if (w_f, h_f) == (target_w, target_h) else \
            cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        vf = av.VideoFrame.from_ndarray(frame_r, format='bgr24').reformat(format='yuv420p')
        vf.pts = pts
        vf.time_base = encoder.time_base
        pts += 1

        try:
            packets = list(encoder.encode(vf))
        except Exception as e:
            print(f"[H264] encode error: {e}")
            continue

        for packet in packets:
            nals = list(_iter_annexb_nals(bytes(packet)))
            if not nals:
                continue
            ts = int(time.time() * 90000) & 0xFFFFFFFF
            for i, nal in enumerate(nals):
                _send_nal(sock, (gcs_ip, port), nal, gen, i == len(nals) - 1,
                          ts, ssrc, seq_box, rtp_payload)


def start(frame_buffer, gcs_ip_getter, port=5600, stream_width=480, stream_fps=15,
          bitrate_kbps=2000, gop_seconds=0.5, rtp_payload=1200):
    """Spawn the H.264/RTP-over-UDP sender as a daemon thread. Non-blocking.
    gcs_ip_getter: zero-arg callable returning the current GCS IP (or None
    until it's learned), matching tracker-so.py's dynamically-updated
    GCS_IP module global."""
    t = threading.Thread(
        target=_h264_stream_worker,
        args=(frame_buffer, gcs_ip_getter, port, stream_width, stream_fps,
              bitrate_kbps, gop_seconds, rtp_payload),
        daemon=True,
    )
    t.start()
    return t
