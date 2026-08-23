#!/usr/bin/env python3
"""
Shared GCS-side video receivers for both video transports (config.toml
video_mode = "jpeg_udp" or "h264_udp"). Extracted out of gcs.py so a
lightweight client (viewer.py) can receive/decode video without pulling in
gcs.py's full control UI (buttons, click handling, command channel, ...).

Both classes share one read() contract — (ok, frame_copy, frame_id,
frame_gen) — so callers don't need to know which transport is behind the
capture object:
    cap = LiveCapture(port, group) or H264LiveCapture(port, group)
    ok, frame, frame_id, frame_gen = cap.read()
    ...
    cap.close()   # before switching to the other class on the same port

The Pi sends video once, to a UDP multicast group (see config.toml's
video_multicast_group / tracker-so.py) — both classes join that group on
open() so any number of GCS/viewer instances can receive the same stream
without the Pi doing any per-recipient work.
"""
import socket
import struct
import threading

import cv2
import numpy as np

DEFAULT_MULTICAST_GROUP = "239.5.5.5"   # must match config.toml's video_multicast_group


def _join_multicast(sock: socket.socket, port: int, group: str):
    """Bind + join a UDP multicast group. SO_REUSEPORT (when available)
    lets multiple local processes — e.g. gcs.py and viewer.py on the same
    Mac — join independently and each get their own copy of every packet."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(('', port))
    mreq = socket.inet_aton(group) + socket.inet_aton("0.0.0.0")
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

# ── JPEG-over-UDP ────────────────────────────────────────────────────────────
#
# Problem: cv2.VideoCapture.read() returns frames in decode order from an
# internal queue. When the main loop is busy (drawing, key handling), that
# queue grows and read() starts returning frames from seconds ago.
#
# Fix: a daemon thread that drains the queue as fast as the decoder produces
# frames and only ever keeps the most recent one. The main loop always gets
# "now".
#
# The Pi sends JPEG-encoded frames as individual UDP datagrams. Each
# datagram = one complete JPEG image — no stream reassembly needed.

_JPEG_SOI = b"\xff\xd8"   # JPEG Start-Of-Image marker — always the first 2 bytes of a JPEG

def _split_frame_gen(data: bytes):
    """Split a video datagram into (frame_gen_or_None, jpeg_bytes).

    Wire format (see STABILIZATION.md, "stale-click problem"): the Pi may
    prepend a 4-byte big-endian frame_gen counter before the JPEG bytes, so
    a controller GCS can echo it back with select_point() and the Pi can
    look up the exact frame that was clicked on, instead of using whatever
    is live at request time. Detected via the JPEG SOI marker so this stays
    backward compatible with a Pi that isn't sending the header yet.
    """
    if data[:2] == _JPEG_SOI:
        return None, data            # legacy: no header, whole payload is the JPEG
    if len(data) > 4 and data[4:6] == _JPEG_SOI:
        return struct.unpack(">I", data[:4])[0], data[4:]
    return None, data                # unrecognized — best-effort fallback


class LiveCapture:
    def __init__(self, port: int, group: str = DEFAULT_MULTICAST_GROUP):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)  # 1 MB
        _join_multicast(self._sock, port, group)
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


# ── H.264/RTP over UDP ───────────────────────────────────────────────────────
#
# Same read() contract as LiveCapture — (ok, frame, frame_id, frame_gen) —
# so a caller only needs to pick which class to instantiate.
#
# Deliberately no jitter buffer / reordering: packets are handled strictly
# in arrival order, and any sequence-number gap (including the arrival
# after a Wi-Fi drop, since the Pi's sender keeps sendto()-ing through an
# outage) immediately discards the in-progress frame and any partial
# fragment reassembly, then waits for the next keyframe before decoding
# again — real-time video over recovering every frame, and no
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


class H264LiveCapture:
    def __init__(self, port: int, group: str = DEFAULT_MULTICAST_GROUP):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        _join_multicast(self._sock, port, group)
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
        """Return (ok, frame_copy, frame_id, frame_gen). Same contract as LiveCapture."""
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
