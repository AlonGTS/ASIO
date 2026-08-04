# Video Stabilization — Design Notes

Goal: make the video shown at the GCS steady enough that an operator can
reliably click the tracking point, without hurting tracking latency/accuracy
on the Pi.

## Current decision

**Stabilization runs on the GCS (optical flow), not on the Pi via Pixhawk
attitude.** The GCS has CPU headroom to spare; keeping the Pi side simple
and avoiding a cross-system timestamp-sync dependency won out. This is a
change from the original direction — see "Alternative considered" below for
why Pixhawk attitude was attractive and why it's shelved for now, in case
it's worth revisiting later (e.g. if GCS-side CPU becomes tight, or
translational vibration turns out to matter more than expected).

**Status:**
- ✅ Done (Pi side): the stale-click fix — `frame_gen` tagging, the raw-frame
  rolling buffer, and tracker replay in `/select_point`. Implemented, not
  yet run on real hardware.
- ⬜ To do (GCS side, remote): strip the new `frame_gen` header from the
  video stream, do the optical-flow stabilization for display, and echo
  `frame_gen` back on click (inverting any local display transform first).
  Contract below.
- ⬜ Not started: Pixhawk attitude buffer / timestamp sync (shelved, see
  below).

## The stale-click problem (fixed on the Pi)

`flask_app.py`'s `/select_point` handler used to have a latent bug
independent of stabilization: the GCS sends normalized click coordinates
`(nx, ny)`, and the Pi applied them to whatever frame was live *at the
moment the request was processed* — with no notion of which frame the
operator actually saw. Any added latency (network, GCS-side stabilization,
round-trip) only made the mismatch between "frame operator clicked on" and
"frame Pi is currently on" larger.

Fixed by two changes in `tracker-so.py`/`flask_app.py`:

1. **Every frame now carries an id.** `state.frame_gen` (already existed,
   bumped once per captured frame) is now attached to the published stream
   frame and to a rolling history buffer of raw MAIN frames
   (`_frame_history` in `tracker-so.py`, ~2 seconds deep at the active
   capture fps — comfortably more than one round trip).
2. **`/select_point` accepts an optional `frame_gen` field.** If present and
   still in the buffer, the Pi initializes the tracker on *that* historical
   frame (the one the operator actually saw), then replays
   `tracker.update()` through each buffered frame captured since then, in
   order — the same per-frame step the live loop already does — so it
   arrives at the live frame already converged instead of jumping the whole
   round-trip gap in one step (which risks landing outside a correlation
   tracker's search window). If `frame_gen` is omitted, already aged out of
   the buffer, or the replay loses the target partway through, it falls back
   to the original behavior (init directly on the live frame) — never worse
   than before.

### Wire contract the GCS needs to implement

**Video stream (`jpeg_udp` mode, the active `video_mode` in `config.toml`):**
each UDP datagram sent to `gcs_udp_port` is now

```
[4-byte big-endian uint32 frame_gen][JPEG bytes]
```

instead of raw JPEG. The GCS decoder must strip the first 4 bytes, decode
the rest as JPEG as before, and remember that `frame_gen` value as "the id
of the frame currently being displayed."

*(WebRTC mode: `webrtc_server.FrameBuffer.put()` now accepts an explicit
`gen` and tracker-so.py passes `frame_gen` through it, but there's no
data-channel signaling wired up yet to get that id to the browser. Not
needed unless `video_mode` switches to `"webrtc"`.)*

**Click submission:** `POST /select_point` (port 5000) gets one new optional
form field alongside the existing ones:

| field | required? | notes |
|---|---|---|
| `nx`, `ny` | preferred | normalized `[0..1]`, unchanged |
| `x`, `y` | fallback if no `nx`/`ny` | absolute pixels, unchanged |
| `frame_gen` | **new, optional** | the `frame_gen` of the frame that was on screen when the operator clicked |

Important: the coordinate system for `nx`/`ny`/`x`/`y` is the frame as
*received from the Pi* (the buffered history frame and the published stream
frame share the same `frame_gen` and the same pixel geometry — overlays
drawn on the stream frame don't resize/shift anything). If the GCS applies
its own optical-flow stabilization warp before displaying the frame, **it
must invert that warp on the click coordinates before sending them** — send
back where the click falls on the original received frame, not on the
GCS's locally-smoothed display. Omitting `frame_gen` entirely still works
(falls back to old behavior); omitting the un-warp step will silently send
wrong coordinates rather than erroring, so this part is easy to get wrong
quietly — worth testing explicitly with the camera panning during a click.

## Alternative considered: Pixhawk attitude (shelved)

The camera is rigidly mounted to the airframe — no gimbal (see
`mavlink_client.py`'s notes on why `send_attitude_target()` uses pure rate
damping instead of a self-level controller: this airframe has no meaningful
"level" attitude). Because of that rigid mount, the Pixhawk's attitude
(roll/pitch/yaw) *is* the camera's rotation, which would have made it a
better stabilization signal than computing motion from the video itself:

- **Cheaper**: a rotation warp from known angle deltas vs. per-frame feature
  detection + optical flow + RANSAC.
- **More robust**: optical flow can be fooled by the tracked target's own
  motion dominating the frame (especially in `bMoovingTgt` mode); attitude
  data doesn't care what's in the image.
- **Much lower lag**: optical flow needs to smooth a noisy motion estimate
  before undoing it — causal filtering or a lookahead buffer adds roughly
  150-350ms of perceptible settle lag at 15fps. Attitude data is already a
  clean signal (no smoothing needed) — its only lag is MAVLink relay time
  (serial → mavproxy → UDP loopback), estimated at single-digit to ~20ms.

Shelved because the GCS has CPU to spare and doing it there avoids a
cross-system timestamp-sync dependency (Pixhawk ATTITUDE messages vs. camera
frame timestamps, two different clocks-in-code that would need to be kept
aligned on the Pi). If revisited, note it wouldn't have been a complete fix
either:

- **Translational vibration**: gyro/attitude only carries rotation, not
  linear displacement. In practice likely a small contributor for a
  standoff target (apparent angular error from a linear shift Δx at range D
  is ~Δx/D — shrinks with distance, unlike rotational jitter which is
  range-independent), but would matter more for close-range targets.
- **FC/camera mount compliance**: the Pixhawk's IMU measures motion at *its*
  location; any flex/compliance between FC and camera mount is invisible to
  it.
- **Motion blur**: real vibration frequencies (tens–100+ Hz) alias past both
  a ~20Hz attitude stream and a 15fps video stream — that shows up as blur
  *within* a single exposure, which no post-capture stabilization (attitude-
  or optical-flow-based) can undo.

The `ATTITUDE` MAVLink message carries `roll`/`pitch`/`yaw` (EKF-filtered
Euler angles) and `rollspeed`/`pitchspeed`/`yawspeed` (raw gyro rates) if
this gets revisited — see git history on this file for the fuller writeup
(timestamp-sync plan, per-axis image-correction mapping) that was here
before this rewrite.
