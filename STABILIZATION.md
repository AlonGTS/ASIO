# Video Stabilization — Design Notes

Goal: make the video shown at the GCS steady enough that an operator can
reliably click the tracking point, without hurting tracking latency/accuracy
on the Pi.

Not implemented yet — this is the design discussion to review from the GCS
before we touch `tracker-so.py`.

## Why Pixhawk attitude instead of optical flow

The camera is rigidly mounted to the airframe — no gimbal (see
`mavlink_client.py`'s notes on why `send_attitude_target()` uses pure rate
damping instead of a self-level controller: this airframe has no meaningful
"level" attitude). Because of that rigid mount, the Pixhawk's attitude
(roll/pitch/yaw) *is* the camera's rotation, which makes it a better
stabilization signal than computing motion from the video itself:

- **Cheaper**: a rotation warp from known angle deltas vs. per-frame feature
  detection + optical flow + RANSAC on the Pi's CPU.
- **More robust**: optical flow can be fooled by the tracked target's own
  motion dominating the frame (especially in `bMoovingTgt` mode); attitude
  data doesn't care what's in the image.
- **Much lower lag**: optical flow needs to smooth a noisy motion estimate
  before undoing it — causal filtering or a lookahead buffer adds roughly
  150-350ms of perceptible settle lag at 15fps. Attitude data is already a
  clean signal (no smoothing needed) — its only lag is MAVLink relay time
  (serial → mavproxy → UDP loopback), estimated at single-digit to ~20ms.

### What it doesn't fix

- **Translational vibration**: gyro/attitude only carries rotation, not
  linear displacement. Double-integrating accelerometer data to get position
  drifts too fast to be usable. In practice this is likely a small
  contributor for a standoff target (apparent angular error from a linear
  shift Δx at range D is ~Δx/D — it shrinks with distance, unlike rotational
  jitter which is range-independent), but would matter more for close-range
  targets.
- **FC/camera mount compliance**: the Pixhawk's IMU measures motion at *its*
  location. If the camera isn't rigidly bonded to the same structure (any
  flex, standoff, damping between FC and camera), differential vibration
  between the two is invisible to this correction. Worth physically
  verifying mount rigidity.
- **Motion blur**: real vibration frequencies (tens–100+ Hz) alias past both
  the 20Hz attitude stream and the 15fps video stream — that shows up as
  blur *within* a single exposure, which no post-capture stabilization
  (attitude-based or optical-flow-based) can undo. Only shorter exposure or
  physical damping helps here.

### Which attitude fields to use

The MAVLink `ATTITUDE` message carries both:
- `roll` / `pitch` / `yaw` — EKF-filtered Euler angles.
- `rollspeed` / `pitchspeed` / `yawspeed` — raw body-frame gyro rates.

Mapping to image correction (camera boresighted along the airframe):
- **Roll** → pure in-plane image rotation (rotate frame by −Δroll).
- **Pitch/yaw** → apparent translation of the scene across the image plane,
  not rotation. Converting angle → pixels needs focal length/FOV
  (`shift_px ≈ Δangle × frame_width / horizontal_FOV`), and is only a valid
  small-angle approximation near frame center.

The raw gyro rates update with less filter lag than the fused Euler angles,
so integrating rates between two frame timestamps may track fast jitter
better than diffing filtered roll/pitch/yaw. Worth keeping both in the
shared attitude buffer.

## Timestamp sync (Pi-side, needed regardless of GCS changes)

Both sides can share one clock for free: pymavlink stamps every received
message with `msg._timestamp = time.time()` at parse time
(`mavutil.py:371`), and `_telemetry_reader()` in `mavlink_client.py` runs a
tight loop with nothing else blocking it. So an ATTITUDE timestamp and a
`time.time()` call in the camera reader thread land on the same host clock
— no cross-clock calibration needed.

Two gaps to close on the Pi, both currently missing:

1. `_telemetry_reader()` only prints ATTITUDE (if `SHOW_TELEMETRY`); it
   never stores it. Need a small shared ring buffer of the last few
   `(timestamp, roll, pitch, yaw, rollspeed, pitchspeed, yawspeed)` samples,
   guarded by a lock, so frames can be bracketed and interpolated (ATTITUDE
   arrives at ~20Hz vs. the 15fps GCS stream — close enough that nearest-
   neighbor matching alone can be off by up to ±25ms; interpolate instead).
2. `_reader_live_picam()` (`tracker-so.py`) calls `picam2.capture_array()`
   and stores the frame with no timestamp. Need to tag each frame with
   `time.time()` right after `capture_array()` returns.

Unmeasured but likely small: MAVLink relay latency (serial → mavproxy → UDP
loopback) and Picamera2's internal ISP/DMA pipeline delay before
`capture_array()` returns. Should log actual `frame_ts - attitude_ts` deltas
on real hardware before trusting the math.

## The stale-click problem

Independent of stabilization, `flask_app.py`'s `/select_point` handler
already has a latent bug: the GCS sends normalized click coordinates
`(nx, ny)`, and the Pi applies them to `state.current_frame` — whatever is
live *at the moment the request is processed* — with no notion of which
frame the operator actually saw. There's no frame ID or timestamp
round-tripped today.

Any stabilization added on top (Pi-side pre-encode warp, or GCS-side
display smoothing) only adds more delay before the click round-trips back,
making the mismatch between "frame operator clicked on" and "frame Pi is
currently on" larger, not new-in-kind.

### Two separate things to fix

1. **Undoing GCS-side display smoothing** — fully fixable on the GCS alone.
   If the GCS stabilizes the frame before displaying it, it knows (or can
   invert) the transform it applied to that specific frame, and can send
   back click coordinates in the original received frame's space. No
   round-trip needed for this part.

2. **The Pi has moved on since that frame was captured** — cannot be fixed
   on the GCS alone; the GCS has no visibility into the Pi's current live
   frame. Needs a small protocol + Pi-side change:
   - Tag every frame published to the GCS with `state.frame_gen` (already
     exists, bumped every captured frame).
   - GCS echoes that `frame_gen` back alongside the click.
   - Pi keeps a short rolling buffer of raw MAIN frames (last ~1-2s, deep
     enough to cover worst-case round-trip time), keyed by `frame_gen`.
   - On click: find the buffered frame matching the echoed `frame_gen`, and
     `tracker.init()` there — this is the frame the operator actually saw,
     so the bbox is correct for it.
   - **Replay, don't jump**: call `tracker.update()` on each subsequent
     buffered frame in order (N+1, N+2, ... up to live), the same way the
     live loop already does every iteration. Correlation trackers like
     GTSTracker only search a small neighborhood around the previous bbox
     between calls — jumping straight from the old frame to the live frame
     risks landing outside that search window if the round trip was
     300-500ms. Stepping through the buffered frames lets the tracker catch
     the actual motion incrementally, same as it does in normal live
     operation.
   - Once the replay reaches the live frame, install the caught-up tracker
     as `state.tracker` / `state.bbox` and resume normal live tracking.

   This is cheap: a handful of extra `tracker.update()` calls, once per
   click, not ongoing.

   A lighter-weight alternative is a single-shot reprojection using the
   attitude buffer (compute Δroll/Δpitch/Δyaw between the clicked frame's
   timestamp and now, warp the bbox once). Cheaper, but only corrects for
   camera rotation — it won't account for the target's own motion during
   the latency window, which matters in `bMoovingTgt` mode. Buffered replay
   handles both cases since it's the same tracking algorithm already used
   live, so it's the recommended approach; attitude reprojection could seed
   it (narrow the search) but isn't required to start.

## Suggested implementation order

1. Pi: add the attitude ring buffer + frame timestamps (needed by
   everything below; independently testable by logging `frame_ts -
   attitude_ts` deltas).
2. Pi: add `frame_gen` tagging to published frames + the raw-frame rolling
   buffer.
3. Protocol: GCS echoes `frame_gen` back with `/select_point`.
4. Pi: `select_point` looks up the buffered frame instead of
   `state.current_frame`, replays the tracker forward to live.
5. Pi: attitude-based de-rotation warp on the stream-publish path (JPEG/
   WebRTC), decoupled from the LORES tracking frame.
6. GCS: nothing required for the stabilization itself once (5) is done —
   the Pi is sending an already-stabilized frame. If GCS-side smoothing is
   still wanted on top, invert its own transform locally per (1) above
   before sending clicks back.
