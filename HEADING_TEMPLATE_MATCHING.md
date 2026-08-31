# Heading-Compensated Template Matching for General Targets

## Context

The white-target refinement built so far (`_refine_aim_point` /
`_find_cross_centroid` in `tracker-so.py`) is specific to one target shape: a
bright sheet with a dark "+" mark. It works by segmenting shapes
(brightness/contrast thresholds), not by recognizing a picture — which is
why it's rotation-invariant for free and needed no reference image.

This plan is for a **different, more general case**: an arbitrary target
where the operator can supply a reference photo, but the target doesn't have
a simple bright/dark shape to segment. Plain template matching
(`cv2.matchTemplate`) was already ruled out once for the cross mark, because
it isn't rotation-invariant — matching a fixed-orientation reference against
a target that can appear at any angle in frame would need a search across
many rotation angles, too expensive to run per-frame on a Pi.

**The idea that changes this:** the vehicle's real-time heading is already
available via MAVLink `ATTITUDE` messages (`mavlink_client.py`'s
`_telemetry_reader()` already receives `msg.yaw` — currently only printed
when `SHOW_TELEMETRY` is on, never stored). If the operator also records the
heading the reference photo was taken at, the expected rotation between the
reference and the live frame at any moment is just:

```
rotation_needed = current_vehicle_heading − reference_image_heading
```

(This assumes a fixed, non-gimbaled, body-mounted camera — true for this
project's design, where pitch/yaw guidance is sent straight to the flight
controller's attitude control with no separate gimbal.) That turns "search
all 360°" into "check a narrow band around one predicted angle" — cheap
enough to run every frame, the same way the discarded original
template-matching plan predicted *scale* from the tracker's current bbox
size instead of sweeping a wide scale range blindly.

## Design

### 1. Real-time heading capture (`mavlink_client.py`)

Add a module-level `_current_yaw_rad` (mirrors the existing `_launched`
pattern — a plain global other modules read directly), updated in
`_telemetry_reader()`'s existing `ATTITUDE` branch:
```python
elif mtype == 'ATTITUDE':
    global _current_yaw_rad
    _current_yaw_rad = msg.yaw
    ...
```
`None` until the first `ATTITUDE` message arrives — callers must handle that
(no heading yet = can't predict rotation, matching falls back to a wider
search or is simply unavailable that frame).

### 2. Reference image + calibration input

New config (`config.toml [tracking]`), or a small companion sidecar file per
target (needs a decision — see Open Questions):
```toml
target_template_path         = ""     # "" = feature off
target_template_heading_deg  = 0      # compass heading the reference photo was taken at
target_template_ref_bbox_px  = 60     # MAIN-px bbox side length at capture time (scale prediction, same idea as the original template-matching plan)
target_camera_mount_offset_deg = 0    # one-time calibration: camera "up" vs vehicle forward, if not exactly aligned
```
Loaded once at startup exactly like the earlier `aim_template_path` design:
`cv2.imread(..., cv2.IMREAD_GRAYSCALE)`, warn and disable gracefully on
failure, never raise.

### 3. New function `_match_heading_template` (same location/contract as
`_refine_aim_point`/`_find_cross_centroid`)

`(frame, x, y, bw, bh, fallback_cx, fallback_cy) -> (cx, cy, found)` in MAIN
coords — same drop-in shape as the other two detectors, so it plugs into the
same dispatch, periodic recenter, EMA+snap smoothing, drift recovery, and
MAVLink/crosshair code with no changes there.

- **Predicted rotation**: `rotation_needed = current_vehicle_heading -
  target_template_heading_deg - target_camera_mount_offset_deg`, wrapped to
  ±180°. If `_current_yaw_rad` is `None` (no telemetry yet), fall back to
  `(fallback_cx, fallback_cy, False)` — can't safely narrow the search.
- **Rotation band**: search a handful of angles (e.g. 5) spanning
  `rotation_needed ± 15°` — narrow because the heading prediction already
  does the heavy lifting; this only absorbs compass error and small
  mounting misalignment.
- **Predicted scale**: same approach as the original template-matching
  plan — `predicted_scale = max(bw, bh) / target_template_ref_bbox_px`,
  searched over a narrow band (e.g. ±25%, 5 samples) around it.
- **Search window**: crop centered on the current bbox, sized generously
  (e.g. 1.6× max(bw,bh)) like the other detectors' search regions.
- **Per (rotation, scale) pair**: rotate + resize the reference template,
  `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)` (same primitive already
  used in `TrackingQualityMonitor` and the original template-matching
  design — no new OpenCV surface introduced), `cv2.minMaxLoc` for the peak.
  Keep the best-scoring (rotation, scale) combination.
- **Confidence gate**: accept only above a configurable threshold (e.g.
  0.55, tune against real footage); otherwise fall back, same failure
  contract as the other two detectors.

### 4. Integration alongside the existing detectors

This becomes a **third selectable mode**, not a replacement for the
cross-sheet pipeline — the existing blob/cross detectors stay as-is for that
specific target type. Something needs to choose which detector runs (e.g. a
`target_type = "cross_sheet" | "template"` config value, or a GCS control),
since they solve different problems and shouldn't run simultaneously.

### 5. Performance

Grid size is `num_rotations × num_scales` (e.g. 5×5 = 25) `matchTemplate`
calls per search — more than the original scale-only plan's 5, so this
needs its own reduced cadence (like the discarded template-matching plan's
`aim_template_match_every_n_frames`), not every frame. **Must be measured on
real Pi hardware** before picking a default cadence — add a timing print
around the search (same pattern as `_update_ms`) and tune from that
measurement, not an estimate.

## Open Questions (need answers before implementation)

1. **Where does the reference image + its calibration data live?** A single
   `config.toml` entry (simple, but one target per Pi/config) vs. a small
   per-mission sidecar (image + JSON/TOML metadata) selectable at runtime
   from the GCS — the latter is more flexible if targets change between
   flights, at the cost of needing a small file-selection mechanism (GCS
   button, or a `--target` CLI arg like the existing `--video` playback
   flag).
2. **How is the reference heading actually measured/recorded** when the
   operator takes the photo? Options: read it off the vehicle's own compass
   right before flight (if the photo is taken with the aircraft on the
   ground, facing a known direction), a handheld compass, or a phone
   compass app — accuracy here directly limits how narrow the rotation
   search band can safely be.
3. **Compass reliability**: magnetic heading can drift/be disturbed by local
   interference (particularly near the ground, metal structures, motors) —
   worth sanity-checking how trustworthy `ATTITUDE.yaw` is in this
   project's actual flight environment before leaning on it for a search
   window's center.
4. **What target(s) is this actually for?** The design above is generic,
   but concrete example targets would sharpen the confidence threshold,
   search-band widths, and template preprocessing (e.g. does the target
   have enough texture/contrast for NCC to work reliably, or does it need
   edge detection / other preprocessing first?).

## Verification

1. Local playback with a recorded flight segment where heading is roughly
   known/loggable, plus a manually captured reference image and its
   heading, to confirm the rotation-band math actually points in the right
   direction (test with a synthetic frame rotated by a known amount first,
   before trusting real footage).
2. Confirm graceful fallback: no heading yet, missing/failed reference
   image, low-confidence match — all should degrade to the plain bbox
   center, never crash, matching the existing detectors' failure contract.
3. Timing validation on real Pi hardware to set the search cadence, same as
   the original template-matching plan required.
4. Live flight test once cadence/thresholds are validated against real
   footage — watch specifically whether the heading-predicted rotation band
   is wide enough to reliably contain the true angle (compass error +
   mounting tolerance) without being so wide it stops being cheap.
