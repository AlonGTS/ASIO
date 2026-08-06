# Tracker Accuracy & Stability Roadmap

## Goal

Improve the stability of the tracking point and the accuracy of the target's
position in the image — especially when the target is small, near the
center of the frame, and when a smooth, reliable output is needed for an
external controller.

## A note on resolution

Today the frame is 640×480, but the tracker (CSRT) generally runs on a
320×240 downscale.

Moving to a higher resolution *can* improve positional accuracy, but not
automatically.

**Potential benefits:**
1. More pixels on the target.
2. A small angular change shows up as more pixels, so it can be measured
   more precisely.
3. More detail and edges are preserved in the tracked area.
4. Sub-pixel localization becomes easier.

**Limitations:**
1. If the image is blurry, noisy, or heavily compressed, a higher
   resolution doesn't necessarily add real information.
2. Higher resolution can increase processing time and system latency.
3. If the tracking window stays too small relative to the target, the
   tracker can still be unstable.
4. If the tracking error comes from a bad template update, lighting
   change, or a sharp motion, resolution alone won't fix it.

**Important: the sensor is currently read out with pixel binning.** That
means the *available* sensor resolution is significantly higher than the
640×480 (or even the 1640×1232 `main_sizes` ceiling in `config.toml`) in
use today — reaching it is a capture-mode/config change on the Pi, not new
hardware. Binning trades resolution for per-pixel light sensitivity and
noise reduction (it sums/averages neighboring photosites), so moving to a
less-binned or unbinned mode should be expected to increase per-pixel
noise and reduce low-light sensitivity, and it raises bandwidth/processing
cost — worth measuring directly (Experiment 2) rather than assuming either
way. This makes the "higher sensor resolution" fallback in the resolution
decision below a near-term config experiment, not a hardware purchase.

**Recommended conclusion:** don't jump straight to full-resolution tracking
on every frame. It's better to work at two levels:
- Coarse acquisition and tracking at low resolution.
- Precise computation on a small region cropped from the original,
  full-resolution frame.

## Proposed architecture

### Stage 1 — Keep the original frame
Store the original 640×480 frame every cycle, not just the 320×240
downscaled version.

### Stage 2 — Primary tracking
Keep using CSRT on 320×240 for:
- Maintaining lock.
- Handling scale changes.
- Handling relatively large motion.
- Producing a general bounding box.

### Stage 3 — Map the position to full resolution
Multiply the bounding box center and bounds by the resolution ratio to get
640×480 coordinates.

Example:
```
x_full = x_low × 2
y_full = y_low × 2
```

### Stage 4 — Local refinement at full resolution
Crop a small ROI from the original frame around the computed center, e.g.:
- 80×80
- 120×120
- or a size proportional to the target's size

Inside the ROI, refine the position using one of:
1. Normalized Cross Correlation (NCC).
2. Template matching on a gradient image.
3. Lucas-Kanade on a handful of strong points.
4. NCC combined with a weighted centroid of the match peak.

The goal is **not** to replace CSRT — it's to add a Fine Tracking layer
that corrects the center position.

### Stage 5 — Sub-pixel localization
Instead of picking only the pixel with the highest match score, compute the
peak's center over a 3×3 or 5×5 neighborhood.

This gives a position like:
```
x = 317.4
y = 241.7
```
instead of just:
```
x = 317
y = 242
```

This matters especially when a 1-pixel error is already significant.

### Stage 6 — Motion filtering
Add a simple state filter for x, y, vx, vy.

The filter should not replace the measurement — it should:
- Smooth small noise.
- Detect implausible jumps.
- Provide a short prediction for the next frame.
- Allow a smaller local search window.

Recommended: start with a basic constant-velocity Kalman filter, but use it
carefully:
- When measurement confidence is high, weight the measurement heavily.
- When confidence is low, weight the prediction more heavily.
- Don't over-filter the output — that introduces lag.

### Stage 7 — Confidence metric
Compute a confidence score every frame from several signals:
1. NCC value.
2. Ratio between the best peak and the second-best peak.
3. Sharpness of the match peak.
4. Change in bounding-box size.
5. Distance between the CSRT measurement and the refinement result.
6. Agreement between the new position and the prediction.

Suggested breakdown:
- **High confidence** → normal update.
- **Medium confidence** → slow template update.
- **Low confidence** → don't update the template; widen the search area.
- **Very low confidence** → declare lock lost.

### Stage 8 — Template management
One of the main sources of tracker drift is a bad template update. So:
1. Keep the original template from the moment of lock.
2. Keep a dynamic template that updates slowly.
3. Only update the template when confidence is high.
4. Periodically compare against the original template too.
5. If there's a large contradiction, stop the dynamic update.

### Stage 9 — Final precise mode
When all of the following hold:
- The target is near the center.
- The in-image velocity is low.
- Confidence is high.
- The target is large enough.

Switch to Fine Tracking mode:
1. Use a full-resolution ROI.
2. Gradually reduce the template update rate.
3. Rely more on gradients/edges than on absolute brightness.
4. Perform sub-pixel localization.
5. Output both the raw and the filtered position.

### Stage 10 — Separate the tracker from the controller
Separate:
- Image measurement noise.
- The target's motion in the image.
- The controller's response.

The tracker should provide:
- `x_raw`, `y_raw`
- `x_refined`, `y_refined`
- `x_filtered`, `y_filtered`
- `confidence`
- `tracker_state`
- `bbox_size`
- `measurement_age`

This makes it possible to tell whether instability originates in the
tracker or in the controller.

## Experiment plan

### Experiment 1 — Baseline
Record video and log data with the existing system:
- CSRT at 320×240.
- Bounding-box center.
- Processing time.
- Confidence, if available.
- Center movement between frames.

### Experiment 2 — CSRT at full resolution (and less/no binning)
Run CSRT on 640×480 against the same recorded video, and separately capture
a comparison pass at a less-binned/unbinned sensor mode.

Measure:
- Positional error.
- Jitter in the center.
- FPS.
- Latency.
- Drift incidents.
- Per-pixel noise level (expected to rise as binning decreases).

Goal: check whether resolution alone provides a meaningful improvement, and
quantify the noise/light-sensitivity cost of reducing binning.

### Experiment 3 — Low-res CSRT + full-res refinement
Run:
- CSRT at 320×240.
- ROI cropped from 640×480.
- NCC or gradient matching inside the ROI.

**This is the most important experiment** — it's expected to give an
accuracy improvement without doubling the full tracker's load.

### Experiment 4 — Sub-pixel
Add sub-pixel peak fitting and check:
- Standard deviation of the position when the camera and target are nearly
  static.
- Effect on output stability.
- Whether the noise decreases without adding latency.

### Experiment 5 — Kalman
Add a Kalman filter only after there's a good measurement.

Compare:
- raw
- refined
- filtered

Verify the filter reduces noise without introducing motion lag.

### Experiment 6 — Template management
Compare:
- Updating the template every frame.
- Updating only when confidence is high.
- Original template + dynamic template.

## Recommended metrics

1. RMS of the center error, in pixels.
2. Standard deviation when the target is static.
3. Percentage of frames with a jump above threshold.
4. Recovery time after blur or a sharp motion.
5. Percentage of lock losses.
6. End-to-end latency.
7. Actual FPS.
8. Gap between the raw and filtered position.
9. Sharpness of the correlation peak.
10. Estimated angular error, based on the field of view.

## Recommended decision on resolution

The recommended first step is **not** a full move from 320×240 to 640×480.

The recommended step is:
1. Keep running CSRT at 320×240.
2. Take its result as a coarse estimate.
3. Crop an ROI from the original 640×480 frame.
4. Refine the point position inside the ROI.
5. Add sub-pixel localization.
6. Add confidence.
7. Only then add a gentle Kalman filter.

If an experiment shows the target still gets very few pixels even at
640×480, the sensor is already known to support higher resolution than
what's captured today (see the binning note above) — so the next step is
trying a less-binned/unbinned capture mode before considering anything
outside software/config:
- A less-binned or unbinned sensor mode (config change, test the noise
  trade-off first).
- A narrower lens.
- Optical zoom.
- Better focus and exposure time.

Digital upscaling alone does not replace real optical information — but in
this case there may be real optical information already available and
simply not being read out due to the current binning mode.

## Recommended implementation order

- **Step A**: Logging and performance measurement of the existing system.
- **Step B**: Fine Tracking at full resolution inside an ROI.
- **Step C**: Confidence and template management.
- **Step D**: Sub-pixel localization.
- **Step E**: Basic Kalman filter, with confidence-based variable weighting.
- **Step F**: Comparison against full-resolution CSRT (and reduced-binning
  capture, per Experiment 2).
- **Step G**: Only if needed — move to an architecture where the local
  tracker becomes primary and CSRT is used for re-acquisition.

## Expected outcome

The biggest improvement is not expected to come from replacing CSRT, but
from the combination of:

CSRT for acquisition and lock-holding
\+ local refinement at full resolution
\+ sub-pixel localization
\+ confidence
\+ template management that prevents drift
\+ gentle filtering that doesn't add significant latency

## Progress log

### 2026-08-06 — Step A implemented and run

**Step A (logging/baseline instrumentation) is done and committed:**
- `tracker-so.py` writes a per-frame CSV to `logs/baseline_<timestamp>.csv`
  whenever `[logging] baseline_enabled = true` in `config.toml` (on by
  default). Columns: `wall_ts, session_id, session_source, frame_gen,
  main_w/h, lores_w/h, success, cx, cy, bbox_w/h, center_dx/dy/dist,
  update_ms, tq_score, bad_frames, drift_event, inst_fps`. Runs on a
  dedicated thread off a bounded queue — never blocks the tracking loop.
- `session_source` tags *why* each tracker (re)init happened — `click`,
  `local_click`, `nudge`, `resize`, or `bbox_clamp` — via
  `state.last_init_source`, set at every call site that creates a new
  tracker (`flask_app.py` `/select_point` + `/nudge`, `tracker-so.py`'s
  mouse callback, resolution-cycle reinit, and the bbox-clamp reinit).
- `analyze_baseline.py` (repo root) summarizes any log: frame counts,
  drift events, `update_ms`/`fps`/`tq_score` stats, center-jitter stats,
  and — when present — a sessions-by-source breakdown. Run with no args
  for the latest log, or pass a path explicitly.
- Commits: `74d65f8` (baseline logging), `0aef969` (session_source). Both
  pushed to `origin/main`. Tag `stabilized` (`ff52a41`) marks the commit
  right before this work, in case it needs to be rolled back to.

**Three runs collected today, all in `logs/`:**
1. `baseline_20260806_104131.csv` — first smoke test (scenario not recorded
   before `session_source` existed). 1521 frames, 0 drift, jitter mean
   1.83px / p95 8.0px.
2. `baseline_20260806_105602_static_real.csv` — playback, static target,
   real (not synthetic) footage. 2517 frames, 3 auto-detected drift events
   spaced almost exactly ~44.5s apart with near-identical re-acquisition
   coordinates each time — consistent with the tracker reliably failing at
   the *same moment* in a looping clip. Only 3 of 21 session transitions
   were tagged `drift_event`; predates `session_source` so the other 18
   transitions' cause was never confirmed.
3. `baseline_20260806_112640_playback_fixed_enddrift.csv` — playback, fixed
   target; per the operator, the payload/camera ends up pointed at a
   different spot than the tracked point by the end of the clip, causing
   genuine drift there. 1464 frames, 2 drift events. **Key finding: 24 of
   35 sessions (69%) were `bbox_clamp`** — i.e. the CSRT bbox repeatedly
   grew past `max_bb_width`/`max_bb_height` (120×120 in `config.toml`),
   forcing a full tracker reinit (template/state discarded) each time.
   Only 2 sessions were genuine drift-driven re-clicks.

**Open finding to chase next:** `bbox_clamp` churn looks like a bigger
contributor to instability than actual drift on this footage — every
clamp event throws away the tracker's template and restarts cold. Worth
checking before starting Step B whether `max_bb_width`/`max_bb_height` is
simply too tight for this target's apparent size/distance, or whether
CSRT's scale estimation (`number_of_scales`, `scale_step`, etc. in
`config.toml`'s `[gts-track.fixed]`/`[gts-track.moving]`) is overshooting
and growing the box faster than it should.

**Not started yet:** Step B (full-res ROI refinement) — expected to be the
highest-value next step per "Expected outcome" above, but the `bbox_clamp`
finding suggests it may be worth a quick look at the clamp/scale behavior
first, since Step B's accuracy gain will be hard to measure cleanly against
a baseline that's still churning on reinits this often.

**Also flagged, not yet acted on** (from code review earlier in this
work): `webrtc_server.py`'s `FrameBuffer.get()` copies the frame while
holding the lock — worth revisiting once Step B starts touching full-res
frames every iteration instead of just for streaming/recording.
