# Feature tracking, stabilization, and virtual target (gcs.py)

Three features in `gcs.py` build on background feature detection
(`goodFeaturesToTrack` + `calcOpticalFlowPyrLK`), but virtual target and
stabilization now use it differently: stabilization needs a full rigid
(rotation + translation) transform for the whole frame, fit fresh via
RANSAC each call; virtual target just needs a robust average of how the
*same persistent points* the FEATURES diagnostic already tracks moved this
frame (see "Virtual target" below) — no separate fit, no rotation.

## Shared pieces

- `_feat_exclude_mask(frame)` — mask out regions that would contaminate
  "background" motion: the Pi's baked-in status text (top ~22% of frame,
  bottom-left "LAUNCHED" badge) and the Pi's tracking-box overlay
  (color-matched via `cv2.inRange` on its green/orange/red box colors,
  dilated to cover compression edges). Used by both `_feat_update()` (feeds
  virtual target) and `_estimate_global_motion()` (feeds stabilization).
- `_estimate_global_motion(prev_gray, gray, mask=None)` — `goodFeaturesToTrack`
  + `calcOpticalFlowPyrLK` + `estimateAffinePartial2D(..., RANSAC)`. Returns
  `(dx, dy, dangle_deg, n_features, n_inliers)` or `None` if too few inliers.
  Rejects the fitted scale — every consumer rebuilds a scale=1 matrix via
  `_recompose_xya(dx, dy, dangle)` before using it, since accumulating raw
  fitted scale drifts multiplicatively over a long session. **Stabilization
  only** — virtual target used to share this (see its "History" section
  below) but no longer does.
- `_compose` / `_decompose_xya` / `_recompose_xya` / `_wrap_deg` — 2x3 affine
  helpers, used by stabilization. `_wrap_deg` matters because angle is
  circular; without it, an EMA crossing the ±180° boundary produces a
  runaway correction.

## FEATURES diagnostic + shared point tracking (`_feat_update`, `toggle_features`)

`_feat_update()` is the one place that actually runs `goodFeaturesToTrack`
+ `calcOpticalFlowPyrLK` on a *persistent* set of points (re-seeded once
too few survive) — it runs whenever either the FEATURES toggle is on, or a
virtual target is selected, whichever needs it. `toggle_features()` only
controls `draw_feature_points()` — whether those tracked points are drawn
as green dots — not whether tracking itself runs.

Every frame, alongside tracking, it also computes `_feat_last_delta`: the
*median* x/y displacement of whichever points survived (already excluded
from contaminated zones before this, so a plain median of what's left is
enough — no RANSAC/affine fit needed). This is the single shared "how did
the background move this frame" signal — virtual target (below) is its
only other consumer. One tracking pass serves both instead of duplicating
the optical-flow work per consumer.

## Whole-frame stabilization (`_stabilize`)

Maintains a cumulative (x, y, angle) trajectory (`_stab_cum`), low-pass
filtered into `_stab_smooth_xya`. The correction applied to the frame is the
difference between raw and smoothed trajectory, so genuine slow panning
still comes through — only the high-frequency jitter gets removed.

Adaptive alpha: blends between `_STAB_SLOW_ALPHA` (0.05, strong smoothing)
and `_STAB_FAST_ALPHA` (0.45, quick catch-up) based on how large the
per-frame motion is (`_STAB_BIG_MOVE_PX` / `_STAB_BIG_MOVE_ANGLE`). Without
this, a large genuine camera pan lagged far behind the smoothed trajectory,
producing a large correction and exposing a big `BORDER_REPLICATE` strip
("removes part of the video in big moves"). With it, sustained big pans see
correction magnitude drop ~83% vs the fixed-alpha version, while small-jitter
smoothing is barely affected.

Guards: per-frame estimate discarded if `|dx|/|dy| > _STAB_MAX_SHIFT` or
`|dangle| > _STAB_MAX_ANGLE` (bad match, not real motion); resets to
identity if the smoothed-vs-raw gap ever balloons past 4x those limits
(shouldn't happen, but caps worst case instead of ever compounding);
resets `_stab_prev_gray` after a `> _STAB_GAP_RESET_S` gap (dropped/stalled
frames) so the next frame is treated as a fresh reference instead of
estimating motion across the gap.

`_to_raw_coords()` inverts the full `_stab_M` matrix (not just an X/Y
shift), so click coordinates map back correctly even under rotation.

## Virtual target (`_vt_update`, `draw_virtual_target`)

Tracks where a selected point currently is even after it leaves the frame,
so it can be pointed back to on reacquisition. `select_point()` calls
`_vt_reset(x, y)` to anchor it. Every subsequent frame, `_vt_update()`
applies `_feat_last_scale`/`_feat_last_delta` (computed by `_feat_update()`,
which must run first — see the main-loop call order) directly to the point:
`nx, ny = scale*x + dx, scale*y + dy`. If `_VT_MAX_MISSES` (5) consecutive
frames produce no valid delta/scale, the target is cleared rather than left
frozen — mirrors the Pi's own "Drift — re-select target" recovery. Found via
a real recorded session with a hard content cut (the video's own internal
clock progressed smoothly across it — not a dropped-frame gap
`_FEAT_GAP_RESET_S` would catch — but the visible content changed
completely, and the Pi's own tracker flagged the same moment as lost):
without this, the marker froze at its pre-cut position and then kept
"propagating" it through a background it had never actually seen, which
looks like confident nonsense rather than an honest "lost" state.

The tracked background features are themselves anchored to the ground/scene
content, so however *they* moved (scaled, and rotated) this frame is
exactly how a background-anchored point should move too. One mechanism,
used the same way whether the target is currently on-screen or not — no
separate estimation pass, no per-consumer exclusion mask, no
special-casing "in view" vs "out of view".

`_feat_update()` computes a full similarity transform (translation + scale
+ rotation) from the same tracked point set, all from the same per-pair
measurements:
- **Scale and rotation** — for every pair of tracked points, how did the
  vector connecting them change, new frame vs old: `scale_ij =
  dist(new_i,new_j) / dist(old_i,old_j)`, `angle_ij = angle(new_j-new_i) -
  angle(old_j-old_i)`. Median over all pairs whose old-frame baseline is
  far enough apart to give a stable ratio (`_FEAT_MIN_BASELINE_PX`). Both
  are **pivot-independent** — neither needs a "center" chosen up front,
  which is exactly why this rotation estimate doesn't have the amplification
  problem the v1 design did (see History) — that one came from a single
  noisy RANSAC-fit angle applied via a matrix pivoted at the frame's `(0,0)`
  corner, not from rotation itself being inherently unsafe to estimate.
  Both clamped hard per frame (`_FEAT_SCALE_CLAMP` ±15%, `_FEAT_ANGLE_CLAMP`
  ±3°) — scale because it matters on every approach/retreat, not
  occasionally, so an unclamped bad reading would be a frequent problem;
  angle because that's exactly the v1 failure mode, just via a more robust
  estimator this time, not an excuse to skip the safety margin.
- **Translation** — once scale+rotation give a 2x2 linear map `R`, the
  median per-point residual `new_i - R @ old_i`, which is the same
  "smart average" the translation-only version always did, just correctly
  accounting for the fact that different points move differently under
  scale/rotation rather than assuming they all move by the same vector.

Needed because approaching/receding from the target, or an uncompensated
camera roll, aren't shifts — the whole scene scales/rotates around a
center point, so points near the frame edge move differently than points
near the middle, which a plain translation vector can't represent (was
moving the marker away from the real target on approach/retreat, or during
a real camera rotation — see History, v5).

`draw_virtual_target()` draws an on-screen marker when the point is in
view, or an edge arrow + distance label when it isn't (via
`_vt_to_display()`, which forward-applies `_stab_M` so the marker still
lands correctly under stabilization).

### Validation

Live, against a real recorded playback session (`tracker-so.py --mode
playback`, real CSRT tracking, `_H264LiveCapture` decoding the actual RTP
stream): of 100 consecutive frame updates after a real `select_point`, 99
got a valid median delta, and the point stayed within a tight (~±30-40px)
bounded wander around its start — not a runaway divergence — over the
whole 15s window. Re-validated after adding scale (89/90 valid) and again
after adding rotation (105/110 valid; scale stayed in [0.96, 1.02], angle
stayed within ±0.62° with a ~0.01° mean — no systematic bias, no runaway).

Synthetic (`/tmp/real_frame.png`): 5 frames of ~3%/frame zoom-in and
zoom-out each about frame center, sub-pixel error (<0.2px) on an off-center
point in both directions; pure rotation (1.5°/frame x5) and a combined
rotation+scale+translation case, both accurate to ~0.05px; pure translation
re-verified as an exact-match regression throughout.

**Pitfall hit while validating rotation, worth remembering**: the first
version of the rotation synthetic test showed ~40-80px error and looked
like a real bug in the code — it wasn't. `cv2.getRotationMatrix2D`'s 2x2
part is `scale*[[cos,sin],[-sin,cos]]`, the *opposite* sign convention from
the "standard" math-textbook rotation matrix `[[cos,-sin],[sin,cos]]` the
test's hand-written ground-truth function assumed. Recomputing ground
truth by directly applying OpenCV's own returned matrix (instead of an
independently-derived formula) resolved it immediately, down to 0.05px.
Any test that builds a synthetic transform with one OpenCV call and then
checks it against a *separately* hand-derived formula for the "expected"
result is at risk of exactly this — prefer computing ground truth from the
same matrix actually used to build the test frame.

### History

**v1 — full rigid transform (translation + rotation) via a separate RANSAC
affine fit**, reusing `_estimate_global_motion()` (the same estimator
`_stabilize()` uses). Looked directionally correct in synthetic tests, but
live footage showed the marker drifting significantly even fully in FOV —
root-caused to two compounding problems:
1. **Rotation amplification.** `_recompose_xya`'s rotation pivots at the
   frame's `(0,0)` corner, so its error scales with the point's distance
   from there. A live example: `dangle=+0.22°` of ordinary per-frame
   estimation noise alone displaced a point at `(575,445)` by ~2-3px — on
   top of the real translation, in a different direction — *every single
   frame*, with nothing to smooth it out (stabilization is safe from this
   because it applies a smoothed *delta* to the whole frame, not a raw
   per-frame value to one far-off point).
2. **Ill-conditioned translation fits.** On a scene where nearly all
   trackable texture sits along one line (e.g. a single diagonal wash/
   tree-line, confirmed via a screen-recorded test session — the target
   visibly moving down-right while the estimate walked up-left, even fully
   in FOV), a RANSAC affine fit is well-constrained *along* that line but
   barely constrained *across* it — a textbook aperture problem.

**v2 — ground-truth box detection**: while the Pi's own tracking-box
overlay was visible, read its position directly by color
(`cv2.inRange` + contour/fill-ratio filtering to reject false matches) and
skip estimation entirely; only fall back to v1's (by-then translation-only)
propagation when the box wasn't drawn. This sidestepped both v1 problems
whenever the target was in view, and validated well in a *synthetic*
box-detection test — but live testing showed it snapping to the box's
*corners* rather than its center: the thin 2px border fragments under
compression, and reconstructing "the whole box" from scattered color-matched
fragments proved unreliable in practice, not just in the harsher screen-
recording tests used to validate it.

**v3** replaces both: instead of a separate estimation pass or a fragile
box reconstruction, it reuses the *individual* points the FEATURES
diagnostic already tracks (each one persistent, identity-tracked frame to
frame, "stuck" to a real scene location) and takes their median motion
directly — conceptually simpler, reuses existing tracking work instead of
adding a parallel pass, and available far more often (no dependency on the
box being visible or perfectly shaped) — 99/100 frames above vs. v2's
43/80 box-snap rate on a comparable live test.

**Current (v4)** extends v3 with two fixes, both from live Pi footage
(v3 was only validated against recorded playback before this):
- **Fast motion.** A fast whip-pan could lose the target — `_LK_PARAMS`
  widens `calcOpticalFlowPyrLK`'s search window/pyramid (21x21/3 levels ->
  41x41/4 levels), but critically that alone wasn't enough: past a point,
  wider search just made points converge to the wrong (but similar-
  looking) nearby match while still reporting "success" via `status`.
  Measured on a real frame at a 150px synthetic jump: ~90% of "successful"
  points had actually converged to the wrong spot, silently corrupting the
  median with garbage. `_LK_MAX_ERR` filters on LK's own match-residual
  output (previously discarded) to catch this — `err<10` gave the exact
  correct median through 150px, and safely fell back to "too few points,
  skip this frame" beyond that instead of confidently returning a wrong
  answer.
- **Scale (approach/retreat).** See "Virtual target" above — translation
  alone can't represent the marker's true motion as the drone gets closer
  to or farther from the target.

**v5** adds two more fixes, both from a real recorded session with a confirmed-
fixed target:
- **Lost tracking left the marker frozen inside a scene it never saw.** The
  session had a hard content cut — the video's own internal clock progressed
  smoothly across it (not a dropped-frame gap `_FEAT_GAP_RESET_S` would
  catch), but the visible content changed completely, and the Pi's own
  tracker flagged the same moment as lost ("Drift — re-select target"). VT
  just froze at its pre-cut position and kept "propagating" it through
  unrelated content. `_VT_MAX_MISSES` (5 consecutive frames with no valid
  delta) now clears the target instead, mirroring the Pi's own recovery.
- **Rotation**, reintroduced — see "Virtual target" above for the
  pivot-independent, pairwise-median estimator that avoids v1's
  amplification bug. Root-caused from the same session: after a re-select,
  the marker diverged from the (still-tracked, confirmed-fixed) target in a
  steadily *growing*, one-directional way — not random jitter, which is the
  signature of an uncompensated real rotation rather than noise (noise
  wouldn't grow monotonically in one direction) or an independently-moving
  target (ruled out — the target was fixed).

### Known residual risk

No outlier rejection on the scale/rotation/translation estimates beyond the
median's/clamps' own robustness — a scene where the target's own
(independently moving) texture isn't fully excluded from the tracked point
set could still bias it, and there's no confidence gate on point count/
spread before trusting a given frame's estimate. Not yet observed as a
practical problem beyond what v4/v5 already fixed; if it resurfaces, that
confidence gate is the next lever before reaching for anything more
complex.
