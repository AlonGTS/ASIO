# Testing on the Mac, no Pi needed

Two options, depending on what you're testing.

## Option A — GCS-side only (`gcs.py --file`)

Replays a recorded `.mp4` directly through `gcs.py`'s own overlays
(feature tracking, stabilization, virtual target) with no Pi, no
`tracker-so.py`, no clicking-to-track. Good for iterating on that logic
specifically.

```
python3 gcs.py --file gcs_rec_20260813_124204.mp4
```

Loops at end of file. All Pi commands (launch, nudge, select_point, ...)
become silent no-ops — there's nothing on the other end to receive them.

Caveat: these recordings are already fully processed (Pi's baked-in status
text/tracking box, plus whatever GCS-side stabilization/HUD was active when
recorded) — you're not testing against a clean raw feed, just replaying
what was captured.

## Option B — Full loop (`tracker-so.py --mode playback`)

Runs the real thing locally: CSRT tracking, the tracking-box overlay,
`select_point`/nudge/launch, Flask API — `gcs.py` connects to `127.0.0.1`
exactly as if it were a real Pi, completely unmodified. This is what you
want for testing anything that depends on the *actual* tracker (e.g. the
virtual-target drift issue, which needs real box-overlay behavior).

**One-time setup** (already done on this machine, kept here for
reproducing on a fresh machine or after a Python upgrade):

```
pip3 install --break-system-packages psutil waitress aiohttp aiortc \
    flask flask-cors pymavlink cython setuptools
python3 setup.py build_ext --inplace   # compiles gts_tracker.pyx -> gts_tracker.cpython-*.so
```

The `.so` is tied to this Python version/arch (see `gts-tracker-howto.txt`)
— recompile after a Python upgrade.

**Every time**, two terminals:

```
# Terminal 1 — the "Pi"
python3 tracker-so.py --mode playback --video gcs_rec_20260813_124204.mp4 --loop

# Terminal 2 — the GCS
python3 gcs.py --pi 127.0.0.1 --port 5050
```

`--port 5050` matches `flask_port = 5050` added under `[network]` in
`config.toml` — macOS's own AirPlay Receiver squats on the default port
5000, so `tracker-so.py`'s Flask API needs to move off it locally. That
config line and its default (5000) are backward compatible with every real
Pi, which doesn't set it.

Click on the video to select a target, same as flying for real — CSRT
tracks it against the recorded footage, box overlay and all.

### What had to change to make this possible

None of this is Mac-only special-casing — these were genuine bugs/gaps
that would have bitten `--mode playback` on the Pi too, just never
exercised there:

- `tracker-so.py` launched MAVProxy against a real serial port
  unconditionally at import time, regardless of `--mode`. Now skipped
  outside `--mode live` (every `mavlink_client` call is already internally
  gated on `_enabled`, which just stays `False`, so nothing downstream
  needed to change).
- `FrameBuffer.put()` (`webrtc_server.py`) didn't accept the `gen=`
  keyword `tracker-so.py`'s main loop had started passing it (for
  frame_gen correlation, see `STABILIZATION.md`'s stale-click note) — a
  straight `TypeError` on every published frame, any video mode, any
  platform. Fixed to accept an optional `gen` and only auto-increment when
  omitted (preserves `tracker.py`'s older, gen-less caller).
- Flask port hardcoded to 5000; now `config.toml`'s `[network] flask_port`
  (default 5000, unset on every real Pi) can override it.

## Running it from VS Code

`.vscode/launch.json` has a "Local playback (tracker-so + GCS, no Pi)"
compound that launches both processes at once — see the Run and Debug
panel. `tracker-so: playback` doesn't pass `--video` at all, so
`tracker-so.py` always opens a native macOS file picker to choose the
recording — no dropdown to keep in sync with new recordings.

That picker (`_choose_video_file()` in `tracker-so.py`) runs via
`osascript`/AppleScript in a completely separate process, deliberately
**not** Tkinter. An earlier version used Tkinter's `filedialog`, launched
from within the same debugpy-spawned process — that was unreliable in
practice: the "Mahat GCS" window would go unresponsive (spinning-cursor
beachball, clicks not registering), and even after properly destroying the
Tk window, macOS kept the process registered as a second, unlabeled
"Python" app in the Dock for the rest of its life (Tk/Cocoa window
creation does that permanently, not just while a window is open). Running
the dialog in an unrelated `osascript` process sidesteps both problems —
this process never touches Tk/Cocoa itself.
