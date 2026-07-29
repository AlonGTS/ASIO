# ASIO Tracker

Object tracking system with WebRTC streaming and MAVLink flight controller integration.

## Running Manually

Activate the virtual environment and run:

```bash
source /home/mahat/webrtc_venv/bin/activate
cd /home/mahat/ASIO
python3 tracker.py --mode live
```

### Modes

| Flag | Description |
|------|-------------|
| `--mode live` | Live camera feed (default) |
| `--mode record` | Record video |
| `--mode playback` | Playback a recorded video |
| `--video <path>` | Video file path (for playback mode) |
| `--duration <sec>` | Recording duration in seconds |
| `--no-gui` | Disable local OpenCV window (headless/service mode) |

## Autostart Service (Operational Mode)

The tracker runs as a systemd service on boot in headless mode (no local display, streams via WebRTC/Flask).

### Install / Enable

```bash
sudo cp tracker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tracker
```

### Service Commands

```bash
sudo systemctl start tracker      # start without rebooting
sudo systemctl stop tracker       # stop the service
sudo systemctl restart tracker    # restart
sudo systemctl disable tracker    # disable autostart (e.g. for development)
sudo systemctl enable tracker     # re-enable autostart
journalctl -u tracker -f          # view live logs
```

### Development vs Operational

- **Operational**: service is enabled, starts at boot with `--no-gui`
- **Development**: disable the service and run manually (with GUI window)

```bash
# Switch to development mode
sudo systemctl disable tracker
sudo systemctl stop tracker

# Switch back to operational
sudo systemctl enable tracker
sudo systemctl start tracker
```

## Accessing the Web Interface

Once the tracker is running (manually or as a service), open a browser on any device on the same network:

| Interface | URL | Description |
|-----------|-----|-------------|
| **Live view & control** | `http://<pi-ip>:8080` | WebRTC video stream with full control panel |
| **Control API** | `http://<pi-ip>:5000` | Flask REST API (used by the UI; not a browser page) |

Replace `<pi-ip>` with the Raspberry Pi's IP address (e.g. `192.168.1.42`). To find it:

```bash
hostname -I
```

### Web UI controls

| Action | Method |
|--------|--------|
| Start stream | Click **Start** button |
| Select a target | Click on the video |
| Nudge target | Arrow keys (5 px), Shift=10 px, Alt=1 px |
| Reset tracker | **R** key or Reset button |
| Stop tracker | **S** key or Stop button |
| Quit tracker | **Q** key or Quit button |
| Toggle Fixed/Moving target | **M** key or Target button |
| Launch | **L** key or Launch button |
| Cycle MAIN resolution | **X** / **Z** keys or MAIN +/− buttons |
| Cycle tracking resolution | **V** / **C** keys or TRACK +/− buttons |
| Fullscreen | Fullscreen button |

## Deploying to Another Pi

Pulling this repo onto a *different* Raspberry Pi (not the one it was developed
on) needs a few extra steps beyond `git pull`, because some pieces are
architecture/Python-version specific or tied to that Pi's own network setup.

### 1. Get the code

```bash
cd /home/mahat/ASIO   # or wherever you clone it
git pull origin main
```

### 2. Python environment

The tracker runs from a venv created with `--system-site-packages` (so it can
see the apt-installed `opencv`/`picamera2`/`libcamera` packages, which pip
can't build reliably on a Pi):

```bash
python3 -m venv --system-site-packages ~/webrtc_venv
source ~/webrtc_venv/bin/activate
pip install -r requirements.txt
```

If `cv2` or `picamera2` aren't importable after that, install them via apt
first (`sudo apt install python3-opencv python3-picamera2`), then recreate the
venv with `--system-site-packages` so it picks them up.

`tracker.service` hardcodes `/home/mahat/webrtc_venv` in its `PATH` — update
that file if this Pi uses a different venv location or username.

### 3. The compiled tracker (`gts_tracker`)

`gts_tracker.cpython-311-aarch64-linux-gnu.so` is a **prebuilt** binary for
CPython 3.11 on aarch64. Check the new Pi matches:

```bash
python3 --version   # must be 3.11.x
uname -m             # must be aarch64
```

If it doesn't match, the `import GTSTracker` in `tracker-so.py` will fail —
rebuild from source instead:

```bash
pip install Cython setuptools
python3 setup.py build_ext --inplace
```

### 4. Per-Pi settings in `config.toml`

These reflect *this specific Pi's* network, not the code — review and adjust
after pulling:

| Key | What it controls |
|---|---|
| `[network] interface` | Which of `wlan0`/`wlan1` this Pi actually uses |
| `[network.wlan0/wlan1] bind_ip` | This Pi's IP on that interface |
| `[mavlink] pixhawk_port` | Serial device for the flight controller (`/dev/ttyACM0` etc.) |
| `[mavlink] extra_outputs` | GCS/QGC machine IPs that should get MAVLink telemetry |
| `[mavlink] autopilot` | `"custom"` (Simulink flight app) vs `"px4"` (stock PX4 OFFBOARD) |
| `[camera] idle_fps` / `active_fps` | Camera capture rate at boot vs once the GCS bumps it to full (see below) |

`mavproxy_path` in `[mavlink]` usually doesn't need setting — it auto-detects
`mavproxy.py` next to whichever venv is running `tracker-so.py`.

### 5. Camera idle/full FPS

The Pi boots capturing at `idle_fps` (default 5) to keep CPU/heat down, and
only switches to `active_fps` (default 30) when the GCS operator presses the
FPS button (or `F` key) in `gcs.py`. This is a manual toggle, not automatic —
MAVLink keeps flowing at either rate regardless, since it's handled by
`mavproxy` as a separate process, not by the camera loop.

### 6. Restart the service

```bash
sudo systemctl restart tracker
journalctl -u tracker -f
```

## Desktop Autostart (optional)

A `.desktop` entry is also available at `~/.config/autostart/tracker.desktop` which launches the tracker in an `lxterminal` window when the LXDE desktop session starts. This is an alternative for desktop-only use but the systemd service is preferred for reliable operation.
