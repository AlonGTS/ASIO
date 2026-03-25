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

## Desktop Autostart (optional)

A `.desktop` entry is also available at `~/.config/autostart/tracker.desktop` which launches the tracker in an `lxterminal` window when the LXDE desktop session starts. This is an alternative for desktop-only use but the systemd service is preferred for reliable operation.
