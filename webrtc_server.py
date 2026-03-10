#!/usr/bin/env python3
"""
WebRTC server for tracker.py.

Usage from tracker.py:
    import webrtc_server
    fb = webrtc_server.FrameBuffer()
    t  = Thread(target=webrtc_server.start, args=(fb,), daemon=True)
    t.start()
    # then in your frame-production loop:
    fb.put(output_frame)
"""
import asyncio
import time
from fractions import Fraction
from threading import Condition, Lock

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame


# ---------------------------------------------------------------------------
# Shared frame buffer
# ---------------------------------------------------------------------------

class FrameBuffer:
    """Thread-safe container that tracker writes to and GlobalFrameTrack reads from."""

    def __init__(self):
        self._lock = Lock()
        self._cond = Condition(self._lock)
        self._frame = None

    def put(self, frame):
        with self._cond:
            self._frame = frame
            self._cond.notify_all()

    def get(self, timeout=0.05):
        with self._cond:
            if self._frame is None:
                self._cond.wait(timeout=timeout)
            return None if self._frame is None else self._frame.copy()


# ---------------------------------------------------------------------------
# WebRTC video track
# ---------------------------------------------------------------------------

class GlobalFrameTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, frame_buffer: FrameBuffer, target_fps=30):
        super().__init__()
        self._buf = frame_buffer
        self.time_base = Fraction(1, 90000)
        self.frame_interval = 1.0 / target_fps
        self._last_ts = 0.0

    async def recv(self) -> VideoFrame:
        now = time.time()
        if self._last_ts:
            to_sleep = self.frame_interval - (now - self._last_ts)
            if to_sleep > 0:
                await asyncio.sleep(to_sleep)
        self._last_ts = time.time()

        frame = self._buf.get(timeout=0.05)
        if frame is None:
            raise MediaStreamError("No frame available")

        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts = int(self._last_ts * 90000)
        vf.time_base = self.time_base
        return vf


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

WEBRTC_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Mahat Live Video &amp; Control</title>
    <style>
      :root {
        /* Aspect is set dynamically by JS after metadata is available */
        --aspect: 4/3; /* safe default */
      }
      * { box-sizing: border-box; }
      body { font-family: sans-serif; background:#f0f0f0; margin:0; padding:24px; }
      h1 { margin:0 0 16px 0; text-align:center; }

      /* Two columns: video (flex) + control rail (fixed width) */
      .layout {
        display:grid;
        grid-template-columns: minmax(0, 1fr) 280px;
        gap:24px;
        align-items:start;
        max-width: 98vw;
        margin: 0 auto;
      }

      /* The video wrapper grows to fill available viewport space */
      .video-panel {
        display:flex;
        flex-direction:column;
        align-items:center;
        gap:10px;
        min-width: 0; /* allow shrinking on small screens */
      }

      /* This is the key: wrapper scales to viewport with a fixed aspect ratio */
      #wrap {
        width: clamp(480px, calc(100vw - 360px), 1400px);
        /* Fit vertically: leave room for header + controls */
        height: min( calc(100vh - 190px), 85vh );
        /* Keep correct aspect; browser computes height from width if height:auto */
        aspect-ratio: var(--aspect);

        background:#000;
        border:4px solid #333;
        border-radius: 6px;
        display:block;
        position:relative;
        overflow:hidden;
      }

      /* Make the <video> fill the wrapper while preserving content box */
      #video {
        width:100%;
        height:100%;
        object-fit: contain;
        cursor: crosshair;
        display:block;
        background:#000;
      }

      #status { color:#444; font-size:14px; min-height:1.2em; text-align:center; }
      .hint { color:#666; font-size:12px; text-align:center; }

      .rail { display:flex; flex-direction:column; align-items:stretch; gap:12px; }
      .btn { padding:12px 16px; border:0; border-radius:10px; font-size:16px; cursor:pointer; color:white; box-shadow:0 2px 6px rgba(0,0,0,.15); }
      .btn.secondary { background:#607D8B; }
      .btn.go { background:#4CAF50; }
      .btn.quit { background:#f44336; }
      .btn.toggle { background:#2196F3; }
      .btn.full { background:#795548; }

      .dpad { margin-top:4px; display:grid; grid-template-columns:64px 64px 64px; grid-template-rows:64px 64px 64px; gap:10px; justify-content:center; }
      .dpad button { width:64px; height:64px; background:#9E9E9E; border:0; border-radius:12px; color:#fff; font-size:20px; cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,.15); }
      .blank { visibility:hidden; }

      @media (max-width: 980px) {
        .layout { grid-template-columns: 1fr; }
        #wrap { width: min(96vw, 1400px); height: min( calc(100vh - 230px), 86vh ); }
      }
    </style>
  </head>
  <body>
    <h1>Mahat Live Video &amp; Control</h1>
    <div class="layout">
      <div class="video-panel">
        <div id="wrap"><video id="video" autoplay playsinline></video></div>
        <div id="status"></div>
        <div class="hint">
          Click the video to select a target. Use arrow keys to nudge by 5px
          (<kbd>Shift</kbd>=10px, <kbd>Alt</kbd>=1px). <kbd>M</kbd> toggles Fixed/Moving. R/S/Q for Reset/Stop/Quit.
        </div>
      </div>

      <div class="rail">
        <button id="startBtn" class="btn secondary">Start</button>
        <button class="btn go"   onclick="sendCmd('r')">Reset (R)</button>
        <button class="btn go"   onclick="sendCmd('s')">Stop (S)</button>
        <button class="btn quit" onclick="sendCmd('q')">Quit (Q)</button>
        <button id="tgtBtn"  class="btn toggle" onclick="toggleTarget()">Target: Fixed (M)</button>
        <button id="fsBtn"   class="btn full"   onclick="toggleFullscreen()">Fullscreen</button>

        <div class="dpad">
          <span class="blank"></span><button onclick="nudge(0,-5)">&#9650;</button><span class="blank"></span>
          <button onclick="nudge(-5,0)">&#9664;</button><span class="blank"></span><button onclick="nudge(5,0)">&#9654;</button>
          <span class="blank"></span><button onclick="nudge(0,5)">&#9660;</button><span class="blank"></span>
        </div>
      </div>
    </div>

    <script>
      const video   = document.getElementById('video');
      const wrap    = document.getElementById('wrap');
      const startBtn= document.getElementById('startBtn');
      const statusEl= document.getElementById('status');
      const tgtBtn  = document.getElementById('tgtBtn');

      function setStatus(m){ statusEl.textContent = m; }

      // Set CSS aspect-ratio variable from real stream once metadata is known
      function updateAspect(){
        const vw = video.videoWidth, vh = video.videoHeight;
        if (vw && vh) document.documentElement.style.setProperty('--aspect', (vw/vh));
      }
      video.addEventListener('loadedmetadata', updateAspect);
      video.addEventListener('resize', updateAspect);

      async function sendCmd(cmd){
        try{
          const r = await fetch('http://' + location.hostname + ':5000/command', {
            method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
            body:'cmd='+encodeURIComponent(cmd)
          });
          setStatus(r.ok ? 'Sent ' + cmd : 'Cmd failed ' + cmd);
        }catch(e){ setStatus('Cmd error: ' + e); }
      }
      async function nudge(dx,dy){
        try{
          const r = await fetch('http://' + location.hostname + ':5000/nudge', {
            method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
            body:`dx=${dx}&dy=${dy}`
          });
          setStatus(r.ok ? `Nudged (${dx},${dy})` : `Nudge failed`);
        }catch(e){ setStatus('Nudge error: ' + e); }
      }

      // Letterbox-aware click mapping
      video.addEventListener('click', async (e)=>{
        const rect  = video.getBoundingClientRect();
        const style = getComputedStyle(video);
        const bl = parseFloat(style.borderLeftWidth)||0, bt = parseFloat(style.borderTopWidth)||0;
        const br = parseFloat(style.borderRightWidth)||0, bb = parseFloat(style.borderBottomWidth)||0;
        const boxW = rect.width - bl - br, boxH = rect.height - bt - bb;
        const xInBox = Math.max(0, Math.min(boxW, e.clientX - rect.left - bl));
        const yInBox = Math.max(0, Math.min(boxH, e.clientY - rect.top  - bt));

        const vw = video.videoWidth, vh = video.videoHeight;
        if (!vw || !vh){ setStatus('No video metadata yet'); return; }

        const scale = Math.min(boxW / vw, boxH / vh);
        const drawnW = vw * scale, drawnH = vh * scale;
        const offX = (boxW - drawnW) / 2.0, offY = (boxH - drawnH) / 2.0;

        if (xInBox < offX || xInBox > offX+drawnW || yInBox < offY || yInBox > offY+drawnH){
          setStatus('Click inside the video area'); return;
        }
        const x = Math.round((xInBox - offX) / scale);
        const y = Math.round((yInBox - offY) / scale);

        try{
          const r = await fetch('http://' + location.hostname + ':5000/select_point', {
            method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
            body:`x=${x}&y=${y}`
          });
          setStatus(r.ok ? `Selected (${x}, ${y})` : `Select failed`);
        }catch(err){ setStatus('Select error: ' + err); }
      });

      let movingTgt = false;
      async function toggleTarget(){
        movingTgt = !movingTgt;
        tgtBtn.textContent = 'Target: ' + (movingTgt ? 'Moving' : 'Fixed') + ' (M)';
        try{
          const r = await fetch('http://' + location.hostname + ':5000/set_target_mode', {
            method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
            body:'bMoovingTgt=' + (movingTgt ? 1 : 0)
          });
          setStatus(r.ok ? ('Mode: ' + (movingTgt ? 'MOVING' : 'FIXED')) : 'Mode set failed');
        }catch(e){ setStatus('Mode error: ' + e); }
      }

      // Keyboard shortcuts
      window.addEventListener('keydown', (e)=>{
        const step = e.shiftKey ? 10 : (e.altKey ? 1 : 5);
        if (e.key === 'ArrowRight'){ nudge(step,0); e.preventDefault(); }
        if (e.key === 'ArrowLeft'){  nudge(-step,0); e.preventDefault(); }
        if (e.key === 'ArrowUp'){    nudge(0,-step); e.preventDefault(); }
        if (e.key === 'ArrowDown'){  nudge(0, step); e.preventDefault(); }
        if (e.key === 'm' || e.key === 'M') toggleTarget();
        if (e.key === 'r' || e.key === 'R') sendCmd('r');
        if (e.key === 's' || e.key === 'S') sendCmd('s');
        if (e.key === 'q' || e.key === 'Q') sendCmd('q');
      });

      // WebRTC handshake
      async function start(){
        try{
          const pc = new RTCPeerConnection();
          pc.ontrack = (ev)=>{ video.srcObject = ev.streams[0]; setStatus('Streaming\u2026'); };
          const offer = await pc.createOffer({ offerToReceiveVideo: true });
          await pc.setLocalDescription(offer);
          const resp   = await fetch('/offer', { method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
          });
          const answer = await resp.json();
          await pc.setRemoteDescription(answer);
          setStatus('Connected via WebRTC');
        }catch(e){ setStatus('WebRTC error: ' + e); }
      }
      startBtn.addEventListener('click', start);

      // Fullscreen toggle
      function toggleFullscreen(){
        if (!document.fullscreenElement) wrap.requestFullscreen?.();
        else document.exitFullscreen?.();
      }
      window.toggleFullscreen = toggleFullscreen;
    </script>
  </body>
</html>
"""


# ---------------------------------------------------------------------------
# aiohttp server
# ---------------------------------------------------------------------------

_pcs = set()


async def _index(request):
    return web.Response(text=WEBRTC_HTML, content_type="text/html")


def _make_offer_handler(frame_buffer: FrameBuffer):
    async def offer(request):
        params = await request.json()
        pc = RTCPeerConnection()
        _pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_state_change():
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                _pcs.discard(pc)

        pc.addTrack(GlobalFrameTrack(frame_buffer, target_fps=30))
        await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    return offer


async def _on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in _pcs])


def start(frame_buffer: FrameBuffer, port=8080):
    """
    Run the aiohttp/WebRTC server in the calling thread's event loop.
    Call from a daemon Thread so it doesn't block the main loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = web.Application()
    app.on_shutdown.append(_on_shutdown)
    app.router.add_get("/", _index)
    app.router.add_post("/offer", _make_offer_handler(frame_buffer))
    web.run_app(app, host="0.0.0.0", port=port, handle_signals=False)
