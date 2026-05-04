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
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCRtpSender
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
        self._gen = 0   # incremented on every put(); consumers track last seen gen

    def put(self, frame):
        with self._cond:
            self._frame = frame
            self._gen += 1
            self._cond.notify_all()

    def get(self, last_gen=-1, timeout=0.05):
        with self._cond:
            if self._gen == last_gen:   # already delivered this frame; wait for next
                self._cond.wait(timeout=timeout)
            if self._frame is None:
                return None, self._gen
            return self._frame.copy(), self._gen


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
        self._last_gen = -1   # generation of the last delivered frame

    async def recv(self) -> VideoFrame:
        now = time.time()
        if self._last_ts:
            to_sleep = self.frame_interval - (now - self._last_ts)
            if to_sleep > 0:
                await asyncio.sleep(to_sleep)
        self._last_ts = time.time()

        frame, gen = self._buf.get(last_gen=self._last_gen, timeout=0.05)
        self._last_gen = gen
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
      * { box-sizing: border-box; }
      body { font-family: sans-serif; background:#f0f0f0; margin:0; padding:24px; }
      h1 { margin:0 0 16px 0; text-align:center; }

      .layout {
        display:grid;
        grid-template-columns: minmax(0, 1fr) 280px;
        gap:24px;
        align-items:start;
        max-width: 98vw;
        margin: 0 auto;
      }

      .video-panel {
        display:flex;
        flex-direction:column;
        align-items:center;
        gap:10px;
        min-width: 0;
      }

      /* Sized precisely by fitWrap() in JS to match stream aspect ratio */
      #wrap {
        background:#000;
        border:4px solid #333;
        border-radius: 6px;
        display:block;
        position:relative;
        overflow:hidden;
        /* Fallback before stream starts */
        width: 640px;
        height: 480px;
      }

      #video {
        width:100%;
        height:100%;
        display:block;
        background:#000;
        cursor: crosshair;
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
      .btn.launch { background:#FF6F00; font-weight:bold; }

      .dpad { margin-top:4px; display:grid; grid-template-columns:64px 64px 64px; grid-template-rows:64px 64px 64px; gap:10px; justify-content:center; }
      .dpad button { width:64px; height:64px; background:#9E9E9E; border:0; border-radius:12px; color:#fff; font-size:20px; cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,.15); }
      .blank { visibility:hidden; }

      @media (max-width: 980px) {
        .layout { grid-template-columns: 1fr; }
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
        <button class="btn launch" onclick="sendLaunch()">Launch (L)</button>
        <button class="btn go"   onclick="sendCmd('r')">Reset (R)</button>
        <button class="btn go"   onclick="sendCmd('s')">Stop (S)</button>
        <button class="btn quit" onclick="sendCmd('q')">Quit (Q)</button>
        <button id="tgtBtn"  class="btn toggle" onclick="toggleTarget()">Target: Fixed (M)</button>
        <button id="recBtn"  class="btn go"     onclick="toggleRecord()">Record (REC)</button>
        <button id="fsBtn"   class="btn full"   onclick="toggleFullscreen()">Fullscreen</button>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:4px;">
          <button class="btn secondary" onclick="cycleMain(-1)">MAIN &minus;</button>
          <button class="btn secondary" onclick="cycleMain(+1)">MAIN +</button>
          <button class="btn secondary" onclick="cycleLores(-1)">TRACK &minus;</button>
          <button class="btn secondary" onclick="cycleLores(+1)">TRACK +</button>
        </div>

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

      // Resize #wrap to exactly fit the stream's aspect ratio within the viewport
      function fitWrap(){
        const vw = video.videoWidth, vh = video.videoHeight;
        if (!vw || !vh) return;
        const isNarrow = window.innerWidth <= 980;
        const maxW = isNarrow
          ? Math.min(window.innerWidth * 0.96, 1400)
          : Math.min(window.innerWidth - 360, 1400);
        const maxH = window.innerHeight - (isNarrow ? 230 : 190);
        const scale = Math.min(maxW / vw, maxH / vh);
        wrap.style.width  = Math.round(vw * scale) + 'px';
        wrap.style.height = Math.round(vh * scale) + 'px';
      }
      video.addEventListener('loadedmetadata', fitWrap);
      video.addEventListener('resize', fitWrap);
      window.addEventListener('resize', fitWrap);

      async function sendCmd(cmd){
        try{
          const r = await fetch('http://' + location.hostname + ':5000/command', {
            method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
            body:'cmd='+encodeURIComponent(cmd)
          });
          setStatus(r.ok ? 'Sent ' + cmd : 'Cmd failed ' + cmd);
        }catch(e){ setStatus('Cmd error: ' + e); }
      }
      let launched = false;
      function setLaunchBtn(state){
        launched = state;
        const btn = document.querySelector('.btn.launch');
        if(launched){
          btn.textContent = 'Launched (L)';
          btn.style.background = '#2196F3';
        } else {
          btn.textContent = 'Launch (L)';
          btn.style.background = '';
        }
      }
      // Sync launch button with server state on page load
      fetch('http://' + location.hostname + ':5000/status')
        .then(r => r.json())
        .then(d => setLaunchBtn(d.launched))
        .catch(() => {});
      async function sendLaunch(){
        try{
          const r = await fetch('http://' + location.hostname + ':5000/launch', {
            method:'POST'
          });
          if(r.ok){
            setLaunchBtn(!launched);
            setStatus(launched ? 'LAUNCHED' : 'Launch reset');
          } else { setStatus('Launch failed'); }
        }catch(e){ setStatus('Launch error: ' + e); }
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

      // Click mapping: wrap is sized to exact aspect ratio so pixels map 1:1 to stream coords
      video.addEventListener('click', async (e)=>{
        const vw = video.videoWidth, vh = video.videoHeight;
        if (!vw || !vh){ setStatus('No video metadata yet'); return; }
        const rect = video.getBoundingClientRect();
        const nx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const ny = Math.max(0, Math.min(1, (e.clientY - rect.top)  / rect.height));
        const x = Math.round(nx * (vw - 1));
        const y = Math.round(ny * (vh - 1));

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

      async function cycleMain(delta){
        try{
          const r = await fetch('http://' + location.hostname + ':5000/cycle_main', {
            method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
            body:'delta='+delta
          });
          setStatus(r.ok ? 'MAIN res ' + (delta>0?'+':'-') : 'MAIN cycle failed');
        }catch(e){ setStatus('MAIN error: ' + e); }
      }
      async function cycleLores(delta){
        try{
          const r = await fetch('http://' + location.hostname + ':5000/cycle_lores', {
            method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'},
            body:'delta='+delta
          });
          setStatus(r.ok ? 'TRACK res ' + (delta>0?'+':'-') : 'TRACK cycle failed');
        }catch(e){ setStatus('TRACK error: ' + e); }
      }

      // Keyboard shortcuts
      window.addEventListener('keydown', (e)=>{
        const step = e.shiftKey ? 10 : (e.altKey ? 1 : 5);
        if (e.key === 'ArrowRight'){ nudge(step,0); e.preventDefault(); }
        if (e.key === 'ArrowLeft'){  nudge(-step,0); e.preventDefault(); }
        if (e.key === 'ArrowUp'){    nudge(0,-step); e.preventDefault(); }
        if (e.key === 'ArrowDown'){  nudge(0, step); e.preventDefault(); }
        if (e.key === 'm' || e.key === 'M') toggleTarget();
        if (e.key === 'l' || e.key === 'L') sendLaunch();
        if (e.key === 'r' || e.key === 'R') sendCmd('r');
        if (e.key === 's' || e.key === 'S') sendCmd('s');
        if (e.key === 'q' || e.key === 'Q') sendCmd('q');
        if (e.key === 'x' || e.key === 'X') cycleMain(+1);
        if (e.key === 'z' || e.key === 'Z') cycleMain(-1);
        if (e.key === 'v' || e.key === 'V') cycleLores(+1);
        if (e.key === 'c' || e.key === 'C') cycleLores(-1);
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

      // Recording via MediaRecorder (runs entirely in the browser, zero Pi overhead)
      let mediaRecorder = null;
      let recChunks = [];
      const recBtn = document.getElementById('recBtn');

      function toggleRecord(){
        if (mediaRecorder && mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        } else {
          const stream = video.srcObject;
          if (!stream){ setStatus('No stream yet — start WebRTC first'); return; }
          recChunks = [];
          const mimeType = MediaRecorder.isTypeSupported('video/mp4; codecs=avc1')
            ? 'video/mp4; codecs=avc1'
            : 'video/webm; codecs=vp8';
          mediaRecorder = new MediaRecorder(stream, { mimeType });
          mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recChunks.push(e.data); };
          mediaRecorder.onstop = () => {
            const ext  = mimeType.startsWith('video/mp4') ? 'mp4' : 'webm';
            const blob = new Blob(recChunks, { type: mimeType });
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement('a');
            const ts   = new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
            a.href = url; a.download = `mahat_${ts}.${ext}`; a.click();
            URL.revokeObjectURL(url);
            recBtn.textContent = 'Record (REC)';
            recBtn.style.background = '';
            setStatus('Recording saved');
          };
          stream.getTracks().forEach(t => {
            t.onended = () => { if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop(); };
          });
          mediaRecorder.start();
          recBtn.textContent = 'Stop Recording';
          recBtn.style.background = '#f44336';
          setStatus('Recording…');
        }
      }

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

        transceiver = pc.addTransceiver(GlobalFrameTrack(frame_buffer, target_fps=30), direction="sendonly")
        caps = RTCRtpSender.getCapabilities("video")
        h264_codecs = [c for c in caps.codecs if c.mimeType == "video/H264"]
        transceiver.setCodecPreferences(h264_codecs)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        # DEBUG: print negotiated video codec (first rtpmap = chosen codec)
        for line in pc.localDescription.sdp.splitlines():
            if line.startswith("a=rtpmap") and any(c in line.upper() for c in ("H264", "VP8", "VP9", "AV1")):
                print(f"[WebRTC codec] {line}")
                break  # first one is the chosen codec
        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    return offer


async def _on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in _pcs])


def start(frame_buffer: FrameBuffer, port=8080, host="0.0.0.0"):
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
    web.run_app(app, host=host, port=port, handle_signals=False)
