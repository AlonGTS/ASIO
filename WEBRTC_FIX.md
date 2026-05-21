# WebRTC Video Freeze Fix

## Problem
After a 3-second video loss, the stream never recovers.

## Root Cause
There are two bugs working together:

### Bug 1 — `requestVideoFrameCallback` not re-registered after reconnect
`startFrameWatchdogOnce()` registers the frame-callback chain that keeps
`lastFrameTime` updated. But it only runs **once** (guarded by `watchdogStarted`).

After a reconnect, `video.srcObject` is replaced with a new stream, which breaks
the old callback chain. `lastFrameTime` stops updating. The watchdog fires 3 s later,
triggers another reconnect, and the cycle repeats — **infinite reconnect loop**.

### Bug 2 — Reconnect delay is 1000 ms
Adds unnecessary latency before each retry. For drone control, 300 ms is better.

---

## Fix — `webrtc_server.py`

### 1. Add `registerFrameCallback()` function
Insert this **before** `reconnectWebRTC()`:

```javascript
// Register requestVideoFrameCallback on the live video element.
// Must be called every time a new track arrives — not just once on startup —
// because the callback chain is tied to the current srcObject and breaks
// when srcObject is replaced during reconnect.
function registerFrameCallback(){
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    const frameLoop = () => {
      lastFrameTime = Date.now();
      video.requestVideoFrameCallback(frameLoop);
    };
    video.requestVideoFrameCallback(frameLoop);
  }
}
```

### 2. Call `registerFrameCallback()` inside `ontrack`
In `pc.ontrack`, after `setStatus('Streaming…')`:

```javascript
pc.ontrack = (ev)=>{
  video.srcObject = ev.streams[0] ?? new MediaStream([ev.track]);
  video.play().catch(()=>{});
  lastFrameTime = Date.now();
  setStatus('Streaming…');
  // THE KEY FIX: re-register on every new stream so lastFrameTime stays live.
  registerFrameCallback();
};
```

### 3. Change `startFrameWatchdogOnce()` — remove the callback registration from it
Replace the entire `if/else` block that registers `requestVideoFrameCallback`:

**Old:**
```javascript
if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
  const frameLoop = () => {
    lastFrameTime = Date.now();
    video.requestVideoFrameCallback(frameLoop);
  };
  video.requestVideoFrameCallback(frameLoop);
} else {
  // Fallback for older browsers: timeupdate is less accurate but still useful
  video.addEventListener('timeupdate', () => {
    lastFrameTime = Date.now();
  });
}
```

**New:**
```javascript
// Fallback for browsers without requestVideoFrameCallback (older Safari etc.)
// registerFrameCallback() handles the main path; this only runs as a fallback.
if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype)) {
  video.addEventListener('timeupdate', () => { lastFrameTime = Date.now(); });
}
```

### 4. Reduce reconnect delay from 1000 ms → 300 ms
In `reconnectWebRTC()`, change:
```javascript
}, 1000);
```
to:
```javascript
}, 300);
```

### 5. Reduce ICE gathering timeout from 2000 ms → 1000 ms
```javascript
setTimeout(resolve, 1000);   // was 2000
```

---

## Timeline after fix
- t=0 s  — video freezes, watchdog detects after 3 s  
- t=3 s  — `reconnectWebRTC` called, PC closed  
- t=3.3 s — new PC created, offer sent  
- t=4–5 s — ICE complete, answer received, `ontrack` fires  
- t=4–5 s — `registerFrameCallback()` called → `lastFrameTime` resumes ticking  
- **stream stable, watchdog satisfied**

Total recovery: ~1–2 s from the freeze detection firing.

---

## Why not replace WebRTC?
- You're on WiFi → bandwidth matters → H.264 (WebRTC) vs MJPEG (5-10× more data)
- UDP frame loss is fine for video; the problem was the broken reconnect, not the protocol
- Any alternative (MSE, HLS) either adds latency or requires a full server-side encoder
