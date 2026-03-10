"""
Quick smoke-test for gts_tracker.so
Run after building: python3 test_gts_tracker.py
"""
import numpy as np
import cv2
from gts_tracker import GTSTracker

# Synthetic frame: grey 320x240 with a white square target
frame = np.zeros((240, 320, 3), dtype=np.uint8)
frame[100:140, 130:170] = 255   # white 40x40 box

bbox = (130, 100, 40, 40)       # (x, y, w, h)

for mode in ("fixed", "moving"):
    t = GTSTracker(mode=mode)
    assert t.mode == mode

    t.init(frame, bbox)

    # Simulate a few frames (target stays still)
    for i in range(5):
        success, result = t.update(frame)
        assert success, f"[{mode}] update failed on frame {i}"
        assert result is not None

    t.reset()
    success, result = t.update(frame)
    assert not success, "update after reset should return False"

    print(f"[OK] mode={mode}")

print("\nAll tests passed.")
