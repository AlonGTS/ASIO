import cv2
import time

# === Setup ===
cap = cv2.VideoCapture(0)
active_tracker = "CSRT"
tracker = cv2.TrackerCSRT_create()
tracking = False
bbox = None
frame_for_init = None

# ROI drawing state
drawing = False
ix, iy = -1, -1

# Global frame access
current_frame = None  # this will be updated every frame

# === Mouse Callback ===
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bbox, tracking, tracker, frame_for_init, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        frame_for_init = current_frame.copy()

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        bbox = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if bbox and bbox[2] > 10 and bbox[3] > 10:
            if active_tracker == "CSRT":
                tracker = cv2.TrackerCSRT_create()
            elif active_tracker == "KCF":
                tracker = cv2.TrackerKCF_create()
            elif active_tracker == "MIL":
                tracker = cv2.TrackerMIL_create()
            tracker.init(frame_for_init, bbox)
            tracking = True
            print(f"[INFO] Tracking started with bbox: {bbox} using {active_tracker} tracker.")
        else:
            print("[WARNING] Invalid ROI. Draw a bigger box.")

# === Attach Mouse Callback ===
cv2.namedWindow("Tracker")
cv2.setMouseCallback("Tracker", draw_rectangle)

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = frame.copy()  # Store for callback use

    # Info overlays
    cv2.putText(frame, f"Active Tracker: {active_tracker}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    cv2.putText(frame, "Draw ROI | Press 'r' to reset | 'q' to quit",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    # Draw ROI live during selection
    if drawing and bbox is not None:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Track if active
    if tracking:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Tracking lost", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('1'):
        if tracking:
            tracking = False
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame_for_init, bbox)
            print("[INFO] Switched to CSRT tracker.")
            tracking = True
        active_tracker = "CSRT"
    elif key == ord('2'):   
        if tracking:
            tracking = False
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame_for_init, bbox)
            print("[INFO] Switched to KCF tracker.")
            tracking = True
        active_tracker = "KCF"
        if tracking:
            tracking = False
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame_for_init, bbox)
            print("[INFO] Switched to KCF tracker.")
            tracking = True
        active_tracker = "KCF"  
    elif key == ord('3'):
        if tracking:
            tracking = False
            tracker = cv2.TrackerMIL_create()
            tracker.init(frame_for_init, bbox)
            print("[INFO] Switched to MIL tracker.")
            tracking = True
        active_tracker = "MIL"

            
    elif key == ord('r'):
        tracking = False
        bbox = None
        tracker = cv2.TrackerCSRT_create()
        print("[INFO] Tracker reset.")

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()