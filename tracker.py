import cv2

#tracker for raspberry pi 5
# Initialize the video capture
active_tracker = "CSRT"  # Default tracker
tracker_options = {
    "1": ("CSRT", cv2.TrackerCSRT_create),
    "2": ("KCF", cv2.TrackerKCF_create),
    "3": ("MIL", cv2.TrackerMIL_create)
}
cap = cv2.VideoCapture(0)

# Initialize variables for tracking
tracking = False
tracker = cv2.TrackerCSRT_create()  # You can use other trackers like TrackerKCF, TrackerMIL, etc.
bbox = None

import time  # Import time module for tracking calculation time

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Display the active tracker type on the screen
        tracker_text = f"Active Tracker: {active_tracker}"
        cv2.putText(frame, tracker_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if not ret:
        break

    # Display menu on the screen
    menu_text = "Press 's' to select ROI | Press 'q' to quit"
    text_size = cv2.getTextSize(menu_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]  # Increase font scale and thickness
    text_x, text_y = frame.shape[1] - text_size[0] - 10, 30  # Position at the top right
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    cv2.putText(frame, menu_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
   
    if tracking:
        start_time = time.time()  # Start timing the tracker update
        success, bbox = tracker.update(frame)
        end_time = time.time()  # End timing the tracker update
        tracker_time = (end_time - start_time) * 1000  # Convert to milliseconds

        if success:
            # Draw the bounding box
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Tracking failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker calculation time
        time_text = f"Tracker Time: {tracker_time:.2f} ms"
        cv2.putText(frame, time_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Tracker', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit on 'q' key press
        break
    elif key == ord('s'):  # Select ROI on 's' key press
        # Display instructions on the screen
        cv2.putText(frame, "Select ROI and press ENTER or SPACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Tracker', frame)
        cv2.waitKey(1)  # Refresh the frame to show the instructions

        # Allow user to select ROI
        # Ask the user for the type of tracker to use
        print("Select tracker type:")
        print("1. CSRT")
        print("2. KCF")
        print("3. MIL")

        choice = input("Enter the number corresponding to your choice: ")

        # Map user choice to tracker type
        tracker_mapping = {
            "1": cv2.TrackerCSRT_create,
            "2": cv2.TrackerKCF_create,
            "3": cv2.TrackerMIL_create
        }

        if choice in tracker_mapping:
            tracker = tracker_mapping[choice]()
        else:
            print("Invalid choice. Defaulting to CSRT.")
            # Map user choice to tracker type
            tracker_mapping = {
                "1": cv2.TrackerCSRT_create,
                "2": cv2.TrackerKCF_create,
                "3": cv2.TrackerMIL_create

            }

            if choice in tracker_mapping:
                tracker = tracker_mapping[choice]()
            else:
                print("Invalid choice. Defaulting to CSRT.")
                tracker = cv2.TrackerCSRT_create()
        if choice == '1':
            tracker = cv2.TrackerCSRT_create()
        elif choice == '2':
            tracker = cv2.TrackerKCF_create()
        elif choice == '3':
            tracker = cv2.TrackerMIL_create()  
        else:
            print("Invalid choice. Defaulting to CSRT.")
            tracker = cv2.TrackerCSRT_create()

        # Allow user to select ROI
        bbox = cv2.selectROI('Tracker', frame, fromCenter=False, showCrosshair=True)
        tracker = cv2.TrackerCSRT_create()  # Reinitialize the tracker
        tracker.init(frame, bbox)
        tracking = True
    elif key == ord('1'):  # Change to CSRT tracker
        active_tracker = "CSRT"
        tracker = cv2.TrackerCSRT_create()
        if bbox is not None:
            tracker.init(frame, bbox)
    elif key == ord('2'):  # Change to KCF tracker 
        active_tracker = "KCF"
        tracker = cv2.TrackerKCF_create()
        if bbox is not None:
            tracker.init(frame, bbox)   
    elif key == ord('3'):  # Change to MIL tracker
        active_tracker = "MIL"
        tracker = cv2.TrackerMIL_create()
        if bbox is not None:
            tracker.init(frame, bbox)
    elif key == ord('r'):  # Reset the tracker  
        tracking = False
        bbox = None
        tracker = cv2.TrackerCSRT_create()
        print("Tracker reset. Press 's' to select ROI again.")
    elif key == ord('t'):  # Toggle tracker
        if active_tracker == "CSRT":
            active_tracker = "KCF"
            tracker = cv2.TrackerKCF_create()
        elif active_tracker == "KCF":
            active_tracker = "MIL"
            tracker = cv2.TrackerMIL_create()
        else:
            active_tracker = "CSRT"
            tracker = cv2.TrackerCSRT_create()
        print(f"Tracker switched to {active_tracker}. Press 's' to select ROI again.")
        tracking = False
        bbox = None
        tracker = cv2.TrackerCSRT_create()
        print("Tracker reset. Press 's' to select ROI again.")
        # Allow user to select ROI
        bbox = cv2.selectROI('Tracker', frame, fromCenter=False, showCrosshair=True)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        tracking = True


# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()