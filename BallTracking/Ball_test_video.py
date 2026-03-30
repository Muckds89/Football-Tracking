import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/ball_detector_v2/weights/best.pt")

cap = cv2.VideoCapture("input_videos/FirstPart.MP4")

# resize window
cv2.namedWindow("Ball Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Ball Detection", 1280, 720)

frame_skip = 3  # process 1 out of 3 frames
frame_count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 1280))  # resize to square for better detection
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    results = model(frame, conf=0.15, imgsz=1280, verbose=False)
    annotated = results[0].plot()

    # play normal speed
    # cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) + 1000/30)

    cv2.imshow("Ball Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()