from ultralytics import YOLO

# model = YOLO("yolov8n.pt")
model = YOLO("yolov8s.pt")
model.train(
    data="dataset/data.yaml",
    imgsz=640,
    epochs=100,
    batch=4,      # or higher if the GPU allows
    cache=True,
    workers=8,
    name="ball_detector_v3"
)