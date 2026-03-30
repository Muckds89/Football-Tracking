import cv2
from ultralytics import YOLO


class BallTracker:
    def __init__(self, model_path, conf=0.25, ball_class_id=0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.ball_class_id = ball_class_id

    def get_ball_tracks(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        ball_tracks = []
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = self.model.predict(frame, conf=self.conf, verbose=False)
            r = results[0]

            frame_track = None

            if r.boxes is not None and len(r.boxes) > 0:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                boxes_cls = r.boxes.cls.cpu().numpy()
                boxes_conf = r.boxes.conf.cpu().numpy()

                # keep only ball detections
                ball_indices = [i for i, cls_id in enumerate(boxes_cls) if int(cls_id) == self.ball_class_id]

                if ball_indices:
                    # choose highest-confidence ball
                    best_i = max(ball_indices, key=lambda i: boxes_conf[i])
                    x1, y1, x2, y2 = boxes_xyxy[best_i]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    frame_track = {
                        "frame": frame_idx,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "center": [cx, cy],
                        "conf": float(boxes_conf[best_i]),
                        "class_id": int(boxes_cls[best_i]),
                    }

            ball_tracks.append(frame_track)
            frame_idx += 1

        cap.release()
        return ball_tracks
    
