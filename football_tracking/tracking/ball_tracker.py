import logging
import os

import cv2
from ultralytics import YOLO
from football_tracking.utils.video_utils import VideoUtils

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "pipeline.log"), encoding="utf-8")
    ]
)


class BallTracker:
    def __init__(self, model_path, conf=0.25, ball_class_id=0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.ball_class_id = ball_class_id

    def get_ball_tracks(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        frame_skip = 2  # Process every 2nd frame for speed; can be adjusted based on performance needs
        ball_tracks = []
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame in VideoUtils.read_video_stream(video_path):
            if frame_idx % frame_skip == 0:    

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
                        if frame_idx % 1000 == 0:
                            logging.info(f"Tracking progress: {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
            else:
                frame_track = None
            ball_tracks.append(frame_track)
            frame_idx += 1

        # cap.release()
        return ball_tracks
    
