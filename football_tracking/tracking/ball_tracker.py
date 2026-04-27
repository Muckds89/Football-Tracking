import logging
import os

import cv2
from ultralytics import YOLO
from football_tracking.utils.video_utils import VideoUtils
from football_tracking.roi.roi_cropper import ROICropper

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
    def __init__(self, model_path, conf=0.10, ball_class_id=0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.ball_class_id = ball_class_id

    def get_ball_tracks(
        self,
        video_path,
        ball_roi_polygon=None,
        apply_mask=True,
        frame_skip=2,
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        cap.release()

        ball_tracks = []
        frame_idx = 0
        total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

        for frame in VideoUtils.read_video_stream(video_path):
            frame_track = None

            if frame_idx % frame_skip == 0:
                inference_frame = frame
                crop_x = 0
                crop_y = 0

                if ball_roi_polygon is not None:
                    inference_frame, crop_x, crop_y, crop_w, crop_h = ROICropper.crop_frame(
                        frame,
                        ball_roi_polygon,
                        apply_mask=apply_mask,
                    )

                results = self.model.predict(
                    inference_frame,
                    conf=self.conf,
                    iou=0.5,
                    max_det=1,
                    imgsz=1280,
                    verbose=False
                )

                r = results[0]

                if r.boxes is not None and len(r.boxes) > 0:
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                    boxes_cls = r.boxes.cls.cpu().numpy()
                    boxes_conf = r.boxes.conf.cpu().numpy()

                    ball_indices = [
                        i for i, cls_id in enumerate(boxes_cls)
                        if int(cls_id) == self.ball_class_id
                    ]

                    if ball_indices:
                        best_i = max(ball_indices, key=lambda i: boxes_conf[i])

                        x1, y1, x2, y2 = boxes_xyxy[best_i]

                        # Convert crop coordinates back to full-frame coordinates
                        if ball_roi_polygon is not None:
                            x1 += crop_x
                            x2 += crop_x
                            y1 += crop_y
                            y2 += crop_y

                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        frame_track = {
                            "frame": frame_idx,
                            "bbox": [
                                float(x1),
                                float(y1),
                                float(x2),
                                float(y2),
                            ],
                            "center": [cx, cy],
                            "conf": float(boxes_conf[best_i]),
                            "class_id": int(boxes_cls[best_i]),
                        }

                if frame_idx % 1000 == 0:
                    logging.info(
                        f"Tracking progress: {frame_idx}/{total_frames} frames "
                        f"({100 * frame_idx / max(total_frames, 1):.1f}%)"
                    )

            ball_tracks.append(frame_track)
            frame_idx += 1

        return ball_tracks
    
