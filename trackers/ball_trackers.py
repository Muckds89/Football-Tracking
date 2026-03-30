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
    
class Interpolator:
    def __init__(self):
        pass
    def interpolate_ball_tracks(ball_tracks, max_gap=5):
        filled = ball_tracks[:]
        n = len(filled)

        def center_of(track):
            if track is None:
                return None
            if track.get("center") is not None:
                return track["center"]
            if track.get("bbox") is not None:
                x1, y1, x2, y2 = track["bbox"]
                return [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            return None

        i = 0
        while i < n:
            if filled[i] is not None:
                i += 1
                continue

            start_gap = i - 1
            j = i
            while j < n and filled[j] is None:
                j += 1
            end_gap = j

            gap_len = end_gap - i

            if (
                start_gap >= 0 and
                end_gap < n and
                gap_len <= max_gap and
                filled[start_gap] is not None and
                filled[end_gap] is not None
            ):
                c1 = center_of(filled[start_gap])
                c2 = center_of(filled[end_gap])

                if c1 is not None and c2 is not None:
                    for k in range(1, gap_len + 1):
                        alpha = k / (gap_len + 1)
                        cx = int(c1[0] * (1 - alpha) + c2[0] * alpha)
                        cy = int(c1[1] * (1 - alpha) + c2[1] * alpha)

                        filled[i + k - 1] = {
                            "frame": i + k - 1,
                            "center": [cx, cy],
                            "bbox": None,
                            "conf": 0.0,
                            "interpolated": True
                        }

            i = end_gap

        return filled