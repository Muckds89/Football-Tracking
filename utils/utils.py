import cv2
import sys
import numpy as np

sys.path.append('../')

class Utils:


    def read_video(video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    
    def save_video(frames, output_path, fps=25):
        height, width, _ = frames[0].shape

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        for frame in frames:
            out.write(frame)

        out.release()

    def get_center_of_bbox(bbox):
        x1, y1, x2, y2 = bbox
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)
        return x_center, y_center

    def get_center_of_bbox(bbox):
        x1, y1, x2, y2 = bbox
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)
        return x_center, y_center
    
    def get_foot_position(bbox):
        x1, y1, x2, y2 = bbox
        x_center = int((x1 + x2) / 2)
        y_bottom = int(y2)
        return x_center, y_bottom
    

    def measure_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def measure_xy_distance(p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        dx = x2 - x1
        dy = y2 - y1

        return dx, dy

    def draw_rectangle(frame, bbox, color=(0, 255, 0), label=None):
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        if label:
            cv2.putText(
                frame,
                label,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )


    def xyxy_to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


    def get_box_center_xyxy(bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
