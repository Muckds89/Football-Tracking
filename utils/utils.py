import cv2
import sys
import numpy as np
import logging
import os

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

    def _center_of_xyxy(bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @staticmethod
    def _iou_xywh_vs_xyxy(ball_bbox_xywh, obj_bbox_xyxy):
        bx, by, bw, bh = ball_bbox_xywh
        bx2 = bx + bw
        by2 = by + bh

        ox1, oy1, ox2, oy2 = obj_bbox_xyxy

        ix1 = max(bx, ox1)
        iy1 = max(by, oy1)
        ix2 = min(bx2, ox2)
        iy2 = min(by2, oy2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        ball_area = max(1, bw * bh)
        obj_area = max(1, (ox2 - ox1) * (oy2 - oy1))
        union = ball_area + obj_area - inter

        return inter / union if union > 0 else 0.0

    @staticmethod
    def _point_inside_xyxy(point, bbox_xyxy, margin=0):
        x, y = point
        x1, y1, x2, y2 = bbox_xyxy
        return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)



    def write_ball_debug_video(frames, ball_tracks, rois, output_path, fps=25):
        if not frames:
            raise ValueError("No frames provided.")

        h, w = frames[0].shape[:2]

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        roi_polys = {
            name: np.array(points, dtype=np.int32)
            for name, points in rois.items()
        }

        for i, frame in enumerate(frames):
            out = frame.copy()

            # draw ROIs
            for roi_name, poly in roi_polys.items():
                cv2.polylines(out, [poly], True, (0, 255, 0), 2)
                x, y = poly[0]
                cv2.putText(
                    out,
                    roi_name,
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            track = ball_tracks[i] if i < len(ball_tracks) else None

            if track is None:
                cv2.putText(
                    out,
                    "ball: NOT DETECTED",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            else:
                bbox = track.get("bbox")
                center = track.get("center")
                conf = track.get("conf", None)
                is_interp = track.get("interpolated", False)

                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)

                if center is None and bbox is not None:
                    x1, y1, x2, y2 = bbox
                    center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]

                if center is not None:
                    cx, cy = map(int, center)
                    point_color = (0, 165, 255) if is_interp else (0, 0, 255)
                    cv2.circle(out, (cx, cy), 5, point_color, -1)

                    label = "ball(interp)" if is_interp else "ball"
                    cv2.putText(
                        out,
                        f"{label}: ({cx},{cy})",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        point_color,
                        2
                    )

                if conf is not None:
                    cv2.putText(
                        out,
                        f"conf: {conf:.2f}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

            cv2.putText(
                out,
                f"frame: {i}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            writer.write(out)

        writer.release()
        logging.info(f"Saved debug video: {output_path}")

    def build_highlight_windows(events, total_frames, fps, seconds_before=10, seconds_after=10):
        """
        Convert event triggers into non-overlapping highlight windows.
        Each event is expected to have either:
        - "frame"
        - or "start_frame" / "end_frame"

        Returns a list of dicts:
        [
            {
            "event": ...,
            "trigger_frame": ...,
            "start_frame": ...,
            "end_frame": ...
            }
        ]
        """
        if not events:
            return []

        before_frames = int(seconds_before * fps)
        after_frames = int(seconds_after * fps)

        # normalize trigger frame
        normalized = []
        for e in events:
            if "frame" in e:
                trigger_frame = e["frame"]
            elif "start_frame" in e and "end_frame" in e:
                trigger_frame = (e["start_frame"] + e["end_frame"]) // 2
            elif "start_frame" in e:
                trigger_frame = e["start_frame"]
            else:
                continue

            start_frame = max(0, trigger_frame - before_frames)
            end_frame = min(total_frames - 1, trigger_frame + after_frames)

            normalized.append({
                "event": e.get("event", "event"),
                "trigger_frame": trigger_frame,
                "start_frame": start_frame,
                "end_frame": end_frame,
            })

        normalized.sort(key=lambda x: x["start_frame"])

        # skip overlapping windows
        selected = []
        last_end = -1

        for win in normalized:
            if win["start_frame"] <= last_end:
                continue
            selected.append(win)
            last_end = win["end_frame"]

        return selected
    

    def save_highlights_to_video(frames, highlight_windows, output_path, fps):
        """
        Create a highlights video if it doesn't exist.
        Append new clips if it already exists by rewriting a combined file safely.
        """
        if not frames:
            raise ValueError("No frames provided.")

        if not highlight_windows:
            logging.info("No highlight windows to save.")
            return

        height, width = frames[0].shape[:2]
        temp_new_clips = output_path.replace(".mp4", "_newclips.mp4")
        temp_combined = output_path.replace(".mp4", "_combined.mp4")

        # write the newly selected clips first
        writer = cv2.VideoWriter(
            temp_new_clips,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        for idx, win in enumerate(highlight_windows):
            logging.info(
                f"Writing highlight {idx+1}: "
                f"{win['event']} | frames {win['start_frame']} - {win['end_frame']}"
            )

            for frame_idx in range(win["start_frame"], win["end_frame"] + 1):
                frame = frames[frame_idx].copy()

                cv2.putText(
                    frame,
                    f"{win['event']}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"frame {frame_idx}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                writer.write(frame)

        writer.release()

        # if final file does not exist, just move temp file into place
        if not os.path.exists(output_path):
            os.replace(temp_new_clips, output_path)
            logging.info(f"Created highlight video: {output_path}")
            return

        # otherwise append: old video + new clips
        cap_old = cv2.VideoCapture(output_path)
        writer_combined = cv2.VideoWriter(
            temp_combined,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        # copy old content
        while True:
            ret, frame = cap_old.read()
            if not ret:
                break
            writer_combined.write(frame)
        cap_old.release()

        # copy new clips content
        cap_new = cv2.VideoCapture(temp_new_clips)
        while True:
            ret, frame = cap_new.read()
            if not ret:
                break
            writer_combined.write(frame)
        cap_new.release()

        writer_combined.release()

        os.remove(temp_new_clips)
        os.replace(temp_combined, output_path)

        logging.info(f"Appended highlights to existing video: {output_path}")