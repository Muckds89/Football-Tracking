import cv2
import os
import json
import logging
import numpy as np




class VideoUtils:
    @staticmethod
    def read_video_stream(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

        cap.release()

    @staticmethod
    def get_video_metadata(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()
        return metadata

    @staticmethod
    def get_last_frame(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise ValueError(f"No frames in video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1000)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read last frame from: {video_path}")

        return frame

    @staticmethod
    def save_frame(frame, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)
        logging.info(f"Saved frame to {output_path}")


    @staticmethod
    def get_center_of_bbox(bbox):
        x1, y1, x2, y2 = bbox
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)
        return x_center, y_center


    @staticmethod

    def get_center_of_bbox(bbox):
        x1, y1, x2, y2 = bbox
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)
        return x_center, y_center

    @staticmethod
    def get_foot_position(bbox):
        x1, y1, x2, y2 = bbox
        x_center = int((x1 + x2) / 2)
        y_bottom = int(y2)
        return x_center, y_bottom
    
    @staticmethod
    def measure_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    @staticmethod
    def measure_xy_distance(p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        dx = x2 - x1
        dy = y2 - y1

        return dx, dy

    @staticmethod
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


    @staticmethod
    def xyxy_to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


    @staticmethod
    def get_box_center_xyxy(bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @staticmethod
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



    @staticmethod
    def write_ball_debug_video(video_path, ball_tracks, rois, output_path, fps=25):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None:
            fps = video_fps if video_fps and video_fps > 0 else 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        roi_polys = {
            name: np.array(points, dtype=np.int32)
            for name, points in rois.items()
        }

        def draw_frame(out, i, track):
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

        try:
            # write first frame
            out = first_frame.copy()
            first_track = ball_tracks[0] if len(ball_tracks) > 0 else None
            draw_frame(out, 0, first_track)
            writer.write(out)

            # write remaining frames
            for i, frame in enumerate(frames_iter, start=1):
                out = frame.copy()
                track = ball_tracks[i] if i < len(ball_tracks) else None
                draw_frame(out, i, track)
                writer.write(out)
        finally:
            writer.release()

        logging.info(f"Saved debug video: {output_path}")

    @staticmethod
    def write_player_team_debug_video(
        video_path,
        tracks,
        output_path,
        rois=None,
        ball_tracks=None,
        fps=None
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None:
            fps = video_fps if video_fps and video_fps > 0 else 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        roi_polys = {}
        if rois is not None:
            roi_polys = {
                name: np.array(points, dtype=np.int32)
                for name, points in rois.items()
            }

        # default colors for teams
        TEAM_COLORS = {
            "vest_team": (0, 255, 255),   # yellow-ish in BGR
            "other_team": (255, 0, 0),    # blue
            "unknown": (180, 180, 180)    # gray
        }

        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

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

                # draw players
                if frame_idx < len(tracks["players"]):
                    frame_players = tracks["players"][frame_idx]

                    for player_id, info in frame_players.items():
                        bbox = info.get("bbox")
                        if bbox is None:
                            continue

                        x1, y1, x2, y2 = map(int, bbox)

                        team = info.get("team", "unknown")
                        vest_ratio = info.get("vest_ratio", None)
                        has_ball = info.get("has_ball", False)

                        color = TEAM_COLORS.get(team, TEAM_COLORS["unknown"])

                        # player box
                        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

                        # label lines
                        label_1 = f"id:{player_id} team:{team}"

                        if vest_ratio is not None:
                            try:
                                label_2 = f"vest:{float(vest_ratio):.2f}"
                            except (ValueError, TypeError):
                                label_2 = f"vest:{vest_ratio}"
                        else:
                            label_2 = None

                        # background for readability
                        cv2.rectangle(
                            out,
                            (x1, max(0, y1 - 40)),
                            (x1 + 180, y1),
                            (0, 0, 0),
                            -1
                        )

                        cv2.putText(
                            out,
                            label_1,
                            (x1, max(15, y1 - 22)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )

                        if label_2 is not None:
                            cv2.putText(
                                out,
                                label_2,
                                (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )

                        # mark possession
                        if has_ball:
                            cv2.putText(
                                out,
                                "HAS BALL",
                                (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2
                            )
                            cv2.circle(out, (x1 - 8, y1 - 8), 6, (0, 255, 255), -1)

                # draw ball
                if ball_tracks is not None and frame_idx < len(ball_tracks):
                    track = ball_tracks[frame_idx]

                    if track is not None:
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
                                f"ball conf: {conf:.2f}",
                                (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2
                            )

                # frame number
                cv2.putText(
                    out,
                    f"frame: {frame_idx}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                writer.write(out)
                frame_idx += 1

        finally:
            cap.release()
            writer.release()

        logging.info(f"Saved debug video: {output_path}")


