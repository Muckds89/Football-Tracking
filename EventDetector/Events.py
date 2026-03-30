import cv2
import numpy as np


class EventDetector:
    def __init__(self, rois, fps):
        self.rois = rois
        self.fps = fps

        self.center_pitch = np.array(rois["center_pitch"], dtype=np.int32)
        self.left_penalty_box = np.array(rois["left_penalty_box"], dtype=np.int32)
        self.right_penalty_box = np.array(rois["right_penalty_box"], dtype=np.int32)
        self.left_goal = np.array(rois["left_goal"], dtype=np.int32)
        self.right_goal = np.array(rois["right_goal"], dtype=np.int32)

    def point_in_roi(self, point, roi_polygon):
        if point is None:
            return False
        return cv2.pointPolygonTest(
            roi_polygon,
            (float(point[0]), float(point[1])),
            False
        ) >= 0

    def get_ball_center(self, track):
        if track is None:
            return None

        center = track.get("center")
        if center is not None:
            return tuple(center)

        bbox = track.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))

        return None

    def detect_penalty_pressure_merged(
        self,
        ball_tracks,
        side="right",
        window_size=8,
        min_hits=3,
        exit_gap_frames=12,
        min_event_frames=5,
    ):
        if side == "right":
            penalty_roi = self.right_penalty_box
            goal_roi = self.right_goal
            event_name = "right_penalty_pressure"
        else:
            penalty_roi = self.left_penalty_box
            goal_roi = self.left_goal
            event_name = "left_penalty_pressure"

        events = []
        hit_window = []

        event_active = False
        event_start = None
        last_positive_frame = None
        max_hits_seen = 0
        goal_touched = False

        for i, track in enumerate(ball_tracks):
            center = self.get_ball_center(track)

            in_penalty = self.point_in_roi(center, penalty_roi)
            in_goal = self.point_in_roi(center, goal_roi)

            hit = 1 if (in_penalty or in_goal) else 0
            hit_window.append(hit)

            if len(hit_window) > window_size:
                hit_window.pop(0)

            hits_now = sum(hit_window)

            # start event
            if not event_active and hits_now >= min_hits:
                event_active = True
                event_start = max(0, i - window_size + 1)
                last_positive_frame = i
                max_hits_seen = hits_now
                goal_touched = in_goal
                continue

            # maintain event
            if event_active:
                if hit == 1:
                    last_positive_frame = i

                max_hits_seen = max(max_hits_seen, hits_now)
                goal_touched = goal_touched or in_goal

                # close event only after enough absence
                if last_positive_frame is not None and (i - last_positive_frame) >= exit_gap_frames:
                    event_end = last_positive_frame
                    duration_frames = event_end - event_start + 1

                    if duration_frames >= min_event_frames:
                        events.append({
                            "event": event_name,
                            "start_frame": event_start,
                            "end_frame": event_end,
                            "start_time_sec": event_start / self.fps,
                            "end_time_sec": event_end / self.fps,
                            "duration_sec": duration_frames / self.fps,
                            "max_hits_in_window": max_hits_seen,
                            "goal_touched": goal_touched,
                            "side": side,
                        })

                    event_active = False
                    event_start = None
                    last_positive_frame = None
                    max_hits_seen = 0
                    goal_touched = False
                    hit_window = []

        # close event if still active at end of video
        if event_active and last_positive_frame is not None:
            event_end = last_positive_frame
            duration_frames = event_end - event_start + 1

            if duration_frames >= min_event_frames:
                events.append({
                    "event": event_name,
                    "start_frame": event_start,
                    "end_frame": event_end,
                    "start_time_sec": event_start / self.fps,
                    "end_time_sec": event_end / self.fps,
                    "duration_sec": duration_frames / self.fps,
                    "max_hits_in_window": max_hits_seen,
                    "goal_touched": goal_touched,
                    "side": side,
                })

        return events


    def debug_membership(self, ball_tracks, max_frames=None):
        rows = []
        total = len(ball_tracks) if max_frames is None else min(len(ball_tracks), max_frames)

        for i in range(total):
            center = self.get_ball_center(ball_tracks[i])

            row = {
                "frame": i,
                "center": center,
                "in_center": self.point_in_roi(center, self.center_pitch),
                "in_left_box": self.point_in_roi(center, self.left_penalty_box),
                "in_right_box": self.point_in_roi(center, self.right_penalty_box),
                "in_left_goal": self.point_in_roi(center, self.left_goal),
                "in_right_goal": self.point_in_roi(center, self.right_goal),
            }
            rows.append(row)

        return rows


