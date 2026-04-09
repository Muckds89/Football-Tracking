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

    @staticmethod
    def point_in_polygon(point, polygon):
        if point is None or polygon is None or len(polygon) < 3:
            return False

        polygon_np = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(polygon_np, point, False) >= 0

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
        team_ball_control,
        attacking_team,
        side="right",
        window_size=8,
        min_hits=3,
        min_possession_hits=3,
        exit_gap_frames=12,
        min_event_frames=5,
    ):
        if side == "right":
            penalty_roi = self.right_penalty_box
            goal_roi = self.right_goal
            event_name = f"{attacking_team}_right_penalty_pressure"
        else:
            penalty_roi = self.left_penalty_box
            goal_roi = self.left_goal
            event_name = f"{attacking_team}_left_penalty_pressure"

        events = []
        hit_window = []
        possession_window = []

        event_active = False
        event_start = None
        last_positive_frame = None
        max_hits_seen = 0
        goal_touched = False

        for i, track in enumerate(ball_tracks):
            center = self.get_ball_center(track)

            in_penalty = self.point_in_roi(center, penalty_roi)
            in_goal = self.point_in_roi(center, goal_roi)

            team_in_control = team_ball_control[i] if i < len(team_ball_control) else "unknown"
            attacking_control = (team_in_control == attacking_team)

            spatial_hit = 1 if (in_penalty or in_goal) else 0
            possession_hit = 1 if attacking_control else 0
            combined_hit = 1 if spatial_hit and possession_hit else 0

            hit_window.append(combined_hit)
            possession_window.append(possession_hit)

            if len(hit_window) > window_size:
                hit_window.pop(0)
            if len(possession_window) > window_size:
                possession_window.pop(0)

            hits_now = sum(hit_window)
            possession_hits_now = sum(possession_window)

            # start event
            if not event_active and hits_now >= min_hits and possession_hits_now >= min_possession_hits:
                event_active = True
                event_start = max(0, i - window_size + 1)
                last_positive_frame = i
                max_hits_seen = hits_now
                goal_touched = in_goal
                continue

            # maintain event
            if event_active:
                if combined_hit == 1:
                    last_positive_frame = i

                max_hits_seen = max(max_hits_seen, hits_now)
                goal_touched = goal_touched or in_goal

                if last_positive_frame is not None and (i - last_positive_frame) >= exit_gap_frames:
                    event_end = last_positive_frame
                    duration_frames = event_end - event_start + 1

                    if duration_frames >= min_event_frames:
                        events.append({
                            "event": event_name,
                            "team": attacking_team,
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
                    possession_window = []

        if event_active and last_positive_frame is not None:
            event_end = last_positive_frame
            duration_frames = event_end - event_start + 1

            if duration_frames >= min_event_frames:
                events.append({
                    "event": event_name,
                    "team": attacking_team,
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
    
    def detect_kickoff(self, ball_tracks, min_seconds=2.5, max_move_px=20):
        min_frames = int(min_seconds * self.fps)

        streak_start = None
        prev_center = None
        streak_len = 0

        for i, track in enumerate(ball_tracks):
            if not track:
                streak_start = None
                prev_center = None
                streak_len = 0
                continue

            center = track.get("center")
            if center is None:
                streak_start = None
                prev_center = None
                streak_len = 0
                continue

            if not self.point_in_polygon(center, self.rois["center_pitch"]):
                streak_start = None
                prev_center = None
                streak_len = 0
                continue

            if prev_center is None:
                streak_start = i
                prev_center = center
                streak_len = 1
                continue

            dx = center[0] - prev_center[0]
            dy = center[1] - prev_center[1]
            dist2 = dx * dx + dy * dy

            if dist2 <= max_move_px * max_move_px:
                streak_len += 1
            else:
                streak_start = i
                streak_len = 1

            prev_center = center

            if streak_len >= min_frames:
                return {
                    "frame": streak_start,
                    "time_sec": streak_start / self.fps
                }

        return None

    def build_highlight_windows(self,events, fps, seconds_before=20, seconds_after=10):
        if not events:
            return []

        raw_windows = []
        before_frames = int(seconds_before * fps)
        after_frames = int(seconds_after * fps)

        for ev in events:
            trigger_frame = ev["start_frame"]
            raw_windows.append({
                "event": ev["event"],
                "start_frame": max(0, trigger_frame - before_frames),
                "end_frame": trigger_frame + after_frames,
                "trigger_frame": trigger_frame,
                "side": ev.get("side"),
                "goal_touched": ev.get("goal_touched", False),
            })

        raw_windows.sort(key=lambda x: x["start_frame"])

        merged = [raw_windows[0]]

        for win in raw_windows[1:]:
            last = merged[-1]

            if win["start_frame"] <= last["end_frame"]:
                last["end_frame"] = max(last["end_frame"], win["end_frame"])
                if win.get("goal_touched"):
                    last["goal_touched"] = True
            else:
                merged.append(win)

        return merged
    
    def detect_penalty_pressure_for_config(
        self,
        ball_tracks,
        team_ball_control,
        team_attack_directions,
        window_size=8,
        min_hits=3,
        min_possession_hits=3,
        exit_gap_frames=12,
        min_event_frames=5,
    ):
        events = []

        for team_name, side in team_attack_directions.items():
            events.extend(
                self.detect_penalty_pressure_merged(
                    ball_tracks=ball_tracks,
                    team_ball_control=team_ball_control,
                    attacking_team=team_name,
                    side=side,
                    window_size=window_size,
                    min_hits=min_hits,
                    min_possession_hits=min_possession_hits,
                    exit_gap_frames=exit_gap_frames,
                    min_event_frames=min_event_frames,
                )
            )

        events = sorted(events, key=lambda x: x["start_frame"])
        return events