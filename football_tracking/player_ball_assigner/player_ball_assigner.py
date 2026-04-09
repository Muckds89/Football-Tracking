import sys
from typing import Counter 
sys.path.append('../')
from football_tracking.utils.video_utils import  VideoUtils#
import math
import numpy as np

class PlayerBallAssigner:
    def __init__(
        self,
        max_ball_player_distance: float = 60.0,
        carry_forward_on_missing_ball: bool = True,
    ):
        self.max_ball_player_distance = max_ball_player_distance
        self.carry_forward_on_missing_ball = carry_forward_on_missing_ball
    
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = VideoUtils.get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = VideoUtils.measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = VideoUtils.measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player
    
    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _player_feet_point(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y2)

    @staticmethod
    def _dist(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def infer_team_ball_control(self, tracks, ball_tracks_filled, max_dist=80):
        team_ball_control = []

        for frame_num, frame_players in enumerate(tracks["players"]):
            if frame_num >= len(ball_tracks_filled):
                team_ball_control.append("unknown")
                continue

            ball_info = ball_tracks_filled[frame_num]
            if ball_info is None or ball_info.get("bbox") is None:
                if len(team_ball_control) == 0:
                    team_ball_control.append("unknown")
                else:
                    team_ball_control.append(team_ball_control[-1])
                continue

            ball_center = self._bbox_center(ball_info["bbox"])

            best_player_id = None
            best_dist = float("inf")

            for player_id, info in frame_players.items():
                bbox = info.get("bbox")
                if bbox is None:
                    continue

                player_point = self._player_feet_point(bbox)
                d = self._dist(ball_center, player_point)

                if d < best_dist:
                    best_dist = d
                    best_player_id = player_id

            # reset has_ball flags
            for player_id in frame_players:
                frame_players[player_id]["has_ball"] = False

            if best_player_id is not None and best_dist <= max_dist:
                frame_players[best_player_id]["has_ball"] = True
                team = frame_players[best_player_id].get("team", "unknown")

                if team in [None, "", "unknown"]:
                    if len(team_ball_control) == 0:
                        team_ball_control.append("unknown")
                    else:
                        team_ball_control.append(team_ball_control[-1])
                else:
                    team_ball_control.append(team)
            else:
                if len(team_ball_control) == 0:
                    team_ball_control.append("unknown")
                else:
                    team_ball_control.append(team_ball_control[-1])

        return np.array(team_ball_control, dtype=object)

    @staticmethod
    def smooth_team_control( team_ball_control, window_size=5):
        if team_ball_control is None or len(team_ball_control) == 0:
            return team_ball_control

        smoothed = []
        n = len(team_ball_control)

        for i in range(n):
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2 + 1)

            window = [
                t for t in team_ball_control[start:end]
                if t not in [None, "", "unknown"]
            ]

            if len(window) == 0:
                smoothed.append(team_ball_control[i])
            else:
                smoothed.append(Counter(window).most_common(1)[0][0])

        return np.array(smoothed, dtype=object)
    
    @staticmethod
    def _get_player_control_point(bbox):
        """
        Use feet / lower-center rather than bbox center.
        This is usually better for ball possession in football.
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y2)

    @staticmethod
    def _euclidean_distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])