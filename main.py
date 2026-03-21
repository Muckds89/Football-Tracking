from utils import Utils
from trackers import Tracker
from trackers import BallTracker
import cv2
import numpy as np
import os, sys
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

from ultralytics import YOLO

import logging
logging.basicConfig(level=logging.INFO)


def main():
    # Read Video
    
    video_frames = Utils.read_video('input_videos/Goal2.mp4')
    # reduce to the first 60 frames for debug
    video_frames = video_frames[:60]
    logging.info(f"Total frames read: {len(video_frames)}")
    output_dir = "debug_frames"
    os.makedirs(output_dir, exist_ok=True)

    # define goals
    LEFT_GOAL = {
        "x1": 0,
        "y1": 200,
        "x2": 200,
        "y2": 500
    }

    RIGHT_GOAL = {
        "x1": 1000,
        "y1": 200,
        "x2": 1280,
        "y2": 500
    }
    logging.info(f"Video resolution: {video_frames[0].shape[1]}x{video_frames[0].shape[0]}")
    logging.info(f"Left Goal: {LEFT_GOAL}")
    logging.info(f"Right Goal: {RIGHT_GOAL}")

    # Initialize Tracker
    tracker = Tracker('yolov8n.pt')
    logging.info("Initialized YOLOv8n tracker")

    # YOLO Model for Players
    player_tracks = tracker.get_object_track_debug(
        video_frames,
        output_dir=output_dir,
        read_from_stub=False,
        stub_path="stubs/track_stub_debug.pkl"
    )
    print(type(player_tracks))
    print(player_tracks.keys())

    print(type(player_tracks["players"]))
    print(type(player_tracks["players"][0]))
    print("frame 0 players:", player_tracks["players"][0])

    # CV2/CSRT Tracker for Ball
    ball_tracks = BallTracker('yolov8n.pt').track_ball_with_csrt(
        video_frames, 
        init_frame_idx=0,
        init_bbox=(2210, 1343, 60, 60))
    
    print(type(ball_tracks))         # dict  

    print(type(ball_tracks["ball"])) # list

    # Write debug video with tracks
    BallTracker('yolov8n.pt').draw_tracks_on_video(
        video_frames,
        player_tracks,
        ball_tracks,
        output_path="output_videos/output_players_ball.mp4"
    )
    sys.exit(1)

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    

    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)


    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save video
    Utils.save_video(output_video_frames, 'output_videos/goal_fixed1.mp4')

if __name__ == '__main__':
    # log time taken for each step
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")