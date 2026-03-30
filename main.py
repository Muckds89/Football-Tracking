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

from roi_utils import save_rois


video_path = "input_videos/Goal1.mp4"
output_dir = "output_videos"
# trim video to first 300 frames for testing
frames_ = Utils.read_video(video_path)
max_frames = len(list(frames_))
logging.info(f"Total frames in video: {max_frames}")


def main(frames, display_width=1280, display_height=720):

    ROI_NAMES = [
        "midfield_spot",
        "left_penalty_box",
        "right_penalty_box",
        "left_goal",
        "right_goal",
    ]

    current_points = []
    rois = {}
    current_roi_index = 0


    def mouse_callback(event, x, y, flags, param):
        global current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))

    frame = frames[-1]  # Take the last frame for ROI drawing
    # if frm is None:
    #     logging.error("Can't read the last frame of the video.")
    #     return

    # Ora puoi usare 'frm' per le ROI
    cv2.namedWindow("Draw ROIs", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Draw ROIs", display_width, display_height)
    cv2.setMouseCallback("Draw ROIs", mouse_callback)    

    while True:
        display = frame.copy()

        for roi_name, polygon in rois.items():
            pts = polygon[:]
            if len(pts) >= 2:
                cv2.polylines(display, [cv2.UMat(cv2.convexHull(cv2.UMat(np.array(pts, dtype='int32')))).get()], True, (0, 255, 0), 2)

        for i, pt in enumerate(current_points):
            cv2.circle(display, pt, 4, (0, 0, 255), -1)
            if i > 0:
                cv2.line(display, current_points[i - 1], pt, (255, 0, 0), 2)

        if current_roi_index < len(ROI_NAMES):
            text = f"Draw ROI: {ROI_NAMES[current_roi_index]} | ENTER=save ROI | U=undo point | Q=quit"
        else:
            text = "All ROIs completed. Press S to save, Q to quit."

        cv2.putText(display, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Draw ROIs", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u") and current_points:
            current_points.pop()

        elif key == 13:  # Enter
            if current_roi_index < len(ROI_NAMES) and len(current_points) >= 3:
                rois[ROI_NAMES[current_roi_index]] = current_points[:]
                current_points = []
                current_roi_index += 1

        elif key == ord("s"):
            save_rois(rois, "rois.json")
            print("Saved rois.json")
            break

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


    # logging.info(f"Video resolution: {video_frames[0].shape[1]}x{video_frames[0].shape[0]}")
    # logging.info(f"Left Goal: {LEFT_GOAL}")
    # logging.info(f"Right Goal: {RIGHT_GOAL}")

    # # Initialize Tracker
    # tracker = Tracker('yolov8n.pt')
    # take the last fraim from frame generator
    #     # logging.info("Initialized YOLOv8n tracker")

    # YOLO Model for Players
    # player_tracks = tracker.get_object_track_debug(
    #     video_frames,
    #     output_dir=output_dir,
    #     read_from_stub=False,
    #     stub_path="stubs/track_stub_debug.pkl"
    # )
    # print(type(player_tracks))
    # print(player_tracks.keys())

    # print(type(player_tracks["players"]))
    # print(type(player_tracks["players"][0]))
    # print("frame 0 players:", player_tracks["players"][0])

    # CV2/CSRT Tracker for Ball
    # ball_tracker = BallTracker("yolov8n.pt", yolo_conf=0.15)

    # ball_tracks = ball_tracker.track_ball_hybrid(
    #     video_frames,
    #     player_tracks=player_tracks,
    #     init_frame_idx=0,
    #     init_bbox=bbox,
    #     max_frames=max_frames
    # )
    
    # print(type(ball_tracks))         # dict  

    # print(type(ball_tracks["ball"])) # list

    # Write debug video with tracks
    # BallTracker('yolov8n.pt').draw_tracks_on_video(
    #     video_frames,
    #     player_tracks,
    #     ball_tracks,
    #     output_path="output_videos/output_players_ball.mp4"
    # )
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
    main(frames_, display_width=1280, display_height=720)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")