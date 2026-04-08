import cv2
import json
from ultralytics import YOLO
import BallTracker
import Interpolator
import VideoUtils
import logging
logging.basicConfig(level=logging.INFO)

model = YOLO("../../models/best.pt")

with open("output_videos/rois.json", "r") as f:
    rois = json.load(f)

# cap = cv2.VideoCapture("../../input_videos/Video Project 13.mp4")
video_path = "../../input_videos/Video Project 13.mp4"
frames_ = VideoUtils.read_video(video_path)
if not frames_:
    logging.error(f"Failed to read video: {video_path}")

# resize window
# cv2.namedWindow("Ball Detection", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Ball Detection", 1280, 720)

frame_skip = 2  # process 1 out of 2 frames
frame_count = 0

# Interpolate Ball Positions
ball_tracks = BallTracker(model_path="models/best.pt").get_ball_tracks(video_path    
)
logging.info(f"Extracted ball tracks for {len(ball_tracks)} frames.")

# Interpolate Ball Positions
ball_tracks_filled = Interpolator.interpolate_ball_tracks(ball_tracks, max_gap=5)
logging.info(f"Video used for frames: {video_path}")
logging.info(f"Number of ball tracks: {len(ball_tracks_filled)}")

# ball_tracks["ball"] = Tracker(model).interpolate_ball_positions(ball_tracks["ball"])
VideoUtils.write_ball_debug_video(frames_, ball_tracks_filled, rois, "output_videos/debug_filled.mp4", fps=25)




