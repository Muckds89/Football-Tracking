import cv2
import os
import logging
import json
import sys
from ultralytics import YOLO


from football_tracking.roi.roi_manager import ROIManager
from football_tracking.utils.video_utils import VideoUtils
from football_tracking.events.event_detector import EventDetector
from football_tracking.tracking.ball_tracker import BallTracker
from football_tracking.utils.interpolator import Interpolator

from dataclasses import dataclass


logging.basicConfig(level=logging.INFO)

ROI_NAMES = [
    "center_pitch",
    "left_penalty_box",
    "right_penalty_box",
    "left_goal",
    "right_goal",
]


def main():
    video_path = "input_videos/DJI_20260329183038_0002_D.mp4"
    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    frames_ = Utils.read_video(video_path)
    if not frames_:
        logging.error(f"Failed to read video: {video_path}")
        return

    last_frame = frames_[-1]
    last_frame_path = os.path.join(output_dir, "last_frame.jpg")
    roi_json_path = os.path.join(output_dir, "rois.json")

    cv2.imwrite(last_frame_path, last_frame)
    logging.info(f"Saved last frame for ROI annotation: {last_frame_path}")

    roi_utils = ROIUtils(ROI_NAMES)
    rois = roi_utils.annotate_rois(last_frame_path, output_json=roi_json_path)
    # read the saved ROIs from JSON to ensure they are correctly saved
    if os.path.exists(roi_json_path):
        with open(roi_json_path, "r") as f:
            # format json in single line to avoid issues with trailing commas
            rois = f.read().replace("\n", "").replace(" ", "")
            # rois = f.read()
        logging.info(f"Saved ROIs in JSON: {rois}")
    else:
        logging.error(f"ROI JSON file not found: {roi_json_path}")

    logging.info(f"Final ROIs: {rois}")

    # load rois
    with open("output_videos/rois.json", "r") as f:
        rois = json.load(f)

    fps = 25  # or get from the video

    event_detector = EventDetector(rois, fps)
    model = YOLO("models/best.pt")



    # Interpolate Ball Positions
    ball_tracks = BallTracker(model_path="models/best.pt").get_ball_tracks(video_path    
    )
    logging.info(f"Extracted ball tracks for {len(ball_tracks)} frames.")
    # Utils.write_ball_debug_video(
    #     frames=frames_,
    #     ball_tracks=ball_tracks,
    #     rois=rois,
    #     output_path="output_videos/goal1_ball_debug.mp4",
    #     fps=25
    # )
    # Interpolate Ball Positions
    ball_tracks_filled = Interpolator.interpolate_ball_tracks(ball_tracks, max_gap=5)
    logging.info(f"Video used for frames: {video_path}")
    logging.info(f"Number of frames loaded: {len(frames_)}")
    logging.info(f"Number of ball tracks: {len(ball_tracks_filled)}")

    # ball_tracks["ball"] = Tracker(model).interpolate_ball_positions(ball_tracks["ball"])
    Utils.write_ball_debug_video(frames_, ball_tracks_filled, rois, "output_videos/debug_filled.mp4", fps=25)
    cv2.imwrite("output_videos/check_frame_49.jpg", frames_[49])
    debug_rows = event_detector.debug_membership(ball_tracks)

    # for row in debug_rows[:50]:
    #     logging.info(row)
    event_detector = EventDetector(rois, fps=25)

    right_events = event_detector.detect_penalty_pressure_merged(
        ball_tracks_filled,
        side="right",
        window_size=8,
        min_hits=3,
        exit_gap_frames=12,   # about 0.5 sec at 25 fps
        min_event_frames=5
    )

    for e in right_events:
        logging.info(e)

    # for event in events:
        # logging.info(event)

    fps = 25
    total_frames = len(frames_)

    highlight_windows = Utils.build_highlight_windows(
        events=right_events,
        total_frames=total_frames,
        fps=fps,
        seconds_before=20,
        seconds_after=10
    )

    logging.info(f"Highlight windows: {highlight_windows}")

    Utils.save_highlights_to_video(
        frames=frames_,
        highlight_windows=highlight_windows,
        output_path="output_videos/highlights.mp4",
        fps=fps
    )


if __name__ == "__main__":
    main()