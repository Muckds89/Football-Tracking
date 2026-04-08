import os
import logging
import json
from ultralytics import YOLO

from football_tracking.io_utils import IOUtils 
from football_tracking.tracking.tracker import Tracker
from football_tracking.utils.video_utils import VideoUtils
from football_tracking.tracking.ball_tracker import BallTracker
from football_tracking.tracking.interpolator import Interpolator
from football_tracking.events.event_detector import EventDetector
from football_tracking.highlights.highlight_writer import export_event_highlights
from football_tracking.roi.roi_manager import ROIManager
from football_tracking.team_assigner.team_assigner import TeamAssigner


def process_video(video_path: str, config):
    logging.info(f"Processing video: {video_path}")

    video_name = os.path.basename(video_path)

    # 1. Load frames
    frames = VideoUtils.read_video(video_path)
    total_frames = len(frames)

    # 2. ROI handling
    roi_manager = ROIManager(config.roi_dir)

    if roi_manager.roi_exists(video_name):
        rois = roi_manager.load_rois(video_name)
        logging.info("Loaded existing ROIs")
    else:
        raise Exception(f"ROIs not found for {video_name}. Please create them first.")

    # 3. Players Detection
    tracker = Tracker("yolov8n.pt")
    logging.info("Initialized YOLOv8n tracker")

    tracks = tracker.get_object_track_debug(
        frames,
        output_dir=config.output_dir,
        read_from_stub=False,
        stub_path="stubs/track_stub_debug.pkl"
    )
    logging.info(f"Extracted player tracks for {len(tracks['players'])} frames")

    # 4. Assign teams to players
    team_assigner = TeamAssigner()

    if len(tracks["players"]) > 0 and len(tracks["players"][0]) > 0:
        team_assigner.assign_team_color(frames[-1], tracks["players"][-1])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                frames[frame_num],
                track["bbox"],
                player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]
            logging.debug(f"Frame {frame_num} | Player {player_id} assigned to team {team} with color {team_assigner.team_colors[team]}")

    logging.info("Assigned teams to players")

    # 3. Ball tracking
    ball_tracker = BallTracker(config.model_path)
    ball_tracks = ball_tracker.get_ball_tracks(video_path)

    detected = sum(1 for t in ball_tracks if t is not None)
    logging.info(f"Ball detected in {detected}/{len(ball_tracks)} frames")

    # 4. Interpolation
    interpolator = Interpolator()
    ball_tracks_filled = interpolator.interpolate(ball_tracks)

    # 5. Event detection
    event_detector = EventDetector(rois, fps=config.fps)

    events = event_detector.detect_penalty_pressure(
        ball_tracks_filled,
        window_size=8,
        min_hits=3
    )

    logging.info(f"Detected {len(events)} raw events")

    # 6. Export highlights
    output_highlight_path = os.path.join(
        config.output_dir,
        "highlights",
        f"{video_name}_highlights.mp4"
    )

    export_event_highlights(
        frames=frames,
        events=events,
        output_path=output_highlight_path,
        fps=config.fps,
        seconds_before=config.seconds_before,
        seconds_after=config.seconds_after
    )

    # 7. Save events JSON
    event_json_path = os.path.join(
        config.output_dir,
        "events",
        f"{video_name}_events.json"
    )

    IOUtils.save_json(events, event_json_path)

    logging.info(f"Finished processing {video_name}")

    return {
        "video": video_name,
        "frames": total_frames,
        "detections": detected,
        "events": len(events)
    }


def process_new_videos(config):
    input_dir = config.input_dir
    manifest_path = config.processed_manifest_path

    # load processed videos
    processed = IOUtils.load_json(manifest_path)
    IOUtils.save_json({"processed": processed}, manifest_path)

    all_videos = [
        f for f in os.listdir(input_dir)
        if f.endswith(".mp4")
    ]

    new_videos = [v for v in all_videos if v not in processed]

    logging.info(f"Found {len(new_videos)} new videos")

    for video_name in new_videos:
        video_path = os.path.join(input_dir, video_name)

        try:
            process_video(video_path, config)

            processed.append(video_name)

            # save manifest after each success
            IOUtils.save_json({"processed": processed}, manifest_path)

        except Exception as e:
            logging.error(f"Failed processing {video_name}: {e}")