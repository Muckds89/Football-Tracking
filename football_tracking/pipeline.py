import os
import logging
import json
from ultralytics import YOLO

from football_tracking.io_utils import IOUtils 
from football_tracking.utils.video_utils import VideoUtils
from football_tracking.tracking.ball_tracker import BallTracker
from football_tracking.tracking.interpolator import Interpolator
from football_tracking.events.event_detector import EventDetector
from football_tracking.highlights.highlight_writer import export_event_highlights
from football_tracking.roi.roi_manager import ROIManager


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