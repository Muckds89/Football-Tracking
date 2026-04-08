import os
import logging
import json
import sys
from sympy import fps
from ultralytics import YOLO

from football_tracking.events import event_detector
from football_tracking.io_utils import IOUtils 
from football_tracking.utils.video_utils import VideoUtils
from football_tracking.tracking.ball_tracker import BallTracker
from football_tracking.tracking.interpolator import Interpolator
from football_tracking.events.event_detector import EventDetector
from football_tracking.highlights.highlight_writer import HIGHVideoUtils
from football_tracking.roi.roi_manager import ROIManager
from football_tracking.roi.roi_drawer_colab import annotate_rois_colab
from football_tracking.roi.roi_drawer_local import ROIUtils


import traceback
import time

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "pipeline.log"), encoding="utf-8")
    ]
)

try:


    def process_video(video_path: str, config):
        logging.info(f"Processing video: {video_path}")

        video_name = os.path.basename(video_path)
        video_stem = video_name.rsplit(".", 1)[0]

        # 1. Load frames
        start_time = time.time()
        meta = VideoUtils.get_video_metadata(video_path)
        total_frames = meta["frame_count"]
        fps = meta["fps"]
        logging.info(f"Video metadata: {meta}")
        last_frame = None
        last_frame = VideoUtils.get_last_frame(video_path)
        logging.info(f"Step 1 Load Frames done in {time.time() - start_time:.1f}s")

        # 2. ROI handling
        start_time = time.time()
        roi_manager = ROIManager(config.roi_dir)
        if roi_manager.roi_exists(video_name):
            rois = roi_manager.load_rois(video_name)
        else:
            reference_dir = os.path.join(config.output_dir, "reference_frames")
            IOUtils.ensure_dir(reference_dir)

            reference_frame_path = os.path.join(reference_dir, f"{video_stem}_last_frame.jpg")
            VideoUtils.save_frame(last_frame, reference_frame_path)

            if config.environment == "local":
                ROI_NAMES = [
                    "center_pitch",
                    "left_penalty_box",
                    "right_penalty_box",
                    "left_goal",
                    "right_goal",
                ]
                rois = ROIUtils(ROI_NAMES).annotate_rois_local(reference_frame_path)
            elif config.environment == "colab":
                rois = annotate_rois_colab(reference_frame_path)
            else:
                raise ValueError(f"Unknown environment: {config.environment}")

            roi_manager.save_rois(video_name, rois)
            logging.info(f"Created ROIs for {video_name}")
        logging.info(f"Step 2 ROI handling done in {time.time() - start_time:.1f}s")
            
        # 3. Ball tracking
        start_time = time.time()
        ball_tracker = BallTracker(config.model_path)
        ball_tracks = ball_tracker.get_ball_tracks(video_path)

        detected = sum(1 for t in ball_tracks if t is not None)
        logging.info(f"Ball detected in {detected}/{len(ball_tracks)} frames")
        logging.info(f"Step 3 Ball tracking done in {time.time() - start_time:.1f}s")

        # 4. Interpolation
        start_time = time.time()
        interpolator = Interpolator()
        ball_tracks_filled = interpolator.interpolate_ball_tracks(ball_tracks)
        logging.info(f"Step 4 Interpolation done in {time.time() - start_time:.1f}s")

        # 5. Event detection
        start_time = time.time()
        event_detector = EventDetector(rois, fps=config.fps)


        kickoff = event_detector.detect_kickoff(ball_tracks_filled)

        if kickoff is None:
            logging.warning("No kickoff detected, using all frames")
            filtered_tracks = ball_tracks_filled
            start_frame_for_events = 0
        else:
            kickoff_frame = kickoff["frame"]
            kickoff_event = {
                "event": "kickoff",
                "start_frame": max(0, kickoff_frame - int(5 * fps)),
                "end_frame": min(total_frames - 1, kickoff_frame + int(8 * fps)),
                "start_time_sec": max(0, kickoff_frame - int(5 * fps)) / fps,
                "end_time_sec": min(total_frames - 1, kickoff_frame + int(8 * fps)) / fps,
            }
            start_frame_for_events = kickoff["frame"]
            filtered_tracks = ball_tracks_filled[start_frame_for_events:]
            logging.info(f"Kickoff detected at frame {start_frame_for_events}")


        raw_events = event_detector.detect_penalty_pressure_merged(
            filtered_tracks,
            window_size=8,
            min_hits=3
        )

        logging.info(f"Detected {len(raw_events)} raw events")
        logging.info(f"Step 5 Event detection done in {time.time() - start_time:.1f}s")

        # if kickoff used, shift raw event frames back to original video frames
        for ev in raw_events:
            ev["start_frame"] += start_frame_for_events
            ev["end_frame"] += start_frame_for_events
            ev["start_time_sec"] = ev["start_frame"] / fps
            ev["end_time_sec"] = ev["end_frame"] / fps

        highlight_windows = event_detector.build_highlight_windows(
            raw_events,
            fps=fps,
            seconds_before=20,
            seconds_after=10
        )
        if kickoff_event is not None:
            highlight_windows = [kickoff_event] + highlight_windows
        # 6. Export highlights
        start_time = time.time()
        output_highlight_path = os.path.join(
            config.output_dir,
            "highlights",
            f"{video_name}_highlights.mp4"
        )

        HIGHVideoUtils().save_highlights_to_video(
            video_path=video_path,
            highlight_windows=highlight_windows,
            output_path=output_highlight_path,
            fps=fps    )
        logging.info(f"Step 6 Export highlights done in {time.time() - start_time:.1f}s")

        # 7. Save events JSON
        start_time = time.time()
        event_json_path = os.path.join(
            config.output_dir,
            "events",
            f"{video_name}_events.json"
        )

        IOUtils.save_json(raw_events, event_json_path)

        logging.info(f"Finished processing {video_name}")
        logging.info(f"Step 7 Save events JSON done in {time.time() - start_time:.1f}s")
        return {
            "video": video_name,
            "frames": total_frames,
            "detections": detected,
            "events": len(raw_events)
        }


    def process_new_videos(config):
        input_dir = config.input_dir
        manifest_path = config.processed_manifest_path


        # load processed videos
        if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0:
            IOUtils.save_json({"processed": []}, manifest_path)

        manifest = IOUtils.load_json(manifest_path, default={"processed": []})
        processed = manifest.get("processed", [])

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
                IOUtils.ensure_dir(os.path.dirname(config.processed_manifest_path))
                data = IOUtils.load_json(manifest_path)
                IOUtils.save_json({"processed": processed}, manifest_path)

                if "processed" not in data:
                    data = {"processed": []}

                processed = data["processed"]
            except Exception as e:
                logging.error(f"Failed processing {video_name}: {e}")
                traceback.print_exc()   #

except Exception as e:
    logging.error(f"Failed processing:  {e}")
    traceback.print_exc()