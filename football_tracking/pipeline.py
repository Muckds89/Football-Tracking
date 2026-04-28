from logging import config
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
from football_tracking.roi.roi_cropper import ROICropper
from football_tracking.tracking.tracker import Tracker
from football_tracking.team_assigner.team_assigner import TeamAssigner
from football_tracking.player_ball_assigner.player_ball_assigner import PlayerBallAssigner



import traceback
import time

import shutil



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


        # checkpoints 
        cache_dir = os.path.join(config.output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        tracks_path = os.path.join(cache_dir, f"{video_stem}_tracks.pkl")
        team_tracks_path = os.path.join(cache_dir, f"{video_stem}_team_tracks.pkl")
        ball_tracks_path = os.path.join(cache_dir, f"{video_stem}_ball_tracks.pkl")
        ball_interp_path = os.path.join(cache_dir, f"{video_stem}_ball_interp.pkl")
        raw_events_path = os.path.join(cache_dir, f"{video_stem}_raw_events.json")
        windows_path = os.path.join(cache_dir, f"{video_stem}_highlight_windows.json")

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
                    "ball_crop_roi",   # pitch + goals polygon used for ball model
                ]
                rois = ROIUtils(ROI_NAMES).annotate_rois_local(reference_frame_path)
            elif config.environment == "colab":
                rois = annotate_rois_colab(reference_frame_path)
            else:
                raise ValueError(f"Unknown environment: {config.environment}")

            roi_manager.save_rois(video_name, rois)
            logging.info(f"Created ROIs for {video_name}")
        ball_roi_polygon = ROICropper.normalize_polygon(rois, "ball_crop_roi")
        logging.info(f"Step 2 ROI handling done in {time.time() - start_time:.1f}s")

        # 3. Players Tracking
        start_time = time.time()
        tracker = Tracker("yolov8n.pt")
        logging.info("Initialized YOLOv8n tracker")
        #make folder for stubs if not exists
        os.makedirs(config.stubs_dir, exist_ok=True)
        stub_path = os.path.join(config.stubs_dir, "track_stub_debug.pkl")

        if os.path.exists(tracks_path):
            tracks = IOUtils.load_pickle(tracks_path)
            logging.info("Loaded cached tracks")
        else:
            tracks = tracker.get_player_tracks_from_video(
                video_path=video_path,
                read_from_stub=False,
                stub_path=stub_path,
                batch_size=16
            )
            IOUtils.save_pickle(tracks, tracks_path)

        logging.info(f"Step 3 Player tracking done in {time.time() - start_time:.1f}s")

        # 4. Assign Teams to Players
        start_time = time.time()
        if os.path.exists(team_tracks_path):
            tracks = IOUtils.load_pickle(team_tracks_path)
            logging.info("Loaded cached tracks")
        else:   
            tracks = TeamAssigner().assign_player_teams_from_video(video_path, tracks)
            logging.info("Assigned teams to player tracks")
            tracks = TeamAssigner().smooth_player_teams(tracks)    
            logging.info("Smoothed player team assignments")
            IOUtils.save_pickle(tracks, team_tracks_path)

        logging.info(f"Step 4 Player team assignment done in {time.time() - start_time:.1f}s")

        # 5. Ball tracking
        start_time = time.time()
        ball_tracker = BallTracker(config.model_path)
        if os.path.exists(ball_tracks_path):
            ball_tracks = IOUtils.load_pickle(ball_tracks_path)
            logging.info("Loaded cached tracks")
        else: 
            ball_tracks = ball_tracker.get_ball_tracks(
                video_path=video_path,
                ball_roi_polygon=ball_roi_polygon,
                apply_mask=True,
                frame_skip=2
            )
            IOUtils.save_pickle(ball_tracks, ball_tracks_path)

        detected = sum(1 for t in ball_tracks if t is not None)
        logging.info(f"Ball detected in {detected}/{len(ball_tracks)} frames")
        logging.info(f"Step 5 Ball tracking done in {time.time() - start_time:.1f}s")

        # 6. Interpolation
        start_time = time.time()
        interpolator = Interpolator()
        if os.path.exists(ball_interp_path):
            ball_tracks_filled = IOUtils.load_pickle(ball_interp_path)
            logging.info("Loaded cached tracks")
        else: 
            ball_tracks_filled = interpolator.interpolate_ball_tracks(ball_tracks)
            IOUtils.save_pickle(ball_tracks_filled, ball_interp_path)
        if False:
        # debug video with tracks and teams
        # if True:
            debug_output_path = os.path.join(
                config.output_dir,
                "debugs",
                f"{video_stem}_ball_interpolation_and_players_debug.mp4"
            )
            VideoUtils.write_player_team_debug_video(
                video_path=video_path,
                tracks=tracks,
                output_path=debug_output_path,
                fps=fps,
                ball_tracks=ball_tracks_filled
            )

        logging.info(f"Step 6 Interpolation done in {time.time() - start_time:.1f}s")


        # 7. Infer team in ball control
        start_time = time.time()
        team_ball_control = PlayerBallAssigner().infer_team_ball_control(tracks, ball_tracks_filled)
        team_ball_control = PlayerBallAssigner().smooth_team_control(team_ball_control)
        logging.info(f"Step 7 Infer team ball control done in {time.time() - start_time:.1f}s")


        # 8. Event detection
        start_time = time.time()
        event_detector = EventDetector(rois, fps=config.fps)

        kickoff = None
        kickoff_event = None
        if config.has_kickoff:
            kickoff = event_detector.detect_kickoff(ball_tracks_filled)

        if kickoff is None:
            logging.info("Kickoff disabled or not detected, using all frames")
            start_frame_for_events = 0
            filtered_tracks = ball_tracks_filled
            filtered_team_ball_control = team_ball_control
        else:
            start_frame_for_events = kickoff["frame"]
            filtered_tracks = ball_tracks_filled[start_frame_for_events:]
            filtered_team_ball_control = team_ball_control[start_frame_for_events:]
            kickoff_event = {
                "event": "kickoff",
                "start_frame": max(0, start_frame_for_events - int(5 * fps)),
                "end_frame": min(total_frames - 1, start_frame_for_events + int(8 * fps)),
                "start_time_sec": max(0, start_frame_for_events - int(5 * fps)) / fps,
                "end_time_sec": min(total_frames - 1, start_frame_for_events + int(8 * fps)) / fps,
                "goal_touched": False,
            }
            logging.info(f"Kickoff detected at frame {start_frame_for_events}")



        raw_events = []

        raw_events.extend(
            event_detector.detect_penalty_pressure_merged(
                ball_tracks=filtered_tracks,
                team_ball_control=filtered_team_ball_control,
                attacking_team="vest_team",
                side=config.team_attack_directions["vest_team"],
                window_size=8,
                min_hits=3,
                min_possession_hits=3,
            )
        )

        raw_events.extend(
            event_detector.detect_penalty_pressure_merged(
                ball_tracks=filtered_tracks,
                team_ball_control=filtered_team_ball_control,
                attacking_team="other_team",
                side=config.team_attack_directions["other_team"],
                window_size=8,
                min_hits=3,
                min_possession_hits=3,
            )
        )
        logging.info(f"Detected {len(raw_events)} raw events")
        logging.info(f"Step 8 Event detection done in {time.time() - start_time:.1f}s")

        # if kickoff used, shift raw event frames back to original video frames
        for ev in raw_events:
            ev["start_frame"] += start_frame_for_events
            ev["end_frame"] += start_frame_for_events
            ev["start_time_sec"] = ev["start_frame"] / fps
            ev["end_time_sec"] = ev["end_frame"] / fps

        goal_touched_events = [
            ev for ev in raw_events
            if ev.get("goal_touched") is True
        ]

        logging.info(f"Detected {len(goal_touched_events)} goal-touched raw events")    

        highlight_windows_full = event_detector.build_highlight_windows(
            raw_events,
            fps=fps,
            seconds_before=15,
            seconds_after=5
        )

        highlight_windows_goal_touched = event_detector.build_highlight_windows(
            goal_touched_events,
            fps=fps,
            seconds_before=8,
            seconds_after=4
        )

        if kickoff_event is not None:
            highlight_windows_full = [kickoff_event] + highlight_windows_full



        # 9. Export highlights
        start_time = time.time()
        output_highlight_path_full = os.path.join(
            config.output_dir,
            "highlights",
            f"{video_name}_highlights_full.mp4"
        )


        clips_dir_full = os.path.join(config.output_dir, "clips", f"{video_stem}_full")
        clips_dir_goal_touched = os.path.join(config.output_dir, "clips", f"{video_stem}_goal_touched")

        if os.path.exists(clips_dir_full):
            shutil.rmtree(clips_dir_full)
        if os.path.exists(clips_dir_goal_touched):
            shutil.rmtree(clips_dir_goal_touched)

        os.makedirs(clips_dir_full, exist_ok=True)
        os.makedirs(clips_dir_goal_touched, exist_ok=True)

        highlight_windows_json_path = os.path.join(
            config.output_dir,
            "highlights",
            f"{video_name}_highlight_windows.json"
        )

        IOUtils.save_json(highlight_windows_full, highlight_windows_json_path)
        logging.info(f"Saved highlight windows JSON to {highlight_windows_json_path}")

        HIGHVideoUtils().save_highlights_as_clips_parallel(
            video_path=video_path,
            highlight_windows=highlight_windows_full,
            clips_dir=clips_dir_full,
            fps=fps,
            workers=4
        )
        HIGHVideoUtils().concat_clips_ffmpeg(
        clips_dir_full,
        output_highlight_path_full
        )
        logging.info(f"Saved highlights video to {output_highlight_path_full}")
        
        # 11. Shorter highlight windows for goal-touched events

        logging.info(f"Step 9 Export highlights done in {time.time() - start_time:.1f}s")
        output_highlight_shorter_path = os.path.join(
            config.output_dir,
            "highlights",
            f"{video_name}_highlights_shorter.mp4"
        )
        HIGHVideoUtils().save_highlights_as_clips_parallel(
            video_path=video_path,
            highlight_windows=highlight_windows_goal_touched,
            clips_dir=clips_dir_goal_touched,
            fps=fps,
            workers=4
        )
        HIGHVideoUtils().concat_clips_ffmpeg(
            clips_dir_goal_touched,
            output_highlight_shorter_path
        )


        # 12. Save events JSON
        start_time = time.time()
        highlight_windows_full_json_path = os.path.join(
            config.output_dir,
            "highlights",
            f"{video_stem}_highlight_windows_full.json"
        )

        highlight_windows_goal_touched_json_path = os.path.join(
            config.output_dir,
            "highlights",
            f"{video_stem}_highlight_windows_goal_touched.json"
        )

        IOUtils.save_json(highlight_windows_full, highlight_windows_full_json_path)
        IOUtils.save_json(highlight_windows_goal_touched, highlight_windows_goal_touched_json_path)

        logging.info(f"Saved full highlight windows JSON to {highlight_windows_full_json_path}")
        logging.info(f"Saved goal-touched highlight windows JSON to {highlight_windows_goal_touched_json_path}")


        logging.info(f"Finished processing {video_name}")
        logging.info(f"Step 10 Save events JSON done in {time.time() - start_time:.1f}s")
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
            if f.endswith(".mp4") or f.endswith(".MP4")
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
    traceback.print_exc()#

