    import time

    import cv2
    import os
    import logging
    import json
    import sys
    from ultralytics import YOLO

    sys.path.append("../")
    from football_tracking.io_utils import IOUtils
    from football_tracking.roi.roi_manager import ROIManager
    from football_tracking.utils.video_utils import VideoUtils
    from football_tracking.events.event_detector import EventDetector
    from football_tracking.tracking.ball_tracker import BallTracker
    from football_tracking.tracking.interpolator import Interpolator
    from football_tracking.roi.roi_drawer_local import ROIUtils
    from football_tracking.roi.roi_drawer_colab import annotate_rois_colab

    from dataclasses import dataclass
    environment = "local"  # or "colab"

    logging.basicConfig(level=logging.INFO)

    ROI_NAMES = [
        "center_pitch",
        "left_penalty_box",
        "right_penalty_box",
        "left_goal",
        "right_goal",
    ]


    def main():
        video_path = "../input_videos/Video Project 13.mp4"
        video_name = os.path.basename(video_path)
        video_stem = os.path.splitext(video_name)[0]
        output_dir = "output_videos"
        os.makedirs(output_dir, exist_ok=True)

        frames_ = VideoUtils.read_video_stream(video_path)
        if not frames_:
            logging.error(f"Failed to read video: {video_path}")
            return

        last_frame = VideoUtils.get_last_frame(video_path)
        last_frame_path = os.path.join(output_dir, "last_frame.jpg")
        roi_json_path = os.path.join(output_dir, "rois.json")

        cv2.imwrite(last_frame_path, last_frame)
        logging.info(f"Saved last frame for ROI annotation: {last_frame_path}")

        # 2. ROI handling
        start_time = time.time()
        roi_manager = ROIManager('../rois')
        if roi_manager.roi_exists(video_name):
            rois = roi_manager.load_rois(video_name)
        else:
            reference_dir = os.path.join(output_dir, "reference_frames")
            IOUtils.ensure_dir(reference_dir)

            reference_frame_path = os.path.join(reference_dir, f"{video_stem}_last_frame.jpg")
            VideoUtils.save_frame(last_frame, reference_frame_path)

            if environment == "local":
                ROI_NAMES = [
                    "center_pitch",
                    "left_penalty_box",
                    "right_penalty_box",
                    "left_goal",
                    "right_goal",
                ]
                rois = ROIUtils(ROI_NAMES).annotate_rois_local(reference_frame_path)
            elif environment == "colab":
                rois = annotate_rois_colab(reference_frame_path)
            else:
                raise ValueError(f"Unknown environment: {environment}")

            roi_manager.save_rois(video_name, rois)
            logging.info(f"Created ROIs for {video_name}")
        logging.info(f"Step 2 ROI handling done in {time.time() - start_time:.1f}s")


        fps = 25  # or get from the video



        # Interpolate Ball Positions
        ball_tracks = BallTracker(model_path="../models/best.pt").get_ball_tracks(video_path    
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
        interpolator = Interpolator()
        ball_tracks_filled = interpolator.interpolate_ball_tracks(ball_tracks, max_gap=5)
        logging.info(f"Video used for frames: {video_path}")
        logging.info(f"Number of ball tracks: {len(ball_tracks_filled)}")

        # ball_tracks["ball"] = Tracker(model).interpolate_ball_positions(ball_tracks["ball"])
        VideoUtils.write_ball_debug_video(frames_, ball_tracks_filled, rois, "output_videos/debug_filled.mp4", fps=25)
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

        highlight_windows = VideoUtils.build_highlight_windows(
            events=right_events,
            total_frames=total_frames,
            fps=fps,
            seconds_before=20,
            seconds_after=10
        )

        logging.info(f"Highlight windows: {highlight_windows}")

        VideoUtils.save_highlights_to_video(
            frames=frames_,
            highlight_windows=highlight_windows,
            output_path="output_videos/highlights.mp4",
            fps=fps
        )


    if __name__ == "__main__":
        main()