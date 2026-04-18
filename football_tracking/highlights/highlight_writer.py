import os
import cv2
import logging
from football_tracking.io_utils import IOUtils
from multiprocessing import Pool
import os
import subprocess
import json
from pathlib import Path
import logging
    
class HIGHVideoUtils:
    def __init__(self):
        pass
    def build_highlight_windows(self,events, total_frames, fps, seconds_before=10, seconds_after=10):
        """
        Convert event triggers into non-overlapping highlight windows.
        Each event is expected to have either:
        - "frame"
        - or "start_frame" / "end_frame"

        Returns a list of dicts:
        [
            {
            "event": ...,
            "trigger_frame": ...,
            "start_frame": ...,
            "end_frame": ...
            }
        ]
        """
        if not events:
            return []

        before_frames = int(seconds_before * fps)
        after_frames = int(seconds_after * fps)

        # normalize trigger frame
        normalized = []
        for e in events:
            if "frame" in e:
                trigger_frame = e["frame"]
            elif "start_frame" in e and "end_frame" in e:
                trigger_frame = (e["start_frame"] + e["end_frame"]) // 2
            elif "start_frame" in e:
                trigger_frame = e["start_frame"]
            else:
                continue

            start_frame = max(0, trigger_frame - before_frames)
            end_frame = min(total_frames - 1, trigger_frame + after_frames)

            normalized.append({
                "event": e.get("event", "event"),
                "trigger_frame": trigger_frame,
                "start_frame": start_frame,
                "end_frame": end_frame,
            })

        normalized.sort(key=lambda x: x["start_frame"])

        # skip overlapping windows
        selected = []
        last_end = -1

        for win in normalized:
            if win["start_frame"] <= last_end:
                continue
            selected.append(win)
            last_end = win["end_frame"]

        return selected
    

    def save_highlights_to_video(self,video_path, highlight_windows, output_path, fps):
        """
        Create a highlights video from a source video without loading all frames into RAM.
        Append new clips if the output already exists by rewriting a combined file safely.
        """
        if not highlight_windows:
            logging.info("No highlight windows to save.")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_new_clips = output_path.replace(".mp4", "_newclips.mp4")
        temp_combined = output_path.replace(".mp4", "_combined.mp4")

        IOUtils.ensure_dir(os.path.dirname(output_path))

        writer = cv2.VideoWriter(
            temp_new_clips,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        for idx, win in enumerate(highlight_windows):
            start_frame = max(0, win["start_frame"])
            end_frame = min(total_frames - 1, win["end_frame"])

            logging.info(
                f"Writing highlight {idx+1}: "
                f"{win['event']} | frames {start_frame} - {end_frame}"
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame

            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_out = frame.copy()

                cv2.putText(
                    frame_out,
                    f"{win['event']}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2
                )
                # cv2.putText(
                #     frame_out,
                #     f"frame {current_frame}",
                #     (20, 80),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.8,
                #     (255, 255, 255),
                #     2
                # )

                writer.write(frame_out)
                current_frame += 1

        writer.release()
        cap.release()

        if not os.path.exists(output_path):
            os.replace(temp_new_clips, output_path)
            logging.info(f"Created highlight video: {output_path}")
            return

        cap_old = cv2.VideoCapture(output_path)
        if not cap_old.isOpened():
            raise ValueError(f"Cannot open existing highlight video: {output_path}")

        writer_combined = cv2.VideoWriter(
            temp_combined,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        while True:
            ret, frame = cap_old.read()
            if not ret:
                break
            writer_combined.write(frame)

        cap_old.release()

        cap_new = cv2.VideoCapture(temp_new_clips)
        if not cap_new.isOpened():
            raise ValueError(f"Cannot open temp highlight video: {temp_new_clips}")

        while True:
            ret, frame = cap_new.read()
            if not ret:
                break
            writer_combined.write(frame)

        cap_new.release()
        writer_combined.release()

        os.remove(temp_new_clips)
        os.replace(temp_combined, output_path)

        logging.info(f"Appended highlights to existing video: {output_path}")

    def save_highlights_as_clips(self, video_path, highlight_windows, output_dir, fps):
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(output_dir, exist_ok=True)

        for idx, win in enumerate(highlight_windows):
            clip_path = os.path.join(output_dir, f"clip_{idx:04d}.mp4")

            # ✅ skip if already done (resume support)
            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
                logging.info(f"Skipping existing clip {idx}")
                continue

            writer = cv2.VideoWriter(
                clip_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            start_frame = max(0, win["start_frame"])
            end_frame = min(total_frames - 1, win["end_frame"])

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            current_frame = start_frame

            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.putText(
                    frame,
                    f"{win['event']}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2
                )

                writer.write(frame)
                current_frame += 1

            writer.release()
            logging.info(f"Saved clip {idx}: {clip_path}")

        cap.release()



    
    def write_single_clip(self,args):
        video_path, win, clip_path, fps = args

        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
            return f"skip {clip_path}"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"error opening video for {clip_path}"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            clip_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()
            return f"error opening writer for {clip_path}"

        start_frame = max(0, win["start_frame"])
        end_frame = min(total_frames - 1, win["end_frame"])

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(
                    frame,
                    f"{win['event']}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2
                )
            writer.write(frame)
            current_frame += 1

        writer.release()
        cap.release()
        return f"done {clip_path}"
    


    def save_highlights_as_clips_parallel(self,video_path, highlight_windows, clips_dir, fps, workers=4):
        os.makedirs(clips_dir, exist_ok=True)

        jobs = []
        for idx, win in enumerate(highlight_windows):
            clip_path = os.path.join(clips_dir, f"clip_{idx:04d}.mp4")
            jobs.append((video_path, win, clip_path, fps))

        with Pool(processes=workers) as pool:
            for result in pool.imap_unordered(self.write_single_clip, jobs):
                print(result)

    @staticmethod
    def concat_clips_ffmpeg( clips_dir, output_path):
        clip_files = sorted(
            f for f in os.listdir(clips_dir)
            if f.endswith(".mp4") and f.startswith("clip_")
        )

        list_file = os.path.join(clips_dir, "clips.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for clip in clip_files:
                f.write(f"file '{os.path.join(clips_dir, clip)}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path
        ]
        subprocess.run(cmd, check=True)

    
    @staticmethod
    def concat_goal_touched_from_windows(highlight_windows_json, clips_dir, output_path, ffmpeg="ffmpeg"):
        highlight_windows_json = Path(highlight_windows_json)
        clips_dir = Path(clips_dir)
        output_path = Path(output_path)

        with open(highlight_windows_json, "r", encoding="utf-8") as f:
            highlight_windows = json.load(f)

        # Clips sorted by filename order
        clip_files = sorted([p for p in clips_dir.glob("*.mp4") if p.is_file()])

        print(f"Highlight windows: {len(highlight_windows)}")
        print(f"Clips found: {len(clip_files)}")

        if len(clip_files) < len(highlight_windows):
            print("Warning: fewer clips than highlight windows")
        elif len(clip_files) > len(highlight_windows):
            print("Warning: more clips than highlight windows")

        selected_clips = []
        selected_rows = []

        for idx, win in enumerate(highlight_windows):
            if idx >= len(clip_files):
                print(f"Missing clip for window index {idx}")
                continue

            if win.get("goal_touched", False):
                selected_clips.append(clip_files[idx])
                selected_rows.append({
                    "window_index": idx,
                    "clip_name": clip_files[idx].name,
                    "event": win.get("event"),
                    "start_frame": win.get("start_frame"),
                    "end_frame": win.get("end_frame"),
                    "trigger_frame": win.get("trigger_frame"),
                    "side": win.get("side"),
                    "goal_touched": win.get("goal_touched"),
                })

        print(f"Selected goal-touched clips: {len(selected_clips)}")

        if not selected_clips:
            raise RuntimeError("No goal-touched clips selected")

        audit_path = clips_dir / "goal_touched_selected_from_windows.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(selected_rows, f, indent=2)

        concat_file = clips_dir / "concat_goal_touched.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for clip in selected_clips:
                f.write(f"file '{clip.resolve().as_posix()}'\n")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            ffmpeg,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-an",
            "-movflags", "+faststart",
            str(output_path)
        ]

        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        print("\nFFMPEG STDERR:\n", result.stderr)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with code {result.returncode}")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Output file missing or empty")

        print(f"Saved output to: {output_path}")
        print(f"Saved audit to: {audit_path}")