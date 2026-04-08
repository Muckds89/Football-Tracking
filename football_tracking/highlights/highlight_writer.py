import os
import cv2
import logging
from football_tracking.io_utils import IOUtils
    
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
                cv2.putText(
                    frame_out,
                    f"frame {current_frame}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

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