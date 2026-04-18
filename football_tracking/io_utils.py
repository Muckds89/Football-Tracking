import os
import json
import logging

import pickle





class IOUtils:

    @staticmethod
    def merge_event_windows(events, merge_gap_sec=3.0):
        if not events:
            return []

        events = sorted(events, key=lambda e: e["start_time_sec"])
        merged = [events[0].copy()]

        for ev in events[1:]:
            last = merged[-1]

            if ev["start_time_sec"] <= last["end_time_sec"] + merge_gap_sec:
                last["end_frame"] = max(last["end_frame"], ev["end_frame"])
                last["end_time_sec"] = max(last["end_time_sec"], ev["end_time_sec"])
                last["duration_sec"] = last["end_time_sec"] - last["start_time_sec"]
                last["max_hits_in_window"] = max(
                    last.get("max_hits_in_window", 0),
                    ev.get("max_hits_in_window", 0)
                )
                last["goal_touched"] = last.get("goal_touched", False) or ev.get("goal_touched", False)
            else:
                merged.append(ev.copy())

        return merged

    @staticmethod
    def filter_events(events, goal_touched_only=False, allowed_event_names=None):
        filtered = events

        if allowed_event_names is not None:
            filtered = [e for e in filtered if e.get("event") in allowed_event_names]

        if goal_touched_only:
            filtered = [e for e in filtered if e.get("goal_touched") is True]

        return filtered

    @staticmethod
    def save_pickle(obj, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)
        
    @staticmethod
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def save_json(data, path):
        dir_name = os.path.dirname(path)
        if dir_name:
            IOUtils.ensure_dir(dir_name)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()

        logging.info(f"Saved JSON: {path}")

    @staticmethod
    def load_json(path, default=None):
        if default is None:
            default = {}

        if not os.path.exists(path):
            return default

        if os.path.getsize(path) == 0:
            logging.warning(f"JSON file is empty: {path}")
            return default

        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                logging.warning(f"JSON file contains no data: {path}")
                return default
            return json.loads(content)

    @staticmethod
    def file_exists(path):
        return os.path.exists(path)

    @staticmethod
    def list_videos(folder, extensions=(".mp4", ".avi", ".mov")):
        return [
            f for f in os.listdir(folder)
            if f.lower().endswith(extensions)
        ]