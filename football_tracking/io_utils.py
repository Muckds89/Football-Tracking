import os
import json
import logging


class IOUtils:

    @staticmethod
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def save_json(data, path):
        IOUtils.ensure_dir(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved JSON: {path}")

    @staticmethod
    def load_json(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def file_exists(path):
        return os.path.exists(path)

    @staticmethod
    def list_videos(folder, extensions=(".mp4", ".avi", ".mov")):
        return [
            f for f in os.listdir(folder)
            if f.lower().endswith(extensions)
        ]