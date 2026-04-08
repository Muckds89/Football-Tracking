import os
import json
import logging


class IOUtils:

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