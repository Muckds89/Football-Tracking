

import os
import json



class ROIManager:
    def __init__(self, roi_dir):
        self.roi_dir = roi_dir

    def get_roi_path(self, video_name):
        base = video_name.rsplit(".", 1)[0]
        return os.path.join(self.roi_dir, f"{base}_rois.json")

    def roi_exists(self, video_name):
        return os.path.exists(self.get_roi_path(video_name))

    def load_rois(self, video_name):
        path = self.get_roi_path(video_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"ROI file not found: {path}")
        with open(path, "r") as f:
            return json.load(f)
    
    def save_rois(self, video_name, rois):
        path = self.get_roi_path(video_name)
        with open(path, "w") as f:
            json.dump(rois, f, indent=2)