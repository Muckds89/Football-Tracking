

import os
import json
from football_tracking.utils.io_utils import IOUtils




class ROIManager:
    def __init__(self, roi_dir):
        self.roi_dir = roi_dir
    

    def load_rois(self, video_name):
        return IOUtils.load_json(self.get_roi_path(video_name))

    def save_rois(self, video_name, rois):
        IOUtils.save_json(rois, self.get_roi_path(video_name))

    def get_roi_path(self, video_name):
        base = video_name.rsplit(".", 1)[0]
        return os.path.join(self.roi_dir, f"{base}_rois.json")

    def roi_exists(self, video_name):
        return os.path.exists(self.get_roi_path(video_name))

