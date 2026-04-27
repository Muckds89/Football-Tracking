import cv2
import numpy as np


class ROICropper:
    @staticmethod
    def normalize_polygon(rois, roi_name="ball_crop_roi"):
        polygon = rois[roi_name]

        if isinstance(polygon, dict):
            polygon = polygon.get("points", polygon.get("polygon"))

        polygon = np.array(polygon, dtype=np.int32)

        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError(f"Invalid ROI polygon shape for {roi_name}: {polygon.shape}")

        return polygon

    @staticmethod
    def apply_mask(frame, polygon):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return cv2.bitwise_and(frame, frame, mask=mask)

    @staticmethod
    def crop_frame(frame, polygon, apply_mask=True):
        if apply_mask:
            frame = ROICropper.apply_mask(frame, polygon)

        x, y, w, h = cv2.boundingRect(polygon)
        cropped = frame[y:y + h, x:x + w]

        return cropped, x, y, w, h

    @staticmethod
    def box_to_full_frame(box, crop_x, crop_y):
        if box is None:
            return None

        x1, y1, x2, y2 = box

        return [
            float(x1 + crop_x),
            float(y1 + crop_y),
            float(x2 + crop_x),
            float(y2 + crop_y),
        ]