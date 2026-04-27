import os
import cv2
import json
import time
import logging
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from football_tracking.roi.roi_drawer_local import ROIUtils
from football_tracking.io_utils import IOUtils


def get_last_image_path(frames_dir):
    image_files = [
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        raise FileNotFoundError(f"No images found in: {frames_dir}")

    image_files = sorted(image_files)
    return os.path.join(frames_dir, image_files[-1])


def normalize_roi_polygon(rois, roi_name="mask_roi"):
    polygon = rois[roi_name]

    if isinstance(polygon, dict):
        polygon = polygon.get("points", polygon.get("polygon"))

    polygon = np.array(polygon, dtype=np.int32)

    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError(f"Invalid ROI polygon shape: {polygon.shape}")

    return polygon


def get_or_draw_roi_from_frames(
    frames_dir,
    roi_output_path,
    roi_name="mask_roi",
):
    if os.path.exists(roi_output_path):
        with open(roi_output_path, "r") as f:
            data = json.load(f)

        polygon = np.array(data[roi_name], dtype=np.int32)
        logging.info(f"Loaded ROI from {roi_output_path}")
        return polygon

    reference_image_path = get_last_image_path(frames_dir)

    logging.info(f"Drawing ROI on: {reference_image_path}")

    rois = ROIUtils([roi_name]).annotate_rois_local(reference_image_path)

    polygon = normalize_roi_polygon(rois, roi_name)

    IOUtils.ensure_dir(os.path.dirname(roi_output_path))

    with open(roi_output_path, "w") as f:
        json.dump({roi_name: polygon.tolist()}, f, indent=2)

    logging.info(f"Saved ROI to {roi_output_path}")

    return polygon


def apply_roi_mask(image, polygon):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return cv2.bitwise_and(image, image, mask=mask)


def crop_to_polygon_bbox(image, polygon):
    x, y, w, h = cv2.boundingRect(polygon)
    cropped = image[y:y + h, x:x + w]
    return cropped, x, y, w, h


def crop_frames_with_roi(
    frames_dir,
    output_frames_dir,
    roi_output_path,
    roi_name="mask_roi",
    apply_mask=True,
):
    start_time = time.time()

    polygon = get_or_draw_roi_from_frames(
        frames_dir=frames_dir,
        roi_output_path=roi_output_path,
        roi_name=roi_name,
    )

    IOUtils.ensure_dir(output_frames_dir)

    image_files = [
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_files = sorted(image_files)

    logging.info(f"Cropping {len(image_files)} images")

    for i, filename in enumerate(image_files, 1):
        img_path = os.path.join(frames_dir, filename)
        out_path = os.path.join(output_frames_dir, filename)

        image = cv2.imread(img_path)

        if image is None:
            logging.warning(f"Could not read image: {img_path}")
            continue

        if apply_mask:
            image = apply_roi_mask(image, polygon)

        cropped, crop_x, crop_y, crop_w, crop_h = crop_to_polygon_bbox(
            image,
            polygon,
        )

        cv2.imwrite(out_path, cropped)

        if i % 500 == 0:
            logging.info(f"Processed {i}/{len(image_files)}")

    logging.info(f"Done in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    frames_dir = "../../new_video_frames/April_26/Images"

    output_frames_dir = "../../new_video_frames/April_26_roi"

    roi_output_path = "../../roi/April_26_mask_roi.json"

    crop_frames_with_roi(
        frames_dir=frames_dir,
        output_frames_dir=output_frames_dir,
        roi_output_path=roi_output_path,
        roi_name="mask_roi",
        apply_mask=True,
    )