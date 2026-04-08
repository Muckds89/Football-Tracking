import cv2
import json
from typing import Dict, List, Tuple


class ROIUtils:
    def __init__(self, roi_names: List[str]):
        self.roi_names = roi_names
        self.current_points: List[Tuple[int, int]] = []
        self.rois: Dict[str, List[Tuple[int, int]]] = {}
        self.current_roi_index = 0

    def draw_polygon(self, img, points, color=(0, 255, 0), closed=False):
        for pt in points:
            cv2.circle(img, pt, 4, (0, 0, 255), -1)

        for i in range(1, len(points)):
            cv2.line(img, points[i - 1], points[i], color, 2)

        if closed and len(points) > 2:
            cv2.line(img, points[-1], points[0], color, 2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))

    def save_rois_to_json(self, output_path: str):
        serializable = {
            roi_name: [list(pt) for pt in polygon]
            for roi_name, polygon in self.rois.items()
        }
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

    def annotate_rois_local(
        self,
        image_path: str,
        output_json: str = "rois.json",
        display_width: int = 1280,
        display_height: int = 720,
    ):
        self.current_points = []
        self.rois = {}
        self.current_roi_index = 0

        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        cv2.namedWindow("Draw ROIs", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Draw ROIs", display_width, display_height)
        cv2.setMouseCallback("Draw ROIs", self.mouse_callback)

        while True:
            display = frame.copy()

            # Draw already saved ROIs in green
            for roi_name, polygon in self.rois.items():
                self.draw_polygon(display, polygon, color=(0, 255, 0), closed=True)
                if polygon:
                    cv2.putText(
                        display,
                        roi_name,
                        polygon[0],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            # Draw current ROI in blue
            self.draw_polygon(display, self.current_points, color=(255, 0, 0), closed=False)

            if self.current_roi_index < len(self.roi_names):
                current_name = self.roi_names[self.current_roi_index]
                text1 = f"Drawing: {current_name}"
                text2 = "Left click=add point | n=next ROI | u=undo | r=reset current | s=save all | q=quit"
            else:
                text1 = "All ROIs completed"
                text2 = "Press s to save or q to quit"

            cv2.putText(display, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, text2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow("Draw ROIs", display)
            key = cv2.waitKey(20) & 0xFF

            # Undo last point
            if key == ord("u"):
                if self.current_points:
                    self.current_points.pop()

            # Reset current ROI points
            elif key == ord("r"):
                self.current_points = []

            # Save current ROI and move to next one
            elif key == ord("n"):
                if self.current_roi_index < len(self.roi_names):
                    if len(self.current_points) < 2:
                        print(f"ROI '{self.roi_names[self.current_roi_index]}' needs at least 2 points.")
                        continue

                    roi_name = self.roi_names[self.current_roi_index]
                    self.rois[roi_name] = self.current_points.copy()
                    print(f"Saved ROI: {roi_name} -> {self.current_points}")

                    self.current_points = []
                    self.current_roi_index += 1

                    if self.current_roi_index == len(self.roi_names):
                        print("All ROIs have been annotated. Press 's' to save.")

            # Save JSON
            elif key == ord("s"):
                # optionally include current ROI before saving
                if self.current_roi_index < len(self.roi_names) and len(self.current_points) >= 2:
                    roi_name = self.roi_names[self.current_roi_index]
                    self.rois[roi_name] = self.current_points.copy()
                    print(f"Saved ROI: {roi_name} -> {self.current_points}")
                    self.current_points = []
                    self.current_roi_index += 1

                self.save_rois_to_json(output_json)
                print(f"Saved all ROIs to {output_json}")

            # Quit
            elif key == ord("q"):
                break

        cv2.destroyAllWindows()
        return self.rois