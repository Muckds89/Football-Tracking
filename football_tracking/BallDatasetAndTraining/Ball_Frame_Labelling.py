import os
import cv2
from pathlib import Path

CLASS_ID = 0  # ball


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_image_files(images_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = []
    for name in sorted(os.listdir(images_dir)):
        p = os.path.join(images_dir, name)
        if os.path.isfile(p) and Path(name).suffix.lower() in exts:
            files.append(p)
    return files


def load_existing_label(label_path):
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()

    if not line:
        return None

    parts = line.split()
    if len(parts) != 5:
        return None

    cls_id, xc, yc, w, h = parts
    return int(cls_id), float(xc), float(yc), float(w), float(h)


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    cx = xc * img_w
    cy = yc * img_h

    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    return x1, y1, x2, y2


def draw_help(canvas, image_name, idx, total):
    help_lines = [
        f"{idx + 1}/{total}  {image_name}",
        "Drag mouse: draw box",
        "s: save label",
        "n: skip image",
        "d: delete label",
        "r: redraw",
        "q: quit",
    ]

    y = 25
    for line in help_lines:
        cv2.putText(
            canvas,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28


def label_images(images_dir="dataset/images", labels_dir="dataset/labels"):
    ensure_dir(labels_dir)

    image_paths = get_image_files(images_dir)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    window_name = "Ball Labeler"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1400, 900)

    drawing = False
    start_pt = None
    current_rect = None

    state = {
        "display": None,
        "base": None,
        "rect": None,
    }

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_pt, current_rect, state

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
            current_rect = None

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x1, y1 = start_pt
            current_rect = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
            state["display"] = state["base"].copy()
            if current_rect is not None:
                rx1, ry1, rx2, ry2 = current_rect
                cv2.rectangle(state["display"], (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = start_pt
            current_rect = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
            state["rect"] = current_rect
            state["display"] = state["base"].copy()
            rx1, ry1, rx2, ry2 = current_rect
            cv2.rectangle(state["display"], (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)

    cv2.setMouseCallback(window_name, mouse_callback)

    idx = 0
    while idx < len(image_paths):
        image_path = image_paths[idx]
        image_name = os.path.basename(image_path)
        label_path = os.path.join(labels_dir, Path(image_name).stem + ".txt")

        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping unreadable image: {image_path}")
            idx += 1
            continue

        h, w = img.shape[:2]
        canvas = img.copy()

        draw_help(canvas, image_name, idx, len(image_paths))

        existing = load_existing_label(label_path)
        rect = None
        if existing is not None:
            _, xc, yc, bw, bh = existing
            rect = yolo_to_xyxy(xc, yc, bw, bh, w, h)
            x1, y1, x2, y2 = rect
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                canvas,
                "existing label",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        state["base"] = canvas
        state["display"] = canvas.copy()
        state["rect"] = rect

        while True:
            cv2.imshow(window_name, state["display"])
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                return

            elif key == ord("r"):
                state["base"] = img.copy()
                draw_help(state["base"], image_name, idx, len(image_paths))
                state["display"] = state["base"].copy()
                state["rect"] = None

            elif key == ord("d"):
                if os.path.exists(label_path):
                    os.remove(label_path)
                    print(f"Deleted label: {label_path}")
                idx += 1
                break

            elif key == ord("n"):
                idx += 1
                break

            elif key == ord("s"):
                rect = state["rect"]
                if rect is None:
                    print("No box drawn. Draw a rectangle first.")
                    continue

                x1, y1, x2, y2 = rect
                if x2 <= x1 or y2 <= y1:
                    print("Invalid box.")
                    continue

                xc, yc, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)

                with open(label_path, "w", encoding="utf-8") as f:
                    f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                print(f"Saved: {label_path}")
                idx += 1
                break

    cv2.destroyAllWindows()
    print("Finished labeling.")


if __name__ == "__main__":
    label_images("../../new_video_frames/April_26_roi/Images", "../../new_video_frames/April_26_roi/labels")