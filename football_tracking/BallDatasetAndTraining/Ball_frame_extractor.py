import cv2
import os

def extract_frames(video_path, output_dir, every_n=None, max_frames=None, prefix=None):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            out_path = os.path.join(output_dir, f"{prefix}_{saved_idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_idx += 1

            if max_frames is not None and saved_idx >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"Saved {saved_idx} frames to {output_dir}")

if __name__ == "__main__":
    video_path = "../../input_videos/DJI_20260426191314_0016_D.MP4"
    output_dir = "../../new_video_frames/April_26/Images"
    extract_frames(video_path, output_dir, every_n=10, max_frames=1500, prefix="April26_1_")

