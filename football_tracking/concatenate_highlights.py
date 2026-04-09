import os
import re
import cv2

def extract_last_number(filename):
    matches = re.findall(r'(\d+)', filename)
    return int(matches[-1]) if matches else -1


def concatenate_videos(input_dir, output_path):
    video_files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".mp4")
    ]

    if not video_files:
        raise ValueError("No .mp4 files found in directory")

    # Sort by last number in filename
    video_files = sorted(video_files, key=extract_last_number)

    print("Ordered files:")
    for f in video_files:
        print(f)

    first_video_path = os.path.join(input_dir, video_files[0])
    cap = cv2.VideoCapture(first_video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open {first_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # Loop through videos
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        print(f"Adding: {video_file}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Skipping {video_file} (cannot open)")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        cap.release()

    writer.release()
    print(f"\nSaved concatenated video to: {output_path}")


if __name__ == "__main__":
    input_dir = "path/to/highlights"
    output_path = "combined_highlights.mp4"

    concatenate_videos(input_dir, output_path)