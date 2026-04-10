import re
import subprocess
from pathlib import Path
import imageio_ffmpeg
import sys
sys.path.append('../')
import os



def extract_last_number(filename):
    matches = re.findall(r'(\d+)', Path(filename).stem)
    return int(matches[-1]) if matches else -1


def normalize_video(input_path, output_path, width=1920, height=1080, fps=30):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(input_path),
        "-vf", f"scale={width}:{height},fps={fps}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path)
    ]

    subprocess.run(cmd, check=True)


def concatenate_videos(input_dir, output_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    input_dir = Path(input_dir)

    video_files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() == ".mp4"],
        key=lambda p: extract_last_number(p.name)
    )

    if not video_files:
        raise ValueError("No videos found")

    print("Ordered files:")
    for v in video_files:
        print(v.name)

    # temp folder for normalized clips
    temp_dir = input_dir / "normalized"
    temp_dir.mkdir(exist_ok=True)

    normalized_files = []

    # Step 1 — normalize
    for i, video in enumerate(video_files):
        out_path = temp_dir / f"norm_{i}.mp4"
        print(f"Normalizing: {video.name}")
        normalize_video(video, out_path)
        normalized_files.append(out_path)

    # Step 2 — create concat list
    list_file = temp_dir / "concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for vf in normalized_files:
            f.write(f"file '{vf.as_posix()}'\n")

    # Step 3 — concatenate
    cmd = [
        ffmpeg,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path)
    ]

    subprocess.run(cmd, check=True)

    print(f"\nSaved: {output_path}")




if __name__ == "__main__":
    input_dir = "../output_videos/highlights"
    output_path = '../output_videos/combined/29032026_combined_highlights.mp4'

    concatenate_videos(input_dir, output_path)