import os
import sys
import subprocess

def convert_video(input_path, output_path):
    command = [
        "ffmpeg",
        "-y",
        "-err_detect", "ignore_err",
        "-i", input_path,
        "-vf", "yadif,scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ac", "2",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Successfully convert: {output_path}")
    except subprocess.CalledProcessError:
        print(f"Skip error file: {input_path}")


def convert_folder(input_folder, output_folder):
    if not os.path.exists(input_folder):
        print(f"Folder input is not exist: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):

        input_path = os.path.join(input_folder, filename)
        if not os.path.isfile(input_path):
            continue

        name_without_ext = os.path.splitext(filename)[0]
        output_filename = name_without_ext + ".mp4"
        output_path = os.path.join(output_folder, output_filename)

        print(f"Converting: {filename}")
        convert_video(input_path, output_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_video.py input_folder output_folder")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    convert_folder(input_folder, output_folder)