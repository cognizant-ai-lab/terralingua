import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.resolve().absolute()


def create_video(
    input_pattern="%05d.png", output_file="video.mp4", fps=10, crf=18, preset="medium"
):
    scale_filter = "scale=500:500:flags=neighbor"
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file without asking
        "-framerate",
        str(fps),
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-preset",
        preset,  # options: ultrafast, fast, medium, slow, slower
        "-crf",
        str(crf),  # lower is better quality (18-28 range typical)
        "-pix_fmt",
        "yuv420p",  # ensures compatibility
        output_file,
        "-vf",
        scale_filter,
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
