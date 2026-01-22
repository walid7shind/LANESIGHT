import subprocess

URL = "https://www.youtube.com/watch?v=Hue3V65zQRQ&t=19431s"
START = "05:24:00"
DURATION = "30"  # seconds
OUT = "test_images/clip.mp4"

cmd = [
    "yt-dlp",
    "-f", "bv*+ba/best",
    "--download-sections", f"*{START}-{DURATION}",
    "--merge-output-format", "mp4",
    "-o", OUT,
    URL
]

subprocess.run(cmd, check=True)
