import subprocess
from pathlib import Path


def visualize_profile(fp: Path):
    # run snakeviz in subprocess
    if fp.is_file():
        subprocess.run(["snakeviz", fp.as_posix()], check=True)
    else:
        print(f"File {fp} does not exist. Skipping visualization.")
