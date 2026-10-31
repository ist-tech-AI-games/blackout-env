"""
Record the Unity window from the X display (Xvfb) with ffmpeg x11grab.

Run the env at time_scale 1 while recording: at higher scales the game runs faster
than the capture rate and the video skips frames.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # static build with x11grab, for hosts without ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`")


class ScreenRecorder:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        display = os.environ.get("DISPLAY")
        if not display:
            raise RuntimeError("DISPLAY is not set: run under xvfb-run to record")
        self.display = display
        self.width, self.height, self.fps = width, height, fps
        self.ffmpeg = find_ffmpeg()
        self._proc: subprocess.Popen | None = None

    def start(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            [self.ffmpeg, "-loglevel", "error", "-y",
             "-f", "x11grab", "-video_size", f"{self.width}x{self.height}",
             "-framerate", str(self.fps), "-i", self.display,
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out_path)],
            stdin=subprocess.PIPE,
        )

    def stop(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            proc.communicate(b"q", timeout=30)  # 'q' lets ffmpeg finalize the mp4
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
