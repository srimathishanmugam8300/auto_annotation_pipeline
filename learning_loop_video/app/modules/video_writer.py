from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from app.utils.exceptions import ExportError
from app.utils.io import ensure_dir


CODEC_MAP = {
    "h264": "libx264",
    "hevc": "libx265",
    "h265": "libx265",
    "mpeg4": "mpeg4",
}


class FFmpegVideoWriter:
    def __init__(self, output_path: Path, fps: float, codec: str, logger) -> None:
        self.output_path = output_path
        self.fps = max(fps, 1e-3)
        self.codec = CODEC_MAP.get(codec.lower(), codec)
        self.logger = logger
        self._process: subprocess.Popen | None = None
        self._width: int | None = None
        self._height: int | None = None

    @property
    def is_open(self) -> bool:
        return self._process is not None

    def open(self, width: int, height: int) -> None:
        ensure_dir(self.output_path.parent)
        self._width = width
        self._height = height

        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            self.codec,
            "-pix_fmt",
            "yuv420p",
            str(self.output_path),
        ]

        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ExportError("ffmpeg executable was not found on the system PATH") from exc

    def write(self, frame: np.ndarray) -> None:
        if self._process is None or self._process.stdin is None:
            raise ExportError("Video writer was not opened before writing frames")

        assert self._width is not None and self._height is not None
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
        self._process.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self) -> None:
        if self._process is None:
            return

        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()

        stderr_output = ""
        if self._process.stderr is not None:
            stderr_output = self._process.stderr.read().decode("utf-8", errors="ignore")

        return_code = self._process.wait()
        self._process = None
        if return_code != 0:
            raise ExportError(f"ffmpeg failed to write output video: {stderr_output.strip()}")

    def abort(self) -> None:
        if self._process is None:
            return
        self._process.kill()
        self._process.wait()
        self._process = None
