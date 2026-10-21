from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from app.schemas.domain import FramePacket, VideoInfo
from app.utils.exceptions import VideoReadError


class VideoLoader:
    def __init__(self, video_path: Path | str, start_sec: float = 0.0, end_sec: float | None = None) -> None:
        self.video_path = Path(video_path)
        self.start_sec = max(start_sec, 0.0)
        self.end_sec = end_sec
        self._capture: cv2.VideoCapture | None = None
        self._video_info: VideoInfo | None = None

    def __enter__(self) -> "VideoLoader":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def video_info(self) -> VideoInfo:
        if self._video_info is None:
            self.open()
        assert self._video_info is not None
        return self._video_info

    def open(self) -> None:
        if self._capture is not None and self._capture.isOpened():
            return

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise VideoReadError(f"Unable to open video: {self.video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 30.0

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if total_frames <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise VideoReadError(f"Video metadata is invalid or corrupt: {self.video_path}")

        duration_sec = total_frames / fps
        start_frame = min(int(self.start_sec * fps), total_frames - 1)
        raw_end_frame = total_frames - 1 if self.end_sec is None else int(self.end_sec * fps)
        end_frame = min(max(raw_end_frame, start_frame), total_frames - 1)

        if not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            capture.release()
            raise VideoReadError(f"Unable to seek to frame {start_frame} in {self.video_path}")

        self._capture = capture
        self._video_info = VideoInfo(
            source_path=self.video_path,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration_sec=duration_sec,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def iter_frames(self) -> Iterator[FramePacket]:
        self.open()
        assert self._capture is not None
        info = self.video_info

        source_frame_index = info.start_frame
        while source_frame_index <= info.end_frame:
            ok, frame = self._capture.read()
            if not ok:
                break

            yield FramePacket(
                source_frame_index=source_frame_index,
                sampled_frame_index=-1,
                timestamp_sec=source_frame_index / info.fps,
                frame=frame,
            )
            source_frame_index += 1

    def iter_frame_batches(self, batch_size: int) -> Iterator[list[FramePacket]]:
        batch: list[FramePacket] = []
        for packet in self.iter_frames():
            batch.append(packet)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
