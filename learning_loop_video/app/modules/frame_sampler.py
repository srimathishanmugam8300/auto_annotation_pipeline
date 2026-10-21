from __future__ import annotations

from collections.abc import Iterable, Iterator

import cv2
import numpy as np

from app.core.config import SamplingConfig
from app.schemas.domain import FramePacket


class FrameSampler:
    def __init__(self, config: SamplingConfig, source_fps: float) -> None:
        self.config = config
        self.source_fps = max(source_fps, 1e-6)

    @property
    def interval_sec(self) -> float:
        if self.config.mode == "fps":
            return 1.0 / max(self.config.value, 1e-6)
        return self.config.value

    def output_fps(self) -> float:
        if self.config.mode == "fps":
            return min(self.config.value, self.source_fps)
        return min(max(1.0 / max(self.config.value, 1e-6), 1e-6), self.source_fps)

    def estimate_sample_count(self, source_frame_count: int) -> int:
        if source_frame_count <= 0:
            return 0
        duration = source_frame_count / self.source_fps
        return max(int(duration / self.interval_sec), 1)

    def sample(self, frames: Iterable[FramePacket]) -> Iterator[FramePacket]:
        next_due_sec = 0.0
        sampled_index = 0
        last_signature: np.ndarray | None = None

        for packet in frames:
            if packet.timestamp_sec + 1e-9 < next_due_sec:
                continue

            if self.config.redundancy_threshold > 0:
                current_signature = self._signature(packet.frame)
                if last_signature is not None:
                    mean_abs_diff = float(
                        np.mean(np.abs(current_signature.astype(np.int16) - last_signature.astype(np.int16)))
                    )
                    if mean_abs_diff < self.config.redundancy_threshold:
                        continue
                last_signature = current_signature

            packet.sampled_frame_index = sampled_index
            sampled_index += 1
            yield packet
            next_due_sec = packet.timestamp_sec + self.interval_sec

    @staticmethod
    def _signature(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
