from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class MaskData:
    binary_mask: np.ndarray | None = None
    polygons: list[list[tuple[float, float]]] = field(default_factory=list)
    bbox_origin: tuple[int, int] = (0, 0)
    source_size: tuple[int, int] | None = None


@dataclass(slots=True)
class DetectionResult:
    label: str
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None
    mask: MaskData | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


@dataclass(slots=True)
class FramePacket:
    source_frame_index: int
    sampled_frame_index: int
    timestamp_sec: float
    frame: np.ndarray

    @property
    def stem(self) -> str:
        return f"frame_{self.sampled_frame_index:06d}"


@dataclass(slots=True)
class VideoInfo:
    source_path: Path
    fps: float
    width: int
    height: int
    total_frames: int
    duration_sec: float
    start_frame: int
    end_frame: int

    @property
    def sliced_frame_count(self) -> int:
        return max(self.end_frame - self.start_frame + 1, 0)

    @property
    def sliced_duration_sec(self) -> float:
        return max((self.end_frame - self.start_frame + 1) / max(self.fps, 1e-6), 0.0)


class JobState(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    progress: float = 0.0
    message: str = "Queued"
    error: str | None = None
    output_dir: str | None = None
    archive_path: str | None = None
    metadata_path: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

