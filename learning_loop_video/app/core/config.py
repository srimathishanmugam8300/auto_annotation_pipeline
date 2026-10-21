from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: Literal["fps", "interval"]
    value: float = Field(gt=0)
    redundancy_threshold: float = Field(default=0.0, ge=0.0)


class TimeRangeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    start_sec: float = Field(default=0.0, ge=0.0)
    end_sec: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_time_range(self) -> "TimeRangeConfig":
        if self.end_sec is not None and self.end_sec <= self.start_sec:
            raise ValueError("time_range.end_sec must be greater than time_range.start_sec")
        return self


class SegmentationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    model: str = "sam_vit_b"
    mask_type: Literal["binary", "polygon"] = "polygon"
    prompt_mode: Literal["box", "point"] = "box"


class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "yolov8"
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    batch_size: int = Field(default=1, ge=1, le=8)
    max_image_size: int = Field(default=960, ge=320, le=1920)


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    algorithm: Literal["bytetrack"] = "bytetrack"
    persist_ids: bool = True
    lost_track_buffer: int = Field(default=30, ge=1, le=300)
    minimum_matching_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    save_frames: bool = True
    save_labels: bool = True
    save_video: bool = True
    video_codec: str = "h264"
    skip_empty_frames: bool = True


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    video_path: str
    labels: list[str]
    sampling: SamplingConfig
    time_range: TimeRangeConfig
    processing_mode: Literal["dataset", "video", "hybrid"]
    annotation_format: Literal["yolo", "coco"]
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("labels")
    @classmethod
    def normalize_labels(cls, labels: list[str]) -> list[str]:
        normalized = [label.strip().lower() for label in labels if label and label.strip()]
        if not normalized:
            raise ValueError("labels must contain at least one label")
        return list(dict.fromkeys(normalized))

    @field_validator("video_path")
    @classmethod
    def validate_video_path(cls, video_path: str) -> str:
        if not Path(video_path).exists():
            raise ValueError(f"video_path does not exist: {video_path}")
        return video_path

    @property
    def wants_dataset(self) -> bool:
        return self.processing_mode in {"dataset", "hybrid"}

    @property
    def wants_video(self) -> bool:
        return self.processing_mode in {"video", "hybrid"}

