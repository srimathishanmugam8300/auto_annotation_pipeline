from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.schemas.domain import DetectionResult, FramePacket
from app.utils.exceptions import ExportError
from app.utils.io import ensure_dir, make_zip_archive, write_json


class ExportManager:
    def __init__(self, job_dir: Path) -> None:
        self.job_dir = ensure_dir(job_dir)
        self.input_dir = ensure_dir(self.job_dir / "input")
        self.images_dir = ensure_dir(self.job_dir / "images")
        self.labels_dir = ensure_dir(self.job_dir / "labels")
        self.masks_dir = ensure_dir(self.job_dir / "masks")
        self.video_dir = ensure_dir(self.job_dir / "video")
        self.output_video_path = self.video_dir / "output.mp4"

    def write_config(self, config_payload: dict[str, Any]) -> Path:
        return write_json(self.job_dir / "config_used.json", config_payload)

    def save_frame(self, packet: FramePacket, frame: np.ndarray | None = None) -> str:
        image = packet.frame if frame is None else frame
        target_path = self.images_dir / f"{packet.stem}.jpg"
        if not cv2.imwrite(str(target_path), image):
            raise ExportError(f"Failed to write frame image: {target_path}")
        return self._relative(target_path)

    def save_yolo_labels(self, packet: FramePacket, lines: list[str]) -> str:
        label_path = self.labels_dir / f"{packet.stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")
        return self._relative(label_path)

    def save_binary_masks(self, packet: FramePacket, detections: list[DetectionResult]) -> list[str]:
        saved_paths: list[str] = []
        for index, detection in enumerate(detections):
            if detection.mask is None or detection.mask.binary_mask is None:
                continue
            target_path = self.masks_dir / f"{packet.stem}_{index:02d}.png"
            mask_image = (detection.mask.binary_mask.astype(np.uint8)) * 255
            if not cv2.imwrite(str(target_path), mask_image):
                raise ExportError(f"Failed to write mask image: {target_path}")
            relative_path = self._relative(target_path)
            detection.metadata["mask_file"] = relative_path
            saved_paths.append(relative_path)
        return saved_paths

    def write_annotations_json(self, payload: dict[str, Any]) -> Path:
        return write_json(self.job_dir / "annotations.json", payload)

    def write_tracking_manifest(self, payload: list[dict[str, Any]]) -> Path:
        return write_json(self.job_dir / "tracking_manifest.json", payload)

    def write_metadata(self, payload: dict[str, Any]) -> Path:
        return write_json(self.job_dir / "metadata.json", payload)

    def create_archive(self) -> Path:
        return make_zip_archive(self.job_dir, self.job_dir.parent / f"{self.job_dir.name}.zip")

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.job_dir).as_posix()
