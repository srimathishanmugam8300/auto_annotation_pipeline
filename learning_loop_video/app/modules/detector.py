from __future__ import annotations

from pathlib import Path
from threading import RLock

import cv2
import numpy as np

from app.core.config import DetectionConfig
from app.schemas.domain import DetectionResult
from app.utils.exceptions import ModelLoadingError, PipelineError


YOLO_MODEL_ALIASES = {
    "yolov8": "yolov8n.pt",
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
}


class YoloDetector:
    _model_cache: dict[str, object] = {}
    _model_names: dict[str, dict[int, str]] = {}
    _cache_lock = RLock()

    def __init__(self, config: DetectionConfig, target_labels: list[str], models_dir: Path, logger) -> None:
        self.config = config
        self.target_labels = {label.lower() for label in target_labels}
        self.models_dir = models_dir
        self.logger = logger
        self._model_ref = self._resolve_model_reference(config.model)

    def _resolve_model_reference(self, raw_model_ref: str) -> str:
        candidate = Path(raw_model_ref)
        if candidate.exists():
            return str(candidate)

        alias = raw_model_ref.lower()
        if alias in YOLO_MODEL_ALIASES:
            local_candidate = self.models_dir / YOLO_MODEL_ALIASES[alias]
            if local_candidate.exists():
                return str(local_candidate)
            return YOLO_MODEL_ALIASES[alias]

        return raw_model_ref

    def _get_model(self):
        with self._cache_lock:
            cached = self._model_cache.get(self._model_ref)
            if cached is not None:
                return cached

        try:
            from ultralytics import YOLO

            model = YOLO(self._model_ref)
            names = getattr(model, "names", {}) or {}
        except Exception as exc:
            raise ModelLoadingError(f"Failed to load YOLO model '{self._model_ref}': {exc}") from exc

        with self._cache_lock:
            self._model_cache[self._model_ref] = model
            self._model_names[self._model_ref] = {int(key): str(value) for key, value in names.items()}
        return model

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[DetectionResult]]:
        if not frames:
            return []

        model = self._get_model()
        prepared_frames: list[np.ndarray] = []
        inverse_scales: list[float] = []

        for frame in frames:
            prepared, inverse_scale = self._prepare_frame(frame)
            prepared_frames.append(prepared)
            inverse_scales.append(inverse_scale)

        try:
            results = model.predict(
                source=prepared_frames,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                device="cpu",
                verbose=False,
            )
        except Exception as exc:
            raise PipelineError(f"YOLO inference failed: {exc}") from exc

        parsed_results: list[list[DetectionResult]] = []
        for frame, result, inverse_scale in zip(frames, results, inverse_scales):
            parsed_results.append(self._parse_result(result, frame.shape[:2], inverse_scale))
        return parsed_results

    def _prepare_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        longest_side = max(height, width)
        if longest_side <= self.config.max_image_size:
            return frame, 1.0

        scale = self.config.max_image_size / float(longest_side)
        resized_width = max(int(width * scale), 1)
        resized_height = max(int(height * scale), 1)
        resized_frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        return resized_frame, 1.0 / scale

    def _parse_result(self, result, frame_shape: tuple[int, int], inverse_scale: float) -> list[DetectionResult]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.empty((0, 4), dtype=np.float32)
        confidences = boxes.conf.cpu().numpy() if len(boxes) else np.empty((0,), dtype=np.float32)
        class_ids = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.empty((0,), dtype=np.int64)
        label_map = getattr(result, "names", None) or self._model_names.get(self._model_ref, {})
        frame_height, frame_width = frame_shape

        detections: list[DetectionResult] = []
        for bbox, confidence, class_id in zip(xyxy, confidences, class_ids):
            label = str(label_map.get(int(class_id), class_id)).lower()
            if self.target_labels and label not in self.target_labels:
                continue

            x1, y1, x2, y2 = (bbox * inverse_scale).tolist()
            x1 = min(max(x1, 0.0), float(frame_width))
            x2 = min(max(x2, 0.0), float(frame_width))
            y1 = min(max(y1, 0.0), float(frame_height))
            y2 = min(max(y2, 0.0), float(frame_height))
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                DetectionResult(
                    label=label,
                    class_id=int(class_id),
                    confidence=float(confidence),
                    bbox=(x1, y1, x2, y2),
                )
            )

        return detections
