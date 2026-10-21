from __future__ import annotations

from pathlib import Path
from threading import RLock

import cv2
import numpy as np

from app.core.config import SegmentationConfig
from app.schemas.domain import DetectionResult, MaskData
from app.utils.exceptions import ModelLoadingError, PipelineError


SAM_MODEL_ALIASES = {
    "sam_vit_b": ("vit_b", "sam_vit_b_01ec64.pth"),
}


class SamSegmenter:
    _sam_model_cache: dict[str, object] = {}
    _cache_lock = RLock()

    def __init__(self, config: SegmentationConfig, models_dir: Path, logger) -> None:
        self.config = config
        self.models_dir = models_dir
        self.logger = logger
        self._predictor = None
        self._model_type, self._checkpoint = self._resolve_checkpoint()

    def _resolve_checkpoint(self) -> tuple[str, Path]:
        raw_ref = Path(self.config.model)
        if raw_ref.exists():
            return "vit_b", raw_ref

        alias = self.config.model.lower()
        if alias in SAM_MODEL_ALIASES:
            model_type, filename = SAM_MODEL_ALIASES[alias]
            checkpoint = self.models_dir / filename
            return model_type, checkpoint

        raise ModelLoadingError(
            f"Unsupported SAM model reference '{self.config.model}'. "
            "Use a checkpoint path or the 'sam_vit_b' alias."
        )

    def _get_predictor(self):
        if self._predictor is not None:
            return self._predictor

        with self._cache_lock:
            model = self._sam_model_cache.get(str(self._checkpoint))
            if model is None:
                if not self._checkpoint.exists():
                    raise ModelLoadingError(
                        f"SAM checkpoint not found at '{self._checkpoint}'. "
                        "Download the ViT-B weights into the models directory."
                    )

                try:
                    from segment_anything import sam_model_registry

                    model = sam_model_registry[self._model_type](checkpoint=str(self._checkpoint))
                    model.to(device="cpu")
                except Exception as exc:
                    raise ModelLoadingError(f"Failed to load SAM checkpoint '{self._checkpoint}': {exc}") from exc
                self._sam_model_cache[str(self._checkpoint)] = model

        try:
            from segment_anything import SamPredictor

            self._predictor = SamPredictor(model)
        except Exception as exc:
            raise ModelLoadingError(f"Failed to initialize SAM predictor: {exc}") from exc
        return self._predictor

    def segment_frame(self, frame: np.ndarray, detections: list[DetectionResult]) -> list[DetectionResult]:
        if not detections:
            return detections

        predictor = self._get_predictor()
        frame_height, frame_width = frame.shape[:2]

        for detection in detections:
            x1, y1, x2, y2 = self._clip_bbox(detection.bbox, frame_width, frame_height)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            predictor.set_image(crop_rgb)
            crop_height, crop_width = crop.shape[:2]

            try:
                mask = self._predict_crop_mask(predictor, crop_width, crop_height)
            except Exception as exc:
                raise PipelineError(f"SAM segmentation failed: {exc}") from exc

            polygons = self._extract_polygons(mask, x1, y1)
            detection.mask = MaskData(
                binary_mask=mask.astype(bool),
                polygons=polygons,
                bbox_origin=(x1, y1),
                source_size=(frame_width, frame_height),
            )

        return detections

    def _predict_crop_mask(self, predictor, width: int, height: int) -> np.ndarray:
        if self.config.prompt_mode == "point":
            point_coords = np.array([[width / 2.0, height / 2.0]])
            point_labels = np.array([1])
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
        else:
            prompt_box = np.array([0, 0, width - 1, height - 1])
            masks, _, _ = predictor.predict(box=prompt_box, multimask_output=False)

        if masks is None or len(masks) == 0:
            return np.zeros((height, width), dtype=bool)
        return masks[0].astype(bool)

    @staticmethod
    def _clip_bbox(bbox: tuple[float, float, float, float], frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        return (
            max(int(np.floor(x1)), 0),
            max(int(np.floor(y1)), 0),
            min(int(np.ceil(x2)), frame_width),
            min(int(np.ceil(y2)), frame_height),
        )

    @staticmethod
    def _extract_polygons(mask: np.ndarray, x_offset: int, y_offset: int) -> list[list[tuple[float, float]]]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[list[tuple[float, float]]] = []
        for contour in contours:
            if cv2.contourArea(contour) < 8.0:
                continue
            epsilon = 0.002 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            if len(simplified) < 3:
                continue
            polygon = [
                (float(point[0][0] + x_offset), float(point[0][1] + y_offset))
                for point in simplified
            ]
            polygons.append(polygon)
        return polygons
