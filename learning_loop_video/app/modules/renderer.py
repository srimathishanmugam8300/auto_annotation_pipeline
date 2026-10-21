from __future__ import annotations

import cv2
import numpy as np

from app.schemas.domain import DetectionResult


class Renderer:
    def render(self, frame: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
        rendered = frame.copy()

        for detection in detections:
            color = self._color_for(detection)
            if detection.mask is not None and detection.mask.binary_mask is not None:
                self._blend_binary_mask(rendered, detection, color)
            elif detection.mask is not None and detection.mask.polygons:
                self._blend_polygon_mask(rendered, detection, color)

            x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)

            label_parts = [detection.label, f"{detection.confidence:.2f}"]
            if detection.track_id is not None:
                label_parts.insert(0, f"ID:{detection.track_id}")
            label = " | ".join(label_parts)
            text_origin = (x1, max(y1 - 8, 12))
            cv2.putText(
                rendered,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        return rendered

    def _blend_binary_mask(self, frame: np.ndarray, detection: DetectionResult, color: tuple[int, int, int]) -> None:
        assert detection.mask is not None and detection.mask.binary_mask is not None
        x_offset, y_offset = detection.mask.bbox_origin
        mask = detection.mask.binary_mask.astype(bool)
        height, width = mask.shape[:2]
        region = frame[y_offset : y_offset + height, x_offset : x_offset + width]
        if region.size == 0:
            return

        blended = region.copy()
        blended[mask] = np.asarray(color, dtype=np.uint8)
        cv2.addWeighted(blended, 0.35, region, 0.65, 0.0, dst=region)

    def _blend_polygon_mask(self, frame: np.ndarray, detection: DetectionResult, color: tuple[int, int, int]) -> None:
        assert detection.mask is not None
        overlay = frame.copy()
        polygons = [np.asarray(polygon, dtype=np.int32) for polygon in detection.mask.polygons if len(polygon) >= 3]
        if not polygons:
            return
        cv2.fillPoly(overlay, polygons, color)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0.0, dst=frame)

    @staticmethod
    def _color_for(detection: DetectionResult) -> tuple[int, int, int]:
        seed = detection.track_id if detection.track_id is not None else detection.class_id + 1
        return (
            int((37 * seed) % 255),
            int((17 * seed + 80) % 255),
            int((29 * seed + 160) % 255),
        )
