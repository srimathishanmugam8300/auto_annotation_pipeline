from __future__ import annotations

from typing import Any

import numpy as np

from app.schemas.domain import DetectionResult, FramePacket


class LabelGenerator:
    def __init__(
        self,
        annotation_format: str,
        categories: list[str],
        segmentation_enabled: bool,
        mask_type: str,
    ) -> None:
        self.annotation_format = annotation_format
        self.categories = categories
        self.segmentation_enabled = segmentation_enabled
        self.mask_type = mask_type
        self._yolo_category_ids = {label: index for index, label in enumerate(categories)}
        self._coco_category_ids = {label: index + 1 for index, label in enumerate(categories)}
        self._coco_images: list[dict[str, Any]] = []
        self._coco_annotations: list[dict[str, Any]] = []
        self._next_annotation_id = 1

    def generate_yolo_lines(self, packet: FramePacket, detections: list[DetectionResult]) -> list[str]:
        frame_height, frame_width = packet.frame.shape[:2]
        lines: list[str] = []

        for detection in detections:
            category_id = self._yolo_category_ids.get(detection.label)
            if category_id is None:
                continue

            polygon = self._best_polygon(detection)
            if self.segmentation_enabled and polygon is not None:
                normalized_points: list[str] = []
                for x_coord, y_coord in polygon:
                    normalized_points.append(f"{x_coord / frame_width:.6f}")
                    normalized_points.append(f"{y_coord / frame_height:.6f}")
                lines.append(" ".join([str(category_id), *normalized_points]))
                continue

            x1, y1, x2, y2 = detection.bbox
            x_center = ((x1 + x2) / 2.0) / frame_width
            y_center = ((y1 + y2) / 2.0) / frame_height
            width = (x2 - x1) / frame_width
            height = (y2 - y1) / frame_height
            lines.append(
                " ".join(
                    [
                        str(category_id),
                        f"{x_center:.6f}",
                        f"{y_center:.6f}",
                        f"{width:.6f}",
                        f"{height:.6f}",
                    ]
                )
            )

        return lines

    def add_coco_image_and_annotations(
        self,
        packet: FramePacket,
        detections: list[DetectionResult],
        image_file: str | None,
    ) -> None:
        frame_height, frame_width = packet.frame.shape[:2]
        image_id = packet.sampled_frame_index + 1
        file_name = image_file or f"images/{packet.stem}.jpg"
        self._coco_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": frame_width,
                "height": frame_height,
                "frame_index": packet.source_frame_index,
                "timestamp_sec": round(packet.timestamp_sec, 6),
            }
        )

        for detection in detections:
            category_id = self._coco_category_ids.get(detection.label)
            if category_id is None:
                continue

            x1, y1, x2, y2 = detection.bbox
            bbox = [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)]
            annotation: dict[str, Any] = {
                "id": self._next_annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": round((x2 - x1) * (y2 - y1), 3),
                "iscrowd": 0,
                "score": round(detection.confidence, 6),
                "attributes": {
                    "track_id": detection.track_id,
                    "mask_type": self.mask_type if self.segmentation_enabled else None,
                },
            }
            polygon_segments = self._all_polygons_as_flat_lists(detection)
            if polygon_segments:
                annotation["segmentation"] = polygon_segments
            self._coco_annotations.append(annotation)
            self._next_annotation_id += 1

    def build_frame_manifest(
        self,
        packet: FramePacket,
        detections: list[DetectionResult],
        image_file: str | None,
        label_file: str | None,
    ) -> dict[str, Any]:
        return {
            "frame_name": packet.stem,
            "source_frame_index": packet.source_frame_index,
            "sampled_frame_index": packet.sampled_frame_index,
            "timestamp_sec": round(packet.timestamp_sec, 6),
            "image_file": image_file,
            "label_file": label_file,
            "detections": [
                {
                    "label": detection.label,
                    "class_id": detection.class_id,
                    "confidence": round(detection.confidence, 6),
                    "bbox": [round(value, 3) for value in detection.bbox],
                    "track_id": detection.track_id,
                    "mask_file": detection.metadata.get("mask_file"),
                    "polygon_count": len(detection.mask.polygons) if detection.mask else 0,
                }
                for detection in detections
            ],
        }

    def finalize(self) -> dict[str, Any] | None:
        if self.annotation_format != "coco":
            return None

        return {
            "images": self._coco_images,
            "annotations": self._coco_annotations,
            "categories": [
                {"id": index + 1, "name": label, "supercategory": "object"}
                for index, label in enumerate(self.categories)
            ],
        }

    @staticmethod
    def _best_polygon(detection: DetectionResult) -> list[tuple[float, float]] | None:
        if detection.mask is None or not detection.mask.polygons:
            return None
        return max(detection.mask.polygons, key=LabelGenerator._polygon_area)

    @staticmethod
    def _all_polygons_as_flat_lists(detection: DetectionResult) -> list[list[float]]:
        if detection.mask is None:
            return []
        segments: list[list[float]] = []
        for polygon in detection.mask.polygons:
            if len(polygon) < 3:
                continue
            flat = [round(coord, 3) for point in polygon for coord in point]
            if len(flat) >= 6:
                segments.append(flat)
        return segments

    @staticmethod
    def _polygon_area(polygon: list[tuple[float, float]]) -> float:
        points = np.asarray(polygon, dtype=np.float32)
        if len(points) < 3:
            return 0.0
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        return float(0.5 * abs(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1))))
