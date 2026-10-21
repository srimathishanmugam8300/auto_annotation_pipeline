from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.config import TrackingConfig
from app.schemas.domain import DetectionResult


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


@dataclass(slots=True)
class _TrackState:
    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    missed_frames: int = 0


class _SimpleIouTracker:
    def __init__(self, matching_threshold: float, lost_track_buffer: int) -> None:
        self.matching_threshold = matching_threshold
        self.lost_track_buffer = lost_track_buffer
        self._tracks: list[_TrackState] = []
        self._next_track_id = 1

    def update(self, detections: list[DetectionResult]) -> list[DetectionResult]:
        unmatched_tracks = set(range(len(self._tracks)))
        assigned_detections: set[int] = set()

        scored_pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                if track.label != detection.label:
                    continue
                score = _iou(track.bbox, detection.bbox)
                if score >= self.matching_threshold:
                    scored_pairs.append((score, track_index, detection_index))

        for _, track_index, detection_index in sorted(scored_pairs, reverse=True):
            if track_index not in unmatched_tracks or detection_index in assigned_detections:
                continue
            detection = detections[detection_index]
            track = self._tracks[track_index]
            detection.track_id = track.track_id
            track.bbox = detection.bbox
            track.missed_frames = 0
            unmatched_tracks.remove(track_index)
            assigned_detections.add(detection_index)

        for track_index in unmatched_tracks:
            self._tracks[track_index].missed_frames += 1

        self._tracks = [track for track in self._tracks if track.missed_frames <= self.lost_track_buffer]

        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            detection.track_id = self._next_track_id
            self._tracks.append(
                _TrackState(track_id=self._next_track_id, label=detection.label, bbox=detection.bbox)
            )
            self._next_track_id += 1

        return detections

    def age_tracks(self) -> None:
        for track in self._tracks:
            track.missed_frames += 1
        self._tracks = [track for track in self._tracks if track.missed_frames <= self.lost_track_buffer]


class ByteTracker:
    def __init__(self, config: TrackingConfig, frame_rate: float, logger) -> None:
        self.config = config
        self.frame_rate = max(frame_rate, 1.0)
        self.logger = logger
        self.backend_name = "bytetrack"
        self._backend = None
        self._fallback = _SimpleIouTracker(
            matching_threshold=max(config.minimum_matching_threshold, 0.1),
            lost_track_buffer=config.lost_track_buffer,
        )
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        try:
            import supervision as sv

            self._backend = sv.ByteTrack(
                minimum_matching_threshold=self.config.minimum_matching_threshold,
                lost_track_buffer=self.config.lost_track_buffer,
                frame_rate=int(round(self.frame_rate)),
            )
        except Exception as exc:
            self.backend_name = "iou_fallback"
            self.logger.warning("Supervision ByteTrack unavailable, using IOU tracker fallback: %s", exc)
            self._backend = None

    def update(self, detections: list[DetectionResult]) -> list[DetectionResult]:
        if not detections:
            if self._backend is None:
                self._fallback.age_tracks()
            return detections

        if self._backend is None:
            tracked = self._fallback.update(detections)
        else:
            tracked = self._update_with_supervision(detections)

        if not self.config.persist_ids:
            for detection in tracked:
                detection.track_id = None
        return tracked

    def _update_with_supervision(self, detections: list[DetectionResult]) -> list[DetectionResult]:
        import supervision as sv

        xyxy = np.asarray([detection.bbox for detection in detections], dtype=np.float32)
        confidence = np.asarray([detection.confidence for detection in detections], dtype=np.float32)
        class_id = np.asarray([detection.class_id for detection in detections], dtype=np.int32)
        sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

        tracked = self._backend.update_with_detections(sv_detections)
        tracker_ids = getattr(tracked, "tracker_id", None)
        tracked_boxes = getattr(tracked, "xyxy", None)
        tracked_classes = getattr(tracked, "class_id", None)
        if tracker_ids is None or tracked_boxes is None:
            return detections

        unmatched = set(range(len(detections)))
        for tracked_box, tracked_class, tracked_id in zip(tracked_boxes, tracked_classes, tracker_ids):
            if tracked_id is None:
                continue
            best_match_index = None
            best_iou = 0.0
            for detection_index in unmatched:
                detection = detections[detection_index]
                if int(tracked_class) != int(detection.class_id):
                    continue
                score = _iou(tuple(map(float, tracked_box.tolist())), detection.bbox)
                if score > best_iou:
                    best_iou = score
                    best_match_index = detection_index
            if best_match_index is not None and best_iou >= 0.3:
                detections[best_match_index].track_id = int(tracked_id)
                unmatched.remove(best_match_index)

        return detections
