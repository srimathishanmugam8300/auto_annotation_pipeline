from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.core.config import PipelineConfig
from app.modules.detector import YoloDetector
from app.modules.export_manager import ExportManager
from app.modules.frame_sampler import FrameSampler
from app.modules.label_generator import LabelGenerator
from app.modules.renderer import Renderer
from app.modules.segmenter import SamSegmenter
from app.modules.tracker import ByteTracker
from app.modules.video_loader import VideoLoader
from app.modules.video_writer import FFmpegVideoWriter
from app.services.job_store import JobStore
from app.utils.logging import get_logger


class PipelineController:
    def __init__(self, models_dir: Path, output_dir: Path, job_store: JobStore) -> None:
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.job_store = job_store
        self.logger = get_logger(self.__class__.__name__)

    def run_job(self, job_id: str, config_payload: dict[str, Any]) -> None:
        try:
            self.process_job(job_id, config_payload)
            metadata_path = self.output_dir / job_id / "metadata.json"
            self.job_store.update(
                job_id,
                status="completed",
                progress=1.0,
                message="Processing completed",
                metadata_path=str(metadata_path),
            )
        except Exception as exc:
            self.logger.exception("Job %s failed", job_id)
            self.job_store.update(
                job_id,
                status="failed",
                progress=1.0,
                message="Processing failed",
                error=str(exc),
            )

    def process_job(self, job_id: str, config_payload: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        config = PipelineConfig.model_validate(config_payload)
        job_dir = self.output_dir / job_id
        export_manager = ExportManager(job_dir)
        export_manager.write_config(config.model_dump(mode="json"))

        self.job_store.update(job_id, status="running", progress=0.02, message="Initializing pipeline")

        video_writer: FFmpegVideoWriter | None = None
        output_fps = 0.0
        tracking_manifest: list[dict[str, Any]] = []

        stats = {
            "sampled_frames": 0,
            "frames_with_detections": 0,
            "rendered_frames": 0,
            "saved_frames": 0,
            "saved_label_files": 0,
            "detections": 0,
        }

        with VideoLoader(
            config.video_path,
            start_sec=config.time_range.start_sec,
            end_sec=config.time_range.end_sec,
        ) as video_loader:
            video_info = video_loader.video_info
            frame_sampler = FrameSampler(config.sampling, source_fps=video_info.fps)
            output_fps = frame_sampler.output_fps()
            detector = YoloDetector(config.detection, config.labels, self.models_dir, self.logger)
            segmenter = SamSegmenter(config.segmentation, self.models_dir, self.logger) if config.segmentation.enabled else None
            tracker = ByteTracker(config.tracking, output_fps, self.logger) if config.tracking.enabled else None
            label_generator = LabelGenerator(
                annotation_format=config.annotation_format,
                categories=config.labels,
                segmentation_enabled=config.segmentation.enabled,
                mask_type=config.segmentation.mask_type,
            )
            renderer = Renderer() if config.wants_video and config.output.save_video else None

            if renderer is not None:
                video_writer = FFmpegVideoWriter(
                    output_path=export_manager.output_video_path,
                    fps=output_fps,
                    codec=config.output.video_codec,
                    logger=self.logger,
                )

            buffer: list[Any] = []
            try:
                for packet in frame_sampler.sample(video_loader.iter_frames()):
                    buffer.append(packet)
                    if len(buffer) >= config.detection.batch_size:
                        self._flush_batch(
                            batch=buffer,
                            config=config,
                            detector=detector,
                            segmenter=segmenter,
                            tracker=tracker,
                            label_generator=label_generator,
                            renderer=renderer,
                            video_writer=video_writer,
                            export_manager=export_manager,
                            video_info=video_info,
                            tracking_manifest=tracking_manifest,
                            stats=stats,
                            job_id=job_id,
                        )
                        buffer = []

                if buffer:
                    self._flush_batch(
                        batch=buffer,
                        config=config,
                        detector=detector,
                        segmenter=segmenter,
                        tracker=tracker,
                        label_generator=label_generator,
                        renderer=renderer,
                        video_writer=video_writer,
                        export_manager=export_manager,
                        video_info=video_info,
                        tracking_manifest=tracking_manifest,
                        stats=stats,
                        job_id=job_id,
                    )
            except Exception:
                if video_writer is not None:
                    video_writer.abort()
                raise
            else:
                if video_writer is not None:
                    video_writer.close()

            annotations_path = None
            if config.annotation_format == "coco" and config.output.save_labels:
                coco_payload = label_generator.finalize() or {}
                annotations_path = export_manager.write_annotations_json(coco_payload)
                stats["saved_label_files"] += 1

            tracking_manifest_path = export_manager.write_tracking_manifest(tracking_manifest)
            metadata = {
                "job_id": job_id,
                "status": "completed",
                "frame_count": stats["sampled_frames"],
                "frames_with_detections": stats["frames_with_detections"],
                "rendered_frames": stats["rendered_frames"],
                "saved_frames": stats["saved_frames"],
                "saved_label_files": stats["saved_label_files"],
                "detection_count": stats["detections"],
                "processing_time_sec": round(time.perf_counter() - started_at, 3),
                "config": config.model_dump(mode="json"),
                "model_details": {
                    "detection_model": config.detection.model,
                    "segmentation_model": config.segmentation.model if config.segmentation.enabled else None,
                    "tracking_algorithm": tracker.backend_name if tracker is not None else None,
                },
                "video": {
                    "source_path": str(video_info.source_path),
                    "source_fps": round(video_info.fps, 6),
                    "output_fps": round(output_fps, 6),
                    "width": video_info.width,
                    "height": video_info.height,
                    "total_frames": video_info.total_frames,
                    "slice_start_frame": video_info.start_frame,
                    "slice_end_frame": video_info.end_frame,
                },
                "outputs": {
                    "job_dir": str(job_dir),
                    "annotations_json": str(annotations_path) if annotations_path else None,
                    "tracking_manifest": str(tracking_manifest_path),
                    "output_video": str(export_manager.output_video_path) if export_manager.output_video_path.exists() else None,
                },
            }
            export_manager.write_metadata(metadata)
            return metadata

    def _flush_batch(
        self,
        batch: list[Any],
        config: PipelineConfig,
        detector: YoloDetector,
        segmenter: SamSegmenter | None,
        tracker: ByteTracker | None,
        label_generator: LabelGenerator,
        renderer: Renderer | None,
        video_writer: FFmpegVideoWriter | None,
        export_manager: ExportManager,
        video_info,
        tracking_manifest: list[dict[str, Any]],
        stats: dict[str, int],
        job_id: str,
    ) -> None:
        frames = [packet.frame for packet in batch]
        detections_per_frame = detector.detect_batch(frames)

        for packet, detections in zip(batch, detections_per_frame):
            if segmenter is not None and detections:
                detections = segmenter.segment_frame(packet.frame, detections)

            if tracker is not None:
                detections = tracker.update(detections)

            has_detections = bool(detections)
            if has_detections:
                stats["frames_with_detections"] += 1
                stats["detections"] += len(detections)

            should_emit_frame = has_detections or not config.output.skip_empty_frames
            image_file = None
            label_file = None

            if config.segmentation.enabled and config.segmentation.mask_type == "binary" and detections:
                export_manager.save_binary_masks(packet, detections)

            if config.wants_dataset and should_emit_frame:
                if config.output.save_frames:
                    image_file = export_manager.save_frame(packet)
                    stats["saved_frames"] += 1

                if config.output.save_labels:
                    if config.annotation_format == "yolo":
                        yolo_lines = label_generator.generate_yolo_lines(packet, detections)
                        label_file = export_manager.save_yolo_labels(packet, yolo_lines)
                        stats["saved_label_files"] += 1
                    else:
                        label_generator.add_coco_image_and_annotations(packet, detections, image_file)

            if should_emit_frame or detections:
                tracking_manifest.append(
                    label_generator.build_frame_manifest(packet, detections, image_file=image_file, label_file=label_file)
                )

            if video_writer is not None and renderer is not None and should_emit_frame:
                if not video_writer.is_open:
                    video_writer.open(video_info.width, video_info.height)
                rendered_frame = renderer.render(packet.frame, detections)
                video_writer.write(rendered_frame)
                stats["rendered_frames"] += 1

            stats["sampled_frames"] += 1
            progress = min(
                (packet.source_frame_index - video_info.start_frame + 1) / max(video_info.sliced_frame_count, 1),
                0.99,
            )
            self.job_store.update(
                job_id,
                status="running",
                progress=progress,
                message=f"Processed sampled frame {packet.sampled_frame_index}",
            )
