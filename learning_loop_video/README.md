# Configurable Video Annotation Pipeline

This project is a production-oriented FastAPI service for configurable video annotation, segmentation, tracking, and export. It is designed for CPU-only execution, lazy model loading, frame streaming, and runtime behavior driven by a user-supplied JSON configuration.

## Folder Structure

```text
project_root/
+-- app/
|   +-- main.py
|   +-- api/
|   |   +-- routes.py
|   +-- core/
|   |   +-- config.py
|   |   +-- pipeline.py
|   +-- modules/
|   |   +-- detector.py
|   |   +-- export_manager.py
|   |   +-- frame_sampler.py
|   |   +-- label_generator.py
|   |   +-- renderer.py
|   |   +-- segmenter.py
|   |   +-- tracker.py
|   |   +-- video_loader.py
|   |   +-- video_writer.py
|   +-- schemas/
|   |   +-- domain.py
|   +-- services/
|   |   +-- job_store.py
|   +-- utils/
|       +-- exceptions.py
|       +-- io.py
|       +-- logging.py
+-- configs/
|   +-- sample_config.json
+-- logs/
+-- models/
|   +-- README.md
+-- outputs/
+-- requirements.txt
+-- README.md
```

## Runtime Design

- API layer: FastAPI handles job submission, status lookup, health checks, and artifact download.
- Orchestration layer: `PipelineController` validates config, builds the execution graph, and coordinates output routing.
- Processing modules: detector, segmenter, tracker, renderer, label generator, writer, and exporter are isolated for testing and swapping.
- Async execution: jobs run in background worker threads so the API stays responsive.
- CPU-safe flow: frames are streamed, sampled, processed in tiny batches, and released immediately.

## Supported Outputs

- Dataset mode: image frames, YOLO label files or `annotations.json`, `tracking_manifest.json`, `metadata.json`
- Video mode: annotated `output.mp4`, `metadata.json`
- Hybrid mode: both sets together

## Local Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Place model weights according to `models/README.md`.
4. Start the API:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Usage

### Submit a job

Use multipart form-data so the service can accept both the uploaded video and the JSON config in a single request.

```powershell
$config = Get-Content configs\sample_config.json -Raw
curl.exe -X POST "http://localhost:8000/process-video" `
  -F "config_json=$config" `
  -F "video=@C:\path\to\input.mp4"
```

If you do not upload a file, set `video_path` in the JSON to an accessible local path.

### Poll job status

```powershell
curl.exe "http://localhost:8000/status/<job_id>"
```

### Download artifacts

```powershell
curl.exe -OJ "http://localhost:8000/download/<job_id>"
```

## Output Layout Per Job

Each job writes to `outputs/<job_id>/`:

- `input/`: uploaded source video when applicable
- `images/`: exported sampled frames
- `labels/`: YOLO labels when enabled
- `masks/`: binary masks when binary segmentation output is enabled
- `video/output.mp4`: annotated video when enabled
- `annotations.json`: COCO export when enabled
- `tracking_manifest.json`: per-frame detection/tracking manifest
- `metadata.json`: frame counts, timing, config snapshot, and model details
- `job_state.json`: API-facing lifecycle state

## Notes on CPU-Only Constraints

- The pipeline never loads the full video into memory.
- Detection supports small-batch inference; the default batch size is conservative for 8 GB RAM hosts.
- SAM only runs on cropped detections, not full frames.
- Empty frames can be skipped with `output.skip_empty_frames`.
- Model instances are loaded lazily and cached per process.

## Scaling to GPU or Cloud

- GPU: pass GPU-capable weights, switch model device handling inside detector/segmenter adapters, and increase batch size via config.
- Queueing: replace the in-memory `JobStore` with Redis plus Celery, Dramatiq, or Arq.
- Storage: move `outputs/` to object storage such as S3, GCS, or Azure Blob and return signed download URLs.
- Distributed processing: split long videos into temporal shards and merge manifests/videos downstream.
- Multi-video batch execution: introduce a batch job schema that fans out to one `PipelineController` run per asset.

## Implementation Notes

- ByteTrack uses `supervision` when available and falls back to an IOU tracker if the dependency is unavailable at runtime.
- YOLO aliases resolve to common Ultralytics weights, while SAM ViT-B expects a local checkpoint in `models/`.
- The config schema is strict for the required contract but allows extra keys for forward-compatible tuning fields.
