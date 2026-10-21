from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from app.core.pipeline import PipelineController
from app.services.job_store import JobStore
from app.utils.io import copy_stream_to_path


router = APIRouter()


def _get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def _get_pipeline_controller(request: Request) -> PipelineController:
    return request.app.state.pipeline_controller


@router.post("/process-video", status_code=status.HTTP_202_ACCEPTED)
async def process_video(
    request: Request,
    config_json: str = Form(...),
    video: UploadFile | None = File(default=None),
):
    try:
        config_payload = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {exc}") from exc

    job_id = str(uuid.uuid4())
    output_root = Path(request.app.state.output_dir)
    job_dir = output_root / job_id
    job_store = _get_job_store(request)
    pipeline_controller = _get_pipeline_controller(request)

    if video is not None:
        job_store.create(job_id, job_dir)
        filename = Path(video.filename or "uploaded_video.mp4").name
        saved_video_path = copy_stream_to_path(video.file, job_dir / "input" / filename)
        config_payload["video_path"] = str(saved_video_path)
    elif not config_payload.get("video_path"):
        raise HTTPException(status_code=400, detail="Either upload a video file or provide video_path in config")
    else:
        job_store.create(job_id, job_dir)

    background_task = asyncio.create_task(asyncio.to_thread(pipeline_controller.run_job, job_id, config_payload))
    request.app.state.background_tasks.add(background_task)
    background_task.add_done_callback(request.app.state.background_tasks.discard)

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/status/{job_id}",
        "download_url": f"/download/{job_id}",
    }


@router.get("/status/{job_id}")
async def get_status(job_id: str, request: Request):
    job_store = _get_job_store(request)
    try:
        state = job_store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}") from exc
    return state.model_dump(mode="json")


@router.get("/download/{job_id}")
async def download_job(job_id: str, request: Request):
    job_store = _get_job_store(request)
    try:
        state = job_store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}") from exc

    if state.status != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed yet")

    output_root = Path(request.app.state.output_dir)
    export_archive_path = output_root / f"{job_id}.zip"

    if not export_archive_path.exists():
        from app.modules.export_manager import ExportManager

        manager = ExportManager(output_root / job_id)
        export_archive_path = await asyncio.to_thread(manager.create_archive)
        job_store.update(job_id, archive_path=str(export_archive_path))

    return FileResponse(
        path=export_archive_path,
        media_type="application/zip",
        filename=f"{job_id}.zip",
    )
