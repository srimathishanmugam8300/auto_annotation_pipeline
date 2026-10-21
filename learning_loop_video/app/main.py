from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.routes import router
from app.core.pipeline import PipelineController
from app.services.job_store import JobStore
from app.utils.io import ensure_dir
from app.utils.logging import configure_logging


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dir(OUTPUT_DIR)
    ensure_dir(MODELS_DIR)
    ensure_dir(LOG_DIR)
    configure_logging(LOG_DIR)

    job_store = JobStore(OUTPUT_DIR)
    app.state.output_dir = str(OUTPUT_DIR)
    app.state.job_store = job_store
    app.state.pipeline_controller = PipelineController(MODELS_DIR, OUTPUT_DIR, job_store)
    app.state.background_tasks = set()
    yield


app = FastAPI(
    title="Configurable Video Annotation Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
