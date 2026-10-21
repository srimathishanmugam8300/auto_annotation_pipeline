from __future__ import annotations

from pathlib import Path
from threading import RLock

from app.schemas.domain import JobState, utc_now
from app.utils.io import ensure_dir, write_json


class JobStore:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self._states: dict[str, JobState] = {}
        self._lock = RLock()

    def create(self, job_id: str, output_dir: Path) -> JobState:
        state = JobState(job_id=job_id, output_dir=str(output_dir))
        with self._lock:
            self._states[job_id] = state
        self._persist(state)
        return state

    def update(self, job_id: str, **changes: object) -> JobState:
        with self._lock:
            state = self.get(job_id)
            data = state.model_dump()
            data.update(changes)
            data["updated_at"] = utc_now()
            updated_state = JobState.model_validate(data)
            self._states[job_id] = updated_state
        self._persist(updated_state)
        return updated_state

    def get(self, job_id: str) -> JobState:
        with self._lock:
            state = self._states.get(job_id)
        if state is not None:
            return state

        state_file = self.output_root / job_id / "job_state.json"
        if state_file.exists():
            state = JobState.model_validate_json(state_file.read_text(encoding="utf-8"))
            with self._lock:
                self._states[job_id] = state
            return state
        raise KeyError(job_id)

    def _persist(self, state: JobState) -> None:
        output_dir = Path(state.output_dir or self.output_root / state.job_id)
        ensure_dir(output_dir)
        write_json(output_dir / "job_state.json", state.model_dump(mode="json"))
