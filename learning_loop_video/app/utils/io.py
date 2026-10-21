from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, BinaryIO


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def copy_stream_to_path(stream: BinaryIO, target: Path, chunk_size: int = 1024 * 1024) -> Path:
    ensure_dir(target.parent)
    with target.open("wb") as destination:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            destination.write(chunk)
    return target


def make_zip_archive(source_dir: Path, target_zip: Path) -> Path:
    ensure_dir(target_zip.parent)
    base_name = target_zip.with_suffix("")
    archive_path = shutil.make_archive(str(base_name), "zip", root_dir=source_dir)
    return Path(archive_path)

