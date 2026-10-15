"""The claim loop's liveness record, `<data>/worker/status.json`."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .locations import data_dir
from .repository import release_info


def status_path(repository: Path) -> Path:
    return data_dir(repository) / "worker/status.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def status_writer(repository: Path, metadata: Callable[[], dict]) -> Callable[[], None]:
    """Return a function that rewrites the status file atomically."""
    path = status_path(repository)
    started_at = _now()

    def write() -> None:
        git = metadata()
        info = release_info(repository) or {}
        record = {
            "version": info.get("version"),
            "commit": git.get("commit"),
            "ref": git.get("branch"),
            "pid": os.getpid(),
            "started_at": started_at,
            "updated_at": _now(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(temp, path)

    return write
