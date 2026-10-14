"""Crash-resumable JSON state for a generation request."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path


def operation_state_path(output_root: Path, kind: str,
                         key_prefix: str | None) -> Path | None:
    """Return a Windows-safe state path for a queued operation, if any."""
    if not key_prefix:
        return None
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key_prefix).strip("._")
    if not safe:
        raise ValueError("key_prefix does not contain a usable state identifier")
    return output_root / "requests" / f"{kind}-{safe}.state.json"


class JsonRunState:
    def load(self, path: Path) -> dict:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"idempotency_key": str(uuid.uuid4()), "jobs": []}

    def save(self, path: Path, state: dict) -> None:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=path.parent,
                    prefix=f".{path.name}.", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(state, temporary, indent=2)
                temporary.flush()
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
