"""Where the reference models sit on the worker.

`manifests/worker-models.toml` puts each one under ComfyUI's `models/` at the
folders named here; `CHAGAMA_REFERENCE_MODELS` points elsewhere (a Mac run).
"""

from __future__ import annotations

import os
from pathlib import Path

MODELS_ENV = "CHAGAMA_REFERENCE_MODELS"
TAGGER_HEAD = "reference/anima-tagger"
TAGGER_TRUNK = "reference/caformer_b36.dbv4-full"
VERIFIER = "reference/d1-3B"
DWPOSE = "reference/dwpose"


class ReferenceUnavailable(RuntimeError):
    """A reference model or its Python packages are not on this machine."""


def models_dir() -> Path:
    override = os.environ.get(MODELS_ENV, "").strip()
    if override:
        return Path(override)
    try:
        import folder_paths
    except ImportError as error:
        raise ReferenceUnavailable(
            f"{MODELS_ENV} is not set and ComfyUI's folder_paths is not importable"
        ) from error
    return Path(folder_paths.models_dir)


def model_path(folder: str, required: tuple[str, ...]) -> Path:
    path = models_dir() / folder
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ReferenceUnavailable(
            f"{path} lacks {', '.join(missing)}; run the fetch worker models workflow")
    return path
