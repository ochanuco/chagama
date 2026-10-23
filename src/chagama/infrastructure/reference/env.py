"""The reference analysis env and the subprocess that runs in it.

The Anima Tagger and d1 need packages ComfyUI's Python must not be given, so
the worker runs them in their own env, which `scripts/worker/reference-env.ps1`
builds at `<data>/reference-env/<key>`. `<key>` is the same hash of the
requirements files on both sides, so a release only finds an env built for it.
`CHAGAMA_REFERENCE_PYTHON` names any other interpreter (a Mac run).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..locations import data_dir
from .models import MODELS_ENV, ReferenceUnavailable

PYTHON_ENV = "CHAGAMA_REFERENCE_PYTHON"
REQUIREMENTS = ("scripts/worker/reference-env/requirements.txt",
                "scripts/worker/reference-env/requirements-standalone.txt")
TIMEOUT_SECONDS = 15 * 60
STDERR_TAIL = 2000
ENTRY = ("import sys; sys.path.insert(0, sys.argv[1]); "
         "from chagama.infrastructure.reference.analyze import main; "
         "main(sys.argv[2:])")


def env_key(repository: Path) -> str:
    digest = hashlib.sha256(b"".join(
        (repository / name).read_bytes() for name in REQUIREMENTS))
    return digest.hexdigest()[:12]


def env_python(repository: Path) -> Path:
    override = os.environ.get(PYTHON_ENV, "").strip()
    if override:
        return Path(override)
    env = data_dir(repository) / "reference-env" / env_key(repository)
    if not (env / "ready").is_file():
        raise ReferenceUnavailable(
            f"the reference env is not built at {env}; run scripts/worker/reference-env.ps1 "
            "(the deploy's prepare step, or the reference-env op)")
    return env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run_analysis(python: Path, src: Path, image: bytes, *,
                 models_dir: Path | None = None,
                 timeout: float = TIMEOUT_SECONDS,
                 run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict:
    """The analysis of one picture, as `analyze.analyze` returns it."""
    with tempfile.TemporaryDirectory(prefix="chagama-reference-") as directory:
        picture = Path(directory) / "picture"
        answer = Path(directory) / "analysis.json"
        picture.write_bytes(image)
        env = {**os.environ, "PYTHONUTF8": "1"}
        if models_dir is not None:
            env[MODELS_ENV] = str(models_dir)
        try:
            done = run([str(python), "-c", ENTRY, str(src), str(picture), str(answer)],
                       env=env, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as error:
            raise ReferenceUnavailable(f"reference env python not found: {python}") from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"reference analysis timed out after {timeout:.0f} s") from error
        if done.returncode:
            tail = (done.stderr or "").strip()[-STDERR_TAIL:]
            raise RuntimeError(f"reference analysis exited {done.returncode}: {tail}")
        return json.loads(answer.read_text(encoding="utf-8"))
