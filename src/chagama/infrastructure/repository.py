"""Repository provenance adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path


def discover_repository(cwd: Path | None = None) -> Path:
    """Return the containing Git root, independent of the caller's directory."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=cwd or Path.cwd(),
        check=True, capture_output=True, text=True,
    )
    return Path(result.stdout.strip())


def git_metadata(repository: Path) -> dict:
    def output(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repository, check=True,
            capture_output=True, text=True,
        ).stdout.strip()

    return {
        "commit": output("rev-parse", "HEAD"),
        "branch": output("branch", "--show-current"),
        "dirty": bool(output("status", "--porcelain")),
    }
