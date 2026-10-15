"""Where persistent data and secrets live."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ENV = "CHAGAMA_DATA"
CONFIG_ENV = "CHAGAMA_CONFIG"


def data_dir(repository: Path) -> Path:
    """`$CHAGAMA_DATA`, else `<repository>/.local/_nogit`."""
    override = os.environ.get(DATA_ENV, "").strip()
    return Path(override) if override else repository / ".local/_nogit"


def config_dir(repository: Path) -> Path:
    """`$CHAGAMA_CONFIG`, else `<repository>/.local`."""
    override = os.environ.get(CONFIG_ENV, "").strip()
    return Path(override) if override else repository / ".local"
