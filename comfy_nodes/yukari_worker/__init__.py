"""ComfyUI custom node registration for hosting the worker inside ComfyUI.

ComfyUI reaches this package through a junction/symlink named
``yukari_worker`` under its own ``custom_nodes/``, so ``__file__`` -- not
the current working directory -- is what has to resolve back to this repo's
``src/``. ``CHAGAMA_SRC`` overrides that for a checkout laid out
differently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Runs before src/ is importable, so the legacy name is read inline rather
# than through chagama.envcompat; drop it with that module.
_src = os.environ.get("CHAGAMA_SRC") or os.environ.get("COMFYUI_RECIPES_SRC")
_src_path = Path(_src) if _src else Path(__file__).resolve().parents[2] / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from .agent import start  # noqa: E402

# This pack registers no nodes: it exists only so ComfyUI's own process can
# host the worker's claim loop as a background thread.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

start()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
