"""ComfyUI custom node registration for Yukari's redraw, deliver, dof and lens graphs.

ComfyUI reaches this package through a junction/symlink named
``yukari_finalize`` under its own ``custom_nodes/``, so ``__file__`` -- not
the current working directory -- is what has to resolve back to this repo's
``src/``. ``COMFYUI_RECIPES_SRC`` overrides that for a checkout laid out
differently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_src = os.environ.get("COMFYUI_RECIPES_SRC")
_src_path = Path(_src) if _src else Path(__file__).resolve().parents[2] / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
