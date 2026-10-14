"""Environment variable lookup that still honours the pre-rename names.

The fallback exists only until the GPU machine's environment is migrated to
the CHAGAMA_* names; delete LEGACY and getenv() then.
"""

from __future__ import annotations

import os

LEGACY = {
    "CHAGAMA_WORKER": "COMFYUI_RECIPES_WORKER",
    "CHAGAMA_WORKER_ID": "COMFYUI_RECIPES_WORKER_ID",
    "CHAGAMA_REPOSITORY": "COMFYUI_RECIPES_REPOSITORY",
    "CHAGAMA_SRC": "COMFYUI_RECIPES_SRC",
    "CHAGAMA_MODEL_DIR": "COMFY_RECIPES_MODEL_DIR",
}


def getenv(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None and name in LEGACY:
        value = os.environ.get(LEGACY[name])
    return value
