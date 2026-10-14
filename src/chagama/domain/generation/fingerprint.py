"""Content-addressed identity for an assembled prompt pair."""

from __future__ import annotations

import hashlib
import json


def prompt_fingerprint(recipe: str, pose: str, positive: str, negative: str) -> str:
    payload = json.dumps(
        {"recipe": recipe, "pose": pose, "positive": positive, "negative": negative},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()
