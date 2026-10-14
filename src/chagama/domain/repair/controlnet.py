"""Control-signal vocabulary for the repair reroll's ControlNet hook."""

from __future__ import annotations

CONTROL_MODELS = {
    "lineart": "noob-lineart-anime-fp16.safetensors",
}

DEFAULT_CONTROL_STRENGTH = 0.8


def control_model(word: str) -> str:
    if word not in CONTROL_MODELS:
        valid = ", ".join(sorted(CONTROL_MODELS))
        raise ValueError(f"unknown control word: {word!r} (one of {valid})")
    return CONTROL_MODELS[word]
