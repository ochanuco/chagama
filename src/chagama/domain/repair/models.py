"""Vocabulary of crop-only checkpoints a repair reroll's `model` option may name."""

from __future__ import annotations

MODELS = {
    "anima": "sudachiAnima_v10.safetensors",
    "anima-hassaku": "hassakuAnima_v13.safetensors",
    "anima-base": "anima_baseV10.safetensors",
}


def resolve_model(word: str) -> str:
    if word not in MODELS:
        valid = ", ".join(sorted(MODELS))
        raise ValueError(f"unknown model: {word!r}, must be one of {valid}")
    return MODELS[word]
