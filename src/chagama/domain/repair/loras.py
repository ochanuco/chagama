"""Part-specific LoRA vocabulary for the repair reroll."""

from __future__ import annotations

from collections.abc import Sequence

PART_LORAS = {
    "feet": "feet-xl-ill.safetensors",
    "hands": "hands-xl-ill.safetensors",
}

DEFAULT_PART_LORA_WEIGHT = 0.8


def part_loras(parts: Sequence[str], weight: float | None) -> tuple[tuple[str, float], ...]:
    """One `(lora_name, weight)` per `parts` entry with a `PART_LORAS` entry,
    in the order of `parts`. `()` when `weight is None`; a part with no
    `PART_LORAS` entry is skipped.
    """
    if weight is None:
        return ()
    return tuple((PART_LORAS[part], weight) for part in parts if part in PART_LORAS)
