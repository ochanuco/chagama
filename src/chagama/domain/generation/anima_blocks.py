"""Where the blocks of the 28-block Anima sit in a 40-block Anima 2.9B, which
inserts twelve new blocks among the old ones (the table the
ComfyUI-Anima-LoRA-ControlNet-Patch node pack remaps LoRAs and LLLites by)."""

from __future__ import annotations

from collections.abc import Sequence

BASE_BLOCKS = 28
INSERTED_28_40 = (2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36)
EXPANDED_BLOCKS = BASE_BLOCKS + len(INSERTED_28_40)
MAP_28_40 = tuple(index for index in range(EXPANDED_BLOCKS)
                  if index not in INSERTED_28_40)


def remap(layers: Sequence[int], blocks: int) -> list[int]:
    """`layers`, block indices of the 28-block model, as indices of a model
    with `blocks` blocks."""
    if blocks == BASE_BLOCKS:
        return list(layers)
    if blocks == EXPANDED_BLOCKS:
        return [MAP_28_40[layer] for layer in layers]
    raise ValueError(f"no block map from {BASE_BLOCKS} to {blocks} Anima blocks")
