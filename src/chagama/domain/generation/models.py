"""Pure generation values shared by recipes and graph adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptPair:
    positive: str
    negative: str


@dataclass(frozen=True)
class HiresSpec:
    width: int
    height: int
    denoise: float
    negative: str
    positive: str | None = None


@dataclass(frozen=True)
class RenderSpec:
    model_path: str
    prompts: PromptPair
    width: int
    height: int
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float
    filename_prefix: str
    hires: HiresSpec | None = None
    loras: tuple[tuple[str, float], ...] = ()
    # Leading steps sampled at `cfg`, the rest at CFG 1.0, same noise
    # schedule. None or 0 means a plain single KSampler.
    guided_steps: int | None = None
    # (name, text) breakdown of `prompts.positive`, in order -- joining the
    # texts reproduces it byte for byte. Empty when the recipe has no named
    # parts.
    positive_parts: tuple[tuple[str, str], ...] = ()
    # Legacy part name -> `positive_parts` names that composed it, for a
    # `prompt.positive.<part>` patch that still names a legacy part.
    part_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
