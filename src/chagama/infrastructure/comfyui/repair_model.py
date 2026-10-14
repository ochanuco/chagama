"""A `model_hooks` entry that reroutes a repair reroll onto another checkpoint.

The Anima loader shape mirrors `anima_graph.build_graph`: a `UNETLoader` for
the checkpoint, a shared `CLIPLoader type=qwen_image` text encoder and a
shared `VAELoader qwen_image_vae`, none of which the source graph carries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from ...domain.repair.models import resolve_model
from ...domain.yukari.prompt_style import CFG, SAMPLER, SCHEDULER, STEPS
from .anima_graph import CLIP_NAME, VAE_NAME
from .repair_graph import RerollHook, RerollRefs


def anima_model_hook(word: str) -> RerollHook:
    """A `model_hooks` entry that samples the reroll on the named Anima checkpoint."""
    checkpoint = resolve_model(word)

    def hook(graph: dict, allocate: Callable[[], str], refs: RerollRefs) -> RerollRefs:
        unet_id = allocate()
        graph[unet_id] = {"class_type": "UNETLoader", "inputs": {
            "unet_name": checkpoint, "weight_dtype": "default"}}
        clip_id = allocate()
        graph[clip_id] = {"class_type": "CLIPLoader", "inputs": {
            "clip_name": CLIP_NAME, "type": "qwen_image"}}
        vae_id = allocate()
        graph[vae_id] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}}
        return replace(
            refs, model=[unet_id, 0], positive_clip=[clip_id, 0],
            negative_clip=[clip_id, 0], vae=[vae_id, 0],
            sampler={"steps": STEPS, "cfg": CFG, "sampler_name": SAMPLER,
                     "scheduler": SCHEDULER})

    return hook
