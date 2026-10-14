"""ControlNet conditioning hook for the repair reroll (see `repair_graph.py`
`RerollHook`): reroutes `refs.positive`/`refs.negative` through a ControlNet
fed by a pre-staged reference hint image, rather than anything derived from
the crop itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from ...domain.repair.controlnet import control_model
from .repair_graph import RerollHook, RerollRefs


def control_hook(word: str, strength: float, reference_image: str) -> RerollHook:
    """A conditioning hook applying `control_model(word)` at `strength`,
    conditioned on `reference_image` (already staged on the ComfyUI worker).
    """
    model_name = control_model(word)

    def hook(graph: dict, allocate: Callable[[], str], refs: RerollRefs) -> RerollRefs:
        load_image = allocate()
        graph[load_image] = {"class_type": "LoadImage", "inputs": {
            "image": reference_image}}
        loader = allocate()
        graph[loader] = {"class_type": "ControlNetLoader", "inputs": {
            "control_net_name": model_name}}
        apply = allocate()
        graph[apply] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": refs.positive, "negative": refs.negative,
            "control_net": [loader, 0], "image": [load_image, 0],
            "strength": strength, "start_percent": 0.0, "end_percent": 1.0}}
        return replace(refs, positive=[apply, 0], negative=[apply, 1])

    return hook
