"""ComfyUI graphs that set a delivery's layers on a backdrop and light them by it."""

from __future__ import annotations

import json
from collections.abc import Callable

from ...domain.yukari.delivery_style import BackdropPalette
from .refinement_graph import (
    DELIVERED_SUFFIX,
    LAYER_BACKDROP_SUFFIX,
    LAYER_FIGURE_SUFFIX,
    LAYER_OUTLINE_SUFFIX,
)

STAGE_SUFFIX = "-set"
LIT_SUFFIX = "-lit"
TRAIL_SUFFIX = "-trailed"
LAYER_FRONT_SUFFIX = "-layer-front"


def _builder() -> tuple[dict, Callable[[], str], Callable[[str], str]]:
    graph: dict = {}
    cursor = 1

    def allocate() -> str:
        nonlocal cursor
        node_id = str(cursor)
        cursor += 1
        return node_id

    def load(image: str) -> str:
        node_id = allocate()
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": image}}
        return node_id

    return graph, allocate, load


def _save(graph: dict, allocate, ref: list, filename_prefix: str) -> None:
    graph[allocate()] = {"class_type": "SaveImage", "inputs": {
        "images": ref, "filename_prefix": filename_prefix}}


def stage_graph(figure_image: str, outline_image: str, prefix: str, *,
                photo_image: str | None = None, backdrop: str | None = None,
                shadow_from: str | None = None, outlines: list | None = None,
                place: tuple[float, float] = (0.5, 1.0), height: float = 1.0,
                light: tuple[str, str] | None = None, frame: dict | None = None,
                backdrop_palette: BackdropPalette | None = None) -> dict:
    """Set the uploaded layer images on the uploaded `photo_image`, or on
    the `backdrop` pattern when there is none (in the `frame` when one is
    given, painted in `backdrop_palette`), tinted to the `light` (scene,
    from) when one is given. The picture is saved under `prefix` +
    STAGE_SUFFIX and its three layers beside it."""
    graph, allocate, load = _builder()
    figure_id, outline_id = load(figure_image), load(outline_image)
    inputs = {"figure": [figure_id, 0], "figure_mask": [figure_id, 1],
              "outline": [outline_id, 0], "outline_mask": [outline_id, 1]}
    if photo_image is not None:
        inputs.update({"mode": "photo", "photo": [load(photo_image), 0],
                       "place_x": place[0], "place_y": place[1], "height": height})
    else:
        inputs.update({"mode": "pattern", "backdrop": backdrop or "",
                       "shadow_from": shadow_from or "",
                       "outlines": json.dumps(outlines) if outlines is not None else "",
                       **({"light_scene": light[0], "light_from": light[1]} if light else {}),
                       **({"frame": json.dumps(frame)} if frame is not None else {}),
                       **({"backdrop_palette": json.dumps(backdrop_palette._asdict())}
                          if backdrop_palette is not None else {})})
    stage_id = allocate()
    graph[stage_id] = {"class_type": "YukariStage", "inputs": inputs}
    _save(graph, allocate, [stage_id, 0], prefix + STAGE_SUFFIX)
    for output, suffix in ((1, LAYER_FIGURE_SUFFIX), (2, LAYER_OUTLINE_SUFFIX),
                           (3, LAYER_BACKDROP_SUFFIX)):
        _save(graph, allocate, [stage_id, output], prefix + suffix)
    return graph


def lighting_graph(figure_image: str, outline_image: str, backdrop_image: str | None,
                   prefix: str, *, direction: str | None, strength: float,
                   scene: str | None = None, tint_backdrop: bool = False,
                   front_image: str | None = None) -> dict:
    """Light the uploaded layer images by their uploaded backdrop, in the
    light `scene` when one is given (tinting the backdrop when
    `tint_backdrop`). The picture is saved under `prefix` + LIT_SUFFIX and
    the layers beside it (no backdrop layer without a backdrop)."""
    graph, allocate, load = _builder()
    figure_id, outline_id = load(figure_image), load(outline_image)
    backdrop_id = load(backdrop_image) if backdrop_image is not None else None
    light_id = allocate()
    inputs = {
        "figure": [figure_id, 0], "figure_mask": [figure_id, 1],
        "outline": [outline_id, 0], "outline_mask": [outline_id, 1],
        "direction": direction or "",
        "strength": strength, "scene": scene or "", "tint_backdrop": tint_backdrop}
    if backdrop_id is not None:
        inputs["backdrop"] = [backdrop_id, 0]
    graph[light_id] = {"class_type": "YukariRelight", "inputs": inputs}
    lit_ref = [light_id, 0]
    if front_image is not None:
        lit_ref = over_front(graph, allocate, lit_ref, load(front_image))
    _save(graph, allocate, lit_ref, prefix + LIT_SUFFIX)
    _save(graph, allocate, [light_id, 1], prefix + LAYER_FIGURE_SUFFIX)
    _save(graph, allocate, [light_id, 2], prefix + LAYER_OUTLINE_SUFFIX)
    if backdrop_id is not None:
        _save(graph, allocate, [light_id, 3], prefix + LAYER_BACKDROP_SUFFIX)
    return graph


def saturation_graph(figure_image: str, outline_image: str, prefix: str, *,
                     backdrop_image: str | None = None, strength: float) -> dict:
    """Compress the uploaded layer images' saturation by `strength`, over
    the uploaded `backdrop_image` when there is one. The picture is saved
    under `prefix` + DELIVERED_SUFFIX and the layers beside it, the
    backdrop as it came."""
    graph, allocate, load = _builder()
    figure_id, outline_id = load(figure_image), load(outline_image)
    inputs = {"figure": [figure_id, 0], "figure_mask": [figure_id, 1],
              "outline": [outline_id, 0], "outline_mask": [outline_id, 1],
              "strength": strength}
    backdrop_id = load(backdrop_image) if backdrop_image is not None else None
    if backdrop_id is not None:
        inputs["backdrop"] = [backdrop_id, 0]
    saturation_id = allocate()
    graph[saturation_id] = {"class_type": "YukariSaturation", "inputs": inputs}
    _save(graph, allocate, [saturation_id, 0], prefix + DELIVERED_SUFFIX)
    _save(graph, allocate, [saturation_id, 1], prefix + LAYER_FIGURE_SUFFIX)
    _save(graph, allocate, [saturation_id, 2], prefix + LAYER_OUTLINE_SUFFIX)
    if backdrop_id is not None:
        _save(graph, allocate, [backdrop_id, 0], prefix + LAYER_BACKDROP_SUFFIX)
    return graph


def trail_graph(figure_image: str, outline_image: str, backdrop_image: str, prefix: str, *,
                trail: dict, back_box: list[float] | None) -> dict:
    """Draw the `trail` across the uploaded staged layers: the picture under
    `prefix` + TRAIL_SUFFIX, the backdrop with the back part painted in and
    the front part beside it."""
    graph, allocate, load = _builder()
    figure_id, outline_id, backdrop_id = load(figure_image), load(outline_image), load(backdrop_image)
    trail_id = allocate()
    inputs = {"figure": [figure_id, 0], "figure_mask": [figure_id, 1],
              "outline": [outline_id, 0], "outline_mask": [outline_id, 1],
              "backdrop": [backdrop_id, 0], "trail": json.dumps(trail, sort_keys=True)}
    if back_box is not None:
        inputs["back_box"] = json.dumps(back_box)
    graph[trail_id] = {"class_type": "YukariTrail", "inputs": inputs}
    _save(graph, allocate, [trail_id, 0], prefix + TRAIL_SUFFIX)
    _save(graph, allocate, [trail_id, 1], prefix + LAYER_BACKDROP_SUFFIX)
    _save(graph, allocate, [trail_id, 2], prefix + LAYER_FRONT_SUFFIX)
    return graph


def over_front(graph: dict, allocate: Callable[[], str], picture_ref: list,
               front_id: str) -> list:
    """`picture_ref` with the uploaded front layer `front_id` laid over it."""
    over_id = allocate()
    graph[over_id] = {"class_type": "YukariOver", "inputs": {
        "image": picture_ref, "layer": [front_id, 0], "layer_mask": [front_id, 1]}}
    return [over_id, 0]
