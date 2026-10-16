"""ComfyUI graph for the delivery of a picture Generation: cut, decorate, save."""

from __future__ import annotations

from collections.abc import Callable

from ...domain.yukari.delivery_style import STROKE_CHOICES
from ..imaging import backdrops
from .refinement_graph import (
    _depth_nodes,
    _matte_nodes,
    delivery_tail,
    foreground_node,
    matting_node,
    save_mask,
    see_through_node,
)

ALPHA_SUFFIX = "-alpha"
DEPTH_SUFFIX = "-depth"
FOREGROUND_SUFFIX = "-foreground"


def alpha_nodes(graph: dict, allocate: Callable[[], str], source_ref: list,
                matte_model: str, prefix: str,
                alpha_image: str | None) -> list:
    """The ViTMatte alpha of `source_ref`: loaded from the uploaded
    `alpha_image`, or cut here and saved under `prefix` + ALPHA_SUFFIX.
    Returns the alpha mask ref."""
    if alpha_image is not None:
        load_alpha = allocate()
        graph[load_alpha] = {"class_type": "LoadImage", "inputs": {
            "image": alpha_image}}
        to_mask = allocate()
        graph[to_mask] = {"class_type": "ImageToMask", "inputs": {
            "image": [load_alpha, 0], "channel": "red"}}
        return [to_mask, 0]
    coarse_ref = _matte_nodes(graph, allocate, source_ref, matte_model)
    alpha_ref = matting_node(graph, allocate, source_ref, coarse_ref)
    save_mask(graph, allocate, alpha_ref, prefix + ALPHA_SUFFIX)
    return alpha_ref


def depth_nodes(graph: dict, allocate: Callable[[], str], source_ref: list,
                prefix: str, depth_image: str | None,
                size: tuple[int, int] | None = None) -> list:
    """The depth of `source_ref`: loaded from the uploaded `depth_image`, or
    built here at the picture's `size` and saved under `prefix` + DEPTH_SUFFIX.
    Returns the depth ref."""
    if depth_image is not None:
        load_depth = allocate()
        graph[load_depth] = {"class_type": "LoadImage", "inputs": {
            "image": depth_image}}
        return [load_depth, 0]
    if size is None:
        raise ValueError("depth_nodes needs a depth_image or the source size")
    depth_ref = _depth_nodes(graph, allocate, source_ref, size)
    save_depth = allocate()
    graph[save_depth] = {"class_type": "SaveImage", "inputs": {
        "images": depth_ref, "filename_prefix": prefix + DEPTH_SUFFIX}}
    return depth_ref


def cut_graph(source_image: str, matte_model: str, prefix: str,
              size: tuple[int, int]) -> dict:
    """The cut of the uploaded `source_image` as a deliver makes it: the
    alpha and depth saved under `prefix` + ALPHA_SUFFIX / DEPTH_SUFFIX, and
    the depth inferred at the source's `size`, and the figure's colour under that alpha under `prefix` + FOREGROUND_SUFFIX."""
    graph: dict = {}
    cursor = 1

    def allocate() -> str:
        nonlocal cursor
        node_id = str(cursor)
        cursor += 1
        return node_id

    load = allocate()
    graph[load] = {"class_type": "LoadImage", "inputs": {"image": source_image}}
    source_ref = [load, 0]
    alpha_ref = alpha_nodes(graph, allocate, source_ref, matte_model, prefix, None)
    depth_nodes(graph, allocate, source_ref, prefix, None, size)
    graph[allocate()] = {"class_type": "SaveImage", "inputs": {
        "images": foreground_node(graph, allocate, source_ref, alpha_ref),
        "filename_prefix": prefix + FOREGROUND_SUFFIX}}
    return graph


def deliver_graph(source_image: str, matte_model: str, prefix: str, *,
                  alpha_image: str | None = None,
                  keep_scene: bool,
                  transparent: bool, backdrop: str | None,
                  stroke_light: str | None, outlines: list[dict],
                  light_scene: str | None, light_from: str | None,
                  see_through: float = 1.0) -> dict:
    """The deliver graph over the uploaded `source_image`.

    `alpha_image` is the uploaded alpha cut asset to load instead of
    computing; without it the graph cuts the source itself and
    saves the result under `prefix` + ALPHA_SUFFIX. `see_through` scales the
    key showing through see-through clothing after the colour is solved;
    the saved alpha stays unscaled.
    """
    if stroke_light is not None and stroke_light not in STROKE_CHOICES:
        valid = ", ".join(repr(key) for key in STROKE_CHOICES)
        raise ValueError(f"stroke_light must be null or one of {valid}, got {stroke_light!r}")
    if backdrop is not None and not backdrops.is_backdrop(backdrop):
        valid = ", ".join(repr(key) for key in sorted(backdrops.PATTERNS))
        raise ValueError(
            f"backdrop must be null, a #RRGGBB colour or one of {valid}, got {backdrop!r}")
    graph: dict = {}
    cursor = 1

    def allocate() -> str:
        nonlocal cursor
        node_id = str(cursor)
        cursor += 1
        return node_id

    load_id = allocate()
    graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": source_image}}
    source_ref = [load_id, 0]

    alpha_ref = alpha_nodes(
        graph, allocate, source_ref, matte_model, prefix, alpha_image)

    image_ref = source_ref
    silhouette_ref = None
    if not keep_scene:
        image_ref = foreground_node(graph, allocate, image_ref, alpha_ref)
        if see_through != 1.0:
            silhouette_ref = alpha_ref
            alpha_ref = see_through_node(
                graph, allocate, source_ref, alpha_ref, see_through)

    delivery_tail(
        graph, allocate, image_ref, alpha_ref, prefix,
        keep_scene=keep_scene, transparent=transparent, backdrop=backdrop,
        stroke_light=stroke_light, outlines=outlines,
        light_scene=light_scene, light_from=light_from,
        silhouette_ref=silhouette_ref)
    return graph
