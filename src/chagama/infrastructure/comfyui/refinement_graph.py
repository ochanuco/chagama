"""Stages the delivery graphs share."""

from __future__ import annotations

import json
from collections.abc import Callable

# The delivered composite: background cut, outline bands.
DELIVERED_SUFFIX = "-delivered"
# What the delivered composite was made of, saved beside it.
LAYER_FIGURE_SUFFIX = "-layer-figure"
LAYER_OUTLINE_SUFFIX = "-layer-outline"
LAYER_BACKDROP_SUFFIX = "-layer-backdrop"
# A matte_model of this form names a ComfyUI-RMBG model instead of a core
# background-removal model file.
RMBG_MATTE_PREFIX = "rmbg:"
DEPTH_NODE = "DepthAnythingV2Preprocessor"
DEPTH_CKPT = "depth_anything_v2_vitl.pth"
# Depth is inferred at the picture's own size; a picture over this many
# pixels is shrunk to fit it, keeping its aspect ratio.
DEPTH_MAX_PIXELS = 2048 * 2560
DEPTH_MIN_RESOLUTION = 64


def depth_resolution(size: tuple[int, int]) -> int:
    """The `resolution` input of the depth node (the target short side) for a
    picture of `size`: its own short side, or the short side of it shrunk to
    DEPTH_MAX_PIXELS. Never above the picture's own, rounded down."""
    width, height = size
    short = min(width, height)
    if width * height > DEPTH_MAX_PIXELS:
        short = int(short * (DEPTH_MAX_PIXELS / (width * height)) ** 0.5)
    return max(short, DEPTH_MIN_RESOLUTION)


def _matte_nodes(graph: dict, allocate: Callable[[], str], image_ref: list,
                 matte_model: str) -> list:
    if matte_model.startswith(RMBG_MATTE_PREFIX):
        node_id = allocate()
        graph[node_id] = {"class_type": "BiRefNetRMBG", "inputs": {
            "image": image_ref, "model": matte_model[len(RMBG_MATTE_PREFIX):],
            "sensitivity": 1.0, "mask_blur": 0, "mask_offset": 0,
            "invert_output": False, "refine_foreground": False,
            "background": "Alpha", "background_color": "#222222"}}
        return [node_id, 1]
    bg_loader = allocate()
    graph[bg_loader] = {"class_type": "LoadBackgroundRemovalModel", "inputs": {
        "bg_removal_name": matte_model}}
    remove = allocate()
    graph[remove] = {"class_type": "RemoveBackground", "inputs": {
        "bg_removal_model": [bg_loader, 0], "image": image_ref}}
    return [remove, 0]


def save_mask(graph: dict, allocate: Callable[[], str], mask_ref: list,
              filename_prefix: str) -> None:
    to_image = allocate()
    graph[to_image] = {"class_type": "MaskToImage", "inputs": {"mask": mask_ref}}
    save = allocate()
    graph[save] = {"class_type": "SaveImage", "inputs": {
        "images": [to_image, 0], "filename_prefix": filename_prefix}}


def matting_node(graph: dict, allocate: Callable[[], str], image_ref: list,
                 matte_ref: list) -> list:
    """The ViTMatte alpha of the picture. Returns the alpha ref."""
    node_id = allocate()
    graph[node_id] = {"class_type": "YukariMatting", "inputs": {
        "image": image_ref, "matte": matte_ref}}
    return [node_id, 0]


def foreground_node(graph: dict, allocate: Callable[[], str], image_ref: list,
                    alpha_ref: list) -> list:
    """The figure's colour under the alpha. Returns the foreground ref."""
    node_id = allocate()
    graph[node_id] = {"class_type": "YukariForeground", "inputs": {
        "image": image_ref, "alpha": alpha_ref}}
    return [node_id, 0]


def see_through_node(graph: dict, allocate: Callable[[], str], image_ref: list,
                     alpha_ref: list, gain: float) -> list:
    """The alpha with the key through the see-through inside scaled by
    `gain`. Returns the alpha ref."""
    node_id = allocate()
    graph[node_id] = {"class_type": "YukariSeeThrough", "inputs": {
        "image": image_ref, "alpha": alpha_ref, "gain": gain}}
    return [node_id, 0]


def _depth_nodes(graph: dict, allocate: Callable[[], str], image_ref: list,
                 size: tuple[int, int]) -> list:
    depth_id = allocate()
    graph[depth_id] = {"class_type": DEPTH_NODE, "inputs": {
        "image": image_ref, "ckpt_name": DEPTH_CKPT,
        "resolution": depth_resolution(size)}}
    return [depth_id, 0]


def delivery_tail(graph: dict, allocate: Callable[[], str], image_ref: list,
                  matte_ref: list, prefix: str, *,
                  keep_scene: bool, transparent: bool, backdrop: str | None,
                  stroke_light: str | None, outlines: list[dict],
                  light_scene: str | None, light_from: str | None,
                  silhouette_ref: list | None = None) -> list:
    """Appends the composite stage onto `graph` (mutated), saving the
    delivered picture and its layers.
    `silhouette_ref` is the mask the bands follow when it is not the matte.
    Returns the delivered ref."""
    deliver_id = allocate()
    graph[deliver_id] = {"class_type": "YukariDeliver", "inputs": {
        "image": image_ref, "matte": matte_ref, "keep_scene": keep_scene,
        "transparent": transparent, "stroke_light": stroke_light or "",
        "backdrop": backdrop or "", "matted": not keep_scene,
        "outlines": json.dumps(outlines),
        **({"light_scene": light_scene, "light_from": light_from}
           if light_scene else {}),
        **({"silhouette": silhouette_ref} if silhouette_ref is not None else {})}}
    delivered_ref = [deliver_id, 0]
    save_delivered = allocate()
    graph[save_delivered] = {"class_type": "SaveImage", "inputs": {
        "images": delivered_ref, "filename_prefix": prefix + DELIVERED_SUFFIX}}
    layers = [(LAYER_FIGURE_SUFFIX, 2), (LAYER_OUTLINE_SUFFIX, 3)]
    if keep_scene or not transparent:
        layers.append((LAYER_BACKDROP_SUFFIX, 4))
    for suffix, output in layers:
        save = allocate()
        graph[save] = {"class_type": "SaveImage", "inputs": {
            "images": [deliver_id, output],
            "filename_prefix": prefix + suffix}}
    return delivered_ref
