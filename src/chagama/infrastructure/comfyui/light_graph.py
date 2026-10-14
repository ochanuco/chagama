"""ComfyUI graph for the light pass: the finished picture, underpainted with
a scene's light and shade, re-sampled lightly with the scene's words."""

from __future__ import annotations

import copy
from collections.abc import Mapping

from ...domain.yukari import delivery_style
from .base_graph import BaseRoles, sampler_settings, source_prompts
from .deliver_graph import alpha_nodes, depth_nodes
from .refinement_graph import _depth_nodes, _matte_nodes

OUTPUT_CLASSES = ("SaveImage", "PreviewImage")


def light_words(scene: str, direction: str) -> str:
    return ", ".join((
        delivery_style.LIGHT_WORDS.format(
            direction=delivery_style.LIGHT_DIRECTION_WORDS[direction]),
        delivery_style.LIGHT_SCENES[scene]["words"]))


def light_graph(graph: Mapping, roles: BaseRoles, source_image: str,
                matte_model: str, scene: str, direction: str,
                prefix: str, *, cut: bool = False,
                alpha_image: str | None = None,
                depth_image: str | None = None) -> dict:
    """A copy of `graph` without its outputs, so its own sampling does not
    run again, that lights the uploaded `source_image` and re-samples it with
    the source's seed and negative and a positive that adds the scene's words.

    With `cut`, the matte is the ViTMatte alpha and the depth is the cut
    asset of the picture: loaded from `alpha_image` / `depth_image` when
    given, otherwise built here and saved under `prefix` + "-alpha" /
    "-depth". Without it the matte is the coarse background-removal one."""
    result = {key: node for key, node in copy.deepcopy(dict(graph)).items()
              if node.get("class_type") not in OUTPUT_CLASSES}
    settings = sampler_settings(result, roles.sampler_id)
    positive_text, _ = source_prompts(graph)
    sampler_inputs = result[roles.sampler_id]["inputs"]
    model_ref = sampler_inputs["model"]
    clip_ref = result[roles.positive_id]["inputs"].get("clip", ["4", 1])
    vae_ref = result[roles.decode_id]["inputs"].get("vae", ["4", 2])
    cursor = max(int(key) for key in result if key.isdecimal()) + 1

    def allocate() -> str:
        nonlocal cursor
        node_id = str(cursor)
        cursor += 1
        return node_id

    load_id = allocate()
    result[load_id] = {"class_type": "LoadImage", "inputs": {"image": source_image}}
    image_ref = [load_id, 0]
    if cut:
        matte_ref = alpha_nodes(
            result, allocate, image_ref, matte_model, prefix, alpha_image)
        depth_ref = depth_nodes(
            result, allocate, image_ref, prefix, depth_image)
    else:
        depth_ref = _depth_nodes(result, allocate, image_ref)
        matte_ref = _matte_nodes(result, allocate, image_ref, matte_model)
    light_id = allocate()
    result[light_id] = {"class_type": "YukariLight", "inputs": {
        "image": image_ref, "depth": depth_ref, "matte": matte_ref,
        "direction": direction, "scene": scene}}
    encode_id = allocate()
    result[encode_id] = {"class_type": "VAEEncode", "inputs": {
        "pixels": [light_id, 0], "vae": vae_ref}}
    positive_id = allocate()
    result[positive_id] = {"class_type": "CLIPTextEncode", "inputs": {
        "clip": clip_ref,
        "text": f"{positive_text}, {light_words(scene, direction)}"}}
    sample_id = allocate()
    result[sample_id] = {"class_type": "KSampler", "inputs": {
        "model": model_ref, "positive": [positive_id, 0],
        "negative": sampler_inputs["negative"],
        "latent_image": [encode_id, 0], "seed": settings["seed"],
        "steps": settings["steps"], "cfg": settings["cfg"],
        "sampler_name": settings["sampler_name"],
        "scheduler": settings["scheduler"],
        "denoise": delivery_style.LIGHT_DENOISE}}
    decode_id = allocate()
    result[decode_id] = {"class_type": "VAEDecode", "inputs": {
        "samples": [sample_id, 0], "vae": vae_ref}}
    save_id = allocate()
    result[save_id] = {"class_type": "SaveImage", "inputs": {
        "images": [decode_id, 0], "filename_prefix": prefix}}
    return result
