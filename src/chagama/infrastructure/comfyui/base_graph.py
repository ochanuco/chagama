"""Structural resolution of a base graph's semantic roles.

Node IDs vary by recipe and pass (a plain base samples through "3", a
masked_redraw base through "19"), so every role is found by walking the
graph's own edges rather than assuming a fixed ID.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass

# Nodes a finished picture can pass through, unmodified, between decode and
# SaveImage. Excludes a delivery's own matte/delivered branches, so they never
# resolve as the picture's save.
PASSTHROUGH = ("JoinImageWithAlpha", "LayeredDiffusionDecode", "InpaintStitchImproved")


def is_ref(value: object) -> bool:
    return isinstance(value, list) and len(value) == 2


def consumers(graph: Mapping) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {key: [] for key in graph}
    for key, node in graph.items():
        for value in node.get("inputs", {}).values():
            if is_ref(value) and value[0] in graph:
                result[value[0]].append(key)
    return result


def reaches(graph: Mapping, consumers_map: Mapping[str, list[str]], start: str,
           class_type: str) -> bool:
    stack = list(consumers_map.get(start, []))
    seen: set[str] = set()
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        if graph[node_id].get("class_type") == class_type:
            return True
        stack.extend(consumers_map.get(node_id, []))
    return False


def find_decode(graph: Mapping) -> str:
    """The one VAEDecode that is the source's finished picture."""
    consumers_map = consumers(graph)
    candidates = [
        key for key, node in graph.items()
        if node.get("class_type") == "VAEDecode" and consumers_map.get(key)
        and not reaches(graph, consumers_map, key, "VAEEncode")]
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one final VAEDecode reachable without a "
            f"re-sample, found {len(candidates)}: {sorted(candidates)}")
    return candidates[0]


_SAMPLER_CLASSES = ("KSampler", "KSamplerAdvanced")


def find_sampler(graph: Mapping, decode_id: str) -> str:
    """The KSampler/KSamplerAdvanced feeding `decode_id`, through any
    Latent*/SetLatentNoiseMask hop."""
    node_id = graph[decode_id]["inputs"]["samples"][0]
    seen: set[str] = set()
    while graph[node_id].get("class_type") not in _SAMPLER_CLASSES:
        if node_id in seen:
            raise ValueError(
                f"could not trace a KSampler upstream of VAEDecode {decode_id!r}")
        seen.add(node_id)
        inputs = graph[node_id].get("inputs", {})
        if "samples" not in inputs:
            raise ValueError(
                f"could not trace a KSampler upstream of VAEDecode {decode_id!r}")
        node_id = inputs["samples"][0]
    return node_id


def chain_origin(graph: Mapping, sampler_id: str) -> str:
    """`sampler_id` itself, or the first (noise-adding) stage of a guided
    KSamplerAdvanced pair whose `latent_image` chains back to it."""
    node_id = sampler_id
    while True:
        inputs = graph[node_id].get("inputs", {})
        latent_ref = inputs.get("latent_image")
        if not (is_ref(latent_ref) and latent_ref[0] in graph):
            return node_id
        if graph[latent_ref[0]].get("class_type") != "KSamplerAdvanced":
            return node_id
        node_id = latent_ref[0]


_SETTINGS_KEYS = ("seed", "steps", "cfg", "sampler_name", "scheduler")


def sampler_settings(graph: Mapping, sampler_id: str) -> dict:
    """The seed/steps/cfg/sampler/scheduler the picture at `sampler_id` was
    drawn with; for a guided KSamplerAdvanced pair, seed and cfg come from
    the first stage. Keys absent from the node are absent here."""
    inputs = graph[sampler_id]["inputs"]
    settings = {key: inputs[key] for key in _SETTINGS_KEYS if key in inputs}
    if graph[sampler_id].get("class_type") != "KSamplerAdvanced":
        return settings
    origin_inputs = graph[chain_origin(graph, sampler_id)]["inputs"]
    if "noise_seed" in origin_inputs:
        settings["seed"] = origin_inputs["noise_seed"]
    if "cfg" in origin_inputs:
        settings["cfg"] = origin_inputs["cfg"]
    return settings


def source_prompts(graph: Mapping) -> tuple[str, str]:
    """The source's own final positive/negative CLIPTextEncode text."""
    decode_id = find_decode(graph)
    sampler_id = find_sampler(graph, decode_id)
    inputs = graph[sampler_id]["inputs"]
    positive = graph[inputs["positive"][0]]["inputs"]["text"]
    negative = graph[inputs["negative"][0]]["inputs"]["text"]
    return positive, negative


@dataclass(frozen=True)
class BaseRoles:
    save_id: str
    decode_id: str
    sampler_id: str
    positive_id: str
    negative_id: str
    stitched: bool


def _saves_downstream(graph: Mapping, consumers_map: Mapping[str, list[str]],
                      node_id: str) -> list[tuple[str, bool]]:
    """SaveImage nodes reachable from `node_id` through PASSTHROUGH only;
    each result's bool is whether the walk crossed InpaintStitchImproved
    (the sampler's latent is then the inpaint crop, not the whole canvas)."""
    found: list[tuple[str, bool]] = []
    for consumer_id in consumers_map.get(node_id, []):
        class_type = graph[consumer_id].get("class_type")
        if class_type == "SaveImage":
            found.append((consumer_id, False))
        elif class_type in PASSTHROUGH:
            found.extend(
                (save_id, crossed or class_type == "InpaintStitchImproved")
                for save_id, crossed in _saves_downstream(graph, consumers_map, consumer_id))
    return found


def base_roles(graph: Mapping) -> BaseRoles:
    """Resolve a base graph's sampler, decode, save and prompt roles."""
    try:
        decode_id = find_decode(graph)
    except ValueError as exc:
        raise ValueError(
            f"base graph's SaveImage must be fed by a VAEDecode: {exc}") from exc
    consumers_map = consumers(graph)
    saves = _saves_downstream(graph, consumers_map, decode_id)
    if len(saves) != 1:
        raise ValueError(
            "expected exactly one SaveImage reachable from the decode "
            f"through {PASSTHROUGH}, found {len(saves)}: "
            f"{sorted(save_id for save_id, _ in saves)}")
    save_id, stitched = saves[0]
    sampler_id = find_sampler(graph, decode_id)
    sampler_inputs = graph[sampler_id].get("inputs", {})
    if "positive" not in sampler_inputs or "negative" not in sampler_inputs:
        raise ValueError(
            f"KSampler {sampler_id!r} has no positive/negative prompt refs")
    return BaseRoles(
        save_id=save_id, decode_id=decode_id, sampler_id=sampler_id,
        positive_id=sampler_inputs["positive"][0],
        negative_id=sampler_inputs["negative"][0], stitched=stitched)


def hires_graph(graph: Mapping, roles: BaseRoles, pixels: int, denoise: float,
                prefix: str) -> dict:
    """A copy of `graph` that latent-upscales the finished sampler's output
    to `pixels` in area, keeping its aspect ratio, and re-samples it with the
    first-pass seed, before the decode. The EmptyLatentImage is left
    untouched."""
    if any(node.get("class_type") == "LatentUpscale" for node in graph.values()):
        raise ValueError("この graph は既に hires 済みです")
    result = copy.deepcopy(dict(graph))
    origin = chain_origin(result, roles.sampler_id)
    latent_ref = result[origin]["inputs"].get("latent_image")
    if not (is_ref(latent_ref) and latent_ref[0] in result
            and result[latent_ref[0]].get("class_type") == "EmptyLatentImage"):
        raise ValueError("元の canvas (EmptyLatentImage) を特定できません")
    canvas = result[latent_ref[0]]["inputs"]
    width, height = canvas["width"], canvas["height"]
    scale = math.sqrt(pixels / (width * height))
    target_width = round(width * scale / 8) * 8
    target_height = round(height * scale / 8) * 8
    if target_width < 8 or target_height < 8:
        raise ValueError(
            f"hires の大きさが小さすぎます: {target_width}x{target_height}")
    settings = sampler_settings(result, roles.sampler_id)
    sampler_inputs = result[roles.sampler_id]["inputs"]
    next_id = max(int(key) for key in result if key.isdecimal()) + 1
    upscale_id, resample_id = str(next_id), str(next_id + 1)
    result[upscale_id] = {"class_type": "LatentUpscale", "inputs": {
        "samples": [roles.sampler_id, 0], "upscale_method": "bicubic",
        "width": target_width, "height": target_height, "crop": "disabled"}}
    result[resample_id] = {"class_type": "KSampler", "inputs": {
        "model": sampler_inputs["model"],
        "positive": sampler_inputs["positive"],
        "negative": sampler_inputs["negative"],
        "latent_image": [upscale_id, 0], "seed": settings["seed"],
        "steps": settings["steps"], "cfg": settings["cfg"],
        "sampler_name": settings["sampler_name"],
        "scheduler": settings["scheduler"], "denoise": denoise}}
    result[roles.decode_id]["inputs"]["samples"] = [resample_id, 0]
    result[roles.save_id]["inputs"]["filename_prefix"] = prefix
    return result
