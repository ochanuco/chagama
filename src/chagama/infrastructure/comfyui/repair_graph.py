"""ComfyUI graph transformation for the repair pass: a masked local redraw.

Crop the masked region, resample it in place with the source's own model,
LoRA and prompts, and stitch it back -- instead of the whole-canvas redraw
`refinement_graph.redraw_graph` builds.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from .base_graph import (
    PASSTHROUGH as _PASSTHROUGH,
    consumers as _consumers,
    find_decode as _find_decode,
    find_sampler as _find_sampler,
    is_ref as _is_ref,
    sampler_settings as _sampler_settings,
)

_INPAINT_CROP_DEFAULTS = {
    "downscale_algorithm": "lanczos",
    "upscale_algorithm": "lanczos",
    "preresize": False,
    "preresize_mode": "ensure minimum resolution",
    "preresize_min_width": 1024,
    "preresize_min_height": 1024,
    "preresize_max_width": 16384,
    "preresize_max_height": 16384,
    "mask_fill_holes": True,
    "mask_expand_pixels": 0,
    "mask_invert": False,
    "mask_blend_pixels": 32,
    "mask_hipass_filter": 0.1,
    "extend_for_outpainting": False,
    "extend_up_factor": 1.0,
    "extend_down_factor": 1.0,
    "extend_left_factor": 1.0,
    "extend_right_factor": 1.0,
    "context_from_mask_extend_factor": 1.5,
    "output_resize_to_target_size": True,
    "output_padding": "32",
    "device_mode": "gpu (much faster)",
}


@dataclass(frozen=True)
class RerollRefs:
    """The node refs the reroll's sampler is wired from, handed to hooks.

    A hook returns a copy (`dataclasses.replace`) with the refs it rerouted;
    anything it adds to the graph goes through the same `allocate`. Model
    hooks run before the reroll's text encode, so `positive`/`negative` are
    still None there; conditioning hooks see them filled in.
    """
    model: list
    positive_clip: list
    negative_clip: list
    vae: list
    cropped_image: list
    cropped_mask: list
    positive: list | None = None
    negative: list | None = None
    # KSampler inputs a hook wants over the source pass's own (`steps`, `cfg`,
    # `sampler_name`, `scheduler`); seed and denoise stay the caller's.
    sampler: Mapping[str, object] | None = None


RerollHook = Callable[[dict, Callable[[], str], RerollRefs], RerollRefs]


def _upstream(graph: Mapping, refs: list[list | None]) -> set[str]:
    """Every node reachable by following input refs from `refs`' target nodes."""
    keep: set[str] = set()
    stack = [ref[0] for ref in refs if ref is not None]
    while stack:
        node_id = stack.pop()
        if node_id in keep:
            continue
        keep.add(node_id)
        for value in graph[node_id].get("inputs", {}).values():
            if _is_ref(value) and value[0] in graph:
                stack.append(value[0])
    return keep


def _splice_reroll(graph: dict, allocate: Callable[[], str], *, image_ref: list,
                   mask_name: str, positive: str, negative: str,
                   model_ref: list, positive_clip_ref: list,
                   negative_clip_ref: list, vae_ref: list, steps: int, cfg: float,
                   sampler_name: str, scheduler: str, seed: int, denoise: float,
                   size: int, mask_expand_pixels: int = 0,
                   mask_blend_pixels: int = 32,
                   loras: Sequence[tuple[str, float]] = (),
                   model_hooks: Sequence[RerollHook] = (),
                   conditioning_hooks: Sequence[RerollHook] = ()) -> tuple[str, list]:
    """Adds the crop/resample/stitch reroll subgraph to `graph` (mutated).

    Returns `(crop_id, repaired_ref)`: `crop_id` so a caller that spliced
    `image_ref` from its own graph's output can exclude the crop's own input
    when rewiring every other consumer of that output; `repaired_ref` is the
    stitch node's `[id, 0]` output.

    `loras` chains `LoraLoader` nodes onto `model_ref`/`positive_clip_ref`
    ahead of the reroll's own KSampler/CLIPTextEncode. `negative_clip_ref`
    rides the same chain only when it started out equal to
    `positive_clip_ref` -- otherwise it keeps its own ref untouched.

    `model_hooks` run once the crop exists and before the text encode: a hook
    that swaps `model`/`positive_clip`/`negative_clip`/`vae` puts the reroll on
    another checkpoint (the LoRA chain above stays on the source's model and
    is simply left behind). `conditioning_hooks` run after the text encode and
    may reroute `positive`/`negative`, e.g. through a ControlNet fed from
    `cropped_image`.
    """
    if loras:
        share_negative = negative_clip_ref == positive_clip_ref
        clip_ref = positive_clip_ref
        for lora_name, weight in loras:
            lora_id = allocate()
            graph[lora_id] = {"class_type": "LoraLoader", "inputs": {
                "model": model_ref, "clip": clip_ref, "lora_name": lora_name,
                "strength_model": weight, "strength_clip": weight}}
            model_ref, clip_ref = [lora_id, 0], [lora_id, 1]
        positive_clip_ref = clip_ref
        if share_negative:
            negative_clip_ref = clip_ref

    load_mask = allocate()
    graph[load_mask] = {"class_type": "LoadImage", "inputs": {"image": mask_name}}
    to_mask = allocate()
    graph[to_mask] = {"class_type": "ImageToMask", "inputs": {
        "image": [load_mask, 0], "channel": "red"}}
    crop = allocate()
    graph[crop] = {"class_type": "InpaintCropImproved", "inputs": {
        **_INPAINT_CROP_DEFAULTS,
        "mask_expand_pixels": mask_expand_pixels,
        "mask_blend_pixels": mask_blend_pixels,
        "image": image_ref, "mask": [to_mask, 0],
        "output_target_width": size, "output_target_height": size}}
    stitcher_ref = [crop, 0]
    refs = RerollRefs(model=model_ref, positive_clip=positive_clip_ref,
                      negative_clip=negative_clip_ref, vae=vae_ref,
                      cropped_image=[crop, 1], cropped_mask=[crop, 2])
    for hook in model_hooks:
        refs = hook(graph, allocate, refs)

    positive_id = allocate()
    graph[positive_id] = {"class_type": "CLIPTextEncode", "inputs": {
        "clip": refs.positive_clip, "text": positive}}
    negative_id = allocate()
    graph[negative_id] = {"class_type": "CLIPTextEncode", "inputs": {
        "clip": refs.negative_clip, "text": negative}}
    refs = replace(refs, positive=[positive_id, 0], negative=[negative_id, 0])
    for hook in conditioning_hooks:
        refs = hook(graph, allocate, refs)

    encode = allocate()
    graph[encode] = {"class_type": "VAEEncode", "inputs": {
        "pixels": refs.cropped_image, "vae": refs.vae}}
    noise_mask = allocate()
    graph[noise_mask] = {"class_type": "SetLatentNoiseMask", "inputs": {
        "samples": [encode, 0], "mask": refs.cropped_mask}}
    sample = allocate()
    graph[sample] = {"class_type": "KSampler", "inputs": {
        "model": refs.model, "positive": refs.positive,
        "negative": refs.negative, "latent_image": [noise_mask, 0],
        "steps": steps, "cfg": cfg, "sampler_name": sampler_name,
        "scheduler": scheduler, **(refs.sampler or {}),
        "seed": seed, "denoise": denoise}}
    decode = allocate()
    graph[decode] = {"class_type": "VAEDecode", "inputs": {
        "samples": [sample, 0], "vae": refs.vae}}
    stitch = allocate()
    graph[stitch] = {"class_type": "InpaintStitchImproved", "inputs": {
        "stitcher": stitcher_ref, "inpainted_image": [decode, 0]}}
    return crop, [stitch, 0]


def _redraw_pass(graph: Mapping) -> dict:
    """The final sampler/decode's own model, CLIP, VAE refs and sampling settings."""
    decode_id = _find_decode(graph)
    sampler_id = _find_sampler(graph, decode_id)
    sampler_inputs = graph[sampler_id]["inputs"]
    model_ref = sampler_inputs["model"]
    apply_node = graph.get(model_ref[0], {})
    if apply_node.get("class_type") == "LayeredDiffusionApply":
        model_ref = apply_node["inputs"]["model"]
    settings = _sampler_settings(graph, sampler_id)
    return {
        "decode_id": decode_id,
        "model_ref": model_ref,
        "positive_clip_ref": graph[sampler_inputs["positive"][0]]["inputs"]["clip"],
        "negative_clip_ref": graph[sampler_inputs["negative"][0]]["inputs"]["clip"],
        "vae_ref": graph[decode_id]["inputs"]["vae"],
        "steps": settings["steps"],
        "cfg": settings["cfg"],
        "sampler_name": settings["sampler_name"],
        "scheduler": settings["scheduler"],
        "seed": settings["seed"],
    }


def _layerdiffuse_tail(graph: Mapping, consumers_map: Mapping[str, list[str]],
                       decode_id: str) -> tuple[str, str, str] | None:
    """`(ld_decode_id, invert_id, join_id)` for a `LayeredDiffusionDecode`
    consuming `decode_id`'s own image directly (a layerdiffuse tail: node 12
    `LayeredDiffusionApply`, 13 `LayeredDiffusionDecode`, 14 `InvertMask`, 15
    `JoinImageWithAlpha`), or `None` if the source is not shaped that way.
    """
    for node_id in consumers_map.get(decode_id, []):
        node = graph[node_id]
        if node.get("class_type") != "LayeredDiffusionDecode":
            continue
        images_ref = node["inputs"].get("images")
        if not (_is_ref(images_ref) and images_ref[0] == decode_id):
            continue
        invert_id = next(
            (consumer_id for consumer_id in consumers_map.get(node_id, [])
             if graph[consumer_id].get("class_type") == "InvertMask"
             and graph[consumer_id]["inputs"].get("mask") == [node_id, 1]), None)
        if invert_id is None:
            continue
        join_id = next(
            (consumer_id for consumer_id in consumers_map.get(node_id, [])
             if graph[consumer_id].get("class_type") == "JoinImageWithAlpha"
             and graph[consumer_id]["inputs"].get("image") == [node_id, 0]
             and graph[consumer_id]["inputs"].get("alpha") == [invert_id, 0]), None)
        if join_id is None:
            continue
        return node_id, invert_id, join_id
    return None


def _reachable_save(result: Mapping, consumers_map: Mapping[str, list[str]],
                    node_id: str) -> str | None:
    """The SaveImage reachable from `node_id` through `PASSTHROUGH` nodes only."""
    for consumer_id in consumers_map.get(node_id, []):
        class_type = result[consumer_id].get("class_type")
        if class_type == "SaveImage":
            return consumer_id
        if class_type in _PASSTHROUGH:
            found = _reachable_save(result, consumers_map, consumer_id)
            if found is not None:
                return found
    return None


def _pruned_reroll(source: Mapping, *, image_name: str, mask_name: str,
                   positive: str, negative: str, seed: int, denoise: float,
                   size: int, mask_expand_pixels: int, mask_blend_pixels: int,
                   loras: Sequence[tuple[str, float]] = (),
                   model_hooks: Sequence[RerollHook] = (),
                   conditioning_hooks: Sequence[RerollHook] = ()
                   ) -> tuple[dict, dict, str, Callable[[], str], list, str]:
    """Prunes `source` to the redraw pass's own loaders and splices a fresh
    crop/resample/stitch reroll off a staged source image.

    Returns `(result, source_copy, decode_id, allocate, repaired_ref,
    load_image)`: `result` is the pruned graph with the reroll already
    added, `source_copy` is the full (unpruned) deep copy of `source` a
    caller reattaching its own downstream tail still needs, `decode_id` is
    that tail's own root in `source_copy`, and `load_image` is the staged
    source's own `LoadImage` node id (its alpha channel feeds a
    layerdiffuse tail's own mask).
    """
    graph = json.loads(json.dumps(source))
    pass_ = _redraw_pass(graph)
    decode_id = pass_["decode_id"]

    # The loaders/LoRA the redraw needs -- not `sampler_id` itself, and not
    # the latent/pass-1 chain feeding it, since neither ref reaches those.
    keep = _upstream(graph, [pass_["model_ref"], pass_["vae_ref"],
                            pass_["positive_clip_ref"], pass_["negative_clip_ref"]])

    result = {node_id: graph[node_id] for node_id in keep}

    ids = itertools.count(max((int(key) for key in graph), default=0) + 1)

    def allocate() -> str:
        return str(next(ids))

    load_image = allocate()
    result[load_image] = {"class_type": "LoadImage", "inputs": {"image": image_name}}

    _, repaired_ref = _splice_reroll(
        result, allocate, image_ref=[load_image, 0], mask_name=mask_name,
        positive=positive, negative=negative, model_ref=pass_["model_ref"],
        positive_clip_ref=pass_["positive_clip_ref"],
        negative_clip_ref=pass_["negative_clip_ref"], vae_ref=pass_["vae_ref"],
        steps=pass_["steps"], cfg=pass_["cfg"], sampler_name=pass_["sampler_name"],
        scheduler=pass_["scheduler"], seed=seed, denoise=denoise, size=size,
        mask_expand_pixels=mask_expand_pixels, mask_blend_pixels=mask_blend_pixels,
        loras=loras, model_hooks=model_hooks, conditioning_hooks=conditioning_hooks)
    return result, graph, decode_id, allocate, repaired_ref, load_image


def _prune_and_splice(source: Mapping, *, image_name: str, mask_name: str,
                      positive: str, negative: str, seed: int, denoise: float,
                      size: int, prefix: str, mask_expand_pixels: int,
                      mask_blend_pixels: int,
                      loras: Sequence[tuple[str, float]] = (),
                      model_hooks: Sequence[RerollHook] = (),
                      conditioning_hooks: Sequence[RerollHook] = ()) -> dict:
    """Shared body of `repair_graph`/`masked_redraw_graph`: prune to the
    redraw pass's own loaders, keep the tail downstream of its decode, and
    splice a fresh crop/resample/stitch reroll off a staged source image.
    """
    result, graph, decode_id, allocate, repaired_ref, load_image = _pruned_reroll(
        source, image_name=image_name, mask_name=mask_name, positive=positive,
        negative=negative, seed=seed, denoise=denoise, size=size,
        mask_expand_pixels=mask_expand_pixels, mask_blend_pixels=mask_blend_pixels,
        loras=loras, model_hooks=model_hooks, conditioning_hooks=conditioning_hooks)
    stitch = repaired_ref[0]

    consumers = _consumers(graph)
    ld_tail = _layerdiffuse_tail(graph, consumers, decode_id)
    dropped = set(ld_tail[:2]) if ld_tail is not None else set()

    tail: set[str] = set()
    seen: set[str] = set()
    stack = list(consumers.get(decode_id, []))
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        if node_id not in dropped:
            tail.add(node_id)
        stack.extend(consumers.get(node_id, []))
    # A tail node can depend on a sibling not itself downstream of the decode
    # (a matte model loader feeding a background-removal node) -- pull those in too.
    dependency_stack = [
        value[0] for node_id in tail
        for value in graph[node_id].get("inputs", {}).values()
        if _is_ref(value) and value[0] in graph and value[0] != decode_id
        and value[0] not in dropped]
    while dependency_stack:
        node_id = dependency_stack.pop()
        # `result`'s ids (pruned loaders plus the reroll's fresh ones) never
        # collide with a pre-existing id from `graph`, so this membership
        # check is the same dedup `keep` would give.
        if node_id in result or node_id in tail or node_id in dropped:
            continue
        tail.add(node_id)
        dependency_stack.extend(
            value[0] for value in graph[node_id].get("inputs", {}).values()
            if _is_ref(value) and value[0] in graph and value[0] != decode_id
            and value[0] not in dropped)

    if ld_tail is not None:
        _, _, join_id = ld_tail
        graph[join_id]["inputs"]["image"] = repaired_ref
        graph[join_id]["inputs"]["alpha"] = [load_image, 1]

    for node_id in tail:
        node = graph[node_id]
        inputs = node.get("inputs", {})
        for key, value in list(inputs.items()):
            if _is_ref(value) and value[0] == decode_id:
                inputs[key] = repaired_ref
        result[node_id] = node

    save_id = _reachable_save(result, _consumers(result), stitch)
    if save_id is not None:
        result[save_id]["inputs"]["filename_prefix"] = prefix
    for node_id in tail:
        node = result[node_id]
        if node.get("class_type") != "SaveImage" or node_id == save_id:
            continue
        node["inputs"]["filename_prefix"] = prefix

    if save_id is None:
        save = allocate()
        result[save] = {"class_type": "SaveImage", "inputs": {
            "images": repaired_ref, "filename_prefix": prefix}}

    return result


def repair_graph(source: Mapping, *, image_name: str, mask_name: str,
                 positive: str, negative: str, seed: int, denoise: float,
                 size: int, prefix: str,
                 loras: Sequence[tuple[str, float]] = (),
                 model_hooks: Sequence[RerollHook] = (),
                 conditioning_hooks: Sequence[RerollHook] = ()) -> dict:
    return _prune_and_splice(
        source, image_name=image_name, mask_name=mask_name, positive=positive,
        negative=negative, seed=seed, denoise=denoise, size=size, prefix=prefix,
        mask_expand_pixels=_INPAINT_CROP_DEFAULTS["mask_expand_pixels"],
        mask_blend_pixels=_INPAINT_CROP_DEFAULTS["mask_blend_pixels"],
        loras=loras, model_hooks=model_hooks, conditioning_hooks=conditioning_hooks)


def masked_redraw_graph(source: Mapping, *, image_name: str, mask_name: str,
                        positive: str, negative: str, seed: int, denoise: float,
                        mask_padding: int, mask_feather: int, size: int,
                        prefix: str,
                        model_hooks: Sequence[RerollHook] = (),
                        conditioning_hooks: Sequence[RerollHook] = ()) -> dict:
    """Like `repair_graph`, but the mask expand/blend pixels are caller-given.

    `parts`/pose-driven regions do not apply here -- the mask is whatever
    rectangles the caller already rendered.
    """
    return _prune_and_splice(
        source, image_name=image_name, mask_name=mask_name, positive=positive,
        negative=negative, seed=seed, denoise=denoise, size=size, prefix=prefix,
        mask_expand_pixels=mask_padding, mask_blend_pixels=mask_feather,
        model_hooks=model_hooks, conditioning_hooks=conditioning_hooks)
