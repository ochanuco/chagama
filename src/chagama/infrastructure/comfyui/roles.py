"""Stable node role names, written to `_meta.title` before a graph is submitted."""

from __future__ import annotations

import re
from collections.abc import Mapping

SAMPLERS = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced")
UPSCALES = ("LatentUpscale", "LatentUpscaleBy")
ENCODES = ("VAEEncode", "VAEEncodeForInpaint", "VAEEncodeTiled")
OUTPUTS = ("SaveImage", "PreviewImage")
LOADER_ROLES = {
    "UNETLoader": "unet_loader",
    "CLIPLoader": "clip_loader",
    "VAELoader": "vae_loader",
    "CheckpointLoaderSimple": "checkpoint_loader",
    "DiffusersLoader": "diffusers_loader",
    "LoraLoader": "lora_loader",
    "LoraLoaderModelOnly": "lora_loader",
    "ControlNetLoader": "controlnet_loader",
}
FIXED_ROLES = {**LOADER_ROLES, "SaveImage": "save_image"}
LATENT_KEYS = ("latent_image", "samples", "latent", "latent_samples")
PROMPT_KEYS = ("conditioning", "positive", "negative", "conditioning_1")


def snake_case(class_type: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", class_type)
    return re.sub(r"[^a-z0-9]+", "_", spaced.lower()).strip("_") or "node"


def _sort_key(node_id: str) -> tuple:
    return (0, int(node_id), "") if node_id.isdigit() else (1, 0, node_id)


def _ref(graph: Mapping, value: object) -> str | None:
    if (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            and value[0] in graph and isinstance(graph[value[0]], Mapping)):
        return value[0]
    return None


def _class(graph: Mapping, node_id: str) -> str:
    return str(graph[node_id].get("class_type", ""))


def _inputs(graph: Mapping, node_id: str) -> Mapping:
    inputs = graph[node_id].get("inputs")
    return inputs if isinstance(inputs, Mapping) else {}


def _topological(graph: Mapping) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()

    def visit(node_id: str) -> None:
        stack = [(node_id, iter(sorted(
            (ref for value in _inputs(graph, node_id).values()
             if (ref := _ref(graph, value))), key=_sort_key)))]
        seen.add(node_id)
        while stack:
            current, parents = stack[-1]
            for parent in parents:
                if parent not in seen:
                    seen.add(parent)
                    stack.append((parent, iter(sorted(
                        (ref for value in _inputs(graph, parent).values()
                         if (ref := _ref(graph, value))), key=_sort_key))))
                    break
            else:
                order.append(current)
                stack.pop()

    for node_id in sorted(graph, key=_sort_key):
        if node_id not in seen and isinstance(graph[node_id], Mapping):
            visit(node_id)
    return order


def _latent_chain(graph: Mapping, sampler_id: str) -> tuple[str, bool, list[str]]:
    """The class of the latent's origin, whether an upscale sits on the way,
    and the samplers upstream of `sampler_id`."""
    upscaled = False
    upstream: list[str] = []
    seen = {sampler_id}
    current = sampler_id
    while True:
        parent = next((ref for key in LATENT_KEYS
                       if (ref := _ref(graph, _inputs(graph, current).get(key)))), None)
        if parent is None or parent in seen:
            return ("" if parent is None else _class(graph, parent)), upscaled, upstream
        seen.add(parent)
        kind = _class(graph, parent)
        if kind in SAMPLERS:
            upstream.append(parent)
        elif kind in UPSCALES:
            upscaled = True
        elif kind.startswith("Empty") or kind in ENCODES or not any(
                key in _inputs(graph, parent) for key in LATENT_KEYS):
            return kind, upscaled, upstream
        current = parent


def _prompt_encoder(graph: Mapping, start: object) -> str | None:
    node_id = _ref(graph, start)
    seen: set[str] = set()
    while node_id is not None and node_id not in seen:
        seen.add(node_id)
        if _class(graph, node_id) == "CLIPTextEncode":
            return node_id
        node_id = next((ref for key in PROMPT_KEYS
                        if (ref := _ref(graph, _inputs(graph, node_id).get(key)))), None)
    return None


def _sampler_roles(graph: Mapping) -> dict[str, str]:
    groups: dict[str, list[tuple[int, str]]] = {}
    for node_id in graph:
        if not isinstance(graph[node_id], Mapping) or _class(graph, node_id) not in SAMPLERS:
            continue
        origin, upscaled, upstream = _latent_chain(graph, node_id)
        if upscaled:
            group = "hires"
        elif origin.startswith("Empty"):
            group = "base"
        elif origin in ENCODES:
            group = "redraw"
        else:
            continue
        groups.setdefault(group, []).append((len(upstream), node_id))
    roles = {}
    for group, members in groups.items():
        for position, (_, node_id) in enumerate(
                sorted(members, key=lambda item: (item[0], _sort_key(item[1])))):
            suffix = "" if position == 0 else f"_stage{position + 1}"
            roles[node_id] = f"{group}_sampler{suffix}"
    return roles


def _proposals(graph: Mapping) -> dict[str, str]:
    consumers: dict[str, set[str]] = {}
    for node_id, node in graph.items():
        for value in _inputs(graph, node_id).values() if isinstance(node, Mapping) else ():
            if ref := _ref(graph, value):
                consumers.setdefault(ref, set()).add(node_id)
    samplers = _sampler_roles(graph)
    proposed = dict(samplers)
    for sampler_id in sorted(samplers, key=lambda item: (samplers[item], _sort_key(item))):
        for key, role in (("positive", "positive_prompt"), ("negative", "negative_prompt")):
            encoder = _prompt_encoder(graph, _inputs(graph, sampler_id).get(key))
            if encoder is not None:
                proposed.setdefault(encoder, role)
    for node_id, node in graph.items():
        if not isinstance(node, Mapping) or node_id in proposed:
            continue
        kind = _class(graph, node_id)
        if kind in FIXED_ROLES:
            proposed[node_id] = FIXED_ROLES[kind]
        elif kind in UPSCALES:
            proposed[node_id] = "hires_upscale"
        elif kind == "VAEDecode" and any(
                _class(graph, consumer) in OUTPUTS for consumer in consumers.get(node_id, ())):
            proposed[node_id] = "vae_decode"
    return proposed


def label_roles(graph: Mapping) -> dict:
    """A copy of `graph` whose nodes carry a role in `_meta.title`.

    Titles a builder already set are kept. The rest follow the topology:
    samplers by where their latent comes from (`base_sampler` from an empty
    latent, `hires_sampler` after a latent upscale, `redraw_sampler` from an
    encoded picture; a further sampler on the same chain adds `_stage2`,
    `_stage3`), prompt encoders by the sampler input they feed, loaders and
    the decode feeding a save by class. Every other node is the snake_case
    class name. A name used twice gets `_2`, `_3` in topological order.
    """
    proposed = _proposals(graph)
    counts: dict[str, int] = {}
    titles: dict[str, str] = {}
    for node_id in _topological(graph):
        node = graph[node_id]
        meta = node.get("_meta")
        if isinstance(meta, Mapping) and meta.get("title"):
            continue
        role = proposed.get(node_id) or snake_case(_class(graph, node_id))
        counts[role] = counts.get(role, 0) + 1
        titles[node_id] = role if counts[role] == 1 else f"{role}_{counts[role]}"
    labelled = {}
    for node_id, node in graph.items():
        if node_id in titles:
            meta = node.get("_meta")
            node = {**node, "_meta": {**(meta if isinstance(meta, Mapping) else {}),
                                      "title": titles[node_id]}}
        labelled[node_id] = node
    return labelled


def role_of(node: Mapping) -> str | None:
    meta = node.get("_meta") if isinstance(node, Mapping) else None
    title = meta.get("title") if isinstance(meta, Mapping) else None
    return title or None
