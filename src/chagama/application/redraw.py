"""Redraw one picture Generation with a single pixel-changing method."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.generation.models import PromptPair
from ..domain.repair.regions import rects_from_fractions
from ..domain.yukari import delivery_style
from ..domain.yukari.delivery_style import Light
from ..domain.yukari.prompt_style import HEIGHT, WIDTH
from ..domain.yukari.recipe import refinement_prompt
from ..infrastructure.comfyui.base_graph import (
    BaseRoles,
    base_roles,
    hires_graph,
    sampler_settings,
)
from ..infrastructure.comfyui.light_graph import light_graph
from ..infrastructure.comfyui.refinement_graph import redraw_graph, sizes
from ..infrastructure.comfyui.timings import note_ingested
from ..infrastructure.imaging.delivery import image_size
from ..infrastructure.imaging.masks import render_soft_mask_png
from ..infrastructure.persistence.run_state import JsonRunState, operation_state_path
from .cut_assets import (
    ALPHA_ROLE,
    DEPTH_ROLE,
    attach_cut,
    current_cut,
    reusable,
    stored_cut,
)
from .ingest import (
    classify_redraw_outputs,
    generation_key,
    open_request,
    record_job,
    upload_generation,
)
from .picture_source import DERIVED_KINDS, check_resample_source, is_delivered

# `render_soft_mask_png`'s feather, as a share of the redraw canvas' longest side.
KEEP_FEATHER_FRACTION = 0.03

# Applies when `hires` is given without a denoise.
HIRES_DENOISE = 0.45

LINEAGE_HOPS = 10
DELIVERED_SOURCE = ("納品済みの絵は redraw できません。"
                    "描き直しの元になる絵を指定してください")


def hires_pixels(hires: int) -> int:
    """The area `hires` asks for: the standard canvas' area with its long
    side at `hires` px. Every aspect ratio gets that many pixels."""
    return round(hires * hires * WIDTH / HEIGHT)


@dataclass(frozen=True)
class RedrawServices:
    management: object
    comfyui: object
    graph_from_png: Callable[[bytes], dict | None]
    git_metadata: Callable[[], dict]
    notifier: object
    output_root: Path
    emit: Callable[[str], None] = print
    image_size: Callable[[bytes], tuple[int, int]] = image_size
    redraw_graph: Callable[..., dict] = redraw_graph
    hires_graph: Callable[..., dict] = hires_graph
    light_graph: Callable[..., dict] = light_graph
    state: JsonRunState = field(default_factory=JsonRunState)


@dataclass(frozen=True)
class _Source:
    context: dict
    picked: bytes
    derived: bool
    graph: dict
    roles: BaseRoles | None
    is_anima: bool


@dataclass(frozen=True)
class _Prepared:
    graph: dict
    seed: int
    parameters: dict
    cut_roles: tuple[str, ...] = ()
    stored: dict = field(default_factory=dict)
    asset_roles: frozenset = frozenset()


def _load_source(generation_id: str, services: RedrawServices,
                 context: dict | None) -> _Source:
    management = services.management
    if context is None:
        context = management.request(
            "GET", f"/api/v1/generations/{generation_id}/context")
    request = context["request"]
    if is_delivered(request):
        raise SystemExit(DELIVERED_SOURCE)
    picked = management.fetch_generation_image(generation_id)
    parameters = request.get("parameters") or {}
    derived = parameters.get("kind") in DERIVED_KINDS
    origin_id, record = generation_id, None
    for _ in range(LINEAGE_HOPS):
        if parameters.get("kind") not in DERIVED_KINDS:
            break
        origin_id = parameters.get("base_generation")
        if not origin_id:
            raise SystemExit("元の絵 (base_generation) をたどれません")
        record = management.request("GET", f"/api/v1/generations/{origin_id}")
        parameters = (record.get("request") or {}).get("parameters") or {}
    if parameters.get("kind") in DERIVED_KINDS:
        raise SystemExit("元の絵までの履歴が長すぎてたどれません")
    if record is None:
        record = management.request("GET", f"/api/v1/generations/{generation_id}")
    graph = (record.get("comfy_job") or {}).get("graph")
    if graph is None:
        graph = services.graph_from_png(
            management.fetch_generation_image(origin_id) if derived else picked)
    if graph is None:
        return _Source(context=context, picked=picked, derived=derived,
                       graph={}, roles=None, is_anima=False)
    if any(node.get("class_type") == "LayeredDiffusionApply"
           for node in graph.values()):
        raise SystemExit("LayerDiffuse 由来の絵は redraw できません")
    return _Source(
        context=context, picked=picked, derived=derived, graph=graph,
        roles=base_roles(graph),
        is_anima=any(node.get("class_type") == "UNETLoader"
                     for node in graph.values()))


def _require_anima(source: _Source) -> BaseRoles:
    if source.roles is None:
        raise SystemExit(
            "この絵には ComfyUI の graph が無いので、描き直しはできません")
    if not source.is_anima:
        raise SystemExit("描き直しができるのは Anima で描いた絵だけです")
    return source.roles


def _prepare_canvas(services: RedrawServices, source: _Source, prefix: str, *, denoise: float | None,
                    size: int | None, latent_route: bool | None,
                    finalizer: str | None, upscale: str | None,
                    keep_regions: Sequence[Sequence[float]],
                    keep_strength: float) -> _Prepared:
    roles = _require_anima(source)
    denoise = delivery_style.REDRAW_DENOISE if denoise is None else denoise
    size = delivery_style.REDRAW_SIZE if size is None else size
    latent_route = bool(latent_route) and not roles.stitched
    loader = finalizer or delivery_style.REDRAW_MODEL
    prompt = refinement_prompt(PromptPair(
        source.graph[roles.positive_id]["inputs"]["text"],
        source.graph[roles.negative_id]["inputs"]["text"]))
    source_image = None
    if source.derived:
        source_image = services.comfyui.upload_image(
            f"{prefix}-{uuid.uuid4().hex[:8]}-source.png", source.picked)
    keep_mask_image = None
    if keep_regions:
        redraw_width, redraw_height = sizes(*services.image_size(source.picked), size)
        keep_rects = rects_from_fractions(keep_regions, redraw_width, redraw_height)
        feather = round(KEEP_FEATHER_FRACTION * max(redraw_width, redraw_height))
        keep_mask_image = services.comfyui.upload_image(
            f"{prefix}-keep-mask.png",
            render_soft_mask_png(redraw_width, redraw_height, keep_rects,
                                 keep_strength, feather))
    graph = services.redraw_graph(
        source.graph, size, denoise, prefix,
        prompt=(prompt.positive, prompt.negative),
        latent_route=latent_route,
        sampler=delivery_style.REDRAW_SAMPLER, loader=loader,
        sampling=(delivery_style.REDRAW_STEPS, delivery_style.REDRAW_CFG),
        source_image=source_image, keep_mask_image=keep_mask_image,
        upscale=upscale or "bicubic",
        redraw_from_source=source.derived and not latent_route,
        canvas=services.image_size(source.picked))
    parameters = {
        "size": size, "denoise": denoise,
        **({"route": "latent"} if latent_route else {}),
        "finalizer": loader,
        **({"upscale": upscale} if upscale else {}),
        **({"keep_regions": [list(region) for region in keep_regions],
            "keep_strength": keep_strength} if keep_regions else {})}
    return _Prepared(graph, sampler_settings(source.graph, roles.sampler_id)["seed"],
                     parameters)


def _prepare_hires(source: _Source, prefix: str, services: RedrawServices, *,
                   hires: int, denoise: float) -> _Prepared:
    check_resample_source(
        "hires", has_graph=source.roles is not None, repaired=source.derived,
        stitched=source.roles is not None and source.roles.stitched,
        is_anima=source.is_anima)
    try:
        graph = services.hires_graph(
            source.graph, source.roles, hires_pixels(hires), denoise, prefix)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return _Prepared(
        graph, sampler_settings(source.graph, source.roles.sampler_id)["seed"],
        {"hires": hires, "denoise": denoise})


def _prepare_light(generation_id: str, services: RedrawServices,
                   source: _Source, prefix: str, light: Light) -> _Prepared:
    check_resample_source(
        "light", has_graph=source.roles is not None, repaired=False,
        stitched=source.roles is not None and source.roles.stitched,
        is_anima=source.is_anima)
    current = current_cut()
    stored, assets = stored_cut(services, generation_id)
    alpha_bytes = reusable(services, generation_id, ALPHA_ROLE, stored, assets, current)
    depth_bytes = reusable(services, generation_id, DEPTH_ROLE, stored, assets, current)
    token = uuid.uuid4().hex[:8]
    upload = services.comfyui.upload_image
    source_image = upload(f"{prefix}-source-{token}.png", source.picked)
    alpha_image = (upload(f"{prefix}-alpha-in-{token}.png", alpha_bytes)
                   if alpha_bytes is not None else None)
    depth_image = (upload(f"{prefix}-depth-in-{token}.png", depth_bytes)
                   if depth_bytes is not None else None)
    try:
        graph = services.light_graph(
            source.graph, source.roles, source_image, delivery_style.MATTE_MODEL,
            light.scene, light.direction, prefix, cut=True,
            alpha_image=alpha_image, depth_image=depth_image)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    cut_roles = tuple(role for role, found in (
        (ALPHA_ROLE, alpha_bytes), (DEPTH_ROLE, depth_bytes)) if found is None)
    return _Prepared(
        graph, sampler_settings(source.graph, source.roles.sampler_id)["seed"],
        {"scene": light.scene, "from": light.direction},
        cut_roles=cut_roles, stored=stored, asset_roles=frozenset(assets))


def redraw(generation_id: str, services: RedrawServices, *, method: str,
           denoise: float | None = None, size: int | None = None,
           latent_route: bool | None = None, finalizer: str | None = None,
           upscale: str | None = None,
           keep_regions: Sequence[Sequence[float]] = (),
           keep_strength: float = 0.25, hires: int | None = None,
           light: Light | None = None, key_prefix: str | None = None,
           request_id: str | None = None, context: dict | None = None) -> dict:
    state_path = operation_state_path(services.output_root, "redraw", key_prefix)
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
    state = services.state.load(state_path) if state_path else {}
    if state.get("status") == "completed" and state.get("result"):
        return state["result"]
    management = services.management
    source = _load_source(generation_id, services, context)

    prefix = f"rdw-{generation_id}"
    if method == "canvas":
        prepared = _prepare_canvas(
            services, source, prefix, denoise=denoise, size=size,
            latent_route=latent_route, finalizer=finalizer, upscale=upscale,
            keep_regions=keep_regions, keep_strength=keep_strength)
    elif method == "hires":
        prepared = _prepare_hires(
            source, prefix, services, hires=hires, denoise=denoise)
    elif method == "light":
        prepared = _prepare_light(generation_id, services, source, prefix, light)
    else:
        raise SystemExit(f"unsupported redraw method: {method!r}")
    graph = prepared.graph

    prompt_id = state.get("prompt_id")
    knows = getattr(services.comfyui, "knows", None)
    if prompt_id and (knows is None or knows(prompt_id)):
        services.emit(f"{prefix} resume {prompt_id}")
    else:
        prompt_id = services.comfyui.submit(graph)
        if state_path:
            state["prompt_id"] = prompt_id
            services.state.save(state_path, state)
    services.emit(f"{prefix} {prompt_id}")

    outputs = classify_redraw_outputs(services.comfyui.wait_for(prompt_id))
    missing = [role for role in ("picture", *prepared.cut_roles)
               if not outputs[role]]
    if missing:
        raise SystemExit(f"{prefix} is missing its {', '.join(missing)} output(s)")
    services.output_root.mkdir(parents=True, exist_ok=True)
    fetched = {}
    for role in ("picture", *prepared.cut_roles):
        out = outputs[role][-1]
        fetched[role] = (out["filename"], services.comfyui.fetch(out))
        (services.output_root / out["filename"]).write_bytes(fetched[role][1])
    name, data = fetched["picture"]

    git = services.git_metadata()
    request = open_request(
        management, request_id=request_id,
        idempotency_key=key_prefix or str(uuid.uuid4()),
        resolution={
            "raw_instruction": f"{generation_id} を描き直し ({method})",
            "recipe": source.context["request"].get("recipe") or "yukari",
            "parameters": {"kind": "redraw", "method": method,
                           "base_generation": generation_id,
                           **prepared.parameters},
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "references": [{"source_generation_id": generation_id,
                            "purpose": "rebuild", "aspect": "composition",
                            "instruction": "この絵の描き直し"}],
        })
    job = record_job(management, request["id"], key_prefix=key_prefix, index=0,
                     seed=prepared.seed, prompt_id=prompt_id, graph=graph,
                     source_generation_id=generation_id if request_id else None)
    rendered = upload_generation(
        management, services.emit, job["id"], seed=prepared.seed, name=name,
        data=data, index=0, idempotency_key=generation_key(key_prefix, 0, 0))
    if prepared.cut_roles:
        attach_cut(management, services.emit, generation_id,
                   key_prefix=key_prefix, fetched=fetched,
                   stored=prepared.stored, roles=set(prepared.asset_roles),
                   current=current_cut(), cut_roles=list(prepared.cut_roles))
    management.request("PATCH", f"/api/v1/jobs/{job['id']}", {"status": "ingested"})
    note_ingested(services.comfyui, prompt_id)
    services.notifier.send(
        f"**redraw {method}** `{generation_id}`\n"
        f"**file** `{name}`\n"
        f"**chimera** {rendered['canonical_url']}", name, data)
    services.emit(f"request {request.get('short_id') or request['id']} done")
    result = {"generation_ids": [rendered["id"]]}
    if state_path:
        state.update({"status": "completed", "result": result})
        services.state.save(state_path, state)
    return result
