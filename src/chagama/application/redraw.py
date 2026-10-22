"""Redraw one picture Generation with a single pixel-changing method."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.yukari import delivery_style
from ..domain.yukari.prompt_style import HEIGHT, WIDTH
from ..infrastructure.comfyui.base_graph import (
    BaseRoles,
    base_roles,
    hires_graph,
    sampler_settings,
)
from ..infrastructure.comfyui.deliver_graph import cut_graph
from ..infrastructure.comfyui.timings import note_ingested
from ..infrastructure.imaging.delivery import image_size
from ..infrastructure.imaging.matting import cutout_png
from ..infrastructure.persistence.run_state import JsonRunState, operation_state_path
from .cut_assets import (
    ALPHA_ROLE,
    CUTOUT_ROLE,
    DEPTH_ROLE,
    attach_cut,
    current_cut,
)
from .ingest import (
    asset_key,
    attach_asset,
    classify_cut_outputs,
    classify_redraw_outputs,
    generation_key,
    open_request,
    record_job,
    upload_generation,
)
from .picture_source import DERIVED_KINDS, check_resample_source, is_delivered

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
    hires_graph: Callable[..., dict] = hires_graph
    cut_graph: Callable[..., dict] = cut_graph
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


def _cut_output(services: RedrawServices, rendered_id: str, picture: tuple[str, bytes], *,
                prefix: str, key_prefix: str | None, state: dict, state_path) -> None:
    """Cut the redrawn `picture` as a deliver of it would and attach the
    alpha, depth and their `cut` description to `rendered_id`, so a deliver
    of it reuses them, with an RGBA `cutout` to show it by."""
    prompt_id = state.get("cut_prompt_id")
    knows = getattr(services.comfyui, "knows", None)
    if prompt_id and (knows is None or knows(prompt_id)):
        services.emit(f"{prefix} resume {prompt_id}")
    else:
        image = services.comfyui.upload_image(
            f"{prefix}-in-{uuid.uuid4().hex[:8]}.png", picture[1])
        prompt_id = services.comfyui.submit(
            services.cut_graph(image, delivery_style.MATTE_MODEL, prefix,
                                services.image_size(picture[1])))
        if state_path:
            state["cut_prompt_id"] = prompt_id
            services.state.save(state_path, state)
    services.emit(f"{prefix} {prompt_id}")
    outputs = classify_cut_outputs(services.comfyui.wait_for(prompt_id))
    missing = [role for role, found in outputs.items() if not found]
    if missing:
        raise SystemExit(f"{prefix} is missing its {', '.join(missing)} output(s)")
    fetched = {role: (found[-1]["filename"], services.comfyui.fetch(found[-1]))
               for role, found in outputs.items()}
    output_key = None if key_prefix is None else f"{key_prefix}:output"
    attach_cut(services.management, services.emit, rendered_id, key_prefix=output_key,
               fetched=fetched, stored={}, roles=set(), current=current_cut(),
               cut_roles=[ALPHA_ROLE, DEPTH_ROLE])
    name = f"{prefix}-cutout.png"
    attach_asset(services.management, rendered_id, role=CUTOUT_ROLE, name=name,
                 data=cutout_png(fetched["foreground"][1], fetched["alpha"][1]),
                 idempotency_key=asset_key(output_key, 0, CUTOUT_ROLE))
    services.emit(f"{name} -> {CUTOUT_ROLE} on {rendered_id}")


def redraw(generation_id: str, services: RedrawServices, *, method: str,
           denoise: float | None = None, hires: int | None = None,
           cut: bool = False, key_prefix: str | None = None,
           request_id: str | None = None, context: dict | None = None) -> dict:
    """Redraw `generation_id` by `method` and record the result; with `cut`,
    also cut the result as a deliver would (see `_cut_output`)."""
    state_path = operation_state_path(services.output_root, "redraw", key_prefix)
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
    state = services.state.load(state_path) if state_path else {}
    if state.get("status") == "completed" and state.get("result"):
        return state["result"]
    management = services.management
    source = _load_source(generation_id, services, context)

    prefix = f"rdw-{generation_id}"
    if method == "hires":
        prepared = _prepare_hires(
            source, prefix, services, hires=hires, denoise=denoise)
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
    if not outputs["picture"]:
        raise SystemExit(f"{prefix} is missing its picture output(s)")
    services.output_root.mkdir(parents=True, exist_ok=True)
    out = outputs["picture"][-1]
    name, data = out["filename"], services.comfyui.fetch(out)
    (services.output_root / name).write_bytes(data)

    git = services.git_metadata()
    request = open_request(
        management, request_id=request_id,
        idempotency_key=key_prefix or str(uuid.uuid4()),
        resolution={
            "raw_instruction": f"{generation_id} を描き直し ({method})",
            "recipe": source.context["request"].get("recipe") or "yukari",
            "parameters": {"kind": "redraw", "method": method,
                           "base_generation": generation_id,
                           **prepared.parameters, **({"cut": True} if cut else {})},
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
    management.request("PATCH", f"/api/v1/jobs/{job['id']}", {"status": "ingested"})
    note_ingested(services.comfyui, prompt_id)
    if cut:
        _cut_output(services, rendered["id"], (name, data), prefix=f"{prefix}-out",
                    key_prefix=key_prefix, state=state, state_path=state_path)
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
