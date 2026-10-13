"""Set a delivery on a backdrop (stage), light it by that backdrop (lighting),
and record each result with the layers it now stands in."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.yukari.delivery_style import STROKE_LIGHTS, Light
from ..infrastructure.comfyui.stage_graph import lighting_graph, stage_graph
from ..infrastructure.comfyui.timings import note_ingested
from ..infrastructure.persistence.run_state import JsonRunState, operation_state_path
from .deliver import inherited_light
from .ingest import (
    asset_key,
    attach_asset,
    classify_lighting_outputs,
    classify_stage_outputs,
    generation_key,
    open_request,
    record_job,
    upload_generation,
)

STAGE_SOURCE = "背景を付けられるのは納品の絵（deliver の出力）だけです。先に deliver してください"
LIGHTING_SOURCE = ("光を当てられるのは背景を付けた絵（stage の出力、または背景ありの納品）"
                   "だけです。先に stage してください")
LIGHTING_TWICE = "この絵にはもう光が当たっています。stage の出力からかけ直してください"
NO_LAYERS = "この絵には層が無いので、作り直してください"
FIGURE_ROLES = ("layer-figure", "layer-outline")
LAYER_ROLES = (*FIGURE_ROLES, "layer-backdrop")


@dataclass(frozen=True)
class StageServices:
    management: object
    comfyui: object
    git_metadata: Callable[[], dict]
    notifier: object
    output_root: Path
    emit: Callable[[str], None] = print
    stage_graph: Callable[..., dict] = stage_graph
    lighting_graph: Callable[..., dict] = lighting_graph
    state: JsonRunState = field(default_factory=JsonRunState)


def _parameters(context: dict) -> dict:
    return context["request"].get("parameters") or {}


def _require_layers(services: StageServices, generation_id: str, roles) -> None:
    present = {asset["role"] for asset in services.management.list_assets(generation_id)}
    if not set(roles) <= present:
        raise SystemExit(NO_LAYERS)


def inherited_shadow(parameters: dict) -> str | None:
    """The direction a transparent delivery would have thrown its drop
    shadow from on a backdrop: its scene light, else its stroke light."""
    if not parameters.get("transparent"):
        return None
    light_from = (parameters.get("light") or {}).get("from")
    if light_from in STROKE_LIGHTS:
        return light_from
    stroke = parameters.get("stroke_light")
    return stroke if stroke in STROKE_LIGHTS else None


def _run(services: StageServices, *, prefix: str, graph: dict, state: dict,
         state_path) -> str:
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
    return prompt_id


def _fetch(services: StageServices, outputs: dict, roles, prefix: str) -> dict:
    missing = [role for role in roles if not outputs[role]]
    if missing:
        raise SystemExit(f"{prefix} is missing its {', '.join(missing)} output(s)")
    services.output_root.mkdir(parents=True, exist_ok=True)
    fetched = {}
    for role in roles:
        out = outputs[role][-1]
        fetched[role] = (out["filename"], services.comfyui.fetch(out))
        (services.output_root / out["filename"]).write_bytes(fetched[role][1])
    return fetched


def _record(services: StageServices, *, generation_id: str, context: dict,
            key_prefix: str | None, request_id: str | None, prompt_id: str,
            graph: dict, picture: tuple[str, bytes], layers: dict,
            raw_instruction: str, parameters: dict, references: list,
            title: str) -> dict:
    management = services.management
    git = services.git_metadata()
    request = open_request(
        management, request_id=request_id,
        idempotency_key=key_prefix or str(uuid.uuid4()),
        resolution={
            "raw_instruction": raw_instruction,
            "recipe": context["request"].get("recipe") or "yukari",
            "parameters": parameters,
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "references": references,
        })
    job = record_job(management, request["id"], key_prefix=key_prefix, index=0,
                     seed=0, prompt_id=prompt_id, graph=graph,
                     source_generation_id=generation_id if request_id else None)
    name, data = picture
    rendered = upload_generation(
        management, services.emit, job["id"], seed=0, name=name, data=data,
        index=0, idempotency_key=generation_key(key_prefix, 0, 0))
    for role, (layer_name, layer_data) in layers.items():
        attach_asset(management, rendered["id"], role=role, name=layer_name,
                     data=layer_data, idempotency_key=asset_key(key_prefix, 0, role))
        services.emit(f"{layer_name} -> {role} on {rendered['id']}")
    management.request("PATCH", f"/api/v1/jobs/{job['id']}", {"status": "ingested"})
    note_ingested(services.comfyui, prompt_id)
    services.notifier.send(
        f"**{title}** `{generation_id}`\n**file** `{name}`\n"
        f"**chimera** {rendered['canonical_url']}", name, data)
    services.emit(f"request {request.get('short_id') or request['id']} done")
    return {"generation_ids": [rendered["id"]]}


def _load_state(services: StageServices, operation: str, key_prefix: str | None):
    state_path = operation_state_path(services.output_root, operation, key_prefix)
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
    return state_path, (services.state.load(state_path) if state_path else {})


def _finish(services: StageServices, state_path, state: dict, result: dict) -> dict:
    if state_path:
        state.update({"status": "completed", "result": result})
        services.state.save(state_path, state)
    return result


def stage(generation_id: str, services: StageServices, *,
          photo_id: str | None = None, place: tuple[float, float] = (0.5, 0.95),
          height: float = 0.7, backdrop: str | None = None,
          shadow_from: str | None = None, inherit_shadow: bool = False,
          light: Light | None = None, inherit_light: bool = False,
          key_prefix: str | None = None, request_id: str | None = None,
          context: dict | None = None) -> dict:
    """Set the delivery `generation_id` in the photo `photo_id`, or on the
    `backdrop` pattern when there is none, in the `light` scene (inherited
    from the lineage's light redraw when `inherit_light`)."""
    state_path, state = _load_state(services, "stage", key_prefix)
    if state.get("status") == "completed" and state.get("result"):
        return state["result"]
    management = services.management
    if context is None:
        context = management.request("GET", f"/api/v1/generations/{generation_id}/context")
    parameters = _parameters(context)
    if parameters.get("kind") != "deliver":
        raise SystemExit(STAGE_SOURCE)
    _require_layers(services, generation_id, FIGURE_ROLES)
    if photo_id is not None:
        light = None
    elif inherit_light:
        light = inherited_light(management, generation_id, context)
    if photo_id is None and inherit_shadow:
        # A delivery in a scene throws its shadow away from the scene's light.
        shadow_from = inherited_shadow(parameters) or (light.direction if light else None)

    prefix = f"set-{generation_id}"
    token = uuid.uuid4().hex[:8]
    uploaded = {role: services.comfyui.upload_image(
        f"{prefix}-{role}-{token}.png", management.fetch_asset(generation_id, role))
        for role in FIGURE_ROLES}
    if photo_id is not None:
        photo_image = services.comfyui.upload_image(
            f"{prefix}-photo-{token}.png", management.fetch_generation_image(photo_id))
        graph = services.stage_graph(uploaded["layer-figure"], uploaded["layer-outline"],
                                     prefix, photo_image=photo_image, place=place,
                                     height=height)
        recorded = {"photo": photo_id, "place": list(place), "height": height}
        raw = f"{generation_id} を写真 {photo_id} に置く"
        references = [{"source_generation_id": photo_id, "purpose": "background",
                       "aspect": "composition", "instruction": "背景の写真"}]
    else:
        graph = services.stage_graph(uploaded["layer-figure"], uploaded["layer-outline"],
                                     prefix, backdrop=backdrop, shadow_from=shadow_from,
                                     outlines=parameters.get("outlines"),
                                     light=(light.scene, light.direction) if light else None)
        recorded = {"backdrop": backdrop, "shadow_from": shadow_from,
                    "light": {"scene": light.scene, "from": light.direction} if light else None}
        raw = f"{generation_id} に背景 {backdrop}"
        references = []

    prompt_id = _run(services, prefix=prefix, graph=graph, state=state,
                     state_path=state_path)
    outputs = classify_stage_outputs(services.comfyui.wait_for(prompt_id))
    fetched = _fetch(services, outputs, ("staged", *LAYER_ROLES), prefix)
    result = _record(
        services, generation_id=generation_id, context=context, key_prefix=key_prefix,
        request_id=request_id, prompt_id=prompt_id, graph=graph,
        picture=fetched["staged"], layers={role: fetched[role] for role in LAYER_ROLES},
        raw_instruction=raw,
        parameters={"kind": "stage", "base_generation": generation_id, **recorded},
        references=[{"source_generation_id": generation_id, "purpose": "rebuild",
                     "aspect": "composition", "instruction": "この納品の絵に背景を付ける"},
                    *references],
        title="stage")
    return _finish(services, state_path, state, result)


def lighting(generation_id: str, services: StageServices, *,
             direction: str | None, strength: float, scene: str | None = None,
             key_prefix: str | None = None, request_id: str | None = None,
             context: dict | None = None) -> dict:
    """Light the staged picture `generation_id`'s figure and outline by its
    backdrop, in the light `scene` when one is given (which also tints a
    pattern backdrop, never a photo)."""
    state_path, state = _load_state(services, "lighting", key_prefix)
    if state.get("status") == "completed" and state.get("result"):
        return state["result"]
    management = services.management
    if context is None:
        context = management.request("GET", f"/api/v1/generations/{generation_id}/context")
    parameters = _parameters(context)
    kind = parameters.get("kind")
    if kind == "lighting":
        raise SystemExit(LIGHTING_TWICE)
    if kind not in ("stage", "deliver") or (kind == "deliver" and parameters.get("transparent")):
        raise SystemExit(LIGHTING_SOURCE)
    _require_layers(services, generation_id, LAYER_ROLES)

    tint_backdrop = scene is not None and not (kind == "stage" and parameters.get("photo"))
    prefix = f"light-{generation_id}"
    token = uuid.uuid4().hex[:8]
    layer_bytes = {role: management.fetch_asset(generation_id, role) for role in LAYER_ROLES}
    uploaded = {role: services.comfyui.upload_image(f"{prefix}-{role}-{token}.png", data)
                for role, data in layer_bytes.items()}
    graph = services.lighting_graph(
        uploaded["layer-figure"], uploaded["layer-outline"], uploaded["layer-backdrop"],
        prefix, direction=direction, strength=strength, scene=scene,
        tint_backdrop=tint_backdrop)
    prompt_id = _run(services, prefix=prefix, graph=graph, state=state,
                     state_path=state_path)
    outputs = classify_lighting_outputs(services.comfyui.wait_for(prompt_id))
    fetched = _fetch(services, outputs, ("lit", *LAYER_ROLES), prefix)
    layers = {role: fetched[role] for role in LAYER_ROLES}
    result = _record(
        services, generation_id=generation_id, context=context, key_prefix=key_prefix,
        request_id=request_id, prompt_id=prompt_id, graph=graph, picture=fetched["lit"],
        layers=layers,
        raw_instruction=f"{generation_id} に背景の光",
        parameters={"kind": "lighting", "base_generation": generation_id,
                    "direction": direction, "strength": strength, "scene": scene},
        references=[{"source_generation_id": generation_id, "purpose": "rebuild",
                     "aspect": "composition", "instruction": "この絵を背景の光で照らす"}],
        title="lighting")
    return _finish(services, state_path, state, result)
