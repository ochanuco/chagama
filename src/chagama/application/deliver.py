"""Deliver one picture Generation: cut it, decorate it, record the delivery."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.yukari import delivery_style
from ..domain.yukari.delivery_style import Light
from ..infrastructure.comfyui.deliver_graph import deliver_graph
from ..infrastructure.comfyui.timings import note_ingested
from ..infrastructure.persistence.run_state import JsonRunState, operation_state_path
from .cut_assets import (
    ALPHA_ROLE,
    attach_cut,
    current_cut,
    reusable,
    stored_cut,
)
from .ingest import (
    LAYER_ROLES,
    asset_key,
    attach_asset,
    classify_deliver_outputs,
    generation_key,
    open_request,
    record_job,
    upload_generation,
)
from .picture_source import is_delivered, stroke_light_conflict

# Passed for stroke_light to mean "use DELIVER_DEFAULTS' value".
RECIPE_DEFAULT = object()

LINEAGE_HOPS = 10
PICTURE_ONLY = "納品済みの絵は deliver できません。納品の元になる絵を指定してください"


@dataclass(frozen=True)
class DeliverServices:
    management: object
    comfyui: object
    graph_from_png: Callable[[bytes], dict | None]
    git_metadata: Callable[[], dict]
    notifier: object
    output_root: Path
    emit: Callable[[str], None] = print
    measure: Callable[[bytes], dict] | None = None
    deliver_graph: Callable[..., dict] = deliver_graph
    state: JsonRunState = field(default_factory=JsonRunState)


def inherited_light(management, generation_id: str, context: dict) -> Light | None:
    """The scene of the nearest `light` redraw in the picture's lineage,
    walking `base_generation` (or what the Generation refines) from the
    source itself."""
    record = context
    for _ in range(LINEAGE_HOPS + 1):
        parameters = (record.get("request") or {}).get("parameters") or {}
        if parameters.get("kind") == "redraw" and parameters.get("method") == "light":
            return Light(parameters["scene"],
                         parameters.get("from", delivery_style.LIGHT_FROM_DEFAULT))
        parent = (parameters.get("base_generation")
                  or (record.get("refines_generation") or {}).get("id"))
        if not parent:
            return None
        record = management.request("GET", f"/api/v1/generations/{parent}")
    return None


def _check_source(generation_id: str, services: DeliverServices, context: dict,
                  picked: bytes) -> None:
    request = context["request"]
    parameters = request.get("parameters") or {}
    kind = parameters.get("kind")
    if is_delivered(request):
        raise SystemExit(PICTURE_ONLY)
    repaired = kind in ("repair", "masked_redraw")
    base_id = parameters["base_generation"] if repaired else generation_id
    record = services.management.request("GET", f"/api/v1/generations/{base_id}")
    graph = (record.get("comfy_job") or {}).get("graph")
    if graph is None:
        graph = services.graph_from_png(
            services.management.fetch_generation_image(base_id)
            if repaired else picked)
    if graph and any(node.get("class_type") == "LayeredDiffusionApply"
                     for node in graph.values()):
        raise SystemExit("LayerDiffuse 由来の絵は deliver できません")


def _request_parameters(generation_id: str, *, repin: bool,
                        keep_scene: bool, transparent: bool,
                        backdrop: str | None, stroke_light: str | None,
                        outlines: list[dict], light: Light | None,
                        see_through: float = 1.0) -> dict:
    return {
        "kind": "deliver",
        "base_generation": generation_id,
        "repin": repin,
        **({"keep_scene": True} if keep_scene else {}),
        **({"transparent": True} if transparent else {}),
        **({"backdrop": backdrop} if backdrop else {}),
        **({"stroke_light": stroke_light} if stroke_light is not None else {}),
        "outlines": outlines,
        **({"light": {"scene": light.scene, "from": light.direction}}
           if light is not None else {}),
        **({"see_through": see_through} if see_through != 1.0 else {}),
    }


def deliver(generation_id: str, services: DeliverServices, *,
            repin: bool = True, keep_scene: bool = False,
            transparent: bool = False,
            backdrop: str | None = delivery_style.DELIVER_DEFAULTS["backdrop"],
            stroke_light: str | None | object = RECIPE_DEFAULT,
            outlines: list[dict] | None = None,
            light: Light | None = None, see_through: float = 1.0,
            key_prefix: str | None = None, request_id: str | None = None,
            context: dict | None = None) -> dict:
    state_path = operation_state_path(services.output_root, "deliver", key_prefix)
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
    state = services.state.load(state_path) if state_path else {}
    if state.get("status") == "completed" and state.get("result"):
        return state["result"]
    management = services.management
    if context is None:
        context = management.request(
            "GET", f"/api/v1/generations/{generation_id}/context")
    picked = management.fetch_generation_image(generation_id)
    _check_source(generation_id, services, context, picked)
    if light is None:
        light = inherited_light(management, generation_id, context)
        if (light is not None and stroke_light in delivery_style.STROKE_LIGHTS
                and stroke_light != light.direction):
            raise SystemExit(stroke_light_conflict(light.direction))
    if stroke_light is RECIPE_DEFAULT:
        stroke_light = (light.direction if light is not None
                        else delivery_style.DELIVER_DEFAULTS["stroke_light"])
    if outlines is None:
        outlines = delivery_style.DELIVER_DEFAULTS["outlines"]

    prefix = f"dlv-{generation_id}"
    current = current_cut()
    stored, roles = stored_cut(services, generation_id)
    alpha_bytes = reusable(services, generation_id, ALPHA_ROLE, stored, roles, current)

    token = uuid.uuid4().hex[:8]
    source_image = services.comfyui.upload_image(f"{prefix}-source-{token}.png", picked)
    alpha_image = (services.comfyui.upload_image(f"{prefix}-alpha-in-{token}.png", alpha_bytes)
                   if alpha_bytes is not None else None)
    graph = services.deliver_graph(
        source_image, delivery_style.MATTE_MODEL, prefix,
        alpha_image=alpha_image,
        repin=repin, keep_scene=keep_scene,
        transparent=transparent, backdrop=backdrop, stroke_light=stroke_light,
        outlines=outlines,
        light_scene=light.scene if light is not None else None,
        light_from=light.direction if light is not None else None,
        see_through=see_through)

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

    outputs = classify_deliver_outputs(services.comfyui.wait_for(prompt_id))
    cut_roles = [ALPHA_ROLE] if alpha_bytes is None else []
    layer_roles = [role for role in LAYER_ROLES
                   if role != "layer-backdrop" or keep_scene or not transparent]
    missing = [role for role in ("delivered", *cut_roles, *layer_roles)
               if not outputs[role]]
    if missing:
        raise SystemExit(
            f"{prefix} is missing its {', '.join(missing)} output(s)")

    services.output_root.mkdir(parents=True, exist_ok=True)
    fetched = {}
    for role in (*cut_roles, "delivered", *layer_roles):
        out = outputs[role][-1]
        fetched[role] = (out["filename"], services.comfyui.fetch(out))
    for name, data in fetched.values():
        (services.output_root / name).write_bytes(data)
    if services.measure is not None:
        summary = services.measure(fetched["delivered"][1])
        status = "FAIL" if summary["fails"] else "pass"
        services.emit(
            f"palette {status}: fig mid {summary['fig_sat_mean']:.1f} "
            f"p90 {summary['fig_sat_p90']:.0f} light {summary['light_sat']:.1f}")

    git = services.git_metadata()
    request = open_request(
        management, request_id=request_id,
        idempotency_key=key_prefix or str(uuid.uuid4()),
        resolution={
            "raw_instruction": f"{generation_id} を納品",
            "recipe": context["request"].get("recipe") or "yukari",
            "parameters": _request_parameters(
                generation_id, repin=repin,
                keep_scene=keep_scene, transparent=transparent,
                backdrop=backdrop, stroke_light=stroke_light,
                outlines=outlines, light=light, see_through=see_through),
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "references": [{"source_generation_id": generation_id,
                            "purpose": "rebuild", "aspect": "composition",
                            "instruction": "この絵の納品"}],
        })
    job = record_job(management, request["id"], key_prefix=key_prefix, index=0,
                     seed=0, prompt_id=prompt_id, graph=graph,
                     source_generation_id=generation_id if request_id else None)
    delivered_name, delivered = fetched["delivered"]
    rendered = upload_generation(
        management, services.emit, job["id"], seed=0, name=delivered_name,
        data=delivered, index=0, idempotency_key=generation_key(key_prefix, 0, 0))
    for role in layer_roles:
        name, data = fetched[role]
        attach_asset(management, rendered["id"], role=role, name=name, data=data,
                     idempotency_key=asset_key(key_prefix, 0, role))
        services.emit(f"{name} -> {role} on {rendered['id']}")

    attach_cut(management, services.emit, generation_id, key_prefix=key_prefix,
               fetched=fetched, stored=stored, roles=roles, current=current,
               cut_roles=cut_roles)

    management.request("PATCH", f"/api/v1/jobs/{job['id']}", {"status": "ingested"})
    note_ingested(services.comfyui, prompt_id)
    services.notifier.send(
        f"**deliver** `{generation_id}`\n"
        f"**file** `{delivered_name}`\n"
        f"**chimera** {rendered['canonical_url']}", delivered_name, delivered)
    services.emit(f"request {request.get('short_id') or request['id']} done")
    result = {"generation_ids": [rendered["id"]]}
    if state_path:
        state.update({"status": "completed", "result": result})
        services.state.save(state_path, state)
    return result
