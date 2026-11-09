"""Deliver one picture Generation: cut it, decorate it, record the delivery."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.yukari import delivery_style
from ..domain.yukari.delivery_style import Light
from ..domain.recipes import DEFAULT_RECIPE, backdrop_palette, deliver_outlines
from ..infrastructure.comfyui.deliver_graph import deliver_graph
from ..infrastructure.comfyui.timings import note_ingested
from ..infrastructure.imaging.matting import guide_alpha
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
SATURATION_TWICE = ("この絵は彩度を調整済みです。強さを変えるなら元の絵から"
                    "saturation をやり直してください")
KEEP_SCENE_SATURATED = "彩度を調整した絵は keep_scene で deliver できません"


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


def _request_parameters(generation_id: str, *,
                        keep_scene: bool, transparent: bool,
                        backdrop: str | None, stroke_light: str | None,
                        outlines: list[dict], light: Light | None,
                        see_through: float = 1.0, saturation: float = 0.0) -> dict:
    return {
        "kind": "deliver",
        "base_generation": generation_id,
        **({"keep_scene": True} if keep_scene else {}),
        **({"transparent": True} if transparent else {}),
        **({"backdrop": backdrop} if backdrop else {}),
        **({"stroke_light": stroke_light} if stroke_light is not None else {}),
        "outlines": outlines,
        **({"light": {"scene": light.scene, "from": light.direction}}
           if light is not None else {}),
        **({"see_through": see_through} if see_through != 1.0 else {}),
        **({"saturation": saturation} if saturation > 0.0 else {}),
    }


def _drawing_under(generation_id: str, services: DeliverServices,
                   context: dict) -> tuple[str, dict, float | None]:
    """The drawing the delivery is cut from, its context, and the strength
    its figure is compressed with when the source is a saturation of that
    drawing (None otherwise): such a delivery is the drawing's, at that
    strength, so the figure is compressed once."""
    parameters = context["request"].get("parameters") or {}
    if parameters.get("kind") == "saturation" and parameters.get("base_generation"):
        base_id = parameters["base_generation"]
        base = services.management.request(
            "GET", f"/api/v1/generations/{base_id}/context")
        if not is_delivered(base["request"]):
            return base_id, base, float(parameters.get("strength", 0.0))
    return generation_id, context, None


def deliver(generation_id: str, services: DeliverServices, *,
            keep_scene: bool = False,
            transparent: bool = False,
            backdrop: str | None = delivery_style.DELIVER_DEFAULTS["backdrop"],
            stroke_light: str | None | object = RECIPE_DEFAULT,
            outlines: list[dict] | None = None,
            light: Light | None = None, see_through: float = 1.0,
            saturation: float = delivery_style.DELIVER_DEFAULTS["saturation"],
            key_prefix: str | None = None, request_id: str | None = None,
            context: dict | None = None) -> dict:
    """Cut the picture out, compress its figure's saturation and draw the
    outlines around it. A saturation of a drawing is delivered from that
    drawing at the saturation's strength, whatever `saturation` says."""
    return _deliver(generation_id, services, keep_scene=keep_scene,
                    transparent=transparent, backdrop=backdrop,
                    stroke_light=stroke_light, outlines=outlines, light=light,
                    see_through=see_through, saturation=saturation,
                    key_prefix=key_prefix, request_id=request_id,
                    context=context, as_saturation=False)


def saturate_drawing(generation_id: str, services: DeliverServices, *,
                     strength: float, key_prefix: str | None = None,
                     request_id: str | None = None,
                     context: dict | None = None) -> dict:
    """The saturation step before the outlines: cut the drawing out with no
    outline and compress its figure at `strength`, as a deliver of it at
    that strength would, and record it as a saturation of the drawing."""
    return _deliver(generation_id, services, keep_scene=False, transparent=True,
                    backdrop=None, stroke_light=None, outlines=[], light=None,
                    see_through=1.0, saturation=strength,
                    key_prefix=key_prefix, request_id=request_id,
                    context=context, as_saturation=True)


def _deliver(generation_id: str, services: DeliverServices, *,
             keep_scene: bool, transparent: bool, backdrop: str | None,
             stroke_light: str | None | object, outlines: list[dict] | None,
             light: Light | None, see_through: float, saturation: float,
             key_prefix: str | None, request_id: str | None,
             context: dict | None, as_saturation: bool) -> dict:
    state_path = operation_state_path(
        services.output_root, "saturation" if as_saturation else "deliver", key_prefix)
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
    state = services.state.load(state_path) if state_path else {}
    if state.get("status") == "completed" and state.get("result"):
        return state["result"]
    management = services.management
    if context is None:
        context = management.request(
            "GET", f"/api/v1/generations/{generation_id}/context")
    drawing_id, drawing, compressed = _drawing_under(generation_id, services, context)
    if as_saturation and compressed is not None:
        raise SystemExit(SATURATION_TWICE)
    picked = management.fetch_generation_image(drawing_id)
    _check_source(drawing_id, services, drawing, picked)
    if compressed is not None:
        if keep_scene:
            raise SystemExit(KEEP_SCENE_SATURATED)
        saturation = compressed
    if light is None and not as_saturation:
        light = inherited_light(management, generation_id, context)
        if (light is not None and stroke_light in delivery_style.STROKE_LIGHTS
                and stroke_light != light.direction):
            raise SystemExit(stroke_light_conflict(light.direction))
    if stroke_light is RECIPE_DEFAULT:
        stroke_light = (light.direction if light is not None
                        else delivery_style.DELIVER_DEFAULTS["stroke_light"])
    if outlines is None:
        outlines = deliver_outlines(drawing["request"].get("recipe"))
    if keep_scene:
        saturation = 0.0

    prefix = f"dlv-{generation_id}"
    current = current_cut()
    stored, roles = stored_cut(services, drawing_id)
    alpha_bytes = reusable(services, drawing_id, ALPHA_ROLE, stored, roles, current)

    token = uuid.uuid4().hex[:8]
    source_image = services.comfyui.upload_image(f"{prefix}-source-{token}.png", picked)
    alpha_image = (services.comfyui.upload_image(f"{prefix}-alpha-in-{token}.png", alpha_bytes)
                   if alpha_bytes is not None else None)
    guide_image = None
    if alpha_image is None:
        guide = guide_alpha(picked)
        if guide is not None:
            guide_image = services.comfyui.upload_image(
                f"{prefix}-guide-{token}.png", guide)
    graph = services.deliver_graph(
        source_image, delivery_style.MATTE_MODEL, prefix,
        alpha_image=alpha_image,
        guide_image=guide_image,
        keep_scene=keep_scene,
        transparent=transparent, backdrop=backdrop, stroke_light=stroke_light,
        outlines=outlines,
        light_scene=light.scene if light is not None else None,
        light_from=light.direction if light is not None else None,
        see_through=see_through, saturation=saturation,
        backdrop_palette=(None if transparent or keep_scene
                          else backdrop_palette(drawing["request"].get("recipe"))))

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
    if as_saturation:
        parameters = {"kind": "saturation", "base_generation": generation_id,
                      "strength": saturation}
        raw_instruction, instruction = (f"{generation_id} の彩度を調整",
                                        "この絵の人物を切り抜いて彩度を調整する")
    else:
        parameters = _request_parameters(
            generation_id,
            keep_scene=keep_scene, transparent=transparent,
            backdrop=backdrop, stroke_light=stroke_light,
            outlines=outlines, light=light, see_through=see_through,
            saturation=saturation)
        raw_instruction, instruction = f"{generation_id} を納品", "この絵の納品"
    request = open_request(
        management, request_id=request_id,
        idempotency_key=key_prefix or str(uuid.uuid4()),
        resolution={
            "raw_instruction": raw_instruction,
            "recipe": context["request"].get("recipe") or DEFAULT_RECIPE,
            "parameters": parameters,
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "references": [{"source_generation_id": generation_id,
                            "purpose": "rebuild", "aspect": "composition",
                            "instruction": instruction}],
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

    attach_cut(management, services.emit, drawing_id, key_prefix=key_prefix,
               fetched=fetched, stored=stored, roles=roles, current=current,
               cut_roles=cut_roles)

    management.request("PATCH", f"/api/v1/jobs/{job['id']}", {"status": "ingested"})
    note_ingested(services.comfyui, prompt_id)
    services.notifier.send(
        f"**{'saturation' if as_saturation else 'deliver'}** `{generation_id}`\n"
        f"**file** `{delivered_name}`\n"
        f"**chimera** {rendered['canonical_url']}", delivered_name, delivered)
    services.emit(f"request {request.get('short_id') or request['id']} done")
    result = {"generation_ids": [rendered["id"]], "saturation": saturation,
              "outlines": outlines}
    if state_path:
        state.update({"status": "completed", "result": result})
        services.state.save(state_path, state)
    return result
