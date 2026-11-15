"""Set a delivery on a backdrop (stage), light it by that backdrop (lighting),
compress its layers' saturation (saturation), and record each result with the
layers it now stands in."""

from __future__ import annotations

import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.yukari.delivery_style import STROKE_LIGHTS, Light
from ..domain.recipes import DEFAULT_RECIPE, backdrop_palette
from ..infrastructure.comfyui.refinement_graph import LAYER_FIGURE_SUFFIX, LAYER_OUTLINE_SUFFIX
from ..infrastructure.comfyui.stage_graph import (
    LAYER_FRONT_SUFFIX,
    lighting_graph,
    saturation_graph,
    stage_graph,
    trail_graph,
)
from ..domain.yukari import trail_style
from ..infrastructure.comfyui.timings import note_ingested
from ..infrastructure.imaging.stage import frame_canvas_share
from ..infrastructure.persistence.run_state import JsonRunState, operation_state_path
from .deliver import inherited_light
from .ingest import (
    asset_key,
    attach_asset,
    classify_lighting_outputs,
    classify_saturation_outputs,
    classify_stage_outputs,
    classify_trail_outputs,
    generation_key,
    open_request,
    record_job,
    upload_generation,
)
from .picture_source import (
    delivery_parameters,
    has_backdrop_layer,
    is_cut_redraw,
    layer_bytes,
    layer_roles_present,
)

STAGE_SOURCE = ("背景を付けられるのは納品の絵（deliver または saturation の出力）か"
                "切り抜いた描き直し（redraw の cut）だけです。先に deliver してください")
SATURATION_SOURCE = ("納品の絵で彩度を調整できるのは deliver の出力だけです。"
                     "描いた絵か描き直した絵なら、その絵の saturation を指定してください")
SATURATION_TWICE = ("この納品は deliver の saturation で彩度を圧縮済みです。"
                    "強さを変えるなら元の絵を saturation を指定して deliver し直してください")
LIGHTING_SOURCE = ("光を当てられるのは stage の出力、deliver / saturation の出力、"
                   "切り抜いた描き直し（redraw の cut）だけです。先に stage してください")
LIGHTING_NOTHING = ("背景の無い絵に当てられるのは direction か scene の光だけです。"
                    "どちらかを指定してください")
LIGHTING_TWICE = "この絵にはもう光が当たっています。stage の出力からかけ直してください"
NO_LAYERS = "この絵には層が無いので、作り直してください"
TRAIL_SOURCE = "軌跡を描けるのは stage の出力（背景を付けた絵）だけです"
TRAIL_TWICE = "この絵にはもう軌跡があります。stage の出力から描き直してください"
FRONT_ROLE = "layer-front"
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
    saturation_graph: Callable[..., dict] = saturation_graph
    trail_graph: Callable[..., dict] = trail_graph
    state: JsonRunState = field(default_factory=JsonRunState)


def _parameters(context: dict) -> dict:
    return context["request"].get("parameters") or {}


def _require_layers(services: StageServices, generation_id: str, context: dict,
                    roles) -> None:
    present = layer_roles_present(services.management, generation_id, context)
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
            "recipe": context["request"].get("recipe") or DEFAULT_RECIPE,
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
          frame: dict | None = None,
          key_prefix: str | None = None, request_id: str | None = None,
          context: dict | None = None) -> dict:
    """Set the delivery `generation_id` in the photo `photo_id`, or on the
    `backdrop` pattern when there is none (filling the `frame`'s box when
    one is given, the sides of its canvas it leaves None resolved from the
    layers),
    in the `light` scene (inherited from the lineage's light redraw when
    `inherit_light`). The result carries the frame as resolved."""
    state_path, state = _load_state(services, "stage", key_prefix)
    if state.get("status") == "completed" and state.get("result"):
        return state["result"]
    management = services.management
    if context is None:
        context = management.request("GET", f"/api/v1/generations/{generation_id}/context")
    parameters = _parameters(context)
    if parameters.get("kind") not in ("deliver", "saturation") and not is_cut_redraw(parameters):
        raise SystemExit(STAGE_SOURCE)
    _require_layers(services, generation_id, context, FIGURE_ROLES)
    parameters = delivery_parameters(management, context)
    if photo_id is not None:
        light = None
    elif inherit_light:
        light = inherited_light(management, generation_id, context)
    if photo_id is None and inherit_shadow:
        # A delivery in a scene throws its shadow away from the scene's light.
        shadow_from = inherited_shadow(parameters) or (light.direction if light else None)

    prefix = f"set-{generation_id}"
    token = uuid.uuid4().hex[:8]
    layers = layer_bytes(management, generation_id, context, FIGURE_ROLES)
    if frame is not None and (frame.get("canvas") is None or None in frame["canvas"]):
        frame = {**frame, "canvas": frame_canvas_share(
            layers["layer-figure"], layers["layer-outline"], frame)}
    uploaded = {role: services.comfyui.upload_image(f"{prefix}-{role}-{token}.png", data)
                for role, data in layers.items()}
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
                                     light=(light.scene, light.direction) if light else None,
                                     frame=frame,
                                     backdrop_palette=backdrop_palette(
                                         context["request"].get("recipe")))
        recorded = {"backdrop": backdrop, "shadow_from": shadow_from,
                    "light": {"scene": light.scene, "from": light.direction} if light else None,
                    **({"frame": frame} if frame is not None else {})}
        raw = (f"{generation_id} を枠に入れる（枠の中は {backdrop}）" if frame is not None
               else f"{generation_id} に背景 {backdrop}")
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
    if frame is not None:
        result["frame"] = frame
    return _finish(services, state_path, state, result)


def lighting(generation_id: str, services: StageServices, *,
             direction: str | None, strength: float, scene: str | None = None,
             key_prefix: str | None = None, request_id: str | None = None,
             context: dict | None = None) -> dict:
    """Light the staged picture `generation_id`'s figure and outline by its
    backdrop, in the light `scene` when one is given (which also tints a
    pattern backdrop, never a photo). A transparent picture has no backdrop
    to read, so only `direction` or `scene` shades it, and it stays
    transparent."""
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
    if kind not in ("stage", "trail", "deliver", "saturation") and not is_cut_redraw(parameters):
        raise SystemExit(LIGHTING_SOURCE)
    roles = LAYER_ROLES if has_backdrop_layer(management, context) else FIGURE_ROLES
    if len(roles) == len(FIGURE_ROLES) and direction is None and scene is None:
        raise SystemExit(LIGHTING_NOTHING)
    _require_layers(services, generation_id, context, roles)

    tint_backdrop = scene is not None and not (kind == "stage" and parameters.get("photo"))
    prefix = f"light-{generation_id}"
    token = uuid.uuid4().hex[:8]
    front_roles = ((FRONT_ROLE,) if FRONT_ROLE in layer_roles_present(
        management, generation_id, context) else ())
    layers = layer_bytes(management, generation_id, context, (*roles, *front_roles))
    uploaded = {role: services.comfyui.upload_image(f"{prefix}-{role}-{token}.png", data)
                for role, data in layers.items()}
    graph = services.lighting_graph(
        uploaded["layer-figure"], uploaded["layer-outline"], uploaded.get("layer-backdrop"),
        prefix, direction=direction, strength=strength, scene=scene,
        tint_backdrop=tint_backdrop, front_image=uploaded.get(FRONT_ROLE))
    front = layers.get(FRONT_ROLE)
    prompt_id = _run(services, prefix=prefix, graph=graph, state=state,
                     state_path=state_path)
    outputs = classify_lighting_outputs(services.comfyui.wait_for(prompt_id))
    fetched = _fetch(services, outputs, ("lit", *roles), prefix)
    layers = {role: fetched[role] for role in roles}
    if front is not None:
        layers[FRONT_ROLE] = (f"{prefix}{LAYER_FRONT_SUFFIX}.png", front)
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


def _back_box(parameters: dict) -> list[float] | None:
    """The stage frame's box as shares of the staged picture: the box is in
    shares of the delivery's canvas, which the framed `canvas` spans."""
    frame = parameters.get("frame")
    if not frame or not frame.get("box"):
        return None
    x0, y0, x1, y1 = frame.get("canvas") or (0.0, 0.0, 1.0, 1.0)
    bx0, by0, bx1, by1 = frame["box"]
    return [(bx0 - x0) / (x1 - x0), (by0 - y0) / (y1 - y0),
            (bx1 - x0) / (x1 - x0), (by1 - y0) / (y1 - y0)]


def trail(generation_id: str, services: StageServices, *, template: str,
          points: list, front: list, path: str, rule: str, width: float, color: str,
          head: str, head_at_end: bool, in_frame: bool, seed: int | None,
          key_prefix: str | None = None, request_id: str | None = None,
          context: dict | None = None) -> dict:
    """Draw a trail across the staged picture `generation_id`: the stretches
    behind the figure painted into its backdrop (inside the stage frame's
    box when `in_frame`), the rest over the figure as its front layer. The
    result carries the seed it drew with."""
    state_path, state = _load_state(services, "trail", key_prefix)
    if state.get("status") == "completed" and state.get("result"):
        return state["result"]
    management = services.management
    if context is None:
        context = management.request("GET", f"/api/v1/generations/{generation_id}/context")
    parameters = _parameters(context)
    if parameters.get("kind") == "trail":
        raise SystemExit(TRAIL_TWICE)
    if parameters.get("kind") != "stage":
        raise SystemExit(TRAIL_SOURCE)
    _require_layers(services, generation_id, context, LAYER_ROLES)
    if seed is None:
        seed = state.get("seed")
    if seed is None:
        seed = random.randint(0, trail_style.TRAIL_SEED_MAX)
    state["seed"] = seed
    if state_path:
        services.state.save(state_path, state)

    spec = {"points": points, "front": front, "path": path, "rule": rule, "width": width,
            "color": color, "head": head, "head_at_end": head_at_end, "seed": seed}
    back_box = _back_box(parameters) if in_frame else None
    prefix = f"trail-{generation_id}"
    token = uuid.uuid4().hex[:8]
    layers = layer_bytes(management, generation_id, context, LAYER_ROLES)
    uploaded = {role: services.comfyui.upload_image(f"{prefix}-{role}-{token}.png", data)
                for role, data in layers.items()}
    graph = services.trail_graph(uploaded["layer-figure"], uploaded["layer-outline"],
                                 uploaded["layer-backdrop"], prefix, trail=spec,
                                 back_box=back_box)
    prompt_id = _run(services, prefix=prefix, graph=graph, state=state,
                     state_path=state_path)
    outputs = classify_trail_outputs(services.comfyui.wait_for(prompt_id))
    fetched = _fetch(services, outputs, ("trailed", "layer-backdrop", FRONT_ROLE), prefix)
    recorded = {"template": template, **spec, "in_frame": in_frame}
    result = _record(
        services, generation_id=generation_id, context=context, key_prefix=key_prefix,
        request_id=request_id, prompt_id=prompt_id, graph=graph, picture=fetched["trailed"],
        layers={"layer-figure": (f"{prefix}{LAYER_FIGURE_SUFFIX}.png", layers["layer-figure"]),
                "layer-outline": (f"{prefix}{LAYER_OUTLINE_SUFFIX}.png", layers["layer-outline"]),
                "layer-backdrop": fetched["layer-backdrop"],
                FRONT_ROLE: fetched[FRONT_ROLE]},
        raw_instruction=f"{generation_id} に軌跡（{template}）",
        parameters={"kind": "trail", "base_generation": generation_id, **recorded},
        references=[{"source_generation_id": generation_id, "purpose": "rebuild",
                     "aspect": "composition", "instruction": "この絵に軌跡を描く"}],
        title="trail")
    result["seed"] = seed
    return _finish(services, state_path, state, result)


def saturation(generation_id: str, services: StageServices, *, strength: float,
               key_prefix: str | None = None, request_id: str | None = None,
               context: dict | None = None) -> dict:
    """Compress the saturation of the delivery `generation_id`'s figure and
    outline layers on repin's curve (`strength` 0..1 blends it in) and put
    them back together over its backdrop, if it has one. A drawing is
    compressed before its outlines instead (`deliver.saturate_drawing`)."""
    state_path, state = _load_state(services, "saturation", key_prefix)
    if state.get("status") == "completed" and state.get("result"):
        return state["result"]
    management = services.management
    if context is None:
        context = management.request("GET", f"/api/v1/generations/{generation_id}/context")
    parameters = _parameters(context)
    if parameters.get("kind") != "deliver":
        raise SystemExit(SATURATION_SOURCE)
    if parameters.get("saturation"):
        raise SystemExit(SATURATION_TWICE)
    _require_layers(services, generation_id, context, FIGURE_ROLES)
    has_backdrop = "layer-backdrop" in layer_roles_present(management, generation_id, context)
    roles = LAYER_ROLES if has_backdrop else FIGURE_ROLES

    prefix = f"sat-{generation_id}"
    token = uuid.uuid4().hex[:8]
    uploaded = {role: services.comfyui.upload_image(f"{prefix}-{role}-{token}.png", data)
                for role, data in layer_bytes(management, generation_id, context,
                                              roles).items()}
    graph = services.saturation_graph(
        uploaded["layer-figure"], uploaded["layer-outline"], prefix,
        backdrop_image=uploaded.get("layer-backdrop"), strength=strength)
    prompt_id = _run(services, prefix=prefix, graph=graph, state=state,
                     state_path=state_path)
    outputs = classify_saturation_outputs(services.comfyui.wait_for(prompt_id))
    fetched = _fetch(services, outputs, ("delivered", *roles), prefix)
    result = _record(
        services, generation_id=generation_id, context=context, key_prefix=key_prefix,
        request_id=request_id, prompt_id=prompt_id, graph=graph,
        picture=fetched["delivered"], layers={role: fetched[role] for role in roles},
        raw_instruction=f"{generation_id} の彩度を調整",
        parameters={"kind": "saturation", "base_generation": generation_id,
                    "strength": strength},
        references=[{"source_generation_id": generation_id, "purpose": "rebuild",
                     "aspect": "composition", "instruction": "この納品の絵の彩度を調整する"}],
        title="saturation")
    return _finish(services, state_path, state, result)
