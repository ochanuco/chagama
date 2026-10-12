"""Stand one delivered figure in a photo, blur the scene by depth and record it."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..infrastructure.comfyui.photo_graph import photo_graph
from ..infrastructure.comfyui.timings import note_ingested
from ..infrastructure.persistence.run_state import JsonRunState, operation_state_path
from .cut_assets import DEPTH_ROLE, attach_cut, current_cut, reusable, stored_cut
from .ingest import (
    classify_photo_outputs,
    generation_key,
    open_request,
    record_job,
    upload_generation,
)
from .picture_source import is_delivered

NOT_DELIVERED = "写真に立てられるのは納品の絵だけです。先に deliver してください"
NO_LAYERS = "この納品の絵には層が無いので、納品し直してください"
LAYER_ROLES = ("layer-figure", "layer-outline")


@dataclass(frozen=True)
class PhotoServices:
    management: object
    comfyui: object
    git_metadata: Callable[[], dict]
    notifier: object
    output_root: Path
    emit: Callable[[str], None] = print
    photo_graph: Callable[..., dict] = photo_graph
    state: JsonRunState = field(default_factory=JsonRunState)


def _check_source(services: PhotoServices, generation_id: str, context: dict) -> None:
    """A deliver output that carries its figure and outline layers."""
    request = context["request"]
    parameters = request.get("parameters") or {}
    if not is_delivered(request):
        raise SystemExit(NOT_DELIVERED)
    if parameters.get("kind") != "deliver":
        raise SystemExit(NO_LAYERS)
    present = {asset["role"] for asset in services.management.list_assets(generation_id)}
    if not set(LAYER_ROLES) <= present:
        raise SystemExit(NO_LAYERS)


def _request_parameters(generation_id: str, photo_id: str, *,
                        place: tuple[float, float], height: float,
                        stand_at: tuple[float, float] | None, nearer: float,
                        focus: tuple[float, float] | str, f_number: float,
                        tilt: float, bokeh: float) -> dict:
    return {"kind": "photo", "base_generation": generation_id, "photo": photo_id,
            "place": list(place), "height": height,
            "stand_at": list(stand_at) if stand_at is not None else None,
            "nearer": nearer,
            "focus": focus if isinstance(focus, str) else list(focus),
            "f_number": f_number, "tilt": tilt, "bokeh": bokeh}


def photo(generation_id: str, services: PhotoServices, *, photo_id: str,
          place: tuple[float, float], height: float,
          stand_at: tuple[float, float] | None, nearer: float,
          focus: tuple[float, float] | str, f_number: float, tilt: float = 0.0,
          bokeh: float = 0.0, key_prefix: str | None = None, request_id: str | None = None,
          context: dict | None = None) -> dict:
    state_path = operation_state_path(services.output_root, "photo", key_prefix)
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
    state = services.state.load(state_path) if state_path else {}
    if state.get("status") == "completed" and state.get("result"):
        return state["result"]
    management = services.management
    if context is None:
        context = management.request(
            "GET", f"/api/v1/generations/{generation_id}/context")
    _check_source(services, generation_id, context)

    prefix = f"stage-{generation_id}"
    token = uuid.uuid4().hex[:8]
    current = current_cut()
    stored, photo_roles = stored_cut(services, photo_id)
    depth_bytes = reusable(services, photo_id, DEPTH_ROLE, stored, photo_roles, current)
    depth_image = None
    if depth_bytes is not None:
        depth_image = services.comfyui.upload_image(
            f"{prefix}-depth-in-{token}.png", depth_bytes)
    photo_image = services.comfyui.upload_image(
        f"{prefix}-photo-{token}.png", management.fetch_generation_image(photo_id))
    layers = {role: services.comfyui.upload_image(
        f"{prefix}-{role}-{token}.png", management.fetch_asset(generation_id, role))
        for role in LAYER_ROLES}
    graph = services.photo_graph(
        photo_image, layers["layer-figure"], layers["layer-outline"], prefix,
        depth_image=depth_image, place=place, height=height, stand_at=stand_at,
        nearer=nearer, focus=focus, f_number=f_number, tilt=tilt, bokeh=bokeh)

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

    outputs = classify_photo_outputs(services.comfyui.wait_for(prompt_id))
    cut_roles = [DEPTH_ROLE] if depth_bytes is None else []
    missing = [role for role in ("staged", *cut_roles) if not outputs[role]]
    if missing:
        raise SystemExit(f"{prefix} is missing its {', '.join(missing)} output(s)")

    services.output_root.mkdir(parents=True, exist_ok=True)
    fetched = {}
    for role in (*cut_roles, "staged"):
        out = outputs[role][-1]
        fetched[role] = (out["filename"], services.comfyui.fetch(out))
    for name, data in fetched.values():
        (services.output_root / name).write_bytes(data)

    git = services.git_metadata()
    request = open_request(
        management, request_id=request_id,
        idempotency_key=key_prefix or str(uuid.uuid4()),
        resolution={
            "raw_instruction": f"{generation_id} を写真 {photo_id} に立てる",
            "recipe": context["request"].get("recipe") or "yukari",
            "parameters": _request_parameters(
                generation_id, photo_id, place=place, height=height,
                stand_at=stand_at, nearer=nearer, focus=focus, f_number=f_number,
                tilt=tilt, bokeh=bokeh),
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "references": [
                {"source_generation_id": generation_id, "purpose": "rebuild",
                 "aspect": "composition", "instruction": "この納品の絵を写真に立てる"},
                {"source_generation_id": photo_id, "purpose": "background",
                 "aspect": "composition", "instruction": "背景の写真"}],
        })
    job = record_job(management, request["id"], key_prefix=key_prefix, index=0,
                     seed=0, prompt_id=prompt_id, graph=graph,
                     source_generation_id=generation_id if request_id else None)
    name, data = fetched["staged"]
    rendered = upload_generation(
        management, services.emit, job["id"], seed=0, name=name, data=data,
        index=0, idempotency_key=generation_key(key_prefix, 0, 0))

    attach_cut(management, services.emit, photo_id, key_prefix=key_prefix,
               fetched=fetched, stored=stored, roles=photo_roles,
               current=current, cut_roles=cut_roles)

    management.request("PATCH", f"/api/v1/jobs/{job['id']}", {"status": "ingested"})
    note_ingested(services.comfyui, prompt_id)
    services.notifier.send(
        f"**photo** `{generation_id}` on `{photo_id}`\n**file** `{name}`\n"
        f"**chimera** {rendered['canonical_url']}", name, data)
    services.emit(f"request {request.get('short_id') or request['id']} done")
    result = {"generation_ids": [rendered["id"]]}
    if state_path:
        state.update({"status": "completed", "result": result})
        services.state.save(state_path, state)
    return result
