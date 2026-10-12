"""Blur one delivered picture's layers by depth and record the result."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..infrastructure.comfyui.dof_graph import dof_graph
from ..infrastructure.comfyui.timings import note_ingested
from ..infrastructure.persistence.run_state import JsonRunState, operation_state_path
from .cut_assets import DEPTH_ROLE, attach_cut, current_cut, reusable, stored_cut
from .ingest import (
    classify_dof_outputs,
    generation_key,
    open_request,
    record_job,
    upload_generation,
)
from .picture_source import is_delivered

DOF_OUTPUT = "dof の出力に dof はかけられません。納品の絵を指定してください"
LENS_OUTPUT = ("lens の出力に dof はかけられません。dof を先にかけて、"
               "その絵に lens をかけてください")
NOT_DELIVERED = "dof をかけられるのは納品の絵だけです。先に deliver してください"
NO_LAYERS = "この納品の絵には層が無いので、納品し直してください"


@dataclass(frozen=True)
class DofServices:
    management: object
    comfyui: object
    git_metadata: Callable[[], dict]
    notifier: object
    output_root: Path
    emit: Callable[[str], None] = print
    dof_graph: Callable[..., dict] = dof_graph
    state: JsonRunState = field(default_factory=JsonRunState)


def _layer_roles(parameters: dict) -> list[str]:
    roles = ["layer-figure", "layer-outline"]
    if not parameters.get("transparent"):
        roles.append("layer-backdrop")
    return roles


def _check_source(services: DofServices, generation_id: str,
                  context: dict) -> tuple[str, list[str]]:
    """The delivery's own source and the layer roles it carries."""
    request = context["request"]
    parameters = request.get("parameters") or {}
    if request.get("kind") == "dof" or parameters.get("kind") == "dof":
        raise SystemExit(DOF_OUTPUT)
    if request.get("kind") == "lens" or parameters.get("kind") == "lens":
        raise SystemExit(LENS_OUTPUT)
    if not is_delivered(request):
        raise SystemExit(NOT_DELIVERED)
    if parameters.get("kind") != "deliver":
        raise SystemExit(NO_LAYERS)
    roles = _layer_roles(parameters)
    present = {asset["role"] for asset in
               services.management.list_assets(generation_id)}
    if not set(roles) <= present:
        raise SystemExit(NO_LAYERS)
    source = (parameters.get("base_generation")
              or (context.get("refines_generation") or {}).get("id"))
    if not source:
        raise SystemExit(NO_LAYERS)
    return source, roles


def _request_parameters(generation_id: str, *, focus: tuple[float, float],
                        f_number: float, scope: dict, viewfinder: str) -> dict:
    return {"kind": "dof", "base_generation": generation_id,
            "focus": list(focus), "f_number": f_number, "scope": dict(scope),
            "viewfinder": viewfinder}


def dof(generation_id: str, services: DofServices, *,
        focus: tuple[float, float], f_number: float, scope: dict,
        viewfinder: str = "off", key_prefix: str | None = None,
        request_id: str | None = None, context: dict | None = None) -> dict:
    state_path = operation_state_path(services.output_root, "dof", key_prefix)
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
    state = services.state.load(state_path) if state_path else {}
    if state.get("status") == "completed" and state.get("result"):
        return state["result"]
    management = services.management
    if context is None:
        context = management.request(
            "GET", f"/api/v1/generations/{generation_id}/context")
    source_id, roles = _check_source(services, generation_id, context)

    prefix = f"dof-{generation_id}"
    token = uuid.uuid4().hex[:8]
    current = current_cut()
    stored, source_roles = stored_cut(services, source_id)
    depth_bytes = reusable(services, source_id, DEPTH_ROLE, stored,
                           source_roles, current)
    depth_image = source_image = None
    if depth_bytes is not None:
        depth_image = services.comfyui.upload_image(
            f"{prefix}-depth-in-{token}.png", depth_bytes)
    else:
        source_image = services.comfyui.upload_image(
            f"{prefix}-source-{token}.png",
            management.fetch_generation_image(source_id))
    uploaded = {role: services.comfyui.upload_image(
        f"{prefix}-{role}-{token}.png", management.fetch_asset(generation_id, role))
        for role in roles}
    graph = services.dof_graph(
        uploaded["layer-figure"], uploaded["layer-outline"],
        uploaded.get("layer-backdrop"), prefix, depth_image=depth_image,
        source_image=source_image, focus=focus, f_number=f_number, scope=scope,
        viewfinder=viewfinder)

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

    outputs = classify_dof_outputs(services.comfyui.wait_for(prompt_id))
    cut_roles = [DEPTH_ROLE] if depth_bytes is None else []
    pictures = ["dof", *(["viewfinder"] if viewfinder == "both" else [])]
    missing = [role for role in (*pictures, *cut_roles) if not outputs[role]]
    if missing:
        raise SystemExit(f"{prefix} is missing its {', '.join(missing)} output(s)")

    services.output_root.mkdir(parents=True, exist_ok=True)
    fetched = {}
    for role in (*cut_roles, *pictures):
        out = outputs[role][-1]
        fetched[role] = (out["filename"], services.comfyui.fetch(out))
    for name, data in fetched.values():
        (services.output_root / name).write_bytes(data)

    git = services.git_metadata()
    request = open_request(
        management, request_id=request_id,
        idempotency_key=key_prefix or str(uuid.uuid4()),
        resolution={
            "raw_instruction": f"{generation_id} に dof",
            "recipe": context["request"].get("recipe") or "yukari",
            "parameters": _request_parameters(
                generation_id, focus=focus, f_number=f_number, scope=scope,
                viewfinder=viewfinder),
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "references": [{"source_generation_id": generation_id,
                            "purpose": "rebuild", "aspect": "composition",
                            "instruction": "この納品の絵に dof"}],
        })
    job = record_job(management, request["id"], key_prefix=key_prefix, index=0,
                     seed=0, prompt_id=prompt_id, graph=graph,
                     source_generation_id=generation_id if request_id else None)
    ids, urls = [], []
    for index, role in enumerate(pictures):
        name, data = fetched[role]
        rendered = upload_generation(
            management, services.emit, job["id"], seed=0, name=name, data=data,
            index=index, idempotency_key=generation_key(key_prefix, 0, index))
        ids.append(rendered["id"])
        urls.append(rendered["canonical_url"])

    attach_cut(management, services.emit, source_id, key_prefix=key_prefix,
               fetched=fetched, stored=stored, roles=source_roles,
               current=current, cut_roles=cut_roles)

    management.request("PATCH", f"/api/v1/jobs/{job['id']}", {"status": "ingested"})
    note_ingested(services.comfyui, prompt_id)
    name, data = fetched["dof"]
    services.notifier.send(
        f"**dof** `{generation_id}`\n**file** `{name}`\n**chimera** {urls[0]}",
        name, data)
    services.emit(f"request {request.get('short_id') or request['id']} done")
    result = {"generation_ids": ids}
    if state_path:
        state.update({"status": "completed", "result": result})
        services.state.save(state_path, state)
    return result
