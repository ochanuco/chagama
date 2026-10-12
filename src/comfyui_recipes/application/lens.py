"""Distort one delivered picture through a lens and record the result."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..infrastructure.comfyui.lens_graph import lens_graph
from ..infrastructure.comfyui.timings import note_ingested
from ..infrastructure.persistence.run_state import JsonRunState, operation_state_path
from .ingest import (
    classify_lens_outputs,
    generation_key,
    open_request,
    record_job,
    upload_generation,
)
from .picture_source import is_delivered

NOT_DELIVERED = "lens をかけられるのは納品の絵だけです。先に deliver してください"
VIEWFINDER = ("ビューファインダーを重ねた絵には lens はかけられません。"
              "ビューファインダー無しの dof の絵を指定してください")


@dataclass(frozen=True)
class LensServices:
    management: object
    comfyui: object
    git_metadata: Callable[[], dict]
    notifier: object
    output_root: Path
    emit: Callable[[str], None] = print
    lens_graph: Callable[..., dict] = lens_graph
    state: JsonRunState = field(default_factory=JsonRunState)


def _output_index(context: dict) -> int | None:
    for generation in context.get("generations") or []:
        if generation.get("id") == context.get("id"):
            return generation.get("comfy_output_index")
    return None


def _check_source(context: dict) -> None:
    """A delivered picture, a dof picture or a lens picture, but not one that
    carries the viewfinder overlay, which a lens would bend along with it."""
    request = context["request"]
    parameters = request.get("parameters") or {}
    if not is_delivered(request):
        raise SystemExit(NOT_DELIVERED)
    if parameters.get("kind") == "dof":
        viewfinder = parameters.get("viewfinder", "off")
        if viewfinder == "on" or (viewfinder == "both" and _output_index(context) == 1):
            raise SystemExit(VIEWFINDER)


def _request_parameters(generation_id: str, *, center: tuple[float, float],
                        strength: float) -> dict:
    return {"kind": "lens", "base_generation": generation_id,
            "center": list(center), "strength": strength}


def lens(generation_id: str, services: LensServices, *,
         center: tuple[float, float], strength: float,
         key_prefix: str | None = None, request_id: str | None = None,
         context: dict | None = None) -> dict:
    state_path = operation_state_path(services.output_root, "lens", key_prefix)
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
    state = services.state.load(state_path) if state_path else {}
    if state.get("status") == "completed" and state.get("result"):
        return state["result"]
    management = services.management
    if context is None:
        context = management.request(
            "GET", f"/api/v1/generations/{generation_id}/context")
    _check_source(context)

    prefix = f"lens-{generation_id}"
    token = uuid.uuid4().hex[:8]
    picture = services.comfyui.upload_image(
        f"{prefix}-picture-{token}.png",
        management.fetch_generation_image(generation_id))
    graph = services.lens_graph(picture, prefix, center=center, strength=strength)

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

    outputs = classify_lens_outputs(services.comfyui.wait_for(prompt_id))
    if not outputs["lens"]:
        raise SystemExit(f"{prefix} is missing its lens output")
    out = outputs["lens"][-1]
    name, data = out["filename"], services.comfyui.fetch(out)
    services.output_root.mkdir(parents=True, exist_ok=True)
    (services.output_root / name).write_bytes(data)

    git = services.git_metadata()
    request = open_request(
        management, request_id=request_id,
        idempotency_key=key_prefix or str(uuid.uuid4()),
        resolution={
            "raw_instruction": f"{generation_id} に lens",
            "recipe": context["request"].get("recipe") or "yukari",
            "parameters": _request_parameters(
                generation_id, center=center, strength=strength),
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "references": [{"source_generation_id": generation_id,
                            "purpose": "rebuild", "aspect": "composition",
                            "instruction": "この納品の絵に lens"}],
        })
    job = record_job(management, request["id"], key_prefix=key_prefix, index=0,
                     seed=0, prompt_id=prompt_id, graph=graph,
                     source_generation_id=generation_id if request_id else None)
    rendered = upload_generation(
        management, services.emit, job["id"], seed=0, name=name, data=data,
        index=0, idempotency_key=generation_key(key_prefix, 0, 0))

    management.request("PATCH", f"/api/v1/jobs/{job['id']}", {"status": "ingested"})
    note_ingested(services.comfyui, prompt_id)
    services.notifier.send(
        f"**lens** `{generation_id}`\n**file** `{name}`\n"
        f"**chimera** {rendered['canonical_url']}", name, data)
    services.emit(f"request {request.get('short_id') or request['id']} done")
    result = {"generation_ids": [rendered["id"]]}
    if state_path:
        state.update({"status": "completed", "result": result})
        services.state.save(state_path, state)
    return result
