"""Classify a ComfyUI submission's outputs and record them into chimera."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

from ..infrastructure.comfyui.deliver_graph import ALPHA_SUFFIX, DEPTH_SUFFIX, FOREGROUND_SUFFIX
from ..infrastructure.comfyui.dof_graph import DOF_SUFFIX, VIEWFINDER_SUFFIX
from ..infrastructure.comfyui.lens_graph import LENS_SUFFIX
from ..infrastructure.comfyui.photo_graph import STAGED_SUFFIX
from ..infrastructure.comfyui.refinement_graph import (
    DELIVERED_SUFFIX,
    LAYER_BACKDROP_SUFFIX,
    LAYER_FIGURE_SUFFIX,
    LAYER_OUTLINE_SUFFIX,
)
from ..infrastructure.comfyui.stage_graph import LIT_SUFFIX, STAGE_SUFFIX
from ..infrastructure.imaging.safety import rate_image


def generation_key(key_prefix: str | None, job_index: int,
                   output_index: int) -> str:
    if key_prefix is not None:
        return f"{key_prefix}:job:{job_index}:gen:{output_index}"
    return str(uuid.uuid4())


def asset_key(key_prefix: str | None, job_index: int, role: str) -> str:
    if key_prefix is not None:
        return f"{key_prefix}:job:{job_index}:asset:{role}"
    return str(uuid.uuid4())


LAYER_ROLES = {"layer-figure": LAYER_FIGURE_SUFFIX,
               "layer-outline": LAYER_OUTLINE_SUFFIX,
               "layer-backdrop": LAYER_BACKDROP_SUFFIX}


def _by_suffix(outputs: list, suffixes: dict[str, str]) -> dict[str, list]:
    return {role: [out for out in outputs if suffix in out["filename"]]
            for role, suffix in suffixes.items()}


def classify_deliver_outputs(outputs: list) -> dict[str, list]:
    """A deliver submission's outputs by role: `alpha` when it cut it
    itself, the `delivered` picture and its `layer-*` layers."""
    return _by_suffix(outputs, {"alpha": ALPHA_SUFFIX,
                                "delivered": DELIVERED_SUFFIX, **LAYER_ROLES})


def classify_dof_outputs(outputs: list) -> dict[str, list]:
    """A dof submission's outputs by role: `depth` when it built it itself,
    the `dof` picture and its `viewfinder` twin."""
    return _by_suffix(outputs, {"depth": DEPTH_SUFFIX, "dof": DOF_SUFFIX,
                                "viewfinder": VIEWFINDER_SUFFIX})


def classify_lens_outputs(outputs: list) -> dict[str, list]:
    """A lens submission's outputs by role: the distorted `lens` picture."""
    return _by_suffix(outputs, {"lens": LENS_SUFFIX})


def classify_photo_outputs(outputs: list) -> dict[str, list]:
    """A photo submission's outputs by role: `depth` when it built the
    photo's depth itself, and the `staged` picture."""
    return _by_suffix(outputs, {"depth": DEPTH_SUFFIX, "staged": STAGED_SUFFIX})


def classify_stage_outputs(outputs: list) -> dict[str, list]:
    """A stage submission's outputs by role: the `staged` picture and its
    `layer-*` layers."""
    return _by_suffix(outputs, {"staged": STAGE_SUFFIX, **LAYER_ROLES})


def classify_lighting_outputs(outputs: list) -> dict[str, list]:
    """A lighting submission's outputs by role: the `lit` picture and its
    three layers."""
    return _by_suffix(outputs, {"lit": LIT_SUFFIX, **LAYER_ROLES})


def classify_redraw_outputs(outputs: list) -> dict[str, list]:
    """A redraw submission's outputs by role: the redrawn `picture`, plus
    `alpha` and `depth` when the graph cut them itself."""
    cuts = {"alpha": ALPHA_SUFFIX, "depth": DEPTH_SUFFIX}
    found = {role: [out for out in outputs if suffix in out["filename"]]
             for role, suffix in cuts.items()}
    found["picture"] = [out for out in outputs
                        if not any(suffix in out["filename"]
                                   for suffix in cuts.values())]
    return found


def classify_cut_outputs(outputs: list) -> dict[str, list]:
    """A cut submission's outputs by role: `alpha`, `depth` and `foreground`."""
    return _by_suffix(outputs, {"alpha": ALPHA_SUFFIX, "depth": DEPTH_SUFFIX,
                                "foreground": FOREGROUND_SUFFIX})


def open_request(management, *, request_id: str | None,
                 idempotency_key: str, resolution: dict,
                 run_id: str | None = None) -> dict:
    """Report a request's resolved values, creating an import request when
    no worker-claimed request exists."""
    if request_id is not None:
        return management.request(
            "PUT", f"/api/v1/requests/{request_id}/resolution", resolution)
    payload = {"kind": "import", "status": "done", "created_by": "system",
               "idempotency_key": idempotency_key, **resolution}
    if run_id is not None:
        payload["run_id"] = run_id
    return management.request("POST", "/api/v1/requests", payload)


def create_job(management, request_id: str, *, idempotency_key: str, seed: int,
               index: int, source_generation_id: str | None = None) -> dict:
    payload = {"idempotency_key": idempotency_key, "seed": seed, "index": index}
    if source_generation_id is not None:
        payload["source_generation_id"] = source_generation_id
    return management.request(
        "POST", f"/api/v1/requests/{request_id}/jobs", payload)


def record_job(management, request_id: str, *, key_prefix: str | None,
               index: int, seed: int, prompt_id: str, graph: dict,
               source_generation_id: str | None = None) -> dict:
    job_key = (f"{key_prefix}:job:{index}"
               if key_prefix is not None else str(uuid.uuid4()))
    job = create_job(management, request_id, idempotency_key=job_key,
                     seed=seed, index=index,
                     source_generation_id=source_generation_id)
    management.request(
        "PATCH", f"/api/v1/jobs/{job['id']}",
        {"status": "queued", "comfy_prompt_id": prompt_id, "graph": graph})
    management.request(
        "PATCH", f"/api/v1/jobs/{job['id']}", {"status": "completed"})
    return job


def upload_generation(management, emit, job_id: str, *, seed: int, name: str,
                      data: bytes, index: int,
                      idempotency_key: str | None = None) -> dict:
    rendered = management.request(
        "POST", f"/api/v1/jobs/{job_id}/generations",
        multipart=({"seed": seed, "original_filename": name,
                    "comfy_output_index": index,
                    "idempotency_key": idempotency_key or str(uuid.uuid4())},
                   "image", name, data, "image/png"))
    emit(f"{name} -> {rendered['canonical_url']}")
    rate_generation(management, emit, rendered["id"], data)
    return rendered


def rate_generation(management, emit, generation_id: str, data: bytes) -> None:
    """Rate an uploaded image and send chimera the numbers; never fails the job."""
    try:
        management.request(
            "PUT", f"/api/v1/generations/{generation_id}/safety",
            rate_image(data))
    except (Exception, SystemExit) as error:
        emit(f"safety rating skipped for {generation_id}: {error}")


def attach_asset(management, generation_id: str, *, role: str, name: str,
                 data: bytes, idempotency_key: str | None = None,
                 content_type: str = "image/png") -> None:
    management.request(
        "POST", f"/api/v1/generations/{generation_id}/assets",
        multipart=({"role": role,
                    "idempotency_key": idempotency_key or str(uuid.uuid4())},
                   "file", name, data, content_type))


def import_images(management, emit, *, images: Sequence[Path],
                  resolution: dict, idempotency_key: str,
                  seed: int = 0) -> dict:
    """Register image files as one import request with a single job.

    Resending the same key and resolution registers nothing twice.
    """
    request = open_request(management, request_id=None,
                           idempotency_key=idempotency_key,
                           resolution=resolution)
    job = create_job(management, request["id"],
                     idempotency_key=f"{idempotency_key}:job:0",
                     seed=seed, index=0)
    ids: list[str] = []
    urls: list[str] = []
    for index, path in enumerate(images):
        rendered = upload_generation(
            management, emit, job["id"], seed=seed, name=path.name,
            data=path.read_bytes(), index=index,
            idempotency_key=generation_key(idempotency_key, 0, index))
        ids.append(rendered["id"])
        urls.append(rendered["canonical_url"])
    return {"request_id": request["id"], "short_id": request.get("short_id"),
            "generation_ids": ids, "generation_urls": urls}
