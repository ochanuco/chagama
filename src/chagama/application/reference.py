"""The reference kinds: `reference` reads an uploaded picture into tags,
`reference_render` draws the tags picked from that reading.

`reference` frees ComfyUI's models and has the reference env run the Anima
Tagger and d1 on the GPU in a subprocess; it proposes tags and
lists every candidate. `reference_render` needs no GPU: it turns the picked
tags into part patches, and the reference's pose into a `render.pose`
patch, and queues an ordinary generate.
"""

from __future__ import annotations

import io
import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from PIL import Image

from ..domain.yukari import reference as rules
from ..domain.yukari.recipe import render_spec
from .generate import validate_request
from .worker_channels import Management

RECIPE = "yukari"
# ComfyUI unloads on its own prompt thread after /free returns.
FREE_SETTLE_SECONDS = 2.0
SEED_LIMIT = 2**31
POSE_STRENGTH = 1.0
POSE_ASPECTS = (0.5, 2.0)


class ReferenceImageMissing(Exception):
    """chimera has no picture for this request (yet, or any more)."""


@dataclass(frozen=True)
class ReferenceServices:
    management: Management
    fetch_image: Callable[[str], bytes | None]
    delete_image: Callable[[str], None]
    free_comfyui: Callable[[], None]
    analyze: Callable[[bytes], Mapping]
    emit: Callable[[str], None] = print
    sleep: Callable[[float], None] = time.sleep
    random_seed: Callable[[], int] = field(
        default=lambda: random.randrange(SEED_LIMIT))


def analyze(image: bytes, services: ReferenceServices) -> dict:
    """The `reference` result: the proposed pose and tags, every candidate,
    and the figure's detected pose (None when no figure was found)."""
    services.free_comfyui()
    services.sleep(FREE_SETTLE_SECONDS)
    found = services.analyze(image)
    proposal = rules.plan(found["scores"], found["kept"], found["groups"],
                          found["verdicts"])
    return {
        "generation_ids": [],
        "pose": proposal.pose,
        "candidates": rules.candidate_rows(found["scores"], found["kept"],
                                           found["groups"], found["verdicts"],
                                           proposal),
        "applied": proposal.applied,
        "dropped": proposal.dropped,
        "figure": found.get("figure"),
    }


def pose_patch(figure: Mapping, strength: float) -> dict:
    return {"target": "render.pose", "op": "set",
            "value": {"points": figure["points"], "aspect": figure["aspect"],
                      "strength": strength},
            "reason": "参考画像のポーズを写す"}


def pose_canvas(pose: str, aspect: float) -> tuple[int, int]:
    """The pose's default canvas area at the reference's width over height
    (held to `POSE_ASPECTS`), each side a multiple of 16."""
    default = render_spec(pose, 0, "")
    pixels = default.width * default.height
    aspect = min(max(aspect, POSE_ASPECTS[0]), POSE_ASPECTS[1])
    return (round(math.sqrt(pixels * aspect) / 16) * 16,
            round(math.sqrt(pixels / aspect) / 16) * 16)


def canvas_patches(pose: str, aspect: float) -> list[dict]:
    width, height = pose_canvas(pose, aspect)
    return [{"target": f"render.{side}", "op": "set", "value": value,
             "reason": "参考画像の縦横比で描く"}
            for side, value in (("width", width), ("height", height))]


def generate_request(plan: rules.Plan, *, seed: int, source: str,
                     figure: Mapping | None = None,
                     pose_strength: float = 0.0) -> dict:
    patches = list(plan.patches)
    if figure and pose_strength:
        patches += canvas_patches(plan.pose, figure["aspect"])
        patches.append(pose_patch(figure, pose_strength))
    return {
        "schema_version": 1,
        "request": {"instruction": "参考画像から描く", "count": 1, "seeds": [seed]},
        "generation": {"recipe": RECIPE, "parameters": {"pose": plan.pose},
                       "patches": patches},
        "semantic": {
            "summary": f"参考画像から: {plan.pose}",
            "attributes": {"reference": {
                "source_request": source, "pose": plan.pose,
                "applied": plan.applied,
                "pose_strength": pose_strength if figure else None}},
        },
    }


def reference(row: Mapping, services: ReferenceServices) -> dict:
    """Analyse the uploaded picture; drawing is a later `reference_render`."""
    payload = row.get("payload") or {}
    if payload.get("recipe") != RECIPE:
        raise SystemExit(
            f"reference payload.recipe must be {RECIPE!r}, got {payload.get('recipe')!r}")
    request_id = row["id"]
    data = services.fetch_image(request_id)
    if data is None:
        raise ReferenceImageMissing(f"no reference image for request {request_id}")
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    result = analyze(data, services)
    services.emit(f"reference {request_id}: proposed {result['pose']}, "
                  f"{len(result['candidates'])} candidates")
    try:
        services.delete_image(request_id)
    except (SystemExit, Exception) as error:
        services.emit(f"  ! reference image delete failed: {error}")
    return result


def _selection(payload: Mapping) -> tuple[str, str, dict[str, list[str]]]:
    source = payload.get("reference_request_id")
    pose = payload.get("pose")
    applied = payload.get("applied")
    if not isinstance(source, str) or not source:
        raise SystemExit("reference_render payload.reference_request_id is required")
    if not isinstance(pose, str):
        raise SystemExit("reference_render payload.pose is required")
    if (not isinstance(applied, Mapping)
            or not all(isinstance(tags, list) and all(isinstance(tag, str) for tag in tags)
                       for tags in applied.values())):
        raise SystemExit("reference_render payload.applied must map parts to tag lists")
    return source, pose, {part: list(tags) for part, tags in applied.items()}


def reference_render(row: Mapping, services: ReferenceServices) -> dict:
    """Queue the generate that draws the tags picked from a reference."""
    payload = row.get("payload") or {}
    source_id, pose, applied = _selection(payload)
    source = services.management.request("GET", f"/api/v1/requests/{source_id}") or {}
    if source.get("kind") != "reference" or source.get("status") != "done":
        raise SystemExit(f"request {source_id} is not a done reference")
    try:
        rules.check_selection((source.get("result") or {}).get("candidates") or [],
                              pose, applied)
        plan = rules.build(pose, applied)
    except ValueError as error:
        raise SystemExit(f"reference_render: {error}") from error
    seed = payload.get("seed")
    if seed is None:
        seed = services.random_seed()
    strength = payload.get("pose_strength", POSE_STRENGTH)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) \
            or not 0 <= strength <= 2:
        raise SystemExit("reference_render payload.pose_strength must be between 0 and 2")
    body = generate_request(plan, seed=seed,
                            source=source.get("short_id") or source_id,
                            figure=(source.get("result") or {}).get("figure"),
                            pose_strength=float(strength))
    validate_request(body)
    queued = {"kind": "generate", "created_by": "system",
              "idempotency_key": f"reference_render:{row['id']}", "payload": body}
    if row.get("recipe_ref"):
        queued["recipe_ref"] = row["recipe_ref"]
    created = services.management.request("POST", "/api/v1/requests", queued)
    services.emit(f"reference_render {row['id']}: {pose} -> generate {created['id']}")
    return {"generation_ids": [], "generate_request_id": created["id"],
            "generate_request_short_id": created.get("short_id")}
