"""The `reference` kind: read an uploaded picture into tags and queue the
generate that draws them on the recipe.

The worker never draws here. It frees ComfyUI's models, has the reference env
run the Anima Tagger and d1 on the GPU in a subprocess, then queues an
ordinary generate whose patches carry the picked tags.
"""

from __future__ import annotations

import io
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from PIL import Image

from ..domain.yukari import reference as rules
from .generate import validate_request
from .worker_channels import Management

RECIPE = "yukari"
# ComfyUI unloads on its own prompt thread after /free returns.
FREE_SETTLE_SECONDS = 2.0
SEED_LIMIT = 2**31


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


def read_tags(image: bytes, services: ReferenceServices) -> rules.Plan:
    services.free_comfyui()
    services.sleep(FREE_SETTLE_SECONDS)
    found = services.analyze(image)
    return rules.plan(found["scores"], found["kept"], found["groups"],
                      found["verdicts"])


def generate_request(plan: rules.Plan, *, seed: int, source: str) -> dict:
    return {
        "schema_version": 1,
        "request": {"instruction": "参考画像から描く", "count": 1, "seeds": [seed]},
        "generation": {"recipe": RECIPE, "parameters": {"pose": plan.pose},
                       "patches": plan.patches},
        "semantic": {
            "summary": f"参考画像から: {plan.pose}",
            "attributes": {"reference": {
                "source_request": source, "pose": plan.pose,
                "applied": plan.applied, "dropped": plan.dropped}},
        },
    }


def reference(row: Mapping, services: ReferenceServices) -> dict:
    payload = row.get("payload") or {}
    if payload.get("recipe") != RECIPE:
        raise SystemExit(
            f"reference payload.recipe must be {RECIPE!r}, got {payload.get('recipe')!r}")
    seed = payload.get("seed")
    if seed is None:
        seed = services.random_seed()
    request_id = row["id"]
    data = services.fetch_image(request_id)
    if data is None:
        raise ReferenceImageMissing(f"no reference image for request {request_id}")
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    plan = read_tags(data, services)
    body = generate_request(plan, seed=seed, source=row.get("short_id") or request_id)
    validate_request(body)
    queued = {"kind": "generate", "created_by": "system",
              "idempotency_key": f"reference:{request_id}", "payload": body}
    if row.get("recipe_ref"):
        queued["recipe_ref"] = row["recipe_ref"]
    created = services.management.request("POST", "/api/v1/requests", queued)
    services.emit(f"reference {request_id}: {plan.pose} -> generate {created['id']}")
    try:
        services.delete_image(request_id)
    except (SystemExit, Exception) as error:
        services.emit(f"  ! reference image delete failed: {error}")
    result = {"generation_ids": [], "generate_request_id": created["id"],
              "pose": plan.pose, "applied": plan.applied, "dropped": plan.dropped}
    if created.get("short_id"):
        result["generate_request_short_id"] = created["short_id"]
    return result
