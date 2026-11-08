"""The reference kinds: `reference` reads an uploaded picture into tags,
`reference_render` draws the tags picked from that reading.

`reference` frees ComfyUI's models and has the reference env run the Anima
Tagger and d1 on the GPU in a subprocess; it proposes tags and
lists every candidate. `reference_render` needs no GPU: it turns the picked
tags into part patches, and the reference's pose into a `render.pose` patch
by `pose_follow` (`POSE_FOLLOWS`), and queues an ordinary generate.
"""

from __future__ import annotations

import base64
import io
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from PIL import Image

from ..domain.recipes import DEFAULT_RECIPE, RECIPES
from ..domain.yukari import reference as rules
from ..infrastructure.imaging.pose_skeleton import head_overflow
from .generate import validate_request
from .worker_channels import Management

# The tag rules in `rules` are written for this recipe's persona pose.
RECIPE = DEFAULT_RECIPE
# ComfyUI unloads on its own prompt thread after /free returns.
FREE_SETTLE_SECONDS = 2.0
SEED_LIMIT = 2**31
POSE_STRENGTH = 1.0
DEPTH_BLUR = 0.015
# off: the tags only. loose: the skeleton. firm: the depth, smoothed of its
# hems and folds, with the head and chest left to the tags, or the skeleton for a reference read before depth was kept.
POSE_FOLLOWS = ("off", "loose", "firm")
POSE_ASPECTS = (0.5, 2.0)
# A head the reference crops is drawn whole: the frame grows past it by its
# freed disc and this share of the disc beyond, and the tags that crop it go.
HEADROOM_MARGIN = 0.1
HEAD_CROP_TAGS = ("head out of frame", "close-up")


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
    # WD tagger の生スコア。None なら判定せず、解析のあとすぐ画像を消す。
    rate_image: Callable[[bytes], dict] | None = None
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


def _padded_picture(picture: str, pads: tuple[float, float, float, float]) -> str:
    with Image.open(io.BytesIO(base64.b64decode(picture))) as opened:
        source = opened.convert("L")
    width, height = source.size
    left, top, right, bottom = (round(pad * side) for pad, side in
                                zip(pads, (width, height, width, height)))
    padded = Image.new("L", (width + left + right, height + top + bottom), 0)
    padded.paste(source, (left, top))
    buffer = io.BytesIO()
    padded.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def with_headroom(figure: Mapping) -> Mapping:
    """`figure` with its frame grown so the head disc the depth leaves to the
    prompt lies inside it: the points and aspect moved, the pictures padded
    with black."""
    overflow = head_overflow(figure["points"], figure["aspect"])
    if not any(overflow):
        return figure
    left, top, right, bottom = (pad * (1 + HEADROOM_MARGIN) if pad else 0.0
                                for pad in overflow)
    across, down = 1 + left + right, 1 + top + bottom
    grown = {**figure,
             "points": [[(x + left) / across, (y + top) / down, score]
                        for x, y, score in figure["points"]],
             "aspect": figure["aspect"] * across / down}
    if "depth" in figure:
        grown["depth"] = _padded_picture(figure["depth"], (left, top, right, bottom))
    return grown


def shows_back(applied: Mapping[str, Sequence[str]]) -> bool:
    return any(tag in rules.BACK_TAGS for tags in applied.values() for tag in tags)


def pose_patch(figure: Mapping, follow: str, *,
               applied: Mapping[str, Sequence[str]] | None = None) -> dict:
    applied = applied or {}
    back = shows_back(applied)
    if follow == "firm" and "depth" in figure:
        # A back view's head and chest are her back: the prompt would turn them round.
        looking_back = any(rules.LOOKING_BACK in tags for tags in applied.values())
        value = {"control": "depth", "picture": figure["depth"],
                 "free_head": not back or looking_back, "free_chest": not back,
                 "points": figure["points"], "blur": DEPTH_BLUR}
    else:
        value = {"points": figure["points"]}
    return {"target": "render.pose", "op": "set",
            "value": {**value, "aspect": figure["aspect"], "strength": POSE_STRENGTH},
            "reason": "参考画像のポーズを写す"}


def pose_canvas(pose: str, aspect: float) -> tuple[int, int]:
    """The pose's default canvas area at the reference's width over height
    (held to `POSE_ASPECTS`), each side a multiple of 16."""
    default = RECIPES[RECIPE].recipe.render_spec(pose, 0, "")
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
                     figure: Mapping | None = None, pose_follow: str = "off") -> dict:
    patches = list(plan.patches)
    if figure and pose_follow != "off":
        patches += canvas_patches(plan.pose, figure["aspect"])
        patches.append(pose_patch(figure, pose_follow, applied=plan.applied))
    parameters = {"pose": plan.pose}
    if shows_back(plan.applied):
        parameters["angle"] = "from_behind"
    return {
        "schema_version": 1,
        "request": {"instruction": "参考画像から描く", "count": 1, "seeds": [seed]},
        "generation": {"recipe": RECIPE, "parameters": parameters,
                       "patches": patches},
        "semantic": {
            "summary": f"参考画像から: {plan.pose}",
            "attributes": {"reference": {
                "source_request": source, "pose": plan.pose,
                "applied": plan.applied,
                "pose_follow": pose_follow if figure else None}},
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
    if not _hand_over_for_rating(request_id, data, services):
        try:
            services.delete_image(request_id)
        except (SystemExit, Exception) as error:
            services.emit(f"  ! reference image delete failed: {error}")
    return result


def _hand_over_for_rating(request_id: str, data: bytes,
                          services: ReferenceServices) -> bool:
    """Send chimera the content rating; chimera then keeps the picture for a
    while only when the verdict allows reading it, and deletes it otherwise.
    False (rating unavailable, or chimera refused it) means the caller deletes
    the picture as before."""
    if services.rate_image is None:
        return False
    try:
        rating = services.rate_image(data)
        answer = services.management.request(
            "PUT", f"/api/v1/requests/{request_id}/reference-safety", rating)
    except (SystemExit, Exception) as error:
        services.emit(f"  ! reference safety rating skipped: {error}")
        return False
    verdict = (answer or {}).get("verdict") if isinstance(answer, Mapping) else None
    kept = (answer or {}).get("kept") if isinstance(answer, Mapping) else None
    services.emit(f"  reference safety: {verdict}, image {'kept' if kept else 'deleted'}")
    return True


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


def _without_head_crop(applied: Mapping[str, list[str]]) -> dict[str, list[str]]:
    kept = {part: [tag for tag in tags if tag not in HEAD_CROP_TAGS]
            for part, tags in applied.items()}
    return {part: tags for part, tags in kept.items() if tags}


def reference_render(row: Mapping, services: ReferenceServices) -> dict:
    """Queue the generate that draws the tags picked from a reference."""
    payload = row.get("payload") or {}
    source_id, pose, applied = _selection(payload)
    source = services.management.request("GET", f"/api/v1/requests/{source_id}") or {}
    if source.get("kind") != "reference" or source.get("status") != "done":
        raise SystemExit(f"request {source_id} is not a done reference")
    follow = payload.get("pose_follow", "loose")
    if follow not in POSE_FOLLOWS:
        raise SystemExit(f"reference_render payload.pose_follow must be one of {POSE_FOLLOWS}")
    figure = (source.get("result") or {}).get("figure")
    try:
        rules.check_selection((source.get("result") or {}).get("candidates") or [],
                              pose, applied)
        if figure and follow != "off":
            grown = with_headroom(figure)
            if grown is not figure:
                figure, applied = grown, _without_head_crop(applied)
        plan = rules.build(pose, applied)
    except ValueError as error:
        raise SystemExit(f"reference_render: {error}") from error
    seed = payload.get("seed")
    if seed is None:
        seed = services.random_seed()
    body = generate_request(plan, seed=seed,
                            source=source.get("short_id") or source_id,
                            figure=figure, pose_follow=follow)
    validate_request(body)
    queued = {"kind": "generate", "created_by": "system",
              "idempotency_key": f"reference_render:{row['id']}", "payload": body}
    if row.get("recipe_ref"):
        queued["recipe_ref"] = row["recipe_ref"]
    created = services.management.request("POST", "/api/v1/requests", queued)
    services.emit(f"reference_render {row['id']}: {pose} -> generate {created['id']}")
    return {"generation_ids": [], "generate_request_id": created["id"],
            "generate_request_short_id": created.get("short_id")}
