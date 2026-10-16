"""Rules shared by the use cases that take a picture Generation as input."""

from __future__ import annotations

import io

from PIL import Image

from .cut_assets import ALPHA_ROLE, CUTOUT_ROLE

# "hires-chain" and the request kind "finalize" name pictures delivered before
# deliver existed; their rows are still read.
DELIVERED_KINDS = ("deliver", "saturation", "dof", "lens", "photo", "stage", "lighting",
                   "hires-chain")
# "repair" and "masked_redraw" are no longer created; their stored outputs are
# still valid sources.
DERIVED_KINDS = ("repair", "masked_redraw", "redraw")


def is_delivered(request: dict) -> bool:
    """Whether the request's output is a delivered picture, not a drawing."""
    parameters = request.get("parameters") or {}
    kind = parameters.get("kind")
    return bool(request.get("kind") in ("deliver", "saturation", "dof", "lens", "photo",
                                        "stage", "lighting", "finalize")
                or kind in DELIVERED_KINDS
                or (kind == "repair" and parameters.get("deliver_only")))


FIGURE_LAYER = "layer-figure"
OUTLINE_LAYER = "layer-outline"
BACKDROP_LAYER = "layer-backdrop"
# What a cut redraw stands for as a delivery: transparent, with no outlines.
CUT_REDRAW_DELIVERY = {"transparent": True, "outlines": []}


def is_cut_redraw(parameters: dict) -> bool:
    """Whether the request is a redraw that also cut its result."""
    return parameters.get("kind") == "redraw" and bool(parameters.get("cut"))


def layer_roles_present(management, generation_id: str, context: dict) -> set[str]:
    """The layer roles `layer_bytes` can give for the picture: its own
    layer assets, or for a cut redraw the figure and outline made from its
    cutout (none when the cut assets are missing)."""
    present = {asset["role"] for asset in management.list_assets(generation_id)}
    if is_cut_redraw(context["request"].get("parameters") or {}):
        return ({FIGURE_LAYER, OUTLINE_LAYER}
                if {CUTOUT_ROLE, ALPHA_ROLE} <= present else set())
    return present


def layer_bytes(management, generation_id: str, context: dict, roles) -> dict[str, bytes]:
    """The PNG of each layer in `roles`. A cut redraw's figure is its cutout
    and its outline is fully transparent at the cutout's size."""
    if not is_cut_redraw(context["request"].get("parameters") or {}):
        return {role: management.fetch_asset(generation_id, role) for role in roles}
    cutout = management.fetch_asset(generation_id, CUTOUT_ROLE)
    with Image.open(io.BytesIO(cutout)) as image:
        size = image.size
    blank = io.BytesIO()
    Image.new("RGBA", size, (0, 0, 0, 0)).save(blank, "PNG")
    return {role: cutout if role == FIGURE_LAYER else blank.getvalue() for role in roles}


def delivery_parameters(management, context: dict) -> dict:
    """The request parameters of the delivery the picture stands on: its
    own, for a saturation the delivery it was made from, for a cut redraw
    a transparent delivery with no outlines."""
    parameters = context["request"].get("parameters") or {}
    if is_cut_redraw(parameters):
        return CUT_REDRAW_DELIVERY
    if parameters.get("kind") != "saturation":
        return parameters
    base = management.request(
        "GET", f"/api/v1/generations/{parameters['base_generation']}/context")
    return delivery_parameters(management, base)


def has_backdrop_layer(management, context: dict) -> bool:
    """Whether the picture's layers include a backdrop: not when the
    delivery under it (through saturation and lighting) is transparent."""
    parameters = context["request"].get("parameters") or {}
    kind = parameters.get("kind")
    if is_cut_redraw(parameters):
        return False
    if kind == "deliver":
        return not parameters.get("transparent")
    if kind in ("saturation", "lighting") and parameters.get("base_generation"):
        base = management.request(
            "GET", f"/api/v1/generations/{parameters['base_generation']}/context")
        return has_backdrop_layer(management, base)
    return True


def stroke_light_conflict(direction: str) -> str:
    return (f"stroke_light の向きは light の from（{direction}）と同じで"
            "なければなりません。even なら向きに関係なく使えます")


def check_resample_source(option: str, *, has_graph: bool, repaired: bool,
                          stitched: bool, is_anima: bool) -> None:
    """Refuse a source that `option` cannot re-sample from its stored graph."""
    if not has_graph:
        raise SystemExit(
            f"この絵には ComfyUI の graph が無いので、{option} は使えません")
    if repaired or stitched:
        raise SystemExit(
            f"repair や masked_redraw で直した絵には {option} は使えません")
    if not is_anima:
        raise SystemExit(f"{option} が使えるのは Anima で描いた絵だけです")
