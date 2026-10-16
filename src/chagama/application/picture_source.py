"""Rules shared by the use cases that take a picture Generation as input."""

from __future__ import annotations

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


def delivery_parameters(management, context: dict) -> dict:
    """The request parameters of the delivery the picture stands on: its
    own, or for a saturation the deliver it was made from."""
    parameters = context["request"].get("parameters") or {}
    if parameters.get("kind") != "saturation":
        return parameters
    base = management.request(
        "GET", f"/api/v1/generations/{parameters['base_generation']}/context")
    return base["request"].get("parameters") or {}


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
