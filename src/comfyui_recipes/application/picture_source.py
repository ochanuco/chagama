"""Rules shared by the use cases that take a picture Generation as input."""

from __future__ import annotations

# "hires-chain" and the request kind "finalize" name pictures delivered before
# deliver existed; their rows are still read.
DELIVERED_KINDS = ("deliver", "dof", "lens", "hires-chain")
DERIVED_KINDS = ("repair", "masked_redraw", "redraw")


def is_delivered(request: dict) -> bool:
    """Whether the request's output is a delivered picture, not a drawing."""
    parameters = request.get("parameters") or {}
    kind = parameters.get("kind")
    return bool(request.get("kind") in ("deliver", "dof", "lens", "finalize")
                or kind in DELIVERED_KINDS
                or (kind == "repair" and parameters.get("deliver_only")))


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
