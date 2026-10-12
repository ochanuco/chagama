"""ComfyUI graph for the lens distortion of a delivered picture."""

from __future__ import annotations

LENS_SUFFIX = "-lens"


def lens_graph(picture_image: str, prefix: str, *, center: tuple[float, float],
               strength: float, shape: str = "fisheye",
               radius: float | None = None) -> dict:
    """The lens graph over the uploaded delivered picture, saved under
    `prefix` + LENS_SUFFIX. A `sphere` carries its `radius`."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": picture_image}},
        "2": {"class_type": "YukariLens", "inputs": {
            "image": ["1", 0], "mask": ["1", 1], "center_x": center[0],
            "center_y": center[1], "strength": strength, "shape": shape,
            **({"radius": radius} if shape == "sphere" else {})}},
        "3": {"class_type": "SaveImage", "inputs": {
            "images": ["2", 0], "filename_prefix": prefix + LENS_SUFFIX}},
    }
