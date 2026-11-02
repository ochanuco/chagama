"""ComfyUI graph that stands a delivery's figure in a photo and blurs it by depth."""

from __future__ import annotations

from ...domain.yukari.delivery_style import PHOTO_FOCAL_LENGTH, PHOTO_FOCAL_LENGTHS
from .deliver_graph import depth_nodes

STAGED_SUFFIX = "-staged"


def photo_graph(photo_image: str, figure_image: str, outline_image: str,
                prefix: str, *, depth_image: str | None,
                photo_size: tuple[int, int] | None = None,
                place: tuple[float, float], height: float,
                stand_at: tuple[float, float] | None, nearer: float,
                focus: tuple[float, float] | str, f_number: float,
                tilt: float = 0.0, bokeh: float = 0.0,
                focal_length: str = PHOTO_FOCAL_LENGTH["default"]) -> dict:
    """The photo graph over the uploaded photo and layer images.

    The photo's depth is the uploaded `depth_image`, or is built from the
    photo (of `photo_size`) and saved under `prefix` + DEPTH_SUFFIX. The composite is saved
    under `prefix` + STAGED_SUFFIX.
    """
    graph: dict = {}
    cursor = 1

    def allocate() -> str:
        nonlocal cursor
        node_id = str(cursor)
        cursor += 1
        return node_id

    def load(image: str) -> str:
        node_id = allocate()
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": image}}
        return node_id

    photo_id = load(photo_image)
    figure_id, outline_id = load(figure_image), load(outline_image)
    if depth_image is not None:
        depth_ref = depth_nodes(graph, allocate, [], prefix, depth_image)
    else:
        depth_ref = depth_nodes(graph, allocate, [photo_id, 0], prefix, None, photo_size)

    stage_id = allocate()
    point = focus if focus != "subject" else (0.5, 0.5)
    stand = stand_at if stand_at is not None else (0.5, 0.5)
    graph[stage_id] = {"class_type": "YukariPhotoStage", "inputs": {
        "photo": [photo_id, 0], "depth": depth_ref,
        "figure": [figure_id, 0], "figure_mask": [figure_id, 1],
        "outline": [outline_id, 0], "outline_mask": [outline_id, 1],
        "place_x": place[0], "place_y": place[1], "height": height,
        "stand": "base" if stand_at is None else "point",
        "stand_x": stand[0], "stand_y": stand[1], "nearer": nearer,
        "focus": "subject" if focus == "subject" else "point",
        "focus_x": point[0], "focus_y": point[1], "f_number": f_number,
        "tilt": tilt, "bokeh": bokeh, "focal_mm": PHOTO_FOCAL_LENGTHS[focal_length]}}
    save = allocate()
    graph[save] = {"class_type": "SaveImage", "inputs": {
        "images": [stage_id, 0], "filename_prefix": prefix + STAGED_SUFFIX}}
    return graph
