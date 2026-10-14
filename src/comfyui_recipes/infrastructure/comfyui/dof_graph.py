"""ComfyUI graph for the depth-of-field blur of a delivered picture's layers."""

from __future__ import annotations

from ...domain.yukari.delivery_style import PHOTO_FOCAL_LENGTH, PHOTO_FOCAL_LENGTHS
from .deliver_graph import depth_nodes

DOF_SUFFIX = "-dof"
# The dof picture with the camera viewfinder drawn over it.
VIEWFINDER_SUFFIX = "-viewfinder"


def dof_graph(figure_image: str, outline_image: str, backdrop_image: str | None,
              prefix: str, *, depth_image: str | None, source_image: str | None,
              focus: tuple[float, float], f_number: float, scope: dict,
              viewfinder: str) -> dict:
    """The dof graph over the uploaded layer images of a delivery.

    The depth is the uploaded `depth_image` of the delivery's source, or is
    built from the uploaded `source_image` and saved under `prefix` +
    DEPTH_SUFFIX. `viewfinder` 'on' draws the viewfinder over the saved
    picture; 'both' also saves the plain one under `prefix` + DOF_SUFFIX and
    the overlaid one under `prefix` + VIEWFINDER_SUFFIX.
    """
    if depth_image is None and source_image is None:
        raise ValueError("dof_graph needs a depth_image or a source_image")
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

    figure_id, outline_id = load(figure_image), load(outline_image)
    backdrop_id = load(backdrop_image) if backdrop_image is not None else None
    if depth_image is not None:
        depth_ref = depth_nodes(graph, allocate, [], prefix, depth_image)
    else:
        depth_ref = depth_nodes(graph, allocate, [load(source_image), 0], prefix, None)

    blur_id = allocate()
    graph[blur_id] = {"class_type": "YukariDepthOfField", "inputs": {
        "figure": [figure_id, 0], "figure_mask": [figure_id, 1],
        "outline": [outline_id, 0], "outline_mask": [outline_id, 1],
        "depth": depth_ref, "focus_x": focus[0], "focus_y": focus[1],
        "f_number": f_number, "blur_figure": scope["figure"],
        "blur_outline": scope["outline"], "blur_backdrop": scope["backdrop"],
        **({"backdrop": [backdrop_id, 0]} if backdrop_id is not None else {})}}
    picture_ref = [blur_id, 0]
    saved_ref = picture_ref
    if viewfinder != "off":
        view_id = allocate()
        graph[view_id] = {"class_type": "YukariViewfinder", "inputs": {
            "image": picture_ref, "focus_x": focus[0], "focus_y": focus[1],
            "f_number": f_number}}
        if viewfinder == "on":
            saved_ref = [view_id, 0]
        else:
            save_view = allocate()
            graph[save_view] = {"class_type": "SaveImage", "inputs": {
                "images": [view_id, 0],
                "filename_prefix": prefix + VIEWFINDER_SUFFIX}}
    save = allocate()
    graph[save] = {"class_type": "SaveImage", "inputs": {
        "images": saved_ref, "filename_prefix": prefix + DOF_SUFFIX}}
    return graph


def photo_dof_graph(figure_image: str, outline_image: str, backdrop_image: str,
                    prefix: str, *, depth_image: str | None, photo_image: str | None,
                    place: tuple[float, float], height: float,
                    stand_at: tuple[float, float] | None, nearer: float,
                    focus: tuple[float, float] | str, f_number: float,
                    tilt: float, bokeh: float, viewfinder: str,
                    focal_length: str = PHOTO_FOCAL_LENGTH["default"]) -> dict:
    """The dof graph over the layers of a delivery set in a photo.

    The depth is the photo's uploaded `depth_image`, or is built from the
    uploaded `photo_image` and saved under `prefix` + DEPTH_SUFFIX. The
    viewfinder (needing a focus point) is drawn as `dof_graph` does.
    """
    if depth_image is None and photo_image is None:
        raise ValueError("photo_dof_graph needs a depth_image or a photo_image")
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

    figure_id, outline_id = load(figure_image), load(outline_image)
    backdrop_id = load(backdrop_image)
    if depth_image is not None:
        depth_ref = depth_nodes(graph, allocate, [], prefix, depth_image)
    else:
        depth_ref = depth_nodes(graph, allocate, [load(photo_image), 0], prefix, None)
    point = focus if focus != "subject" else (0.5, 0.5)
    stand = stand_at if stand_at is not None else (0.5, 0.5)
    blur_id = allocate()
    graph[blur_id] = {"class_type": "YukariPhotoDepthOfField", "inputs": {
        "backdrop": [backdrop_id, 0], "depth": depth_ref,
        "figure": [figure_id, 0], "figure_mask": [figure_id, 1],
        "outline": [outline_id, 0], "outline_mask": [outline_id, 1],
        "place_x": place[0], "place_y": place[1], "height": height,
        "stand": "base" if stand_at is None else "point",
        "stand_x": stand[0], "stand_y": stand[1], "nearer": nearer,
        "focus": "subject" if focus == "subject" else "point",
        "focus_x": point[0], "focus_y": point[1], "f_number": f_number,
        "tilt": tilt, "bokeh": bokeh, "focal_mm": PHOTO_FOCAL_LENGTHS[focal_length]}}
    saved_ref = [blur_id, 0]
    if viewfinder != "off" and focus != "subject":
        view_id = allocate()
        graph[view_id] = {"class_type": "YukariViewfinder", "inputs": {
            "image": saved_ref, "focus_x": point[0], "focus_y": point[1],
            "f_number": f_number}}
        if viewfinder == "on":
            saved_ref = [view_id, 0]
        else:
            save_view = allocate()
            graph[save_view] = {"class_type": "SaveImage", "inputs": {
                "images": [view_id, 0],
                "filename_prefix": prefix + VIEWFINDER_SUFFIX}}
    save = allocate()
    graph[save] = {"class_type": "SaveImage", "inputs": {
        "images": saved_ref, "filename_prefix": prefix + DOF_SUFFIX}}
    return graph
