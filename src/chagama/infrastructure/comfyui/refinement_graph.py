"""ComfyUI graph transformation for the canvas redraw, and the stages the delivery graphs share."""

from __future__ import annotations

import json
from collections.abc import Callable

from .base_graph import base_roles, sampler_settings

# The delivered composite: background cut, outline bands.
DELIVERED_SUFFIX = "-delivered"
# What the delivered composite was made of, saved beside it.
LAYER_FIGURE_SUFFIX = "-layer-figure"
LAYER_OUTLINE_SUFFIX = "-layer-outline"
LAYER_BACKDROP_SUFFIX = "-layer-backdrop"
# A matte_model of this form names a ComfyUI-RMBG model instead of a core
# background-removal model file.
RMBG_MATTE_PREFIX = "rmbg:"
DEPTH_NODE = "DepthAnythingV2Preprocessor"
DEPTH_CKPT = "depth_anything_v2_vitl.pth"
DEPTH_RESOLUTION = 1024


def sizes(width: int, height: int, longest_side: int) -> tuple[int, int]:
    longest = max(width, height)
    return (round(longest_side * width / longest / 8) * 8,
            round(longest_side * height / longest / 8) * 8)


def _matte_nodes(graph: dict, allocate: Callable[[], str], image_ref: list,
                 matte_model: str) -> list:
    if matte_model.startswith(RMBG_MATTE_PREFIX):
        node_id = allocate()
        graph[node_id] = {"class_type": "BiRefNetRMBG", "inputs": {
            "image": image_ref, "model": matte_model[len(RMBG_MATTE_PREFIX):],
            "sensitivity": 1.0, "mask_blur": 0, "mask_offset": 0,
            "invert_output": False, "refine_foreground": False,
            "background": "Alpha", "background_color": "#222222"}}
        return [node_id, 1]
    bg_loader = allocate()
    graph[bg_loader] = {"class_type": "LoadBackgroundRemovalModel", "inputs": {
        "bg_removal_name": matte_model}}
    remove = allocate()
    graph[remove] = {"class_type": "RemoveBackground", "inputs": {
        "bg_removal_model": [bg_loader, 0], "image": image_ref}}
    return [remove, 0]


def save_mask(graph: dict, allocate: Callable[[], str], mask_ref: list,
              filename_prefix: str) -> None:
    to_image = allocate()
    graph[to_image] = {"class_type": "MaskToImage", "inputs": {"mask": mask_ref}}
    save = allocate()
    graph[save] = {"class_type": "SaveImage", "inputs": {
        "images": [to_image, 0], "filename_prefix": filename_prefix}}


def matting_node(graph: dict, allocate: Callable[[], str], image_ref: list,
                 matte_ref: list) -> list:
    """The ViTMatte alpha of the picture. Returns the alpha ref."""
    node_id = allocate()
    graph[node_id] = {"class_type": "YukariMatting", "inputs": {
        "image": image_ref, "matte": matte_ref}}
    return [node_id, 0]


def foreground_node(graph: dict, allocate: Callable[[], str], image_ref: list,
                    alpha_ref: list) -> list:
    """The figure's colour under the alpha. Returns the foreground ref."""
    node_id = allocate()
    graph[node_id] = {"class_type": "YukariForeground", "inputs": {
        "image": image_ref, "alpha": alpha_ref}}
    return [node_id, 0]


def see_through_node(graph: dict, allocate: Callable[[], str], image_ref: list,
                     alpha_ref: list, gain: float) -> list:
    """The alpha with the key through the see-through inside scaled by
    `gain`. Returns the alpha ref."""
    node_id = allocate()
    graph[node_id] = {"class_type": "YukariSeeThrough", "inputs": {
        "image": image_ref, "alpha": alpha_ref, "gain": gain}}
    return [node_id, 0]


def _depth_nodes(graph: dict, allocate: Callable[[], str], image_ref: list
                 ) -> list:
    depth_id = allocate()
    graph[depth_id] = {"class_type": DEPTH_NODE, "inputs": {
        "image": image_ref, "ckpt_name": DEPTH_CKPT,
        "resolution": DEPTH_RESOLUTION}}
    return [depth_id, 0]


def _scaled(graph: dict, allocate: Callable[[], str], image_ref: list,
            canvas: tuple[int, int], deliver_size: int | None) -> list:
    width, height = canvas
    longest = max(width, height)
    if deliver_size is None or deliver_size >= longest:
        return image_ref
    node_id = allocate()
    graph[node_id] = {"class_type": "ImageScale", "inputs": {
        "image": image_ref, "upscale_method": "lanczos",
        "width": round(width * deliver_size / longest),
        "height": round(height * deliver_size / longest), "crop": "disabled"}}
    return [node_id, 0]


def delivery_tail(graph: dict, allocate: Callable[[], str], image_ref: list,
                  matte_ref: list, prefix: str, *,
                  keep_scene: bool, transparent: bool, backdrop: str | None,
                  stroke_light: str | None, outlines: list[dict],
                  deliver_size: int | None, canvas: tuple[int, int],
                  light_scene: str | None, light_from: str | None,
                  silhouette_ref: list | None = None) -> list:
    """Appends the composite and downscale stages onto `graph` (mutated),
    saving the delivered picture and its layers at the delivered size.
    `silhouette_ref` is the mask the bands follow when it is not the matte.
    Returns the delivered ref."""
    deliver_id = allocate()
    graph[deliver_id] = {"class_type": "YukariDeliver", "inputs": {
        "image": image_ref, "matte": matte_ref, "keep_scene": keep_scene,
        "transparent": transparent, "stroke_light": stroke_light or "",
        "backdrop": backdrop or "", "matted": not keep_scene,
        "outlines": json.dumps(outlines),
        **({"light_scene": light_scene, "light_from": light_from}
           if light_scene else {}),
        **({"silhouette": silhouette_ref} if silhouette_ref is not None else {})}}
    delivered_ref = _scaled(graph, allocate, [deliver_id, 0], canvas, deliver_size)
    save_delivered = allocate()
    graph[save_delivered] = {"class_type": "SaveImage", "inputs": {
        "images": delivered_ref, "filename_prefix": prefix + DELIVERED_SUFFIX}}
    layers = [(LAYER_FIGURE_SUFFIX, 2), (LAYER_OUTLINE_SUFFIX, 3)]
    if keep_scene or not transparent:
        layers.append((LAYER_BACKDROP_SUFFIX, 4))
    for suffix, output in layers:
        save = allocate()
        graph[save] = {"class_type": "SaveImage", "inputs": {
            "images": _scaled(graph, allocate, [deliver_id, output], canvas,
                              deliver_size),
            "filename_prefix": prefix + suffix}}
    return delivered_ref


def redraw_graph(base: dict, size: int, denoise: float, prefix: str, *,
                 prompt: tuple[str, str] | None = None,
                 latent_route: bool = False,
                 sampler: tuple[str, str] | None = None,
                 loader: str | None = None,
                 sampling: tuple[int, float] | None = None,
                 source_image: str | None = None,
                 keep_mask_image: str | None = None,
                 upscale: str = "bicubic",
                 redraw_from_source: bool = False,
                 canvas: tuple[int, int]) -> dict:
    """`base` re-sampled on a bigger canvas, ending at the base's own single
    SaveImage under `prefix`."""
    if redraw_from_source and not source_image:
        raise ValueError("redraw_from_source requires source_image")
    if upscale not in ("bicubic", "nearest-exact", "bilinear", "lanczos"):
        raise ValueError(f"unsupported upscale method: {upscale!r}")
    unsupported = [key for key in base
                   if not isinstance(key, str) or not key.isdecimal()]
    if unsupported:
        raise ValueError(
            "base graph has unsupported non-numeric node IDs: "
            + ", ".join(map(repr, unsupported)))
    graph = json.loads(json.dumps(base))
    roles = base_roles(graph)
    if latent_route and roles.stitched:
        raise ValueError(
            "latent_route is not supported on a stitched base: its sampler's "
            "latent is the inpaint crop, not the whole picture")
    next_id = max(int(key) for key in graph) + 1
    scale, encode, sample, decode = (
        str(next_id + offset) for offset in range(4))
    # Read off the base pass rather than assumed: a single DiffusersLoader
    # answers model/CLIP/VAE from node 4, a split-file model from three
    # separate loaders.
    model_ref = graph[roles.sampler_id]["inputs"].get("model", ["4", 0])
    # A layerdiffuse base samples through its own LayeredDiffusionApply node,
    # so the redraw's model is what that node itself sampled, not the node.
    apply_node = graph.get(model_ref[0], {})
    if apply_node.get("class_type") == "LayeredDiffusionApply":
        model_ref = apply_node["inputs"]["model"]
    clip_ref = graph[roles.positive_id]["inputs"].get("clip", ["4", 1])
    vae_ref = graph[roles.decode_id]["inputs"].get("vae", ["4", 2])
    if loader:
        # A different checkpoint redraws: its own model, CLIP and VAE, with the
        # base prompts re-encoded through its CLIP.
        loader_id = str(next_id + 10)
        if loader.endswith(".safetensors"):
            graph[loader_id] = {"class_type": "CheckpointLoaderSimple",
                                "inputs": {"ckpt_name": loader}}
        else:
            graph[loader_id] = {"class_type": "DiffusersLoader",
                                "inputs": {"model_path": loader}}
        model_ref, clip_ref, vae_ref = (
            [loader_id, 0], [loader_id, 1], [loader_id, 2])
        if prompt is None:
            prompt = (graph[roles.positive_id]["inputs"]["text"],
                      graph[roles.negative_id]["inputs"]["text"])
    positive, negative = (graph[roles.sampler_id]["inputs"]["positive"],
                         graph[roles.sampler_id]["inputs"]["negative"])
    if prompt:
        positive_id, negative_id = str(next_id + 4), str(next_id + 5)
        graph[positive_id] = {"class_type": "CLIPTextEncode", "inputs": {
            "clip": clip_ref, "text": prompt[0]}}
        graph[negative_id] = {"class_type": "CLIPTextEncode", "inputs": {
            "clip": clip_ref, "text": prompt[1]}}
        positive, negative = [positive_id, 0], [negative_id, 0]
    width, height = sizes(*canvas, size)
    # The routes to the bigger latent draw different pictures: pixel space
    # is faithful, the latent route leaves a staircase on hard contours
    # that the redraw turns into visible stroke.
    if latent_route and not source_image:
        graph[scale] = {"class_type": "LatentUpscale", "inputs": {
            "samples": [roles.sampler_id, 0], "upscale_method": "bicubic",
            "width": width, "height": height, "crop": "disabled"}}
        latent_in = [scale, 0]
    elif latent_route and source_image:
        # A source image outside the base graph (a repaired raw's own
        # picture) replaces the base sampler's latent as the thing upscaled.
        source_load_id = str(next_id + 13)
        graph[source_load_id] = {"class_type": "LoadImage", "inputs": {
            "image": source_image}}
        graph[encode] = {"class_type": "VAEEncode", "inputs": {
            "pixels": [source_load_id, 0], "vae": vae_ref}}
        graph[scale] = {"class_type": "LatentUpscale", "inputs": {
            "samples": [encode, 0], "upscale_method": "bicubic",
            "width": width, "height": height, "crop": "disabled"}}
        latent_in = [scale, 0]
    else:
        if redraw_from_source:
            # A repaired raw's own uploaded picture, not the base graph's
            # un-repaired SaveImage output.
            source_load_id = str(next_id + 13)
            graph[source_load_id] = {"class_type": "LoadImage", "inputs": {
                "image": source_image}}
            image_ref = [source_load_id, 0]
        else:
            image_ref = graph[roles.save_id]["inputs"]["images"]
        graph[scale] = {"class_type": "ImageScale", "inputs": {
            "image": image_ref, "upscale_method": upscale,
            "width": width, "height": height, "crop": "disabled"}}
        graph[encode] = {"class_type": "VAEEncode", "inputs": {
            "pixels": [scale, 0], "vae": vae_ref}}
        latent_in = [encode, 0]
    if keep_mask_image is not None:
        keep_load_id = str(next_id + 14)
        graph[keep_load_id] = {"class_type": "LoadImage", "inputs": {
            "image": keep_mask_image}}
        keep_to_mask_id = str(next_id + 15)
        graph[keep_to_mask_id] = {"class_type": "ImageToMask", "inputs": {
            "image": [keep_load_id, 0], "channel": "red"}}
        keep_noise_mask_id = str(next_id + 16)
        graph[keep_noise_mask_id] = {"class_type": "SetLatentNoiseMask", "inputs": {
            "samples": latent_in, "mask": [keep_to_mask_id, 0]}}
        latent_in = [keep_noise_mask_id, 0]
    # Steps, cfg and seed are the base pass's own: a checkpoint that was tuned
    # at a different cfg must be redrawn the way it was drawn. The sampler is
    # the base pass's own too, unless the caller overrides it.
    base_sampler = sampler_settings(graph, roles.sampler_id)
    sampler_name, scheduler = (
        sampler if sampler is not None
        else (base_sampler.get("sampler_name", "dpmpp_2m"),
              base_sampler.get("scheduler", "karras")))
    steps, cfg = (
        sampling if sampling is not None
        else (base_sampler.get("steps", 30), base_sampler.get("cfg", 5.0)))
    graph[sample] = {"class_type": "KSampler", "inputs": {
        "model": model_ref, "positive": positive, "negative": negative,
        "latent_image": latent_in, "seed": base_sampler["seed"],
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "denoise": denoise}}
    graph[decode] = {"class_type": "VAEDecode", "inputs": {
        "samples": [sample, 0], "vae": vae_ref}}
    graph[roles.save_id]["inputs"]["images"] = [decode, 0]
    graph[roles.save_id]["inputs"]["filename_prefix"] = prefix
    return graph
