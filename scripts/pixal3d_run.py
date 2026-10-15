#!/usr/bin/env python3
"""Turn a chimera turnaround sheet into a textured GLB with Pixal3D multi-view.

    python scripts/pixal3d_run.py <short_id> --out <dir> [--views front=0,left=2,back=4,right=5]

Runs on the GPU box, against the local ComfyUI. The sheet is split into its
figures at the background columns between them; --views picks which figure
(0-based, left to right) feeds each side of the rig. The graph is the core
"Pixal3D: Multi Views to 3D" template with its sampler and bake settings,
fed four separate views and ending in SaveGLB. Writes the views, the split,
the submitted graph and the GLB into --out.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

VIEW_ORDER = ("front", "left", "back", "right")
DEFAULT_VIEWS = "front=0,left=2,back=4,right=5"
# A column belongs to a figure when more than this many pixels differ from the
# white backdrop the sheet is drawn on.
INK_LEVEL = 235
INK_PIXELS = 3
MIN_WIDTH = 40
MARGIN = 12


def parse_views(text: str) -> dict[str, int]:
    views = {}
    for part in text.split(","):
        name, _, index = part.partition("=")
        name = name.strip()
        if name not in VIEW_ORDER or not index.strip().isdigit():
            raise ValueError(f"bad view {part!r}; expected e.g. {DEFAULT_VIEWS}")
        views[name] = int(index)
    if "front" not in views:
        raise ValueError("the front view is required: the mesh is posed to it")
    return {name: views[name] for name in VIEW_ORDER if name in views}


def split_figures(sheet: Image.Image) -> list[tuple[int, int]]:
    """[x0, x1) of each figure, left to right, cut at the empty columns."""
    pixels = np.asarray(sheet.convert("RGB"))
    ink = (pixels < INK_LEVEL).any(axis=2).sum(axis=0) > INK_PIXELS
    runs, start = [], None
    for x, filled in enumerate(ink):
        if filled and start is None:
            start = x
        elif not filled and start is not None:
            runs.append([start, x])
            start = None
    if start is not None:
        runs.append([start, len(ink)])
    # A run too narrow to be a figure (a stray strand, a speck) joins the
    # nearer neighbour instead of standing as one.
    figures = [list(r) for r in runs]
    while len(figures) > 1:
        narrow = [i for i, (x0, x1) in enumerate(figures) if x1 - x0 < MIN_WIDTH]
        if not narrow:
            break
        i = narrow[0]
        gap_left = figures[i][0] - figures[i - 1][1] if i > 0 else None
        gap_right = figures[i + 1][0] - figures[i][1] if i + 1 < len(figures) else None
        j = i - 1 if gap_right is None or (gap_left is not None and gap_left <= gap_right) else i + 1
        lo, hi = sorted((i, j))
        figures[lo:hi + 1] = [[figures[lo][0], figures[hi][1]]]
    figures = [f for f in figures if f[1] - f[0] >= MIN_WIDTH]
    width = len(ink)
    out = []
    for i, (x0, x1) in enumerate(figures):
        left = 0 if i == 0 else (figures[i - 1][1] + x0) // 2
        right = width if i == len(figures) - 1 else (x1 + figures[i + 1][0]) // 2
        out.append((max(left, x0 - MARGIN), min(right, x1 + MARGIN)))
    return out


def build_graph(view_files: dict[str, str], *, prefix: str, fov: float = 20.0,
                texture_size: int = 4096) -> dict:
    """The template's graph in API form, one LoadImage per view."""
    g: dict[str, dict] = {}

    def node(node_id: str, class_type: str, **inputs) -> str:
        g[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    clip = node("clip_vision", "CLIPVisionLoader", clip_name="dino_v3_L_naf_fp32.safetensors")
    bg = node("bg_model", "LoadBackgroundRemovalModel", bg_removal_name="birefnet.safetensors")
    views = {}
    for name, filename in view_files.items():
        load = node(f"load_{name}", "LoadImage", image=filename)
        mask = node(f"mask_{name}", "RemoveBackground", bg_removal_model=[bg, 0], image=[load, 0])
        crop = node(f"crop_{name}", "ImageCropToMask", images=[load, 0], masks=[mask, 0],
                    width=1024, height=1024, pad_factor=1.1, grow_mask=0, background="#000000")
        node(f"save_{name}", "SaveImage", images=[crop, 0], filename_prefix=f"{prefix}/{name}")
        views[name] = [crop, 0]
    cond = node("conditioning", "Pixal3DMultiViewConditioning", clip_vision_model=[clip, 0],
                fov=fov, **views)
    pos, neg = [cond, 0], [cond, 1]

    unet = node("unet", "UNETLoader", unet_name="pixal3d_multiview_int8_convrot.safetensors",
                weight_dtype="default")
    shape_vae = node("shape_vae", "VAELoader", vae_name="trellis_2_shape_vae_bf16.safetensors")
    texture_vae = node("texture_vae", "VAELoader", vae_name="trellis_2_texture_vae_bf16.safetensors")

    def sampler(node_id, model, positive, negative, latent, seed, steps, cfg, scheduler="normal"):
        return node(node_id, "KSampler", model=model, positive=positive, negative=negative,
                    latent_image=latent, seed=seed, steps=steps, cfg=cfg,
                    sampler_name="euler", scheduler=scheduler, denoise=1.0)

    # structure
    s_cfg = node("structure_cfg", "CFGOverride", model=[unet, 0], cfg=1.0,
                 start_percent=0.667, end_percent=1.0)
    s_rescale = node("structure_rescale", "RescaleCFG", model=[s_cfg, 0], multiplier=0.7)
    s_model = node("structure_shift", "ModelSamplingSD3", model=[s_rescale, 0], shift=5.0)
    empty = node("structure_latent", "EmptyTrellis2LatentStructure", batch_size=1)
    s_sample = sampler("structure_sampler", [s_model, 0], pos, neg, [empty, 0], 56, 12, 7.5)
    voxel = node("structure_decode", "VaeDecodeStructureTrellis2", samples=[s_sample, 0],
                 vae=[shape_vae, 0], resolution="32")

    # shape, then its upsample
    shape_stage = node("shape_stage", "Trellis2ShapeStage", positive=pos, negative=neg,
                       voxel=[voxel, 0])
    sh_cfg = node("shape_cfg", "CFGOverride", model=[unet, 0], cfg=1.0,
                  start_percent=0.769, end_percent=1.0)
    sh_model = node("shape_rescale", "RescaleCFG", model=[sh_cfg, 0], multiplier=0.5)
    sh_sample = sampler("shape_sampler", [sh_model, 0], [shape_stage, 0], [shape_stage, 1],
                        [shape_stage, 2], 42, 20, 7.5)
    up = node("upsample_stage", "Trellis2UpsampleStage", positive=[shape_stage, 0],
              negative=[shape_stage, 1], shape_latent=[sh_sample, 0], vae=[shape_vae, 0],
              target_resolution=1536)
    up_sample = sampler("upsample_sampler", [sh_model, 0], [up, 0], [up, 1], [up, 2],
                        42, 12, 7.5, scheduler="simple")
    shape = node("shape_decode", "VaeDecodeShapeTrellis", samples=[up_sample, 0],
                 vae=[shape_vae, 0])

    # texture
    tex_stage = node("texture_stage", "Trellis2TextureStage", positive=[up, 0],
                     negative=[up, 1], shape_latent=[up_sample, 0])
    tex_sample = sampler("texture_sampler", [unet, 0], [tex_stage, 0], [tex_stage, 1],
                         [tex_stage, 2], 43, 12, 1.0)
    colors = node("texture_decode", "VaeDecodeTextureTrellis", samples=[tex_sample, 0],
                  vae=[texture_vae, 0], shape_subdivides=[shape, 1])

    # mesh clean-up, UVs and bakes
    info = node("mesh_info", "GetMeshInfo", mesh=[shape, 0])
    remesh = node("remesh", "RemeshMesh", mesh=[info, 0], resolution=768,
                  sign_mode="udf", **{"sign_mode.qef": False,
                                      "sign_mode.drop_inverted_components": False,
                                      "sign_mode.drop_enclosed_components": False},
                  band=1.0, project_back=0.0, fix_poles=False, smooth_iters=20,
                  drop_small_components=0.01, precluster_max_verts=20_000_000)
    decimate = node("decimate", "DecimateMesh", mesh=[remesh, 0], target_face_count=700_000,
                    placement_mode="midpoint")
    smooth = node("smooth", "MeshSmoothNormals", mesh=[decimate, 0], crease_angle=180.0)
    unwrap = node("unwrap", "UnwrapMesh", mesh=[smooth, 0], segmenter="pec",
                  resolution=texture_size, padding=1, weld_distance=0.0002)
    bake = node("bake_texture", "BakeTextureFromVoxel", mesh=[unwrap, 0],
                voxel_colors=[colors, 0], texture_size=texture_size,
                reference_mesh=[shape, 0])
    normal = node("bake_normal", "BakeNormalMapFromMesh", low_poly=[unwrap, 0],
                  high_poly=[remesh, 0], resolution=2048, cage_distance=0.05,
                  ignore_backfaces=True)
    ao = node("bake_ao", "BakeAmbientOcclusion", low_poly=[unwrap, 0], high_poly=[remesh, 0],
              resolution=1024, samples=64, max_distance=0.71, strength=1.0, bias=0.01)
    node("save_base_color", "SaveImage", images=[bake, 0], filename_prefix=f"{prefix}/base_color")
    textured = node("apply_texture", "ApplyTextureToMesh", mesh=[unwrap, 0],
                    base_color=[bake, 0], metallic=[bake, 1], roughness=[bake, 2],
                    occlusion=[ao, 0], normal_map=[normal, 0])
    final = node("final_smooth", "MeshSmoothNormals", mesh=[textured, 0], crease_angle=180.0)
    node("save_glb", "SaveGLB", mesh=[final, 0], filename_prefix=f"{prefix}/model")
    return g


def saved_files(outputs: dict) -> list[dict]:
    """Every file a node reported saving, whatever key its UI used."""
    files = []
    for node_output in outputs.values():
        for items in node_output.values():
            if isinstance(items, list):
                files.extend(i for i in items if isinstance(i, dict) and i.get("filename"))
    return files


def _keep_awake() -> contextlib.AbstractContextManager:
    if sys.platform != "win32":
        return contextlib.nullcontext()
    from chagama.infrastructure.power.windows import keep_awake
    return keep_awake()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("generation", help="chimera short_id or id of the sheet")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--views", default=DEFAULT_VIEWS)
    parser.add_argument("--fov", type=float, default=20.0)
    parser.add_argument("--repository", type=Path, default=Path.cwd(),
                        help="checkout whose chimera-token is used")
    parser.add_argument("--no-submit", action="store_true",
                        help="split and write the graph, do not run ComfyUI")
    args = parser.parse_args(argv)

    from chagama.infrastructure.chimera.client import ChimeraClient
    from chagama.infrastructure.comfyui.client import ComfyUIClient

    views = parse_views(args.views)
    args.out.mkdir(parents=True, exist_ok=True)
    sheet = Image.open(io.BytesIO(
        ChimeraClient(args.repository).fetch_generation_image(args.generation))).convert("RGB")
    figures = split_figures(sheet)
    (args.out / "split.json").write_text(json.dumps(
        {"size": sheet.size, "figures": figures, "views": views}, indent=2), encoding="utf-8")
    print(f"{args.generation}: {sheet.size[0]}x{sheet.size[1]}, figures at {figures}")
    for name, index in views.items():
        if index >= len(figures):
            print(f"error: {name}={index} but the sheet splits into {len(figures)} figures",
                  file=sys.stderr)
            return 1

    comfy = ComfyUIClient(poll_interval=15, poll_timeout=90 * 60)
    prefix = f"pixal3d/{args.generation}"
    view_files = {}
    for name, index in views.items():
        x0, x1 = figures[index]
        crop = sheet.crop((x0, 0, x1, sheet.size[1]))
        buffer = io.BytesIO()
        crop.save(buffer, "PNG")
        (args.out / f"view_{name}.png").write_bytes(buffer.getvalue())
        view_files[name] = (f"pixal3d_{args.generation}_{name}.png" if args.no_submit else
                            comfy.upload_image(f"pixal3d_{args.generation}_{name}.png",
                                               buffer.getvalue()))
    graph = build_graph(view_files, prefix=prefix, fov=args.fov)
    (args.out / "prompt.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    if args.no_submit:
        return 0

    try:
        prompt_id = comfy.submit(graph, purpose="pixal3d")
    except Exception as error:  # ComfyUI answers a rejected graph with HTTP 400 + node_errors
        body = getattr(error, "read", lambda: b"")()
        print(f"error: submit failed: {error} {body[:4000].decode('utf-8', 'replace')}",
              file=sys.stderr)
        return 1
    print(f"submitted {prompt_id}")
    with _keep_awake():
        outputs = comfy.wait_for_outputs(prompt_id)
    glbs = 0
    for item in saved_files(outputs):
        data = comfy.fetch({**item, "type": item.get("type", "output")})
        (args.out / item["filename"]).write_bytes(data)
        glbs += item["filename"].endswith(".glb")
        print(f"saved {item['filename']} ({len(data) / 1024**2:.1f} MB)")
    if not glbs:
        print("error: ComfyUI finished without a GLB", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
