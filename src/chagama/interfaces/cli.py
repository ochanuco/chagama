"""The single public command-line interface for chagama."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ..application import metadata, safety
from ..application.catalog import build_catalog
from ..application.catalog import publish_catalog as publish_catalog_document
from ..application.deliver import deliver
from ..application.dof import dof
from ..application.generate import generate
from ..application.ingest import import_images
from ..application.lens import lens
from ..application.photo import photo
from ..application.redraw import redraw
from ..application.stage import lighting, stage
from ..application.request_options import (
    REDRAW_DIAL_KEYS,
    deliver_arguments,
    dials_scope,
    dof_arguments,
    lens_arguments,
    photo_arguments,
    stage_arguments,
    lighting_arguments,
    redraw_arguments,
    resolve_dial,
)
from ..application.watch import WatchServices, watch
from ..application.work import fetch_source, work
from ..domain.yukari.costumes import COSTUMES, LEGWEAR_STATES, LEGWEARS
from ..domain.yukari.delivery_style import (
    DOF_F_NUMBER,
    DOF_SCOPE,
    DOF_VIEWFINDER,
    LENS_RADIUS,
    LENS_SHAPE,
    LENS_STRENGTH,
    LIGHTING_STRENGTH,
    LIGHT_FROM_DEFAULT,
    LIGHT_SCENES,
    PHOTO_FOCAL_LENGTH,
    PHOTO_HEIGHT,
    STROKE_CHOICES,
    STROKE_LIGHTS,
    Light,
)
from ..domain.yukari.expressions import EXPRESSIONS
from ..domain.yukari.poses import POSES
from ..domain.yukari.recipe import negative, positive
from ..infrastructure.chimera.client import ChimeraClient
from ..infrastructure.comfyui.client import ComfyUIClient
from ..infrastructure.imaging.backdrops import PATTERNS as BACKDROP_PATTERNS
from ..infrastructure.notifications.discord import DiscordNotifier
from ..infrastructure.repository import discover_repository, git_metadata
from .agent import (
    build_deliver_services,
    build_dof_services,
    build_lens_services,
    build_photo_services,
    build_stage_services,
    build_generate_services,
    build_redraw_services,
    default_worker_id,
    wire_work_services,
)


def _number_or_word(raw: str) -> float | str:
    """argparse type= for a dial-eligible flag: a recipe word passes through
    as a string for `_resolve_word_args` to resolve.
    """
    try:
        return float(raw)
    except ValueError:
        return raw


def _resolve_word_args(chimera: ChimeraClient, generation_id: str, scope: str,
                       values: dict) -> tuple[dict | None, dict]:
    """Resolve any word (string) values in `values` against the source
    generation's recipe dials for `scope`; numbers and `None` pass through.

    Returns the `context` `fetch_source` fetched, or `None`
    if no word was given, so the caller can reuse it without re-fetching.
    """
    if not any(isinstance(value, str) for value in values.values()):
        return None, values
    context, recipe = fetch_source(chimera, generation_id)
    dials = dials_scope(recipe, scope)
    resolved = {}
    for key, value in values.items():
        try:
            resolved[key] = resolve_dial(key, value, dials)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    return context, resolved


def _outlines_from_args(args) -> list[dict] | None:
    if args.no_outlines:
        return []
    if not args.outlines:
        return None
    outlines = []
    for text in args.outlines:
        color, _, width = text.partition(",")
        try:
            outlines.append({"color": color, "width": float(width)})
        except ValueError:
            raise SystemExit(f"--outline must be #RRGGBB,WIDTH, got {text!r}") from None
    return outlines


def _dof_options_from_args(args) -> dict:
    focus = [float(value) for value in args.focus.split(",")]
    scope = None
    if args.scope is not None:
        layers = {layer.strip() for layer in args.scope.split(",") if layer.strip()}
        unknown = sorted(layers - set(DOF_SCOPE))
        if unknown:
            raise SystemExit(f"--scope layers must be among {list(DOF_SCOPE)}, got {unknown}")
        scope = {layer: layer in layers for layer in DOF_SCOPE}
    return {
        "focus": focus,
        **({"f_number": args.f_number} if args.f_number is not None else {}),
        **({"scope": scope} if scope is not None else {}),
        "viewfinder": args.viewfinder,
        **({"stand_at": _xy(args.stand_at, "--stand-at")} if args.stand_at else {}),
        **({"nearer": args.nearer} if args.nearer is not None else {}),
        **({"tilt": args.tilt} if args.tilt is not None else {}),
        **({"bokeh": args.bokeh} if args.bokeh is not None else {}),
        **({"focal_length": args.focal_length} if args.focal_length is not None else {}),
    }


def _lens_options_from_args(args) -> dict:
    options = {}
    if args.center is not None:
        try:
            options["center"] = [float(value) for value in args.center.split(",")]
        except ValueError:
            raise SystemExit(f"--center must be X,Y, got {args.center!r}") from None
    if args.strength is not None:
        options["strength"] = args.strength
    if args.shape is not None:
        options["shape"] = args.shape
    if args.radius is not None:
        options["radius"] = args.radius
    return options


def _xy(text: str, flag: str) -> list[float]:
    try:
        return [float(value) for value in text.split(",")]
    except ValueError:
        raise SystemExit(f"{flag} must be X,Y, got {text!r}") from None


def _photo_options_from_args(args) -> dict:
    options: dict = {"photo": args.photo}
    if args.place is not None:
        options["place"] = _xy(args.place, "--place")
    if args.height is not None:
        options["height"] = args.height
    if args.stand_at is not None:
        options["stand_at"] = _xy(args.stand_at, "--stand-at")
    if args.nearer is not None:
        options["nearer"] = args.nearer
    if args.focus is not None:
        options["focus"] = (args.focus if args.focus == "subject"
                            else _xy(args.focus, "--focus"))
    if args.f_number is not None:
        options["f_number"] = args.f_number
    if args.tilt is not None:
        options["tilt"] = args.tilt
    if args.bokeh is not None:
        options["bokeh"] = args.bokeh
    if args.focal_length is not None:
        options["focal_length"] = args.focal_length
    return options


def _stage_options_from_args(args) -> dict:
    options: dict = {}
    if args.photo is not None:
        options["photo"] = args.photo
    if args.place is not None:
        options["place"] = _xy(args.place, "--place")
    if args.height is not None:
        options["height"] = args.height
    if args.backdrop is not None:
        options["backdrop"] = args.backdrop
    if args.shadow_from is not None:
        options["shadow_from"] = None if args.shadow_from == "none" else args.shadow_from
    if args.light == "none":
        options["light"] = None
    elif args.light:
        light = _light_from_args(args)
        options["light"] = {"scene": light.scene, "from": light.direction}
    if args.frame is not None:
        try:
            options["frame"] = json.loads(args.frame)
        except json.JSONDecodeError as error:
            raise SystemExit(f"--frame must be JSON: {error}") from error
    return options


def _light_from_args(args) -> Light | None:
    if not args.light:
        return None
    scene, *direction = args.light.split(",")
    if scene not in LIGHT_SCENES:
        raise SystemExit(f"--light scene must be one of {sorted(LIGHT_SCENES)}")
    if direction and direction[0] not in STROKE_LIGHTS:
        raise SystemExit(f"--light from must be one of {sorted(STROKE_LIGHTS)}")
    return Light(scene, direction[0] if direction else LIGHT_FROM_DEFAULT)


def _positive_finite_seconds(raw: str) -> float:
    """argparse type= for --interval: rejects 0, negatives, nan and inf."""
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {raw!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(
            f"--interval must be a finite number > 0, got {raw!r}")
    return value


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="chagama")
    commands = root.add_subparsers(dest="command", required=True)

    generate_parser = commands.add_parser("generate", help="run and record a request")
    generate_parser.add_argument("--request", required=True, type=Path)
    generate_parser.add_argument("--dry-run", action="store_true")
    generate_parser.add_argument("--force", action="store_true")

    import_parser = commands.add_parser(
        "import", help="register image files as an import request")
    import_parser.add_argument("images", nargs="+", type=Path)
    import_parser.add_argument("--recipe")
    import_parser.add_argument(
        "--parameters", default="{}", help="JSON object recorded as the request's parameters")
    import_parser.add_argument("--raw-instruction", default="")
    import_parser.add_argument(
        "--references", default="[]",
        help="JSON list of {source_generation_id, purpose, aspect?, instruction?}")
    import_parser.add_argument("--seed", type=int, default=0)
    import_parser.add_argument(
        "--idempotency-key",
        help="defaults to a digest of the image bytes, so a resend registers nothing twice")

    watch_parser = commands.add_parser(
        "watch", help="poll chimera for pending ExperimentRuns and generate them")
    watch_parser.add_argument(
        "--interval", type=_positive_finite_seconds, default=30)
    watch_parser.add_argument("--once", action="store_true")
    watch_parser.add_argument("--dry-run", action="store_true")

    work_parser = commands.add_parser(
        "work", help="claim and execute rows from chimera's requests queue")
    work_parser.add_argument(
        "--interval", type=_positive_finite_seconds, default=30)
    work_parser.add_argument("--once", action="store_true")
    work_parser.add_argument("--dry-run", action="store_true")
    work_parser.add_argument("--worker-id", default=default_worker_id())
    work_parser.add_argument(
        "--kinds", default="generate,redraw,deliver,dof,lens,photo,stage,lighting",
        help="comma-separated request kinds to claim")
    work_parser.add_argument(
        "--no-hub", action="store_true",
        help="poll only; do not open the WorkerHub websocket")
    work_parser.add_argument(
        "--no-catalog", action="store_true",
        help="skip publishing the recipe catalog to chimera at startup")

    redraw_parser = commands.add_parser(
        "redraw", help="redraw one picture generation with a single method")
    redraw_parser.add_argument("generation_id")
    redraw_parser.add_argument(
        "--method", required=True, choices=("canvas", "hires", "light"),
        help="canvas: re-sample on a bigger canvas; hires: re-render the "
             "stored graph at a larger area; light: paint a scene's light and "
             "re-sample")
    redraw_parser.add_argument(
        "--denoise", type=_number_or_word,
        help="canvas: the redraw's denoise (a number or a recipe word); "
             "hires: the second pass's denoise, 0 < d <= 1")
    redraw_parser.add_argument(
        "--size", type=int, metavar="LONGEST",
        help="canvas: longest side of the redraw")
    redraw_route = redraw_parser.add_mutually_exclusive_group()
    redraw_route.add_argument(
        "--latent-route", dest="latent_route", action="store_const",
        const=True, default=None, help="canvas: upscale the latent")
    redraw_route.add_argument(
        "--pixel-route", dest="latent_route", action="store_const",
        const=False, help="canvas: upscale the decoded image (the default)")
    redraw_parser.add_argument(
        "--finalizer", metavar="MODEL",
        help="canvas: model that redraws instead of the recipe's own")
    redraw_parser.add_argument(
        "--upscale", choices=["bicubic", "nearest-exact", "bilinear", "lanczos"],
        help="canvas: pixel-route upscale method")
    redraw_parser.add_argument(
        "--keep-region", dest="keep_regions", action="append",
        metavar="X0,Y0,X1,Y1",
        help="canvas: fractional rectangle [0..1] shielded from the redraw "
             "under a soft noise mask; repeatable")
    redraw_parser.add_argument(
        "--keep-strength", type=float, default=None, metavar="STRENGTH",
        help="canvas: how much the redraw still touches a --keep-region, "
             "0 < s < 1")
    redraw_parser.add_argument(
        "--hires", type=int, default=None, metavar="SIZE",
        help="hires: long side in px of the area a 1024x1640 canvas has")
    redraw_parser.add_argument(
        "--light", metavar="SCENE[,FROM]",
        help="light: the scene "
             f"({', '.join(sorted(LIGHT_SCENES))}) and the direction "
             f"({', '.join(sorted(STROKE_LIGHTS))}; default "
             f"{LIGHT_FROM_DEFAULT}) it is lit from")
    redraw_parser.add_argument(
        "--cut", action="store_true",
        help="also cut the result as a deliver would and attach the cut and an "
             "RGBA cutout to it")

    deliver_parser = commands.add_parser(
        "deliver", help="cut and decorate one picture generation")
    deliver_parser.add_argument("generation_id")
    deliver_parser.add_argument(
        "--no-repin", dest="repin", action="store_false",
        help="leave the palette as drawn instead of repinning it")
    deliver_parser.add_argument(
        "--keep-scene", action="store_true",
        help="deliver the picture uncut, background and all")
    deliver_transparent = deliver_parser.add_mutually_exclusive_group()
    deliver_transparent.add_argument(
        "--transparent", dest="transparent", action="store_const",
        const=True, default=None,
        help="deliver the figure alone as an RGBA cutout")
    deliver_transparent.add_argument(
        "--opaque", dest="transparent", action="store_const", const=False,
        help="composite on the backdrop with the outlines instead")
    deliver_parser.add_argument(
        "--backdrop", metavar="#RRGGBB|" + "|".join(BACKDROP_PATTERNS),
        help="backdrop under the sticker -- a colour or a named pattern")
    deliver_parser.add_argument(
        "--stroke-light", choices=STROKE_CHOICES,
        help="light direction the outermost outline is shaded from, 'even' "
             "for a uniform one")
    deliver_outlines = deliver_parser.add_mutually_exclusive_group()
    deliver_outlines.add_argument(
        "--outline", dest="outlines", action="append", metavar="#RRGGBB,WIDTH",
        help="an outline band, innermost first, its width a percent of the "
             "longest side; repeatable (default: the recipe's own bands)")
    deliver_outlines.add_argument(
        "--no-outlines", action="store_true", help="deliver without outlines")
    deliver_parser.add_argument(
        "--light", metavar="SCENE[,FROM]",
        help="light the delivery as a scene "
             f"({', '.join(sorted(LIGHT_SCENES))}) from a direction "
             f"({', '.join(sorted(STROKE_LIGHTS))}; default "
             f"{LIGHT_FROM_DEFAULT}) and tint the backdrop to match")

    dof_parser = commands.add_parser(
        "dof", help="blur one delivered picture's layers by depth")
    dof_parser.add_argument("generation_id")
    dof_parser.add_argument(
        "--focus", required=True, metavar="X,Y",
        help="focus point as fractions of the picture width and height")
    dof_parser.add_argument(
        "--f-number", type=float, default=None,
        help=f"{DOF_F_NUMBER['min']}..{DOF_F_NUMBER['max']} "
             f"(default {DOF_F_NUMBER['default']})")
    dof_parser.add_argument(
        "--scope", metavar="LAYER[,LAYER]",
        help=f"the layers to blur, among {', '.join(DOF_SCOPE)} (default: all)")
    dof_parser.add_argument(
        "--viewfinder", choices=DOF_VIEWFINDER["values"],
        default=DOF_VIEWFINDER["default"],
        help="draw a camera viewfinder over the picture ('on'), or keep it "
             "plain and add the viewfinder picture as an extra generation ('both')")

    dof_parser.add_argument(
        "--stand-at", metavar="X,Y",
        help="photo-set pictures only: a photo point whose depth the card stands at")
    dof_parser.add_argument(
        "--nearer", type=float, default=None,
        help="photo-set pictures only: pull the card toward the camera, 0..1")
    dof_parser.add_argument(
        "--tilt", type=float, default=None,
        help="photo-set pictures only: the card's top this much farther, -1..1")
    dof_parser.add_argument(
        "--bokeh", type=float, default=None,
        help="photo-set pictures only: open out-of-focus highlights into discs, 0..1")
    dof_parser.add_argument(
        "--focal-length", choices=PHOTO_FOCAL_LENGTH["values"], default=None,
        help="photo-set pictures only: the full-frame lens it is shot through "
             f"(default {PHOTO_FOCAL_LENGTH['default']})")

    lens_parser = commands.add_parser(
        "lens", help="distort one delivered picture through a fisheye or its inverse")
    lens_parser.add_argument("generation_id")
    lens_parser.add_argument(
        "--center", metavar="X,Y",
        help="lens centre as fractions of the picture width and height "
             "(default 0.5,0.5)")
    lens_parser.add_argument(
        "--strength", type=float, default=None,
        help=f"{LENS_STRENGTH['min']}..{LENS_STRENGTH['max']}: positive is a "
             f"fisheye, negative its inverse (default {LENS_STRENGTH['default']})")
    lens_parser.add_argument(
        "--shape", choices=LENS_SHAPE["values"], default=None,
        help="fisheye bends the whole frame; sphere bends only a disc around "
             f"the centre (default {LENS_SHAPE['default']})")
    lens_parser.add_argument(
        "--radius", type=float, default=None,
        help=f"sphere only: {LENS_RADIUS['min']}..{LENS_RADIUS['max']} of the "
             f"shorter side (default {LENS_RADIUS['default']})")

    photo_parser = commands.add_parser(
        "photo", help="stand one delivered figure in a photo and blur it by depth")
    photo_parser.add_argument("generation_id")
    photo_parser.add_argument(
        "--photo", required=True, help="the Generation whose image is the scene")
    photo_parser.add_argument(
        "--place", metavar="X,Y",
        help="where the middle of the card's bottom edge goes (default 0.5,0.95)")
    photo_parser.add_argument(
        "--height", type=float, default=None,
        help=f"card height as a share of the photo's, {PHOTO_HEIGHT['min']}.."
             f"{PHOTO_HEIGHT['max']} (default {PHOTO_HEIGHT['default']})")
    photo_parser.add_argument(
        "--stand-at", metavar="X,Y",
        help="a photo point whose depth the card stands at (default: its own base)")
    photo_parser.add_argument(
        "--nearer", type=float, default=None,
        help="pull the card toward the camera, 0..1 of the scene depth (default 0)")
    photo_parser.add_argument(
        "--focus", metavar="subject|X,Y",
        help="focus on the card ('subject', default) or on a point of the picture")
    photo_parser.add_argument(
        "--tilt", type=float, default=None,
        help="lean the card: its top this much farther than its base, -1..1 "
             "(default 0); focus on the face to blur the body")
    photo_parser.add_argument(
        "--bokeh", type=float, default=None,
        help="open the photo's out-of-focus highlights into bright discs, 0..1 "
             "(default 0)")
    photo_parser.add_argument(
        "--f-number", type=float, default=None,
        help=f"{DOF_F_NUMBER['min']}..{DOF_F_NUMBER['max']} "
             f"(default {DOF_F_NUMBER['default']})")
    photo_parser.add_argument(
        "--focal-length", choices=PHOTO_FOCAL_LENGTH["values"], default=None,
        help="the full-frame lens the photo is shot through; a longer lens "
             f"blurs the scene more (default {PHOTO_FOCAL_LENGTH['default']})")

    stage_parser = commands.add_parser(
        "stage", help="set one delivered picture on a backdrop pattern or in a photo")
    stage_parser.add_argument("generation_id")
    stage_parser.add_argument(
        "--photo", help="the photo Generation to set the delivery in")
    stage_parser.add_argument(
        "--place", metavar="X,Y",
        help="photo only: where the middle of the card's bottom edge goes "
             "(default 0.5,0.95)")
    stage_parser.add_argument(
        "--height", type=float, default=None,
        help=f"photo only: card height as a share of the photo's, "
             f"{PHOTO_HEIGHT['min']}..{PHOTO_HEIGHT['max']} (default {PHOTO_HEIGHT['default']})")
    stage_parser.add_argument(
        "--backdrop",
        help=f"without a photo: a pattern ({', '.join(sorted(BACKDROP_PATTERNS))}) "
             "or #RRGGBB")
    stage_parser.add_argument(
        "--shadow-from", choices=[*sorted(STROKE_LIGHTS), "none"],
        help="without a photo: throw the drop shadow away from this light "
             "(default: the delivery's own light)")
    stage_parser.add_argument(
        "--light", metavar="SCENE[,FROM]|none",
        help="without a photo: tint the backdrop to this scene "
             f"({', '.join(sorted(LIGHT_SCENES))}); 'none' for no scene "
             "(default: the nearest light redraw in the lineage)")
    stage_parser.add_argument(
        "--frame", metavar="JSON",
        help="without a photo: the frame as JSON ({box, sides, line, paper, canvas, "
             "strokes}); the backdrop fills its box")

    lighting_parser = commands.add_parser(
        "lighting", help="light a staged picture's figure by its backdrop")
    lighting_parser.add_argument("generation_id")
    lighting_parser.add_argument(
        "--scene", choices=sorted(LIGHT_SCENES),
        help="light the figure as this scene (and tint a pattern backdrop to it)")
    lighting_parser.add_argument(
        "--direction", choices=sorted(STROKE_LIGHTS),
        help="also shade the figure from this side (default: exposure and colour only)")
    lighting_parser.add_argument(
        "--strength", type=float, default=None,
        help=f"{LIGHTING_STRENGTH['min']}..{LIGHTING_STRENGTH['max']} "
             f"(default {LIGHTING_STRENGTH['default']})")

    catalog_parser = commands.add_parser(
        "catalog", help="print the recipe catalog chimera composes requests from")
    catalog_parser.add_argument(
        "--publish", action="store_true",
        help="also PUT the catalog to chimera and print its response")

    metadata_parser = commands.add_parser("metadata", help="manage generation metadata")
    metadata_commands = metadata_parser.add_subparsers(
        dest="metadata_command", required=True)
    semantic = metadata_commands.add_parser("semantic")
    semantic.add_argument("generation_id")
    semantic.add_argument("file", type=Path)
    tag = metadata_commands.add_parser("tag")
    tag.add_argument("generation_id")
    tag.add_argument("name")
    publish = metadata_commands.add_parser("publish")
    publish.add_argument("generation_id")
    publish.add_argument("--url")
    publish.add_argument("--idempotency-key")
    asset = metadata_commands.add_parser("asset")
    asset.add_argument("generation_id")
    asset.add_argument("role")
    asset.add_argument("file", type=Path)
    asset.add_argument("--region", default="")
    assets = metadata_commands.add_parser("list-assets")
    assets.add_argument("generation_id")

    safety_parser = commands.add_parser(
        "safety", help="rate generations with the WD tagger and send chimera the numbers")
    safety_commands = safety_parser.add_subparsers(
        dest="safety_command", required=True)
    safety_rate = safety_commands.add_parser("rate")
    safety_rate.add_argument("generation_ids", nargs="+", metavar="short_id")
    safety_backfill = safety_commands.add_parser("backfill")
    safety_backfill.add_argument("--published", action="store_true")
    safety_backfill.add_argument("--limit", type=int, default=None)

    yukari_parser = commands.add_parser("yukari", help="inspect the Yukari domain")
    yukari_commands = yukari_parser.add_subparsers(dest="yukari_command", required=True)
    yukari_prompt = yukari_commands.add_parser("prompt")
    yukari_prompt.add_argument("--pose", required=True, choices=sorted(POSES))
    yukari_prompt.add_argument("--costume", choices=sorted(COSTUMES))
    yukari_prompt.add_argument("--expression", choices=sorted(EXPRESSIONS))
    yukari_prompt.add_argument("--legwear", choices=LEGWEARS, default=None)
    yukari_prompt.add_argument("--legwear-state", dest="legwear_state",
                               choices=LEGWEAR_STATES, default=None)
    yukari_prompt.add_argument("--json", action="store_true")
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "yukari":
        prompts = {
            "positive": positive(args.pose, args.costume, args.expression,
                                 args.legwear, args.legwear_state),
            "negative": negative(args.pose, args.costume, args.expression,
                                 args.legwear, args.legwear_state),
        }
        if args.json:
            print(json.dumps(prompts, ensure_ascii=False, indent=2))
        else:
            print(prompts["positive"], "\n\n---\n\n", prompts["negative"])
        return

    repository = discover_repository()
    chimera = ChimeraClient(repository)
    comfyui = ComfyUIClient()
    notifier = DiscordNotifier(repository)

    def repository_metadata() -> dict:
        return git_metadata(repository)

    if args.command == "generate":
        services = build_generate_services(
            chimera, comfyui, notifier, repository, repository_metadata)
        generate(args.request, services, dry_run=args.dry_run, force=args.force)
        return
    if args.command == "import":
        git = repository_metadata()
        key = args.idempotency_key or "import:" + hashlib.sha256(
            b"".join(path.read_bytes() for path in args.images)).hexdigest()
        resolution = {
            "recipe": args.recipe,
            "raw_instruction": args.raw_instruction,
            "parameters": json.loads(args.parameters),
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "references": json.loads(args.references),
        }
        result = import_images(
            chimera, print, images=args.images, resolution=resolution,
            idempotency_key=key, seed=args.seed)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.command == "watch":
        services = build_generate_services(
            chimera, comfyui, notifier, repository, repository_metadata)
        watch_services = WatchServices(management=chimera, generate_services=services)
        watch(watch_services, interval=args.interval, once=args.once,
              dry_run=args.dry_run)
        return
    if args.command == "work":
        work_services = wire_work_services(
            chimera, comfyui, notifier, repository, repository_metadata,
            worker_id=args.worker_id,
            kinds=tuple(kind.strip() for kind in args.kinds.split(",") if kind.strip()),
            hub=not args.no_hub,
        )
        work(work_services, interval=args.interval, once=args.once,
             dry_run=args.dry_run, publish_catalog=not args.no_catalog)
        return
    if args.command == "redraw":
        services = build_redraw_services(
            chimera, comfyui, notifier, repository, repository_metadata)
        light = _light_from_args(args)
        context = None
        denoise = args.denoise
        if args.method == "canvas":
            context, resolved = _resolve_word_args(
                chimera, args.generation_id, "redraw",
                {key: getattr(args, key) for key in REDRAW_DIAL_KEYS})
            denoise = resolved["denoise"]
        options = {
            "method": args.method,
            **({"denoise": denoise} if denoise is not None else {}),
            **({"size": args.size} if args.size is not None else {}),
            **({"route": "latent" if args.latent_route else "pixel"}
               if args.latent_route is not None else {}),
            **({"finalizer": args.finalizer} if args.finalizer else {}),
            **({"upscale": args.upscale} if args.upscale else {}),
            **({"keep_regions": [[float(value) for value in region.split(",")]
                                 for region in args.keep_regions]}
               if args.keep_regions else {}),
            **({"keep_strength": args.keep_strength}
               if args.keep_strength is not None else {}),
            **({"hires": args.hires} if args.hires is not None else {}),
            **({"scene": light.scene, "from": light.direction}
               if light is not None else {}),
            **({"cut": True} if args.cut else {}),
        }
        try:
            arguments = redraw_arguments(options)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        redraw(args.generation_id, services, context=context, **arguments)
        return
    if args.command == "deliver":
        services = build_deliver_services(
            chimera, comfyui, notifier, repository, repository_metadata)
        outlines = _outlines_from_args(args)
        light = _light_from_args(args)
        options = {
            **({} if args.repin else {"repin": False}),
            **({"keep_scene": True} if args.keep_scene else {}),
            **({"transparent": args.transparent}
               if args.transparent is not None else {}),
            **({"backdrop": args.backdrop} if args.backdrop is not None else {}),
            **({"stroke_light": args.stroke_light}
               if args.stroke_light is not None else {}),
            **({"outlines": outlines} if outlines is not None else {}),
            **({"light": {"scene": light.scene, "from": light.direction}}
               if light is not None else {}),
        }
        try:
            arguments = deliver_arguments(options)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        deliver(args.generation_id, services, **arguments)
        return
    if args.command == "dof":
        services = build_dof_services(
            chimera, comfyui, notifier, repository, repository_metadata)
        try:
            arguments = dof_arguments(_dof_options_from_args(args))
        except ValueError as error:
            raise SystemExit(str(error)) from error
        dof(args.generation_id, services, **arguments)
        return
    if args.command == "lens":
        services = build_lens_services(
            chimera, comfyui, notifier, repository, repository_metadata)
        try:
            arguments = lens_arguments(_lens_options_from_args(args))
        except ValueError as error:
            raise SystemExit(str(error)) from error
        lens(args.generation_id, services, **arguments)
        return
    if args.command == "photo":
        services = build_photo_services(
            chimera, comfyui, notifier, repository, repository_metadata)
        try:
            arguments = photo_arguments(_photo_options_from_args(args))
        except ValueError as error:
            raise SystemExit(str(error)) from error
        photo(args.generation_id, services, **arguments)
        return
    if args.command == "stage":
        services = build_stage_services(
            chimera, comfyui, notifier, repository, repository_metadata)
        try:
            arguments = stage_arguments(_stage_options_from_args(args))
        except ValueError as error:
            raise SystemExit(str(error)) from error
        stage(args.generation_id, services, **arguments)
        return
    if args.command == "lighting":
        services = build_stage_services(
            chimera, comfyui, notifier, repository, repository_metadata)
        options = {"direction": args.direction, "scene": args.scene,
                   **({"strength": args.strength} if args.strength is not None else {})}
        try:
            arguments = lighting_arguments(options)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        lighting(args.generation_id, services, **arguments)
        return
    if args.command == "catalog":
        git = repository_metadata()
        document = build_catalog(git)
        print(json.dumps(document, indent=2, ensure_ascii=False))
        if args.publish:
            response = publish_catalog_document(chimera, git, catalog=document)
            print(json.dumps(response, indent=2, ensure_ascii=False))
        return

    if args.command == "safety":
        if args.safety_command == "rate":
            for generation_id in args.generation_ids:
                rating = safety.rate_generation_by_id(chimera, generation_id)
                print(f"{generation_id}: {safety.format_rating(rating)}")
        else:
            safety.backfill(chimera, print, published=args.published,
                            limit=args.limit)
        return

    if args.metadata_command == "semantic":
        metadata.put_semantic(chimera, args.generation_id, args.file)
        print(f"semantic -> {args.generation_id}")
    elif args.metadata_command == "tag":
        metadata.add_tag(chimera, args.generation_id, args.name)
        print(f"tag {args.name!r} -> {args.generation_id}")
    elif args.metadata_command == "publish":
        metadata.record_publication(
            chimera, args.generation_id, url=args.url,
            idempotency_key=args.idempotency_key)
        where = f" ({args.url})" if args.url else ""
        print(f"publish -> {args.generation_id}{where}")
    elif args.metadata_command == "asset":
        row = metadata.upload_asset(
            chimera, args.generation_id, args.role, args.file, args.region)
        where = args.role + (f".{args.region}" if args.region else "")
        print(f"asset {where} -> {args.generation_id} ({row.get('size', '?')} bytes)")
    else:
        for row in metadata.list_assets(chimera, args.generation_id):
            region = row.get("region") or ""
            name = row["role"] + (f".{region}" if region else "")
            print(f"{name}  {row['content_type']}  {row['size']}")


if __name__ == "__main__":
    main()
