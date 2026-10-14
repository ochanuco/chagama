"""Turn a `requests` row's JSON `options` into a use-case's kwargs.

The wire contract is chimera's docs/worker-protocol.md.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..domain.repair.controlnet import CONTROL_MODELS, DEFAULT_CONTROL_STRENGTH
from ..domain.repair.loras import DEFAULT_PART_LORA_WEIGHT
from ..domain.repair.models import MODELS
from ..domain.yukari.delivery_style import (
    DELIVER_DEFAULTS,
    DOF_F_NUMBER,
    DOF_SCOPE,
    DOF_VIEWFINDER,
    LENS_CENTER_DEFAULT,
    LENS_RADIUS,
    LENS_SHAPE,
    LENS_STRENGTH,
    LIGHT_FROM_DEFAULT,
    LIGHT_SCENES,
    LIGHTING_STRENGTH,
    PHOTO_BOKEH,
    PHOTO_FOCAL_LENGTH,
    PHOTO_HEIGHT,
    PHOTO_NEARER,
    PHOTO_PLACE_DEFAULT,
    PHOTO_PLACE_MAX_Y,
    PHOTO_TILT,
    OUTLINE_WIDTH_MAX_PCT,
    OUTLINES_MAX,
    STROKE_CHOICES,
    STROKE_LIGHTS,
    Light,
)
from ..domain.yukari.dials import DIALS
from ..infrastructure.imaging.backdrops import PATTERNS, is_backdrop
from .deliver import RECIPE_DEFAULT
from .picture_source import stroke_light_conflict
from .redraw import HIRES_DENOISE

_KNOWN_DELIVER_OPTIONS = frozenset({
    "repin", "recolor", "skin", "keep_legwear", "keep_scene", "transparent",
    "backdrop", "stroke_light", "outlines", "deliver_size", "light",
    "see_through",
})
# The see-through gain's range; 1 is the cut as solved.
_SEE_THROUGH_RANGE = (1.0, 5.0)

# The photo keys only apply to a delivery set in a photo (stage).
_DOF_PHOTO_OPTIONS = frozenset({"stand_at", "nearer", "tilt", "bokeh", "focal_length"})
_KNOWN_DOF_OPTIONS = frozenset({"focus", "f_number", "scope", "viewfinder"}) | _DOF_PHOTO_OPTIONS

_KNOWN_LENS_OPTIONS = frozenset({"center", "strength", "shape", "radius"})

_KNOWN_PHOTO_OPTIONS = frozenset({
    "photo", "place", "height", "stand_at", "nearer", "focus", "f_number", "tilt", "bokeh",
    "focal_length"})

_KNOWN_STAGE_OPTIONS = frozenset({"photo", "place", "height", "backdrop", "shadow_from",
                                  "light"})
_STAGE_PHOTO_OPTIONS = frozenset({"photo", "place", "height"})
_STAGE_PATTERN_OPTIONS = frozenset({"backdrop", "shadow_from", "light"})

_KNOWN_LIGHTING_OPTIONS = frozenset({"direction", "strength", "scene"})

_KNOWN_REDRAW_OPTIONS = {
    "canvas": frozenset({"denoise", "size", "route", "finalizer", "upscale",
                         "keep_regions", "keep_strength"}),
    "hires": frozenset({"hires", "denoise"}),
    "light": frozenset({"scene", "from"}),
}

_KNOWN_REPAIR_OPTIONS = frozenset({
    "parts", "regions", "denoise", "seeds", "size", "pad", "lora", "model",
    "control", "control_strength",
})

_KNOWN_MASKED_REDRAW_OPTIONS = frozenset({
    "regions", "prompt_patch", "denoise", "mask_padding", "mask_feather",
    "size", "seeds",
})

_REPAIR_PARTS = frozenset({"hands", "feet"})

_MASKED_REDRAW_PROMPT_PATCH_MAX_LENGTH = 4096
_MASKED_REDRAW_SEEDS_MAX = 16

# `generation.recipe` -> its published `dials` block (see domain/*/dials.py).
_RECIPE_DIALS = {
    "yukari": DIALS,
}

# Kept in sync by hand with the resolve_dial() call sites in the *_arguments()
# functions. Public: interfaces/cli.py reads them too, to resolve the same
# keys' words from its own parsed args.
REDRAW_DIAL_KEYS = ("denoise",)
DELIVER_DIAL_KEYS = ("keep_legwear",)
REPAIR_DIAL_KEYS = ("denoise", "lora")
_MASKED_REDRAW_DIAL_KEYS = ("denoise",)


def dials_scope(recipe: str, scope: str) -> Mapping[str, Mapping[str, float]]:
    return _RECIPE_DIALS.get(recipe, {}).get(scope, {})


def resolve_dial(key: str, value: object,
                  dials: Mapping[str, Mapping[str, float]]) -> object:
    """A string option value is looked up in `dials[key]`; every other value
    (a number, `true`, `null`) passes through for the option's own checks.
    """
    if not isinstance(value, str):
        return value
    words = dials.get(key) or {}
    if value not in words:
        raise ValueError(f"unknown {key} word: {value!r}")
    return words[value]


def _resolved_options(options: Mapping, arguments: Mapping,
                      dial_keys: tuple[str, ...]) -> dict:
    """The request's own options, with each dial-eligible key's value
    replaced by what `arguments` actually resolved it to -- the single
    source of truth for what the request ran with.
    """
    return {key: (arguments[key] if key in dial_keys else value)
           for key, value in options.items()}


def _parts_argument(value: object, *, key: str = "parts") -> list[str]:
    if (not isinstance(value, list)
            or any(not isinstance(part, str) for part in value)):
        raise ValueError(f"{key} must be a list of strings, got {value!r}")
    invalid = sorted(set(value) - _REPAIR_PARTS)
    if invalid:
        raise ValueError(f"unknown {key}: {invalid}")
    return value


def _regions_argument(value: object, *, key: str = "regions") -> list[list[float]]:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array, got {type(value).__name__}")
    parsed = []
    for region in value:
        if (not isinstance(region, list) or len(region) != 4
                or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                       for v in region)):
            raise ValueError(
                f"each region must be [x0, y0, x1, y1] numbers, got {region!r}")
        if any(not (0 <= v <= 1) for v in region):
            raise ValueError(f"region values must be within 0..1, got {region!r}")
        parsed.append([float(v) for v in region])
    return parsed


def _outlines_argument(value: object, *, key: str = "outlines") -> list[dict]:
    if value is None:
        return [dict(outline) for outline in DELIVER_DEFAULTS["outlines"]]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be null or an array, got {type(value).__name__}")
    if len(value) > OUTLINES_MAX:
        raise ValueError(f"{key} must have at most {OUTLINES_MAX} entries, got {len(value)}")
    parsed = []
    for index, outline in enumerate(value):
        where = f"{key}[{index}]"
        if not isinstance(outline, Mapping) or set(outline) != {"color", "width"}:
            raise ValueError(f"{where} must be {{\"color\", \"width\"}}, got {outline!r}")
        color, width = outline["color"], outline["width"]
        if (not isinstance(color, str) or len(color) != 7 or color[0] != "#"
                or any(c not in "0123456789abcdefABCDEF" for c in color[1:])):
            raise ValueError(f"{where}.color must be #rrggbb, got {color!r}")
        if not isinstance(width, (int, float)) or isinstance(width, bool):
            raise ValueError(f"{where}.width must be a number, got {width!r}")
        if not (0 < width <= OUTLINE_WIDTH_MAX_PCT):
            raise ValueError(
                f"{where}.width must be > 0 and <= {OUTLINE_WIDTH_MAX_PCT:g}, got {width!r}")
        parsed.append({"color": color.lower(), "width": float(width)})
    return parsed


def _light_argument(value: object, *, key: str = "light") -> Light | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be null or an object, got {type(value).__name__}")
    unknown = sorted(set(value) - {"scene", "from"})
    if unknown:
        raise ValueError(f"{key} has unknown keys: {unknown}")
    scene = value.get("scene")
    if scene not in LIGHT_SCENES:
        raise ValueError(
            f"{key}.scene must be one of {sorted(LIGHT_SCENES)}, got {scene!r}")
    direction = value.get("from", LIGHT_FROM_DEFAULT)
    if direction not in STROKE_LIGHTS:
        raise ValueError(
            f"{key}.from must be one of {sorted(STROKE_LIGHTS)}, got {direction!r}")
    return Light(scene, direction)


def _denoise_argument(value: object, *, key: str = "denoise", max_value: float = 1) -> float:
    if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        raise ValueError(f"{key} must be a number, got {type(value).__name__}")
    if not (0 < value <= max_value):
        raise ValueError(f"{key} must be > 0 and <= {max_value}, got {value!r}")
    return float(value)


def _pixel_argument(value: object, *, key: str, max_value: float) -> float:
    if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        raise ValueError(f"{key} must be a number, got {type(value).__name__}")
    if not (0 <= value <= max_value):
        raise ValueError(f"{key} must be between 0 and {max_value}, got {value!r}")
    return float(value)


def _pad_argument(value: object, *, key: str = "pad") -> float:
    if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        raise ValueError(f"{key} must be a number, got {type(value).__name__}")
    if not (0.5 <= value <= 3):
        raise ValueError(f"{key} must be between 0.5 and 3, got {value!r}")
    return float(value)


def _keep_strength_argument(value: object, *, key: str = "keep_strength") -> float:
    if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        raise ValueError(f"{key} must be a number, got {type(value).__name__}")
    if not (0 < value < 1):
        raise ValueError(f"{key} must be > 0 and < 1, got {value!r}")
    return float(value)


def _crop_size_argument(value: object, *, key: str = "size") -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, got {type(value).__name__}")
    if value < 256 or value % 8 != 0:
        raise ValueError(f"{key} must be a multiple of 8, at least 256, got {value!r}")
    return value


def _part_lora_argument(value: object, *, key: str = "lora") -> float | None:
    if value is True:
        return DEFAULT_PART_LORA_WEIGHT
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not (0 < value <= 2):
            raise ValueError(f"{key} must be > 0 and <= 2, got {value!r}")
        return float(value)
    raise ValueError(
        f"{key} must be null, true or a number, got {type(value).__name__}")


def _model_argument(value: object, *, key: str = "model") -> str | None:
    if value is None:
        return None
    if value not in MODELS:
        valid = ", ".join(repr(word) for word in sorted(MODELS))
        raise ValueError(f"{key} must be null or one of {valid}, got {value!r}")
    return value


def _control_argument(value: object, *, key: str = "control") -> str | None:
    if value is None:
        return None
    if value not in CONTROL_MODELS:
        valid = ", ".join(repr(word) for word in sorted(CONTROL_MODELS))
        raise ValueError(f"{key} must be null or one of {valid}, got {value!r}")
    return value


def _control_strength_argument(value: object, *,
                               key: str = "control_strength") -> float:
    if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        raise ValueError(f"{key} must be a number, got {type(value).__name__}")
    if not (0 < value <= 2):
        raise ValueError(f"{key} must be > 0 and <= 2, got {value!r}")
    return float(value)


def redraw_arguments(options: Mapping,
                     dials: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    """Validate a redraw request's `options` and map it to redraw() kwargs.

    `method` selects which options are accepted; unknown keys or a wrong
    type raise ValueError naming the offending key.
    """
    if not isinstance(options, Mapping):
        raise ValueError(
            f"redraw options must be an object, got {type(options).__name__}")
    method = options.get("method")
    if method not in _KNOWN_REDRAW_OPTIONS:
        raise ValueError(
            f"method must be one of {sorted(_KNOWN_REDRAW_OPTIONS)}, got {method!r}")
    unknown = sorted(set(options) - _KNOWN_REDRAW_OPTIONS[method] - {"method"})
    if unknown:
        raise ValueError(
            f"redraw {method} が受け付けない option です: {', '.join(unknown)}")
    dials = dials or {}

    if method == "light":
        light = _light_argument(
            {"scene": options.get("scene"),
             **({"from": options["from"]} if "from" in options else {})},
            key="light")
        return {"method": method, "light": light}

    if method == "hires":
        hires = options.get("hires")
        if not (isinstance(hires, int) and not isinstance(hires, bool)):
            raise ValueError(f"hires must be an integer, got {hires!r}")
        if hires <= 0:
            raise ValueError(f"hires must be at least 1, got {hires!r}")
        denoise = (_denoise_argument(options["denoise"]) if "denoise" in options
                   else HIRES_DENOISE)
        return {"method": method, "hires": hires, "denoise": denoise}

    denoise = resolve_dial("denoise", options.get("denoise"), dials)
    if denoise is not None:
        denoise = _denoise_argument(denoise)

    route = options.get("route")
    if route is None:
        latent_route = None
    elif route == "latent":
        latent_route = True
    elif route == "pixel":
        latent_route = False
    else:
        raise ValueError(f"route must be null, 'latent' or 'pixel', got {route!r}")

    finalizer = options.get("finalizer")
    if finalizer is not None and not isinstance(finalizer, str):
        raise ValueError(f"finalizer must be null or a string, got {type(finalizer).__name__}")

    size = options.get("size")
    if size is not None and not (isinstance(size, int) and not isinstance(size, bool)):
        raise ValueError(f"size must be null or an integer, got {type(size).__name__}")

    upscale = options.get("upscale")
    if upscale is not None and upscale not in (
            "bicubic", "nearest-exact", "bilinear", "lanczos"):
        raise ValueError(
            "upscale must be null, 'bicubic', 'nearest-exact', 'bilinear' or "
            f"'lanczos', got {upscale!r}")

    return {
        "method": method,
        "denoise": denoise,
        "size": size,
        "latent_route": latent_route,
        "finalizer": finalizer,
        "upscale": upscale,
        "keep_regions": _regions_argument(
            options.get("keep_regions", []), key="keep_regions"),
        "keep_strength": _keep_strength_argument(options.get("keep_strength", 0.25)),
    }


def deliver_arguments(options: Mapping,
                      dials: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    """Validate a deliver request's `options` and map it to deliver() kwargs.

    Absent keys take DELIVER_DEFAULTS; unknown keys or a wrong type raise
    ValueError naming the offending key.
    """
    if not isinstance(options, Mapping):
        raise ValueError(
            f"deliver options must be an object, got {type(options).__name__}")
    unknown = sorted(set(options) - _KNOWN_DELIVER_OPTIONS)
    if unknown:
        raise ValueError(f"deliver が受け付けない option です: {', '.join(unknown)}")
    dials = dials or {}

    def boolean(key: str, default: bool = False) -> bool:
        value = options.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean, got {type(value).__name__}")
        return value

    keep_legwear = resolve_dial("keep_legwear", options.get("keep_legwear"), dials)
    if keep_legwear is True:
        keep_legwear = 0.62
    elif keep_legwear is not None and not (
            isinstance(keep_legwear, (int, float)) and not isinstance(keep_legwear, bool)):
        raise ValueError(
            "keep_legwear must be null, true or a number, got "
            f"{type(keep_legwear).__name__}")

    deliver_size = options.get("deliver_size")
    if deliver_size is not None and not (
            isinstance(deliver_size, int) and not isinstance(deliver_size, bool)):
        raise ValueError(
            f"deliver_size must be null or an integer, got {type(deliver_size).__name__}")
    if deliver_size is not None and deliver_size < 1:
        raise ValueError(f"deliver_size must be at least 1, got {deliver_size!r}")

    transparent = options.get("transparent")
    if transparent is not None and not isinstance(transparent, bool):
        raise ValueError(
            f"transparent must be null or a boolean, got {type(transparent).__name__}")
    keep_scene = boolean("keep_scene")
    if transparent is None and "backdrop" in options and options["backdrop"] is None:
        transparent = True
    transparent = bool(transparent) and not keep_scene

    if "backdrop" not in options:
        backdrop = None if transparent else DELIVER_DEFAULTS["backdrop"]
    else:
        backdrop = options["backdrop"]
        if backdrop is not None:
            if not isinstance(backdrop, str):
                raise ValueError(
                    f"backdrop must be null or a string, got {type(backdrop).__name__}")
            if not is_backdrop(backdrop):
                names = ", ".join(repr(key) for key in sorted(PATTERNS))
                raise ValueError(
                    f"backdrop must be null, a #RRGGBB colour or one of {names}, "
                    f"got {backdrop!r}")

    light = _light_argument(options.get("light"))
    if "stroke_light" not in options:
        stroke_light = light.direction if light is not None else RECIPE_DEFAULT
    else:
        stroke_light = options["stroke_light"]
        if stroke_light is not None and stroke_light not in STROKE_CHOICES:
            valid = ", ".join(repr(key) for key in STROKE_CHOICES)
            raise ValueError(
                f"stroke_light must be null or one of {valid}, got {stroke_light!r}")
    if (light is not None and stroke_light in STROKE_LIGHTS
            and stroke_light != light.direction):
        raise ValueError(stroke_light_conflict(light.direction))

    outlines = _outlines_argument(options.get("outlines"))

    see_through = options.get("see_through")
    if see_through is None:
        see_through = 1.0
    elif not isinstance(see_through, (int, float)) or isinstance(see_through, bool):
        raise ValueError(
            f"see_through must be null or a number, got {type(see_through).__name__}")
    low, high = _SEE_THROUGH_RANGE
    if not low <= see_through <= high:
        raise ValueError(f"see_through must be between {low:g} and {high:g}, got {see_through!r}")

    recolor = boolean("recolor")
    return {
        "repin": boolean("repin", DELIVER_DEFAULTS["repin"]) and not recolor,
        "skin": boolean("skin"),
        "recolor": recolor,
        "keep_legwear": float(keep_legwear) if keep_legwear is not None else None,
        "keep_scene": keep_scene,
        "transparent": transparent,
        "backdrop": backdrop,
        "stroke_light": stroke_light,
        "outlines": outlines,
        "deliver_size": deliver_size,
        "light": light,
        **({"see_through": float(see_through)} if see_through != 1.0 else {}),
    }


def dof_arguments(options: Mapping) -> dict:
    """Validate a dof request's `options` and map it to dof() kwargs.

    `focus` is required; absent keys take the catalog defaults; unknown keys
    or a wrong type raise ValueError naming the offending key.
    """
    if not isinstance(options, Mapping):
        raise ValueError(
            f"dof options must be an object, got {type(options).__name__}")
    unknown = sorted(set(options) - _KNOWN_DOF_OPTIONS)
    if unknown:
        raise ValueError(f"dof が受け付けない option です: {', '.join(unknown)}")
    focus = options.get("focus")
    if (not isinstance(focus, list) or len(focus) != 2
            or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                   for v in focus)):
        raise ValueError(f"focus must be [x, y] numbers, got {focus!r}")
    if any(not (0 <= v <= 1) for v in focus):
        raise ValueError(f"focus values must be within 0..1, got {focus!r}")
    f_number = options.get("f_number", DOF_F_NUMBER["default"])
    if not isinstance(f_number, (int, float)) or isinstance(f_number, bool):
        raise ValueError(f"f_number must be a number, got {f_number!r}")
    if not (DOF_F_NUMBER["min"] <= f_number <= DOF_F_NUMBER["max"]):
        raise ValueError(
            f"f_number must be between {DOF_F_NUMBER['min']} and "
            f"{DOF_F_NUMBER['max']}, got {f_number!r}")
    scope = options.get("scope")
    if scope is None:
        scope = {}
    if not isinstance(scope, Mapping):
        raise ValueError(f"scope must be null or an object, got {type(scope).__name__}")
    unknown = sorted(set(scope) - set(DOF_SCOPE))
    if unknown:
        raise ValueError(f"scope has unknown keys: {unknown}")
    if any(not isinstance(value, bool) for value in scope.values()):
        raise ValueError(f"scope values must be booleans, got {dict(scope)!r}")
    scope = {**DOF_SCOPE, **scope}
    if not any(scope.values()):
        raise ValueError("scope は figure / outline / backdrop のどれか一つはオンにしてください")
    viewfinder = options.get("viewfinder", DOF_VIEWFINDER["default"])
    if viewfinder not in DOF_VIEWFINDER["values"]:
        raise ValueError(
            f"viewfinder must be one of {DOF_VIEWFINDER['values']}, got {viewfinder!r}")
    photo_blur = None
    if _DOF_PHOTO_OPTIONS & set(options):
        photo_blur = {"stand_at": _point(options, "stand_at", None),
                      "nearer": _ranged(options, "nearer", PHOTO_NEARER),
                      "tilt": _ranged(options, "tilt", PHOTO_TILT),
                      "bokeh": _ranged(options, "bokeh", PHOTO_BOKEH),
                      "focal_length": _focal_length(options)}
    return {"focus": (float(focus[0]), float(focus[1])),
            "f_number": float(f_number), "scope": scope, "viewfinder": viewfinder,
            "photo_blur": photo_blur}


def lens_arguments(options: Mapping) -> dict:
    """Validate a lens request's `options` and map it to lens() kwargs.

    Absent keys take the catalog defaults; unknown keys or a wrong type raise
    ValueError naming the offending key.
    """
    if not isinstance(options, Mapping):
        raise ValueError(
            f"lens options must be an object, got {type(options).__name__}")
    unknown = sorted(set(options) - _KNOWN_LENS_OPTIONS)
    if unknown:
        raise ValueError(f"lens が受け付けない option です: {', '.join(unknown)}")
    center = options.get("center", list(LENS_CENTER_DEFAULT))
    if (not isinstance(center, list) or len(center) != 2
            or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                   for v in center)):
        raise ValueError(f"center must be [x, y] numbers, got {center!r}")
    if any(not (0 <= v <= 1) for v in center):
        raise ValueError(f"center values must be within 0..1, got {center!r}")
    strength = options.get("strength", LENS_STRENGTH["default"])
    if not isinstance(strength, (int, float)) or isinstance(strength, bool):
        raise ValueError(f"strength must be a number, got {strength!r}")
    if not (LENS_STRENGTH["min"] <= strength <= LENS_STRENGTH["max"]):
        raise ValueError(
            f"strength must be between {LENS_STRENGTH['min']} and "
            f"{LENS_STRENGTH['max']}, got {strength!r}")
    if strength == 0:
        raise ValueError("strength 0 ではレンズが歪みません。魚眼は正、逆魚眼は負の値です")
    shape = options.get("shape", LENS_SHAPE["default"])
    if shape not in LENS_SHAPE["values"]:
        raise ValueError(f"shape must be one of {LENS_SHAPE['values']}, got {shape!r}")
    radius = None
    if shape == "sphere":
        radius = options.get("radius", LENS_RADIUS["default"])
        if not isinstance(radius, (int, float)) or isinstance(radius, bool):
            raise ValueError(f"radius must be a number, got {radius!r}")
        if not (LENS_RADIUS["min"] <= radius <= LENS_RADIUS["max"]):
            raise ValueError(
                f"radius must be between {LENS_RADIUS['min']} and "
                f"{LENS_RADIUS['max']}, got {radius!r}")
        radius = float(radius)
    elif "radius" in options:
        raise ValueError("radius は shape が sphere のときだけ使えます")
    return {"center": (float(center[0]), float(center[1])),
            "strength": float(strength), "shape": shape, "radius": radius}


def _point(options: Mapping, key: str, default, max_y: float = 1.0):
    value = options.get(key, default)
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in value)):
        raise ValueError(f"{key} must be [x, y] numbers, got {value!r}")
    if not (0 <= value[0] <= 1 and 0 <= value[1] <= max_y):
        bounds = "0..1" if max_y == 1.0 else f"0..1 for x and 0..{max_y} for y"
        raise ValueError(f"{key} values must be within {bounds}, got {value!r}")
    return (float(value[0]), float(value[1]))


def _ranged(options: Mapping, key: str, limits: Mapping) -> float:
    value = options.get(key, limits["default"])
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number, got {value!r}")
    if not (limits["min"] <= value <= limits["max"]):
        raise ValueError(
            f"{key} must be between {limits['min']} and {limits['max']}, got {value!r}")
    return float(value)


def _focal_length(options: Mapping) -> str:
    value = options.get("focal_length", PHOTO_FOCAL_LENGTH["default"])
    if value not in PHOTO_FOCAL_LENGTH["values"]:
        raise ValueError(
            f"focal_length must be one of {PHOTO_FOCAL_LENGTH['values']}, got {value!r}")
    return value


def photo_arguments(options: Mapping) -> dict:
    """Validate a photo request's `options` and map it to photo() kwargs.

    `photo` (the Generation whose image is the scene) is required; absent
    keys take the catalog defaults; unknown keys or a wrong type raise
    ValueError naming the offending key.
    """
    if not isinstance(options, Mapping):
        raise ValueError(
            f"photo options must be an object, got {type(options).__name__}")
    unknown = sorted(set(options) - _KNOWN_PHOTO_OPTIONS)
    if unknown:
        raise ValueError(f"photo が受け付けない option です: {', '.join(unknown)}")
    photo_id = options.get("photo")
    if not isinstance(photo_id, str) or not photo_id:
        raise ValueError(f"photo must be the photo Generation's id, got {photo_id!r}")
    focus = options.get("focus", "subject")
    if focus != "subject":
        focus = _point(options, "focus", None)
        if focus is None:
            raise ValueError('focus must be "subject" or [x, y]')
    return {
        "photo_id": photo_id,
        "place": _point(options, "place", list(PHOTO_PLACE_DEFAULT), PHOTO_PLACE_MAX_Y),
        "height": _ranged(options, "height", PHOTO_HEIGHT),
        "stand_at": _point(options, "stand_at", None),
        "nearer": _ranged(options, "nearer", PHOTO_NEARER),
        "focus": focus,
        "f_number": _ranged(options, "f_number", DOF_F_NUMBER),
        "tilt": _ranged(options, "tilt", PHOTO_TILT),
        "bokeh": _ranged(options, "bokeh", PHOTO_BOKEH),
        "focal_length": _focal_length(options),
    }


_SHADOW_INHERITED = object()


def stage_arguments(options: Mapping) -> dict:
    """Validate a stage request's `options` and map it to stage() kwargs.

    `photo` (a photo Generation's id) sets the delivery in that photo, at
    `place` and `height`; without it the delivery is set on `backdrop`, a
    pattern or #RRGGBB (the deliver default when absent), with the drop
    shadow thrown away from `shadow_from` (a direction, null for none,
    absent to take the delivery's own light) and `light` ({scene, from}
    tinting the backdrop as a delivery in that scene; null for none; absent
    for the nearest light redraw in the picture's lineage). The two sets of
    keys do not mix.
    """
    if not isinstance(options, Mapping):
        raise ValueError(
            f"stage options must be an object, got {type(options).__name__}")
    unknown = sorted(set(options) - _KNOWN_STAGE_OPTIONS)
    if unknown:
        raise ValueError(f"stage が受け付けない option です: {', '.join(unknown)}")
    if "photo" in options:
        mixed = sorted(_STAGE_PATTERN_OPTIONS & set(options))
        if mixed:
            raise ValueError(f"photo と一緒には使えない option です: {', '.join(mixed)}")
        photo_id = options["photo"]
        if not isinstance(photo_id, str) or not photo_id:
            raise ValueError(f"photo must be the photo Generation's id, got {photo_id!r}")
        return {"photo_id": photo_id,
                "place": _point(options, "place", list(PHOTO_PLACE_DEFAULT),
                                PHOTO_PLACE_MAX_Y),
                "height": _ranged(options, "height", PHOTO_HEIGHT)}
    stray = sorted((_STAGE_PHOTO_OPTIONS - {"photo"}) & set(options))
    if stray:
        raise ValueError(f"photo なしでは使えない option です: {', '.join(stray)}")
    backdrop = options.get("backdrop", DELIVER_DEFAULTS["backdrop"])
    if not isinstance(backdrop, str) or not is_backdrop(backdrop):
        raise ValueError(
            f"backdrop must be one of {sorted(PATTERNS)} or a #RRGGBB colour, got {backdrop!r}")
    shadow_from = options.get("shadow_from", _SHADOW_INHERITED)
    if shadow_from is not _SHADOW_INHERITED and shadow_from is not None \
            and shadow_from not in STROKE_LIGHTS:
        raise ValueError(
            f"shadow_from must be null or one of {sorted(STROKE_LIGHTS)}, got {shadow_from!r}")
    return {"backdrop": backdrop,
            "shadow_from": None if shadow_from is _SHADOW_INHERITED else shadow_from,
            "inherit_shadow": shadow_from is _SHADOW_INHERITED,
            "light": _light_argument(options.get("light")),
            "inherit_light": "light" not in options}


def lighting_arguments(options: Mapping) -> dict:
    """Validate a lighting request's `options` and map it to lighting()
    kwargs: `direction` (null or absent for none; a scene's default is nw),
    `strength` and `scene` (sunset / moon, null or absent for none)."""
    if not isinstance(options, Mapping):
        raise ValueError(
            f"lighting options must be an object, got {type(options).__name__}")
    unknown = sorted(set(options) - _KNOWN_LIGHTING_OPTIONS)
    if unknown:
        raise ValueError(f"lighting が受け付けない option です: {', '.join(unknown)}")
    direction = options.get("direction")
    if direction is not None and direction not in STROKE_LIGHTS:
        raise ValueError(
            f"direction must be null or one of {sorted(STROKE_LIGHTS)}, got {direction!r}")
    scene = options.get("scene")
    if scene is not None and scene not in LIGHT_SCENES:
        raise ValueError(f"scene must be null or one of {sorted(LIGHT_SCENES)}, got {scene!r}")
    return {"direction": direction,
            "strength": _ranged(options, "strength", LIGHTING_STRENGTH),
            "scene": scene}


def repair_arguments(options: Mapping,
                     dials: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    """Validate a repair request's `options` and map it to repair() kwargs.

    Unknown keys or a wrong type raise ValueError naming the offending key.
    """
    if not isinstance(options, Mapping):
        raise ValueError(
            f"repair options must be an object, got {type(options).__name__}")
    unknown = sorted(set(options) - _KNOWN_REPAIR_OPTIONS)
    if unknown:
        raise ValueError(f"unknown repair options keys: {unknown}")
    dials = dials or {}

    parts = _parts_argument(options.get("parts", ["hands", "feet"]))
    parsed_regions = _regions_argument(options.get("regions", []))

    if not parts and not parsed_regions:
        raise ValueError("repair needs at least one of parts or regions")

    denoise = _denoise_argument(
        resolve_dial("denoise", options.get("denoise", 0.6), dials))

    seeds = options.get("seeds", [1, 2, 3, 4])
    if (not isinstance(seeds, list) or not seeds
            or any(not isinstance(seed, int) or isinstance(seed, bool)
                   for seed in seeds)):
        raise ValueError(f"seeds must be a non-empty array of integers, got {seeds!r}")

    size = _crop_size_argument(options.get("size", 1024))
    pad = _pad_argument(options.get("pad", 1.0))
    lora = _part_lora_argument(resolve_dial("lora", options.get("lora"), dials))
    model = _model_argument(options.get("model"))
    control = _control_argument(options.get("control"))
    control_strength = _control_strength_argument(
        options.get("control_strength", DEFAULT_CONTROL_STRENGTH))

    return {
        "parts": parts, "regions": parsed_regions, "denoise": denoise,
        "seeds": seeds, "size": size, "pad": pad, "lora": lora, "model": model,
        "control": control, "control_strength": control_strength,
    }


def masked_redraw_arguments(options: Mapping,
                            dials: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    """Validate a masked_redraw request's `options` and map it to
    masked_redraw() kwargs.

    `dials` is the source recipe's `dials.repair` vocabulary; masked_redraw
    has no `dials.masked_redraw` of its own and shares repair's `denoise`.
    """
    if not isinstance(options, Mapping):
        raise ValueError(
            f"masked_redraw options must be an object, got {type(options).__name__}")
    unknown = sorted(set(options) - _KNOWN_MASKED_REDRAW_OPTIONS)
    if unknown:
        raise ValueError(f"unknown masked_redraw options keys: {unknown}")
    dials = dials or {}

    regions = _regions_argument(options.get("regions", []))
    if not regions:
        raise ValueError("masked_redraw needs at least one region")

    prompt_patch = options.get("prompt_patch")
    if not isinstance(prompt_patch, str) or not prompt_patch:
        raise ValueError(
            f"prompt_patch must be a non-empty string, got {prompt_patch!r}")
    if len(prompt_patch) > _MASKED_REDRAW_PROMPT_PATCH_MAX_LENGTH:
        raise ValueError(
            "prompt_patch must be at most "
            f"{_MASKED_REDRAW_PROMPT_PATCH_MAX_LENGTH} characters, got {len(prompt_patch)}")

    denoise = _denoise_argument(
        resolve_dial("denoise", options.get("denoise", 0.45), dials), max_value=0.75)
    mask_padding = _pixel_argument(
        options.get("mask_padding", 0), key="mask_padding", max_value=512)
    mask_feather = _pixel_argument(
        options.get("mask_feather", 32), key="mask_feather", max_value=256)
    size = _crop_size_argument(options.get("size", 1024))

    seeds = options.get("seeds", [1, 2, 3, 4])
    if (not isinstance(seeds, list) or not seeds
            or any(not isinstance(seed, int) or isinstance(seed, bool)
                   for seed in seeds)):
        raise ValueError(f"seeds must be a non-empty array of integers, got {seeds!r}")
    if len(seeds) > _MASKED_REDRAW_SEEDS_MAX:
        raise ValueError(
            f"seeds must have at most {_MASKED_REDRAW_SEEDS_MAX} entries, got {len(seeds)}")

    return {
        "regions": regions, "prompt_patch": prompt_patch, "denoise": denoise,
        "mask_padding": mask_padding, "mask_feather": mask_feather, "size": size,
        "seeds": seeds,
    }
