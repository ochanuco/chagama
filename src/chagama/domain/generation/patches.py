"""Declarative request-time diffs against a resolved RenderSpec."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from .models import RenderSpec

TEXT_TARGETS = ("prompt.positive", "prompt.negative",
                "prompt.hires.positive", "prompt.hires.negative")
# `prompt.positive.<part>` targets one named part instead of the whole
# joined string. `<part>` is a `RenderSpec.positive_parts` name, or a
# `RenderSpec.part_groups` legacy name resolved via `_apply_group`.
PART_TARGET_PREFIX = "prompt.positive."
NUMBER_TARGETS = ("render.cfg", "render.steps", "render.width",
                  "render.height", "render.guided_steps", "hires.denoise")
STRING_TARGETS = ("render.model", "render.sampler", "render.scheduler")
LORAS_TARGETS = ("render.loras",)
TEXT_OPS = ("append", "prepend", "replace", "remove")
_KNOWN_KEYS = frozenset({"target", "op", "value", "old", "reason"})

NUMBER_CONSTRAINTS = {
    "render.steps": "an int >= 1",
    "render.width": "an int >= 64, a multiple of 8",
    "render.height": "an int >= 64, a multiple of 8",
    "render.cfg": "> 0",
    "render.guided_steps": "an int >= 0 (0 means off)",
    "hires.denoise": "0 < value <= 1",
}


@dataclass(frozen=True)
class Patch:
    target: str
    op: str
    value: str | float | int | tuple[tuple[str, float], ...] | None
    old: str | None
    reason: str


def _fail(target: object, message: str) -> None:
    raise ValueError(f"patch {target!r}: {message}")


def _require_str(patch: dict, key: str, target: object) -> str:
    value = patch.get(key)
    if not isinstance(value, str) or not value:
        _fail(target, f"{key!r} must be a non-empty string")
    return value


def _parse_text_patch(patch: dict, target: str) -> Patch:
    op = patch.get("op")
    if op not in TEXT_OPS:
        _fail(target, f"op must be one of {TEXT_OPS}, got {op!r}")
    reason = _require_str(patch, "reason", target)
    if op in ("append", "prepend"):
        if "old" in patch:
            _fail(target, f"op {op!r} does not take 'old'")
        value = _require_str(patch, "value", target)
        return Patch(target, op, value, None, reason)
    if op == "replace":
        old = _require_str(patch, "old", target)
        value = _require_str(patch, "value", target)
        return Patch(target, op, value, old, reason)
    old = _require_str(patch, "old", target)
    if "value" in patch:
        _fail(target, "op 'remove' does not take 'value'")
    return Patch(target, op, None, old, reason)


def _parse_number_patch(patch: dict, target: str,
                        dials: Mapping[str, Mapping[str, float]]) -> Patch:
    op = patch.get("op")
    if op != "set":
        _fail(target, f"op must be 'set' for {target!r}, got {op!r}")
    reason = _require_str(patch, "reason", target)
    value = patch.get("value")
    if isinstance(value, str):
        words = dials.get(target) or {}
        if value not in words:
            _fail(target, f"unknown word {value!r}")
        value = words[value]
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(target, f"value must be a number, got {value!r}")
    constraint = NUMBER_CONSTRAINTS[target]
    if target == "render.steps":
        if not isinstance(value, int) or value < 1:
            _fail(target, f"{target} value must be {constraint}")
    elif target == "render.guided_steps":
        if not isinstance(value, int) or value < 0:
            _fail(target, f"{target} value must be {constraint}")
    elif target in ("render.width", "render.height"):
        if not isinstance(value, int) or value < 64 or value % 8:
            _fail(target, f"{target} value must be {constraint}")
    elif target == "render.cfg":
        if value <= 0:
            _fail(target, f"{target} value must be {constraint}")
    elif value <= 0 or value > 1:
        _fail(target, f"{target} value must satisfy {constraint}")
    return Patch(target, op, value, None, reason)


def _parse_string_patch(patch: dict, target: str) -> Patch:
    op = patch.get("op")
    if op != "set":
        _fail(target, f"op must be 'set' for {target!r}, got {op!r}")
    reason = _require_str(patch, "reason", target)
    value = _require_str(patch, "value", target)
    return Patch(target, op, value, None, reason)


def _parse_loras_patch(patch: dict, target: str) -> Patch:
    op = patch.get("op")
    if op != "set":
        _fail(target, f"op must be 'set' for {target!r}, got {op!r}")
    reason = _require_str(patch, "reason", target)
    value = patch.get("value")
    if not isinstance(value, list) or not value:
        _fail(target, "value must be a non-empty list of [name, strength] pairs")
    parsed = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            _fail(target, f"each entry must be [name, strength], got {pair!r}")
        name, strength = pair
        if (not isinstance(name, str) or not name
                or not name.endswith(".safetensors")):
            _fail(target,
                  "lora name must be a non-empty string ending in "
                  f".safetensors, got {name!r}")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            _fail(target, f"lora strength must be a number, got {strength!r}")
        if not (0 <= strength <= 2):
            _fail(target, f"lora strength must be between 0 and 2, got {strength!r}")
        parsed.append((name, float(strength)))
    return Patch(target, op, tuple(parsed), None, reason)


def parse_patches(raw: object,
                  dials: Mapping[str, Mapping[str, float]] | None = None) -> tuple[Patch, ...]:
    """`dials` is the source recipe's `dials.patches` vocabulary (target ->
    word -> number); a number target with no entry there rejects any word.
    """
    if not isinstance(raw, list):
        raise ValueError("generation.patches must be an array of patch objects")
    dials = dials or {}
    patches = []
    for patch in raw:
        if not isinstance(patch, dict):
            _fail(patch, "patch must be an object")
        unknown = set(patch) - _KNOWN_KEYS
        if unknown:
            _fail(patch.get("target"),
                  f"unknown patch keys: {sorted(unknown)}")
        target = patch.get("target")
        if not isinstance(target, str):
            _fail(target, "'target' must be a string")
        part_name = (target[len(PART_TARGET_PREFIX):]
                    if target.startswith(PART_TARGET_PREFIX) else "")
        if target in TEXT_TARGETS:
            patches.append(_parse_text_patch(patch, target))
        elif part_name and "." not in part_name:
            patches.append(_parse_text_patch(patch, target))
        elif target in NUMBER_TARGETS:
            patches.append(_parse_number_patch(patch, target, dials))
        elif target in STRING_TARGETS:
            patches.append(_parse_string_patch(patch, target))
        elif target in LORAS_TARGETS:
            patches.append(_parse_loras_patch(patch, target))
        else:
            allowed = (TEXT_TARGETS + (PART_TARGET_PREFIX + "<part>",)
                      + NUMBER_TARGETS + STRING_TARGETS + LORAS_TARGETS)
            _fail(target, f"unknown target, must be one of {allowed}")
    return tuple(patches)


def _splice(text: str, target: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"patch needle absent in {target}: {old!r}")
    return text.replace(old, new)


def _apply_text(text: str, patch: Patch) -> str:
    if patch.op == "append":
        return text + patch.value
    if patch.op == "prepend":
        return patch.value + text
    if patch.op == "replace":
        return _splice(text, patch.target, patch.old, patch.value)
    return _splice(text, patch.target, patch.old, "")


def _apply_component(spec: RenderSpec, patch: Patch, component_name: str) -> RenderSpec:
    parts = list(spec.positive_parts)
    names = [name for name, _ in parts]
    index = names.index(component_name)
    name, text = parts[index]
    parts[index] = (name, _apply_text(text, patch))
    joined = "".join(text for _, text in parts)
    prompts = replace(spec.prompts, positive=joined)
    return replace(spec, positive_parts=tuple(parts), prompts=prompts)


def _apply_group(spec: RenderSpec, patch: Patch, group_name: str,
                 members: tuple[str, ...]) -> RenderSpec:
    if patch.op in ("append", "prepend"):
        target = members[-1] if patch.op == "append" else members[0]
        return _apply_component(spec, patch, target)
    by_name = dict(spec.positive_parts)
    for member in members:
        if patch.old in by_name.get(member, ""):
            return _apply_component(spec, patch, member)
    raise ValueError(
        f"patch {patch.target!r}: target text not found within a single "
        f"member of group {group_name!r} ({', '.join(members)}); it may be "
        "split across them -- patch the member components directly")


def _apply_part(spec: RenderSpec, patch: Patch, part_name: str) -> RenderSpec:
    if not spec.positive_parts:
        raise ValueError(
            f"patch {patch.target!r}: this recipe has no prompt parts")
    names = [name for name, _ in spec.positive_parts]
    if part_name in spec.part_groups:
        return _apply_group(spec, patch, part_name, spec.part_groups[part_name])
    if part_name in names:
        return _apply_component(spec, patch, part_name)
    allowed = sorted(set(names) | set(spec.part_groups))
    raise ValueError(
        f"patch {patch.target!r}: unknown part, must be one of {allowed}")


def _apply_one(spec: RenderSpec, patch: Patch) -> RenderSpec:
    if patch.target.startswith(PART_TARGET_PREFIX):
        return _apply_part(spec, patch, patch.target[len(PART_TARGET_PREFIX):])
    if patch.target == "prompt.positive":
        prompts = replace(spec.prompts,
                          positive=_apply_text(spec.prompts.positive, patch))
        return replace(spec, prompts=prompts)
    if patch.target == "prompt.negative":
        prompts = replace(spec.prompts,
                          negative=_apply_text(spec.prompts.negative, patch))
        return replace(spec, prompts=prompts)
    if patch.target in ("prompt.hires.positive", "prompt.hires.negative",
                        "hires.denoise"):
        if spec.hires is None:
            raise ValueError(f"{patch.target} requires request hires")
    if patch.target == "prompt.hires.positive":
        # A pose without a pass-2 positive shares pass 1's node; materializing
        # it here just spells that sharing out before the patch edits it.
        base = (spec.hires.positive if spec.hires.positive is not None
                else spec.prompts.positive)
        hires = replace(spec.hires, positive=_apply_text(base, patch))
        return replace(spec, hires=hires)
    if patch.target == "prompt.hires.negative":
        hires = replace(spec.hires,
                        negative=_apply_text(spec.hires.negative, patch))
        return replace(spec, hires=hires)
    if patch.target == "render.cfg":
        return replace(spec, cfg=float(patch.value))
    if patch.target == "render.steps":
        return replace(spec, steps=patch.value)
    if patch.target == "render.guided_steps":
        return replace(spec, guided_steps=patch.value)
    if patch.target == "render.width":
        return replace(spec, width=patch.value)
    if patch.target == "render.height":
        return replace(spec, height=patch.value)
    if patch.target == "render.model":
        return replace(spec, model_path=patch.value)
    if patch.target == "render.sampler":
        return replace(spec, sampler_name=patch.value)
    if patch.target == "render.scheduler":
        return replace(spec, scheduler=patch.value)
    if patch.target == "render.loras":
        return replace(spec, loras=patch.value)
    hires = replace(spec.hires, denoise=float(patch.value))
    return replace(spec, hires=hires)


def apply_patches(spec: RenderSpec, patches: tuple[Patch, ...]) -> RenderSpec:
    for patch in patches:
        spec = _apply_one(spec, patch)
    return spec
