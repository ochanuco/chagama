"""A reference picture's tags onto a yukari pose: which tags, which pose, which
part patches.

Only the composition, the gesture, the expression and the legwear are taken
from the picture; the character stays the recipe's (the identity part is
never touched).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from ..generation.prompt_lint import tags as bare_tags
from .poses import POSES
from .recipe import negative, positive_parts

# Prompt part -> the Anima Tagger groups (groups.yaml) whose tags it takes.
PART_SOURCES = {
    "framing_tags": ("framing", "viewpoint_angle", "composition_technique"),
    "gesture": ("pose", "gesture"),
    "mouth": ("expression", "emotion"),
    "legwear": ("socks_stockings",),
}
# Gesture tags the tagger scores but groups.yaml leaves ungrouped.
EXTRA_GESTURE = (
    "hand between legs", "hands on own chest", "hand on own chest",
    "hands on own face", "own hands together", "hand on own cheek", "head rest",
    "arm up", "hand up", "leg up", "standing on one leg", "outstretched arm",
    "reaching towards viewer", "incoming food", "knees up", "hugging own legs",
    "leaning forward",
)
# Where she looks, not where the camera is: plain gesture tags, unweighted.
GAZE = frozenset({"looking at viewer", "looking up", "looking down",
                  "looking away", "looking to the side", "eye contact"})
MIN_SCORE = 0.1
MIN_LISTED_SCORE = 0.05
MIN_VERDICT = 0.5
FRAMING_WEIGHT = 1.3
# Tags of those groups that describe furniture, the floor, other people,
# image metadata or explicit content rather than her pose.
DENIED = frozenset({
    "on floor", "shoe soles",
    "on couch", "on chair", "on desk", "against wall", "head on pillow",
    "sitting on lap", "boy on top", "straddling", "pov hands", "pov crotch",
    "censored", "watermark", "dated", "multiple views", "cross-section",
    "outside border", "white border", "solo focus", "size difference",
    "upskirt", "covering privates", "covering breasts", "bikini pull",
    "saliva trail", "drooling", "ahegao", "torogao", "penis awe", "aroused",
    "moaning",
})
SMILES = frozenset({
    "smile", "light smile", "seductive smile", "grin", ":d", ";d", "laughing",
    "open mouth", "one eye closed", "^ ^",
})
SMILE_CONFLICTS = frozenset({"unamused", "expressionless"})

FULL_BODY = frozenset({"full body"})
KNEE_UP = frozenset({"cowboy shot", "feet out of frame", "foot out of frame",
                     "cropped legs"})
CLOSE = frozenset({"upper body", "portrait", "close-up"})
WHOLE_BODY_POSTURE = frozenset({
    "standing", "sitting", "kneeling", "squatting", "lying", "on back",
    "on stomach", "leg up", "standing on one leg", "knees up", "crossed legs",
    "seiza", "wariza", "indian style", "jumping", "walking", "running",
})
FULL_BODY_POSE = "stand"
KNEE_UP_POSE = "coffee"
CLOSE_POSE = "bust"
POSE_CHOICES = (CLOSE_POSE, KNEE_UP_POSE, FULL_BODY_POSE)
REASON = "参考画像のタグ"


@dataclass(frozen=True)
class Plan:
    pose: str
    applied: dict[str, list[str]]
    patches: list[dict]
    dropped: list[str]


def candidates(groups: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Candidate tag -> the part it would fill."""
    part_of = {}
    for part, sources in PART_SOURCES.items():
        for source in sources:
            for tag in groups.get(source, ()):
                if tag not in DENIED:
                    part_of[tag] = "gesture" if tag in GAZE else part
    for tag in EXTRA_GESTURE:
        part_of.setdefault(tag, "gesture")
    return part_of


def questions(scores: Mapping[str, float], kept: Collection[str],
              groups: Mapping[str, Sequence[str]]) -> list[str]:
    """The candidates the tagger left undecided: d1 is asked about these."""
    asked = [tag for tag in candidates(groups)
             if tag not in kept and scores.get(tag, 0.0) >= MIN_SCORE]
    return sorted(asked, key=lambda tag: (-scores[tag], tag))


def extract(scores: Mapping[str, float], kept: Collection[str],
            groups: Mapping[str, Sequence[str]],
            verdicts: Mapping[str, float]) -> dict[str, list[str]]:
    """Accepted tags per part, highest score first: kept by the tagger, or
    scored at least `MIN_SCORE` and confirmed by d1."""
    accepted: dict[str, list[str]] = {part: [] for part in PART_SOURCES}
    for tag, part in candidates(groups).items():
        score = scores.get(tag, 0.0)
        if tag in kept or (score >= MIN_SCORE
                           and verdicts.get(tag, 0.0) >= MIN_VERDICT):
            accepted[part].append(tag)
    for tags in accepted.values():
        tags.sort(key=lambda tag: (-scores.get(tag, 0.0), tag))
    return accepted


def choose_pose(framing: Collection[str], gesture: Collection[str] = ()) -> str:
    if FULL_BODY & set(framing):
        return FULL_BODY_POSE
    if KNEE_UP & set(framing):
        return KNEE_UP_POSE
    if not CLOSE & set(framing) and WHOLE_BODY_POSTURE & set(gesture):
        return FULL_BODY_POSE
    return CLOSE_POSE


def prompt_form(part: str, tag: str) -> str:
    return f"({tag}:{FRAMING_WEIGHT})" if part == "framing_tags" else tag


def _part_patch(part: str, old: str, new: str) -> dict | None:
    target = f"prompt.positive.{part}"
    if old == new:
        return None
    if not old:
        return {"target": target, "op": "append", "value": new, "reason": REASON}
    if not new:
        return {"target": target, "op": "remove", "old": old, "reason": REASON}
    return {"target": target, "op": "replace", "old": old, "value": new,
            "reason": REASON}


def _items(text: str) -> list[tuple[str, str]]:
    """Each comma-separated item as (bare tag, the text that removes it)."""
    items = []
    for match in re.finditer(r"[^,]+(,\s*)?", text):
        bare = bare_tags(match.group(0).rstrip(", "))
        if bare:
            items.append((bare[0].lower(), match.group(0).lstrip()))
    return items


def _remove_items(target: str, text: str, names: Collection[str]) -> list[dict]:
    olds = dict.fromkeys(old for bare, old in _items(text) if bare in names)
    return [{"target": target, "op": "remove", "old": old, "reason": REASON}
            for old in olds]


def build(pose: str, accepted: Mapping[str, Sequence[str]],
          kept: Collection[str] = ()) -> Plan:
    """The patches that put `accepted` (part -> tags) onto `pose`."""
    if pose not in POSE_CHOICES:
        raise ValueError(f"pose must be one of {POSE_CHOICES}, got {pose!r}")
    unknown = set(accepted) - set(PART_SOURCES)
    if unknown:
        raise ValueError(f"unknown parts {sorted(unknown)}")
    accepted = {part: list(dict.fromkeys(accepted.get(part, ())))
                for part in PART_SOURCES}
    if not POSES[pose].legwear:
        accepted["legwear"] = []
    current = dict(positive_parts(pose))
    patches: list[dict] = []
    applied: dict[str, list[str]] = {}

    action = _part_patch("action", current["action"], "")
    if action:
        patches.append(action)
    for part, tags in accepted.items():
        if not tags:
            continue
        text = "".join(f"{prompt_form(part, tag)}, " for tag in tags)
        patch = _part_patch(part, current[part], text)
        if patch:
            patches.append(patch)
        applied[part] = tags

    used = {tag for tags in accepted.values() for tag in tags}
    if used & SMILES:
        patches += _remove_items("prompt.positive.eye_quality",
                                 current["eye_quality"], SMILE_CONFLICTS)
    patches += _remove_items("prompt.negative", negative(pose), used)
    dropped = sorted(set(kept) - used)
    return Plan(pose=pose, applied=applied, patches=patches, dropped=dropped)


def plan(scores: Mapping[str, float], kept: Collection[str],
         groups: Mapping[str, Sequence[str]],
         verdicts: Mapping[str, float]) -> Plan:
    accepted = extract(scores, kept, groups, verdicts)
    pose = choose_pose(accepted["framing_tags"], accepted["gesture"])
    return build(pose, accepted, kept)


def candidate_rows(scores: Mapping[str, float], kept: Collection[str],
                   groups: Mapping[str, Sequence[str]],
                   verdicts: Mapping[str, float], proposal: Plan) -> list[dict]:
    """Every candidate worth showing: scored at least `MIN_LISTED_SCORE`,
    kept, or applied; `d1` is None for a tag d1 was not asked about."""
    rows = []
    for tag, part in candidates(groups).items():
        score = scores.get(tag, 0.0)
        applied = tag in proposal.applied.get(part, ())
        if score < MIN_LISTED_SCORE and tag not in kept and not applied:
            continue
        rows.append({"tag": tag, "part": part, "score": round(score, 4),
                     "kept": tag in kept,
                     "d1": round(verdicts[tag], 4) if tag in verdicts else None,
                     "applied": applied})
    order = list(PART_SOURCES)
    rows.sort(key=lambda row: (order.index(row["part"]), -row["score"], row["tag"]))
    return rows


def check_selection(rows: Sequence[Mapping], pose: str,
                    applied: Mapping[str, Sequence[str]]) -> None:
    """A render may only pick from its reference's candidates, each under the
    part it was listed with."""
    if pose not in POSE_CHOICES:
        raise ValueError(f"pose must be one of {POSE_CHOICES}, got {pose!r}")
    listed = {(row.get("part"), row.get("tag")) for row in rows}
    stray = sorted(f"{part}:{tag}" for part, tags in applied.items()
                   for tag in tags if (part, tag) not in listed)
    if stray:
        raise ValueError(f"not candidates of the reference: {', '.join(stray)}")
