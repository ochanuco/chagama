"""Prompt edits for the repair redraw."""

from __future__ import annotations

from collections.abc import Sequence

REPAIR_DROP_WORDS = (
    "eyes", "eye ", "mouth", "hair", "sidelocks", "bangs", "smile", "smirk",
    "grin", "sigh", "annoyed", "unamused", "tareme", "jitome", "expression",
    "looking at", "head tilt", "blush", "tongue", "wink", "frown", "pout",
    "full body", "from front", "from above", "from side", "from behind", "hood",
)

PART_TAGS = {
    "feet": "(feet:1.2), (soles:1.2), (foot focus:1.2)",
    "hands": "(hands:1.2), (fingers:1.1)",
}


def _bare_tag(tag: str) -> str:
    """A tag's name with weight syntax unwrapped and whitespace stripped."""
    text = tag.strip()
    if text.startswith("(") and text.endswith(")"):
        inner = text[1:-1]
        text = inner.rsplit(":", 1)[0] if ":" in inner else inner
    return text.strip()


def repair_prompt(positive: str, parts: Sequence[str]) -> str:
    """`positive` with face/hair/framing tags dropped and part tags appended."""
    tags = [tag.strip() for tag in positive.split(",")]
    bare_tags = [_bare_tag(tag) for tag in tags]
    kept = [tag for tag, bare in zip(tags, bare_tags)
            if not any(word in bare for word in REPAIR_DROP_WORDS)]
    additions = [PART_TAGS[part] for part in parts if part in PART_TAGS]
    if "feet" in parts and any("pantyhose" in bare for bare in bare_tags):
        additions.append("(pantyhose feet:1.3)")
    return ", ".join(kept + additions)


def masked_redraw_prompt(positive: str, prompt_patch: str) -> str:
    """`positive` with face/hair/framing tags dropped (same as
    `repair_prompt`) and `prompt_patch` appended."""
    tags = [tag.strip() for tag in positive.split(",")]
    bare_tags = [_bare_tag(tag) for tag in tags]
    kept = [tag for tag, bare in zip(tags, bare_tags)
            if not any(word in bare for word in REPAIR_DROP_WORDS)]
    return ", ".join(kept + [prompt_patch])
