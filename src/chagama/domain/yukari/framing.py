"""Camera framing: the tag text and canvas each `Framing` kind carries.

`recipe._components` builds the `framing_tags` component as
`pose.angle + pose.angle_action + FRAMING[pose.framing].text`, with a
request's `angle` and `framing` in place of the pose's; `render_spec`
resolves canvas as `pose.canvas` if set and no `framing` was requested, else
`FRAMING[framing].canvas`, else the recipe default.
"""

from __future__ import annotations

from dataclasses import dataclass

from .components import Framing


@dataclass(frozen=True)
class FramingSpec:
    text: str
    canvas: tuple[int, int] | None = None
    # Leg words pull a closer shot back out to the legs.
    drops_legs: bool = False


FRAMING_LABELS = {
    Framing.FACE: "顔",
    Framing.BUST: "バスト",
    Framing.UPPER: "上半身",
    Framing.COWBOY: "膝上",
    Framing.FULL: "全身",
    Framing.LYING: "寝そべり",
    Framing.CLOSE: "寄り",
}

LEG_TAG_NAMES = frozenset({"thick thighs", "soft thighs", "long legs"})
LEG_BAN = "legs, thighs, pantyhose, "

FRAMING = {
    Framing.BUST: FramingSpec(
        text=("(portrait:1.5), (head and shoulders:1.4), (upper body:1.35), "
              "(face focus:1.3), "),
        canvas=(1280, 2048)),
    Framing.UPPER: FramingSpec(text="(upper body:1.3), "),
    Framing.COWBOY: FramingSpec(text="(cowboy shot:1.3), "),
    Framing.FULL: FramingSpec(text="(full body:1.45), (wide shot:1.3), "),
    Framing.LYING: FramingSpec(text="(lying:1.3), (full body:1.35), "),
    # `cropped` / `out of frame` draw a small inset figure; a tall canvas
    # does not hold the close shot.
    Framing.CLOSE: FramingSpec(text="(close-up:1.15), ", canvas=(1152, 1152),
                               drops_legs=True),
    Framing.FACE: FramingSpec(
        text="(portrait:1.5), (close-up:1.3), (face focus:1.45), "),
}
