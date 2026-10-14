"""Camera framing: the tag text and canvas each `Framing` kind carries.

`recipe._components` builds the `framing_tags` component as
`pose.angle + FRAMING[pose.framing].text`; `render_spec` resolves canvas as
`pose.canvas` if set, else `FRAMING[pose.framing].canvas`, else the recipe
default.
"""

from __future__ import annotations

from dataclasses import dataclass

from .components import Framing


@dataclass(frozen=True)
class FramingSpec:
    text: str
    canvas: tuple[int, int] | None = None


FRAMING = {
    Framing.BUST: FramingSpec(
        text=("(portrait:1.5), (head and shoulders:1.4), (upper body:1.35), "
              "(face focus:1.3), "),
        canvas=(1280, 2048)),
    Framing.UPPER: FramingSpec(text="(upper body:1.3), "),
    Framing.COWBOY: FramingSpec(text="(cowboy shot:1.3), "),
    Framing.FULL: FramingSpec(text="(full body:1.45), (wide shot:1.3), "),
    Framing.LYING: FramingSpec(text="(lying:1.3), (full body:1.35), "),
    Framing.CLOSE: FramingSpec(text="(close-up:1.15), (cropped:1.2), "),
}
