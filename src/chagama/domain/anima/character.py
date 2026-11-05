"""Who is drawn: every character-specific value the Anima recipe assembles,
so one style serves any character that fills a `Character`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .legwear import OFF_LEGWEAR, REMOVING_LEGWEAR, LegwearState


@dataclass(frozen=True)
class Wardrobe:
    costumes: Mapping[str, str]
    # A costume's tags when the request's angle shows her back.
    behind: Mapping[str, str]
    behind_ban: Mapping[str, str]
    # Negative tags a costume adds, in this order, after the character's bans.
    garment_ban: Mapping[str, str]
    costume_ban: Mapping[str, str]
    # Costumes whose garments include a hood or cardigan skip `hood_ban`.
    hooded: frozenset[str]
    hood_ban: str
    # The `opaque` legwear: each costume's own tights.
    opaque: Mapping[str, str]
    # The `sheer` and `sheer-gloss` legwear: one block for every costume.
    sheer: str
    sheer_gloss: str
    sheer_tone_ban: str

    def costume_text(self, costume: str, behind: bool) -> str:
        if behind:
            return self.behind.get(costume, self.costumes[costume])
        return self.costumes[costume]

    def legwear_block(self, costume: str, legwear: str) -> str:
        if legwear == "sheer-gloss":
            return self.sheer_gloss
        if legwear == "sheer":
            return self.sheer
        if legwear != "opaque":
            raise KeyError(legwear)
        return self.opaque[costume]

    def legwear_text(self, costume: str, legwear: str,
                     state: LegwearState) -> str:
        if state is LegwearState.OFF:
            return OFF_LEGWEAR
        text = self.legwear_block(costume, legwear)
        if state is LegwearState.REMOVING:
            return text + REMOVING_LEGWEAR
        return text

    def negative(self, costume: str, behind: bool) -> str:
        return ((self.hood_ban if costume not in self.hooded else "")
                + self.garment_ban.get(costume, "")
                + self.costume_ban.get(costume, "")
                + (self.behind_ban.get(costume, "") if behind else ""))


@dataclass(frozen=True)
class Character:
    tag: str
    series: str
    # Nothing overrides it but a pose's own; the identity guard checks it.
    identity: str
    # Bare-tag form (no weight/parens) -- must match what `prompt_tags`
    # strips tags down to.
    identity_tag_names: frozenset[str]
    costume: str
    eye_shape: str
    nails: str
    body: str
    face: str
    wardrobe: Wardrobe
    # Negative tags in the order `Recipe.negative` places them: `line_ban`
    # after the style's line ban, `eye_ban` and `palette_ban` after its tail,
    # `tail_ban` last of the fixed bans.
    line_ban: str
    eye_ban: str
    palette_ban: str
    tail_ban: str
    # The character alone, for a reference render: no pose, framing, angle
    # or expression of its own. Never listed as a look.
    persona_pose: str
