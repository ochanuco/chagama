"""The interpreter: the style, a character and its pose, costume and
expression records into a prompt pair."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from ..generation.models import HiresSpec, PromptPair, RenderSpec
from ..generation.prompt_lint import tags as prompt_tags
from .angles import angle_text, shows_back
from .character import Character, ExpressionLook, PoseLook
from .components import (PART_GROUPS, Component, Framing, Priority, Section,
                         assemble)
from .expression import BLANK, Expression
from .framing import FRAMING, LEG_BAN, LEG_TAG_NAMES
from .legwear import (OFF_LEGWEAR_BAN, REMOVING_LEGWEAR_BAN, SHEER_BAN,
                      SHEER_GLOSS_BAN, LegwearState)
from .pose import Pose
from .style import (ARTIST_TAG, BACKGROUND, CFG, COLORED_LINE_BAN, COUNT_TAG,
                    DETAIL_BAN, DIGIT_BAN, GARMENT_GLOSS_TAGS, GRADIENT_BAN,
                    GUIDED_STEPS, HEIGHT, HIRES_DENOISE, MODEL, NEGATIVE_TAIL,
                    PROPORTION_BAN, QUALITY_TAG, SAMPLER, SCHEDULER, SCORE_BAN,
                    SHINE_BAN, STEPS, STYLE, TEXT_BAN, THIN_BODY_BAN, WIDTH)

FRAMING_TAG_NAMES = frozenset(
    tag for spec in FRAMING.values() for tag in prompt_tags(spec.text))


def _legwear_state(pose: Pose, legwear_state: str | None) -> LegwearState:
    return (LegwearState(legwear_state) if legwear_state is not None
           else pose.legwear_state)


def _framing(pose: Pose, framing: str | None) -> Framing | None:
    return Framing(framing) if framing is not None else pose.framing


def _framing_text(framing: Framing | None) -> str:
    return FRAMING[framing].text if framing is not None else ""


def _own(override: str | None, layer: str) -> str:
    return override if override is not None else layer


def _over(override: str | None, layer: str | None) -> str | None:
    return override if override is not None else layer


def _drop_tags(text: str, names: frozenset[str]) -> str:
    if not names:
        return text
    return "".join(chunk + ", " for chunk in text.split(", ")
                   if chunk and not set(prompt_tags(chunk)) & names)


def _drops_legs(pose: Pose, requested: str | None) -> bool:
    # A pose that declares the framing itself has already chosen its leg words.
    if requested is None or Framing(requested) is pose.framing:
        return False
    return FRAMING[Framing(requested)].drops_legs


def _without_framing_tags(text: str) -> str:
    return ", ".join(chunk for chunk in text.split(", ")
                     if not set(prompt_tags(chunk)) & FRAMING_TAG_NAMES)


def _dress_pose(pose: Pose, look: PoseLook | None, costume: str) -> Pose:
    look = look or PoseLook()
    return replace(
        pose,
        costume=look.costume or pose.costume or costume,
        body=_over(look.body, pose.body),
        face=_over(look.face, pose.face),
        identity=_over(look.identity, pose.identity),
        negative=pose.negative + look.negative)


def _dress_expression(expression: Expression,
                      look: ExpressionLook | None) -> Expression:
    if look is None:
        return expression
    return replace(
        expression,
        eyes=_own(look.eyes, expression.eyes),
        eye_base=_over(look.eye_base, expression.eye_base),
        face_drop=(look.face_drop if look.face_drop is not None
                   else expression.face_drop))


@dataclass(frozen=True)
class Recipe:
    character: Character
    poses: Mapping[str, Pose]
    expressions: Mapping[str, Expression]

    @classmethod
    def drawing(cls, character: Character, poses: Mapping[str, Pose],
                expressions: Mapping[str, Expression]) -> Recipe:
        """The shared poses and expressions with the character's looks laid over them."""
        return cls(
            character,
            {name: _dress_pose(pose, character.pose_looks.get(name),
                               character.costume)
             for name, pose in poses.items()},
            {name: _dress_expression(expression,
                                     character.expression_looks.get(name))
             for name, expression in expressions.items()})

    @property
    def persona(self) -> Pose:
        return Pose(action="", mood="", gesture="", framing=None,
                    leg_display="", expression=None,
                    costume=self.character.costume)

    def pose(self, name: str) -> Pose:
        if name == self.character.persona_pose:
            return self.persona
        return self.poses[name]

    def _expression(self, pose: Pose, expression: str | None) -> Expression:
        name = expression if expression is not None else pose.expression
        return self.expressions[name] if name is not None else BLANK

    def components(self, pose: str, costume: str | None = None,
                   expression: str | None = None,
                   legwear: str | None = None,
                   legwear_state: str | None = None,
                   angle: str | list[str] | None = None,
                   framing: str | None = None) -> tuple[Component, ...]:
        ch = self.character
        p = self.pose(pose)
        e = self._expression(p, expression)
        c = costume if costume is not None else p.costume
        lw = legwear if legwear is not None else p.legwear_kind
        ls = _legwear_state(p, legwear_state)
        fr = _framing(p, framing)
        no_legs = _drops_legs(p, framing)
        legwear_positive = (ch.wardrobe.legwear_text(c, lw, ls)
                            if p.legwear and not no_legs else "")
        body = _own(p.body, ch.body)
        G, M = Section.GENERAL, Priority.MAIN
        L, T = Priority.LEAD, Priority.TAIL
        declared = (
            Component("quality", Section.QUALITY, M, QUALITY_TAG),
            Component("count", Section.COUNT, M, COUNT_TAG),
            Component("character", Section.CHARACTER, M, ch.tag),
            Component("series", Section.SERIES, M, ch.series),
            Component("artist", Section.ARTIST, M, ARTIST_TAG),
            Component("identity", G, L, _own(p.identity, ch.identity)),
            Component("action", G, M, p.action),
            Component("mouth", G, M, e.mouth),
            Component("mood", G, M, p.mood),
            Component("eye_base", G, L, _own(e.eye_base, ch.eye_shape)),
            Component("eye_quality", G, L,
                      ch.eye_quality[e.eye_quality] + e.eyes),
            Component("gesture", G, M, p.gesture),
            Component("nails", G, M, _own(p.nails, ch.nails)),
            Component("costume", G, M,
                      ch.wardrobe.costume_text(c, shows_back(angle))),
            Component("legwear", G, M, legwear_positive),
            Component("framing_tags", G, L,
                      (angle_text(angle) if angle is not None else p.angle)
                      + p.angle_action + _framing_text(fr)),
            Component("leg_display", G, L, "" if no_legs else p.leg_display),
            Component("body_build", G, L,
                      _drop_tags(body, LEG_TAG_NAMES) if no_legs else body),
            Component("cutout", G, M, _own(p.background, BACKGROUND)),
            Component("face", G, T,
                      _drop_tags(_own(p.face, ch.face), e.face_drop)),
            Component("style", G, T, _own(p.style, STYLE) + ch.style),
        )
        return assemble(declared)

    def positive_parts(self, pose: str, costume: str | None = None,
                       expression: str | None = None,
                       legwear: str | None = None,
                       legwear_state: str | None = None,
                       angle: str | list[str] | None = None,
                       framing: str | None = None
                       ) -> tuple[tuple[str, str], ...]:
        components = self.components(pose, costume, expression, legwear,
                                     legwear_state, angle, framing)
        return tuple((c.name, c.text) for c in components)

    def positive(self, pose: str, costume: str | None = None,
                 expression: str | None = None,
                 legwear: str | None = None,
                 legwear_state: str | None = None,
                 angle: str | list[str] | None = None,
                 framing: str | None = None) -> str:
        return "".join(
            text for _, text in
            self.positive_parts(pose, costume, expression, legwear,
                                legwear_state, angle, framing))

    def identity_tags(self, pose: str, costume: str | None = None,
                      expression: str | None = None) -> frozenset[str]:
        bare = set(prompt_tags(self.positive(pose, costume, expression)))
        return frozenset(self.character.identity_tag_names & bare)

    def negative(self, pose: str, costume: str | None = None,
                 expression: str | None = None,
                 legwear: str | None = None,
                 legwear_state: str | None = None,
                 framing: str | None = None,
                 angle: str | list[str] | None = None) -> str:
        ch = self.character
        p = self.pose(pose)
        if framing is not None and _framing(p, framing) is not p.framing:
            p = replace(p, negative=_without_framing_tags(p.negative),
                        negative_tail=_without_framing_tags(p.negative_tail))
        _ = self._expression(p, expression)
        c = costume if costume is not None else p.costume
        lw = legwear if legwear is not None else p.legwear_kind
        ls = _legwear_state(p, legwear_state)
        _ = ch.wardrobe.costumes[c]
        _ = ch.wardrobe.legwear_block(c, lw)
        shine_ban, sheer_ban = SHINE_BAN, ""
        # A sheer/sheer-gloss kind only earns its own negative edits when the
        # legwear it names is actually worn.
        if lw in ("sheer-gloss", "sheer") and p.legwear and ls is not LegwearState.OFF:
            for tags in GARMENT_GLOSS_TAGS:
                shine_ban = shine_ban.replace(tags, "")
            sheer_ban = SHEER_BAN + (ch.wardrobe.sheer_tone_ban
                                     if lw == "sheer" else "")
        legwear_state_ban = ""
        if p.legwear:
            if ls is LegwearState.REMOVING:
                legwear_state_ban = REMOVING_LEGWEAR_BAN
            elif ls is LegwearState.OFF:
                legwear_state_ban = OFF_LEGWEAR_BAN
        result = (DIGIT_BAN + DETAIL_BAN + COLORED_LINE_BAN + ch.line_ban
                 + THIN_BODY_BAN + p.negative + shine_ban + GRADIENT_BAN
                 + NEGATIVE_TAIL + ch.eye_ban + ch.palette_ban
                 + ch.wardrobe.negative(c, shows_back(angle))
                 + sheer_ban + legwear_state_ban
                 + SCORE_BAN + PROPORTION_BAN + TEXT_BAN + ch.tail_ban)
        # `sheer-gloss` only, tacked on after TEXT_BAN so it is the last
        # thing in the negative prompt.
        if lw == "sheer-gloss" and p.legwear and ls is not LegwearState.OFF:
            result += SHEER_GLOSS_BAN
        if p.negative_tail:
            result += ", " + p.negative_tail
        if _drops_legs(p, framing):
            result += ", " + LEG_BAN.rstrip(", ")
        return result

    def render_spec(self, pose: str, seed: int, prefix: str, hires: int = 0,
                    denoise: float | None = None, costume: str | None = None,
                    expression: str | None = None,
                    legwear: str | None = None,
                    legwear_state: str | None = None,
                    angle: str | list[str] | None = None,
                    framing: str | None = None) -> RenderSpec:
        if not hires and denoise is not None:
            raise ValueError("denoise needs hires")
        p = self.pose(pose)
        pose_canvas = p.canvas if framing is None else None
        resolved = _framing(p, framing)
        width, height = (pose_canvas
                         or (FRAMING[resolved].canvas if resolved is not None else None)
                         or (WIDTH, HEIGHT))
        parts = self.positive_parts(pose, costume, expression, legwear,
                                    legwear_state, angle, framing)
        base_negative = self.negative(pose, costume, expression, legwear,
                                      legwear_state, framing, angle)
        hires_spec = None
        if hires:
            longest = max(width, height)
            hires_width = round(hires * width / longest / 8) * 8
            hires_height = round(hires * height / longest / 8) * 8
            if hires_width < 8 or hires_height < 8:
                raise ValueError(
                    "hires dimensions must both be at least 8 pixels, got "
                    f"{hires_width}x{hires_height}")
            hires_spec = HiresSpec(
                width=hires_width,
                height=hires_height,
                denoise=HIRES_DENOISE if denoise is None else denoise,
                positive=None,
                negative=base_negative,
            )
        return RenderSpec(
            model_path=MODEL,
            prompts=PromptPair("".join(text for _, text in parts), base_negative),
            positive_parts=parts,
            part_groups=PART_GROUPS,
            width=width, height=height, seed=seed, steps=STEPS, cfg=CFG,
            sampler_name=SAMPLER, scheduler=SCHEDULER, denoise=1.0,
            filename_prefix=prefix, hires=hires_spec, loras=p.loras,
            guided_steps=GUIDED_STEPS)
