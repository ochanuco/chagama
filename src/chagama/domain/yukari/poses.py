"""Every yukari pose: its own tag blocks plus the expression/costume it
defaults to, on top of the style and `YUKARI`."""

from __future__ import annotations

from dataclasses import replace

from ..anima.components import Framing
from ..anima.pose import Pose
from .character import YUKARI
from .prompt_style import FACE, HAIR_ORNAMENT_BAN, NO_HAIR_ORNAMENT


POSES = {
    "cinema": Pose(
        action=("(walking:1.2), (holding popcorn:1.45), (popcorn:1.4), "
                "(popcorn bucket:1.3), (holding cup:1.35), "
                "(disposable cup:1.3), (drinking straw:1.25), (cola:1.15), "
                "(holding food:1.1), (holding with both hands:1.1), "),
        mood="(excited:1.1), ",
        gesture=("(looking at viewer:1.2), (sneakers:1.3), "
                 "(white sneakers:1.2), "),
        framing=Framing.COWBOY,
        leg_display="(thighs:1.15), ",
        expression="doya", costume="standard",
        negative=("(sitting:1.3), (eating:1.3), (theater seat:1.1), "
                  "(mug:1.2), (glass:1.2), (bottle:1.2), (bag:1.1), "
                  "(multiple girls:1.3), "),
        ),
    "coffee": Pose(
        action=("(drinking:1.3), (iced coffee:1.4), (plastic cup:1.45), "
                "(clear cup:1.2), (drinking straw:1.4), (holding cup:1.35), "
                "(straw in mouth:1.25), (standing:1.2), "),
        mood="",
        gesture="(looking at viewer:1.1), ",
        framing=Framing.COWBOY,
        leg_display="(thighs:1.2), ",
        expression="resting", costume="standard",
        negative="(mug:1.3), (paper cup:1.2), (hot coffee:1.2), (steam:1.3), "),
    "amae": Pose(
        action="(standing:1.2), ",
        mood="(pleading:1.15), ",
        gesture=("(head tilt:1.2), (leaning forward:1.3), "
                 "(looking at viewer:1.3), (own hands clasped:1.25), "
                 "(hands up:1.1), "),
        framing=Framing.COWBOY,
        leg_display="(thighs:1.2), ",
        expression="doya", costume="standard"),
    "step": Pose(
        action=("(walking:1.3), (stepping:1.3), (big step:1.35), "
                "(wide stride:1.3), (one leg forward:1.3), "
                "(straight leg:1.2), (leg lift:1.15), (arms spread:1.4), "
                "(outstretched arms:1.35), (balancing:1.35), (playful:1.15), "),
        mood="",
        gesture="(looking at viewer:1.1), (sneakers:1.3), (white sneakers:1.2), ",
        framing=Framing.FULL,
        angle="(from side:1.1), ",
        leg_display="(thighs:1.1), ",
        expression="resting", costume="standard",
        negative="(running:1.3), (jumping:1.25), (sitting:1.2), "),
    "stand": Pose(
        action=("(standing:1.5), (own hands together:1.3), (hands up:1.2), "
                "(arched back:1.15), "),
        mood="",
        gesture=("(looking at viewer:1.2), (sneakers:1.3), "
                 "(white sneakers:1.2), "),
        framing=Framing.FULL,
        angle="(from front:1.3), ",
        leg_display="(thighs:1.1), ",
        expression="doya", costume="standard",
        negative="(sitting:1.3), (cowboy shot:1.2), (upper body:1.2), "),
    "dance": Pose(
        action=("(standing:1.4), (dancing:1.4), (knock-kneed:1.3), "
                "(one arm up:1.2), (clenched hand:1.1), "),
        mood="",
        gesture=("(looking at viewer:1.2), (sneakers:1.3), "
                 "(white sneakers:1.2), "),
        framing=Framing.FULL,
        angle="(from front:1.3), ",
        leg_display="(thighs:1.1), ",
        expression="v", costume="standard",
        negative="(sitting:1.3), (cowboy shot:1.2), (upper body:1.2), "),
    "bust": Pose(
        action="",
        mood="",
        gesture="(looking at viewer:1.2), ",
        framing=Framing.BUST,
        angle="(from front:1.3), ",
        leg_display="",
        expression="jitome", costume="standard",
        negative=("(sitting:1.3), (wavy mouth:1.4), (:3:1.3), (pout:1.3), "
                  "(pursed lips:1.3), (puckered lips:1.2), (smile:1.2), "),
        legwear=False,
        body=("(mature female:1.3), (adult:1.2), adult proportions, "
              "(collarbone:1.2), ")),
    "gao": Pose(
        action="(claw pose:1.45), (hands up:1.25), ",
        mood="",
        gesture="(looking at viewer:1.2), ",
        framing=Framing.UPPER,
        angle="(from behind:1.2), ",
        angle_action="(looking back:1.3), (looking over shoulder:1.2), ",
        leg_display="",
        expression="gao", costume="standard",
        canvas=(1024, 1280),
        legwear=False,
        body=("(mature female:1.3), (adult:1.2), (narrow waist:1.25), "
              "adult proportions, long torso, seven heads tall, "),
        face=("(large eyes:1.15), (mature face:1.2), (tareme:1.2), "
              "(thick eyelashes:1.3), (long eyelashes:1.15), "),
        negative="(sitting:1.3), "),
    "anyo": Pose(
        action=("(sitting:1.4), (from below:1.3), (crossed legs:1.3), "
                "(foot out towards viewer:1.4), (pointed toes:1.2), "
                "(foot focus:1.1), seen from below, she sits with crossed "
                "legs and pokes her toes out at the viewer, "),
        mood=("(condescending:1.4), (disdain:1.3), (contempt:1.2), "
              "(chin up:1.2), "),
        gesture=("(looking down at viewer:1.35), (looking at viewer:1.2), "
                 "(no shoes:1.2), (legwear feet:1.1), "),
        framing=Framing.FULL,
        angle="(from front:1.2), (pov:1.2), ",
        angle_action="(foreshortening:1.35), ",
        leg_display="(thighs:1.2), ",
        expression="disdain", costume="standard",
        negative=("(upper body:1.2), (chair:1.4), (furniture:1.3), "
                  "(stool:1.2), (bench:1.2), (smile:1.3), (:3:1.3), "
                  "(smug:1.2), (blush:1.2), (sweatdrop:1.2), ")),
    "pillow": Pose(
        action=("(lying on back:1.5), (lying:1.3), (on back:1.3), "
                "(knees up:1.3), (holding phone above face:1.3), "
                "(hugging a long cushion:1.4), (dark slate blue cushion:1.35), "
                "(solid color fabric:1.2), (seamless velvet bolster:1.2), "
                "(bedding:1.1), (legs wrapped around cushion:1.25), "
                "(cushion between legs:1.2), "),
        mood="",
        gesture=("(looking at phone:1.4), (looking up:1.1), (no shoes:1.2), "
                 "(legwear feet:1.1), "),
        framing=Framing.CLOSE,
        angle="(from above:1.35), (dutch angle:1.3), ",
        leg_display="(thighs:1.1), ",
        expression="dubious", costume="standard",
        canvas=(896, 1440),
        negative="(sitting:1.3), (cowboy shot:1.2), (upper body:1.2), ",
        negative_tail=("(standing:1.4), (sneakers:1.2), (panties:1.3), "
                       "(pantyshot:1.3), (green pillow:1.4), (smile:1.2), "
                       "(looking at viewer:1.2), (eyes on pillow:1.45), "
                       "(face on cushion:1.45), (stuffed toy:1.4), "
                       "(stuffed animal:1.35), (plushie:1.35), (mascot:1.35), "
                       "(creature:1.3), (character print:1.35), "
                       "(dakimakura:1.3)")),
    "avatar": Pose(
        action="",
        mood="",
        gesture="(looking at viewer:1.2), ",
        framing=Framing.FACE,
        angle="(from front:1.2), ",
        leg_display="",
        expression="neutral", costume="standard",
        canvas=(384, 384),
        negative=("(upper body:1.3), (cowboy shot:1.3), (multiple views:1.3), "
                  "(inset:1.3), "),
        legwear=False,
        body="(mature female:1.3), (adult:1.2), ",
        face=("(large eyes:1.15), (mature face:1.2), (tareme:1.2), "
              "(thick eyelashes:1.3), (long eyelashes:1.15), ")),
}

# Lowering the eyelash weights does nothing; only removing them and banning
# eyeliner reads as bare. The double eyelid does not move by any word.
_NOMAKEUP = replace(
    POSES["bust"],
    costume="roomwear",
    identity=YUKARI.identity.replace("hair ornament, ", NO_HAIR_ORNAMENT),
    nails="(natural nails:1.2), ",
    face=FACE.replace(
        "(thick eyelashes:1.3), (long eyelashes:1.15), ",
        "(no makeup:1.4), (bare face:1.3), (natural lips:1.2), "
        "(beautiful face:1.2), (pale skin:1.2), (pale lips:1.15), "),
    negative=(POSES["bust"].negative
              + "(lipstick:1.3), (eyeshadow:1.3), (makeup:1.3), (blush:1.1), "
              "(thick eyelashes:1.2), (eyeliner:1.2), " + HAIR_ORNAMENT_BAN))
POSES["nomakeup"] = _NOMAKEUP
POSES["nomakeup-bangs-up"] = replace(
    _NOMAKEUP,
    identity=(_NOMAKEUP.identity + "(parted bangs:1.3), "
              "(bangs pinned back:1.2), (bobby pin:1.2), forehead, "))

PERSONA_POSE = YUKARI.persona_pose


POSE_LABELS = {
    "amae": "おねだり",
    "anyo": "あんよ",
    "avatar": "アバター",
    "bust": "バストアップ",
    "cinema": "映画館",
    "coffee": "アイスコーヒー",
    "dance": "ダンス",
    "gao": "がおー",
    "nomakeup": "すっぴん",
    "nomakeup-bangs-up": "すっぴん前髪上げ",
    "pillow": "抱き枕",
    "stand": "立ち",
    "step": "大股",
}
