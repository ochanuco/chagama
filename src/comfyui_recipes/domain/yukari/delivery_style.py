"""Yukari's delivery policy: what happens to a render after the model.

The prompt cannot hold these values (the backdrop is not prompt-stable, the
purple marker is a post-process), so the delivery tools apply them instead
and every tool reads its default from here -- one source for what a
delivered picture wears. `delivery_check.py` fingerprints this file, so a
change here changes every future delivery.

Calibration measurements behind each number are in
docs/yukari/delivery_style.md.
"""

from typing import NamedTuple

# The backdrop every delivered picture is repainted to.
BACKDROP = "#ffffff"

# The bands framing the figure, innermost first: each a colour and a width
# as a share of the longest side. The purple marker, outermost, is
# deliberately darker than the hair accent.
OUTLINES = ({"color": "#ffffff", "width": 0.8},
            {"color": "#885b80", "width": 1.04})
OUTLINES_MAX = 6
OUTLINE_WIDTH_MAX_PCT = 5

# Douglas-Peucker epsilon for the band outlines' outer edge, as a percent of
# the longest side, for a hand-cut angular look instead of the smooth
# distance-field ramp. 0 reproduces the smooth ramp exactly; band_alphas
# branches on this.
STROKE_CUT_EPS_PCT = 0.5

# Outermost band width on the side facing the light / length of the
# extrusion away from it, in its uniform widths.
STROKE_LIGHT_THIN = 1.0
STROKE_LIGHT_THICK = 2.8
# Sigma of the distance-field blur the outline normal is read from, in
# outermost band widths -- the smooth-ramp path (STROKE_CUT_EPS_PCT 0) only; the hand-cut
# extrusion sweeps a uniform band instead and does not use it.
STROKE_LIGHT_SMOOTH = 1.0

# Drop shadow throw, in outermost band widths, straight away from the light
# -- the same direction that band is extruded.
STICKER_SHADOW_OFFSET = 1.6
# How much the shadow darkens the backdrop under it, 0..1.
STICKER_SHADOW_DARKEN = 0.28
# Sigma of the distance-field blur each band's edge ramps from, in 2x
# supersample pixels -- fixed, not a share of band width: the staircase
# being rounded off is always one source pixel high regardless of band
# width, so scaling this with band width would over- or under-blur it.
STROKE_EDGE_SMOOTH = 1.0
# Unit vectors toward the light, image coordinates (x right, y down).
_R = 2 ** -0.5
STROKE_LIGHTS = {
    "n": (0.0, -1.0), "ne": (_R, -_R), "e": (1.0, 0.0), "se": (_R, _R),
    "s": (0.0, 1.0), "sw": (-_R, _R), "w": (-1.0, 0.0), "nw": (-_R, -_R),
}
# The `stroke_light` value without a light: a uniform outermost band (null
# is accepted as "even").
STROKE_EVEN = "even"
STROKE_CHOICES = (STROKE_EVEN, *STROKE_LIGHTS)

# `stripes` backdrop: diagonal bands in the hair's lavender with a white
# radial burst. Lengths are shares of the longest side.
STRIPES_BASE = "#d9c6ee"
STRIPES_PITCH = 0.12
STRIPES_CONTRAST = 0.45
STRIPES_BURST_RAYS = 16
STRIPES_BURST = 0.6
STRIPES_BURST_CENTER = (0.5, 0.55)
STRIPES_BURST_REACH = 0.75

# BACKDROP_PINK: ear/sticker/ornament accent. `deep` (pencil/paper-cut
# motifs) is lavender pulled toward the purple stroke rather than a fourth
# free-standing hex, so it tracks either colour if they move.
BACKDROP_PINK = "#f4c9dc"
BACKDROP_DEEP_LAVENDER_SHARE = 0.82
BACKDROP_DEEP_STROKE_SHARE = 0.18

# `waveform`: VOICEROID-style voice-meter bars, loud behind the figure and
# quiet at the edges. Lengths are shares of the longest side.
WAVEFORM_ROW_HEIGHT = 0.13
WAVEFORM_PITCH = 0.018
WAVEFORM_BAR = 0.009
WAVEFORM_CENTER_SPREAD = 0.35
WAVEFORM_AMP_MIN = 0.008
WAVEFORM_AMP_GAIN = 0.052
WAVEFORM_EDGE_SOFT = 0.0015

# `ears`: rabbit-hood ears on a staggered lattice, pink inner ear, alternating
# lean. Lengths are shares of the longest side.
EARS_CELL = 0.13
EARS_LEAN = 0.18
EARS_TILT = 0.1
EARS_OFFSET_X = 0.011
EARS_OFFSET_Y = 0.004
EARS_OUTER_RX = 0.0085
EARS_OUTER_RY = 0.04
EARS_INNER_RX = 0.0042
EARS_INNER_RY = 0.028
EARS_INNER_OFFSET_Y = 0.006
EARS_INNER_ALPHA = 0.55
EARS_EDGE_SOFT = 0.0015

# `phases`: rows of moon phases waxing and waning across the canvas, every
# disc outlined.
PHASES_CELL_X = 0.105
PHASES_CELL_Y = 0.125
PHASES_RADIUS = 0.03
PHASES_RING_WIDTH = 0.0016
PHASES_RING_ALPHA = 0.45
PHASES_EDGE_SOFT = 0.0015

# `hatching`: two diagonal pencil layers, noised for a hand-drawn wobble and
# broken into dashed patches rather than solid lines.
HATCHING_LAYERS = (
    {"direction": (1, -1), "colour": "lavender", "alpha": 0.9, "threshold": 0.42},
    {"direction": (1, 1), "colour": "deep", "alpha": 0.55, "threshold": 0.6},
)
HATCHING_PITCH = 0.011
HATCHING_LINE_HALF_WIDTH = 0.0024
HATCHING_LINE_EDGE_SOFT = 0.0012
HATCHING_WOBBLE_CELLS = 40
HATCHING_WOBBLE_SEED = 10
HATCHING_WOBBLE_AMPLITUDE = 0.006
HATCHING_DASH_FREQ = 90
HATCHING_DASH_STROKE_FREQ = 2.7
HATCHING_DASH_BIAS = 0.55
HATCHING_DASH_EDGE_SOFT = 0.3
HATCHING_PATCH_CELLS = 6
HATCHING_PATCH_SEED = 20
HATCHING_PATCH_EDGE_SOFT = 0.04

# `torn`: hand-cut paper layers behind the figure, white paper edges like the
# delivery's own rim.
TORN_CENTER = (0.5, 0.55)
TORN_SEED = 7
TORN_LAYERS = (
    (1.05, "lavender"), (0.8, "light"), (0.6, "lavender"),
    (0.42, "light"), (0.26, "lavender"),
)
TORN_VERTICES = 13
TORN_JITTER = 0.16
TORN_PAPER_LIP = 0.008
TORN_EDGE_SOFT = 0.002

# `stickers`: scattered crescents, sparkles and hood ears, seeded and sparse.
STICKERS_SEED = 3
STICKERS_CELL = 0.1
STICKERS_JITTER = 0.03
STICKERS_SCALE_RANGE = (0.8, 1.2)
STICKERS_ANGLE_RANGE = (-0.6, 0.6)
STICKERS_PINK_CHANCE = 0.15
STICKERS_MOTIF_RADIUS_SHARE = 0.05
STICKERS_EDGE_SOFT = 0.0015
STICKERS_MOON_RADIUS = 0.022
STICKERS_MOON_BITE_SHARE = 0.9
STICKERS_MOON_BITE_OFFSET = (0.45, 0.2)
STICKERS_SPARKLE_RADIUS = 0.026
STICKERS_SPARKLE_SCALE = 0.08
STICKERS_SPARKLE_EDGE_SOFT = 0.0012
STICKERS_EAR_TILT = 0.1
STICKERS_EAR_OFFSET = 0.009
STICKERS_EAR_RX = 0.007
STICKERS_EAR_RY = 0.032

# `ornament`: Yukari's hair ornament, a ringed disc with a small satellite, as
# a lattice motif.
ORNAMENT_CELL = 0.15
ORNAMENT_RIM_RADIUS = 0.026
ORNAMENT_RIM_ALPHA = 0.9
ORNAMENT_RING_RADIUS = 0.018
ORNAMENT_CENTER_RADIUS = 0.012
ORNAMENT_HUB_ALPHA = 0.8
ORNAMENT_SATELLITE_OFFSET = (0.024, -0.022)
ORNAMENT_SATELLITE_HUB_RADIUS = 0.006
ORNAMENT_EDGE_SOFT = 0.0015

# `dots`: white polka dots on lavender, alternating row offset. Lengths are
# shares of the longest side.
DOTS_CELL = 0.11
DOTS_RADIUS_SHARE = 0.28  # share of DOTS_CELL
DOTS_EDGE_SOFT = 0.004

# `gingham`: two translucent lavender bands, darker where they cross.
GINGHAM_CELL = 0.10
GINGHAM_BAND_SHARE = 0.25  # share of GINGHAM_CELL
GINGHAM_BAND_ALPHA = 0.6
GINGHAM_EDGE_SOFT = 0.004

# `moons`: small white crescents on a staggered lattice, the stripes burst
# behind at reduced strength.
MOONS_CELL = 0.12
MOONS_RADIUS = 0.028
MOONS_HOLE_RADIUS = 0.026
MOONS_HOLE_OFFSET = (0.012, 0.006)
MOONS_BURST = 0.35

# `halftone`: white dots growing toward the lower-left, a faint white
# crescent behind. `HALFTONE_GRADIENT_SPAN` sets how far the growth runs
# before clamping.
HALFTONE_CELL = 0.045
HALFTONE_GRADIENT_SPAN = 1.3
HALFTONE_RADIUS_MIN = 0.08     # share of HALFTONE_CELL
HALFTONE_RADIUS_GROWTH = 0.34  # share of HALFTONE_CELL, added by the gradient
HALFTONE_EDGE_SOFT = 0.0015
HALFTONE_DOT_ALPHA = 0.75
HALFTONE_CRESCENT_CENTER = (0.62, 0.3)
HALFTONE_CRESCENT_RADIUS = 0.34
HALFTONE_CRESCENT_HOLE_OFFSET = (0.12, 0.06)
HALFTONE_CRESCENT_HOLE_RADIUS = 0.32
HALFTONE_CRESCENT_ALPHA = 0.55

# `sunburst`: wide alternating wedges, its own centre (matches the stripes
# burst by eye, not by dependency) and wedge count.
SUNBURST_CENTER = (0.5, 0.55)
SUNBURST_WEDGES = 12
SUNBURST_EDGE_SOFT = 0.01

# `chevron`: horizontal zigzag bands.
CHEVRON_PERIOD = 0.14
CHEVRON_AMPLITUDE = 0.035
CHEVRON_PITCH = 0.1
CHEVRON_EDGE_SOFT = 0.01

# `checker`: diagonal checkerboard, the stripes burst behind at reduced
# strength.
CHECKER_CELL = 0.09
CHECKER_BURST = 0.35

# The catalog thumbnail every backdrop publishes alongside its label.
BACKDROP_THUMBNAIL_WIDTH = 120
BACKDROP_THUMBNAIL_HEIGHT = 192

# Japanese labels for the chimera WebUI's backdrop picker, keyed by
# `backdrops.PATTERNS` name.
BACKDROP_LABELS = {
    "stripes": "斜めストライプ",
    "waveform": "音声波形",
    "ears": "うさ耳",
    "phases": "月の満ち欠け",
    "hatching": "鉛筆ハッチング",
    "torn": "切り紙",
    "stickers": "ステッカー",
    "ornament": "髪飾り",
    "dots": "水玉",
    "gingham": "ギンガムチェック",
    "moons": "三日月",
    "halftone": "網点と月",
    "sunburst": "放射",
    "chevron": "ジグザグ",
    "checker": "斜め市松",
}

# The acceptance band. A pass is not approval -- the human still judges --
# but a FAIL never goes forward. The frame mean tracks the figure's canvas
# share, so the upper bound only catches an explosion; FIGURE_SAT below is
# the composition-independent guard.
SAT_BAND = (30.0, 95.0)
BG_SAT_MAX = 60.0

# The figure's own saturation, measured over non-backdrop pixels at
# V >= FIGURE_MIDTONE_V -- the floor exempts black tights/coat (which sit
# below it) while still catching midtone saturation.
FIGURE_MIDTONE_V = 80
FIGURE_SAT_MEAN_MAX = 95.0
FIGURE_SAT_P90_MAX = 230.0

# Saturation normalization applied to the raw render (scripts/desat.py, HSV
# S alone) before the layered delivery: scale the figure's LIGHT band
# (pale dress/hair, V >= FIGURE_LIGHT_V) saturation by target/measured,
# clamped to at most 1.0 so a pale render is never pushed up.
FIGURE_LIGHT_V = 150
FIGURE_LIGHT_SAT_TARGET = 28.0

# Per-material colour targets, applied by scripts/repin.py: V stays
# untouched (the render's own), only hue and saturation move. Each window's
# saturation target is per V band (light >= FIGURE_LIGHT_V / mid below),
# blended continuously so no band boundary shows.
PALETTE_WINDOWS = (
    {"name": "purple", "hue": (170.0, 225.0), "hue_target": 191.0,
     "sat_light": 50.4, "sat_mid": 45.1},
    {"name": "skin", "hue": (0.0, 48.0), "hue_target": 17.8,
     "sat_light": 45.0, "sat_mid": 75.0},
    {"name": "cyan", "hue": (115.0, 140.0), "hue_target": 128.0,
     "sat_light": 50.4, "sat_mid": 45.1},
)

# The windows `repin` compresses and hue-eases, by name into PALETTE_WINDOWS.
REPIN_CHROMA_WINDOWS = ("purple", "cyan")
# Hue ranges `repin` compresses on the same curve without easing their hue,
# only below REPIN_SAT_ONLY_V_MAX (fading to zero over the last 20 levels).
REPIN_SAT_ONLY_WINDOWS = ((225.0, 255.0),)
REPIN_SAT_ONLY_V_MAX = 210.0
# Light skin (skin window and REPIN_SAT_ONLY_WINDOWS, V from
# REPIN_SKIN_CAP_V up, full 20 levels later) is compressed on this
# (knee, ratio); accents keep ACCENT_KEEP of their excess as elsewhere.
REPIN_SKIN_CAP = (75.0, 0.3)
REPIN_SKIN_CAP_V = 190.0
# The window `repin_skin_png` / `skin_mask` read skin from.
REPIN_SKIN_WINDOW = "skin"

# Skin region must be read off the render the redraw was made from: the
# 2048 redraw re-decides skin and lands it in the purple window instead.
# The pin only ever raises saturation -- lowering it washed the lips out
# with the cheek.
SKIN_SOURCE_S_MIN = 20.0
# Lips are warm and sit inside the skin window; without this ceiling the
# hue pin turns them orange.
SKIN_SOURCE_S_MAX = 60.0
SKIN_SOURCE_V_MIN = 110.0
SKIN_PIN_BLEND = 1.0
# Below this share, the base drew the hair's lavender over the face, not
# skin. The mask fragments left are speckle; pinning them draws cream
# blotches on a lavender cheek instead of leaving the face alone.
SKIN_PIN_MIN_SHARE = 0.08
# One coherent field, not the warm grain along every line.
SKIN_PIN_MIN_AREA = 4096

# repin's compression curve, per V band: (knee, ratio). Saturation below
# the knee is untouched; only the excess is kept, at the ratio. The dark
# band applies to every hue except warm skin shadows (REPIN_WARM_EXEMPT).
REPIN_LIGHT = (28.0, 0.25)   # V >= FIGURE_LIGHT_V
REPIN_MID = (60.0, 0.15)     # midtones inside the purple window
REPIN_DARK = (29.0, 0.08)    # V < 80, any hue
REPIN_WARM_EXEMPT = (0.0, 48.0)
# Hue ranges the dark band leaves alone.
REPIN_DARK_EXEMPT = (REPIN_WARM_EXEMPT, (115.0, 140.0))

# Accents (iris, hair pins) sit far above any field's saturation, so a ramp
# over this S range lets them keep ACCENT_KEEP of their excess and their
# own hue while the fields pin pale. Without it, the eyes wash out to
# white.
ACCENT_RAMP = (150.0, 60.0)  # start, width
ACCENT_KEEP = 0.65
# Saturation alone doesn't identify an accent -- without a value floor, a
# saturated dark hoodie/tights would be read as an iris and preserved.
# Gated at FIGURE_MIDTONE_V so the dark band never claims accent
# protection.
ACCENT_VALUE_RAMP = (FIGURE_MIDTONE_V, 40.0)  # start, width

# Backdrop flatness screen, on the RAW render's corner brightness spread. A
# gradient backdrop starves the flood mask palette.py measures through, so
# every figure number it reports is against a backdrop leak. It screens the
# numbers only -- the cut-out comes from the matte regardless of what the
# backdrop does.
BACKDROP_SPREAD_MAX = 25.0

# The worker-side model that cuts the figure out; has to come from
# something other than colour, since repin moves the figure's colours into
# the backdrop's tolerance before the delivery ever sees them. A `rmbg:`
# name selects a ComfyUI-RMBG model (refinement_graph.RMBG_MATTE_PREFIX); a
# bare file name is a core LoadBackgroundRemovalModel checkpoint.
MATTE_MODEL = "rmbg:BiRefNet-general"
# The band either side of the matte's edge, as a share of the longest side,
# inside which a pixel is figure when it differs from the backdrop by more
# than MATTE_EDGE_TOLERANCE on any channel.
MATTE_EDGE_BAND_PCT = 0.6
MATTE_EDGE_TOLERANCE = 20
# The colour retrace is bounded by the matte model's own soft output: it may
# only add a pixel the model gave any coverage (> SUPPORT) and may not drop
# one the model was sure of (> CERTAIN). A shaded backdrop next to the figure
# fails the tolerance test, and the figure's own light passages pass it.
MATTE_SOFT_SUPPORT = 8
MATTE_SOFT_CERTAIN = 200
# A cast shadow the figure throws on the floor is cut off the silhouette
# where it reaches the outside: a grey pixel (channel spread under CHROMA)
# darker than the local backdrop by between the two DARK bounds. The
# figure's own blacks are darker than the far bound, and a pixel the matte
# model was certain of (> MATTE_SOFT_CERTAIN) is never cut.
SHADOW_CHROMA = 14
SHADOW_DARK_NEAR = 8
SHADOW_DARK_FAR = 90

# Backdrop the figure encloses is cut out of the silhouette even where the
# matte model was certain of it, but only on a green key: green has to be
# the raw backdrop's dominant channel, by at least this. Nothing on the
# figure is green, while its whites and pale hair sit inside
# MATTE_EDGE_TOLERANCE of a grey or light blue backdrop.
ENCLOSED_KEY_MIN_GREEN_EXCESS = 12
# When the corners are not a green key (a drawn frame line closes the green
# off from a white outside), the key is taken from the figure's own green
# pixels instead, once there are at least this many of them. Interior
# linework holds a few pixels within the excess by accident; a pocket the
# matte kept holds thousands.
ENCLOSED_POCKET_MIN_AREA = 256
# The matte model also punches holes through dark figure passages it reads
# as background, such as a cushion seen between the legs. On a green key, an
# enclosed hole in its soft output is figure when under this share of it is
# key-coloured; the gaps a loop of hair closes are nearly all key.
MATTE_HOLE_MIN_KEY_SHARE = 0.5
# A drawn frame the figure steps out of is delivered as a window: with under
# MAX_BORDER_GREEN of the canvas border key-coloured (a green screen is far
# above it), the key-coloured components touching the figure (within
# FIGURE_REACH_PX) and covering at least MIN_AREA_PCT percent of the canvas
# are the frame's inside. The backdrop, bands and shadow stay inside it;
# outside it the raw is kept, bar its own key green. The bands are drawn
# from the figure at least EDGE_PX inside the window, so a frame line or
# bezel the matte took as figure gets none.
FRAME_WINDOW_MAX_BORDER_GREEN = 0.5
FRAME_WINDOW_FIGURE_REACH_PX = 12
FRAME_WINDOW_MIN_AREA_PCT = 2
FRAME_WINDOW_EDGE_PX = 12

# On the keyed-edge path, a white outline the raw drew around the figure
# (`white outline` in the prompt) is painted pure white: a pixel
# within MAX_TINT of a blend of the key and white, or of the key darkened
# to no less than MIN_KEY_SHADE of itself, within DEPTH_BANDS edge
# bands of the matte's edge
# and reached from the outside through such pixels. It applies only when
# those pixels hold at least MIN_EDGE of the figure's edge; without a drawn
# outline only white props touch the edge, at 10% or less.
DRAWN_OUTLINE_MAX_TINT = 14
DRAWN_OUTLINE_DEPTH_BANDS = 2
DRAWN_OUTLINE_MIN_EDGE = 0.5
DRAWN_OUTLINE_MIN_KEY_SHADE = 0.75
DRAWN_OUTLINE_MAX_POCKET_BANDS = 16

# A raw backdrop counts as a chromatic key when its dominant channel
# exceeds the larger of the other two by at least this.
KEY_DESPILL_MIN_EXCESS = 12
# The figure's outermost ring, in pixels, whose coverage ramps by colour
# distance from the local backdrop instead of being 1.
KEY_EDGE_RING_PX = 1
# Coverage in that ring reaches 1 at this multiple of MATTE_EDGE_TOLERANCE
# of colour distance from the local backdrop.
KEY_EDGE_RAMP = 2.0

# The matting stage between the matte model and the delivery: the retraced
# figure, eroded and dilated by TRIMAP_PX, is the trimap's known figure and
# known backdrop, and the model solves the alpha of the band between.
MATTING_MODEL = "hustvl/vitmatte-base-composition-1k"
MATTING_REVISION = "bf486d01a7d9e3dbcc8400f7942835caf0eaf76e"
MATTING_TRIMAP_PX = 12
# The model runs on square tiles of TILE_PX around the unknown band,
# overlapping by OVERLAP_PX and feathered across it: its global attention
# over a whole 2048-px picture does not fit the worker's GPU.
MATTING_TILE_PX = 512
MATTING_TILE_OVERLAP_PX = 64

# Lineart-preserving recolour (infrastructure/imaging/recolor.py). Where
# repin only nudges the render's own saturation, recolor asserts a
# material's colour outright and can therefore fix value too -- a
# washed-out black that repin leaves alone by design.

# A line pixel is darker than its own neighbourhood, not merely dark, so a
# flat dark fill is never mistaken for linework; and darkness is measured
# on the brightest channel, so a magenta stroke doesn't qualify however
# dark it reads.
RECOLOR_LINE_MAX = 140
RECOLOR_LINE_RELIEF = 40
RECOLOR_LINE_WINDOW = 7

# Per-material HSV targets, PIL's 0-255 scale.
RECOLOR_TARGETS = {
    "hair":   (188, 15, 234),
    "hoodie": (221, 23, 64),
    "dress":  (185, 34, 209),
    "skin":   (16, 27, 249),
    "white":  (0, 0, 255),
}

# How much of a region's own value survives the repaint, against the target
# V, so the render's own shading still reads under the asserted colour.
RECOLOR_KEEP_V = {
    "hoodie": 0.8, "hair": 0.6, "dress": 0.6,
    "skin": 0.5, "white": 0.5, "tights": 0.45,
}

# The same for hue and saturation, and it's not optional: pinning a region
# to one hue and one saturation flattens the drawing -- this render's
# shading lives in hue/saturation, not value (purple strokes through the
# hair, the blush on the cheek). The target moves the material; the
# deviation around it stays the render's.
RECOLOR_KEEP_HS = 0.7

# Bounded, or magenta strokes drawn over hands/face -- not line, so they
# ride along inside the skin -- keep enough saturation/hue to read as hot
# pink or teal instead of skin. Anything outside these bounds is pulled
# onto its material's colour.
RECOLOR_S_CEILING = 50
RECOLOR_H_SPREAD = 20

# Legwear has no single target: painted from this gradient, interpolated
# per pixel row against height share so the purple comes up the leg the
# same way regardless of the labelled region's height -- neutral black
# through the knee, purple only near the foot.
RECOLOR_LEG_STOPS = (
    (0.72, (219, 23, 64)),
    (0.82, (0, 6, 32)),
    (0.90, (212, 69, 57)),
    (1.00, (229, 117, 129)),
)
# Below this height (as a share of the figure's own bounding box, not the
# canvas) a dark fill is legwear; above it, the same darkness is the hoodie.
RECOLOR_LEG_CY = 0.72
# Value alone can't find legs on a washed-out render, so a fill this large
# sitting below RECOLOR_LEG_CY is legwear whatever its value -- the area
# separates leg masses from hands/stray hair that fall as low but smaller.
RECOLOR_LEG_MIN_AREA = 0.05

# classify's remaining thresholds. A fill this saturated is an accent --
# the iris, a hair pin -- and keeps its own colour rather than a target.
# Outside this hue window a fill reads as skin regardless of saturation.
# Below RECOLOR_WHITE_S at high value it is a frill; below RECOLOR_HAIR_S
# it is hair; otherwise it is the dress. Below RECOLOR_DARK_V a fill is a
# dark garment, split into hoodie or tights by RECOLOR_LEG_CY above.
RECOLOR_ACCENT_S = 150
# Saturation alone keeps the wrong things: thin magenta strokes drawn as
# interior detail (finger creases, mouth, collarbone) are saturated enough
# to be held back as accents, shipping hot pink fingers. A real accent is
# thick as well as saturated -- the iris and hair pins survive this many
# erosions where a thin stroke does not, taking its surroundings' colour
# instead.
RECOLOR_ACCENT_ERODE = 3
RECOLOR_SKIN_HUE = (48, 240)
RECOLOR_WHITE_S = 8
RECOLOR_HAIR_S = 45
RECOLOR_DARK_V = 120

# What the canvas redraw runs at.
REDRAW_SIZE = 2560
REDRAW_DENOISE = 0.4
REDRAW_MODEL = "hassaku-il-v22"
REDRAW_SAMPLER = ("dpmpp_2m", "karras")
REDRAW_STEPS = 30
REDRAW_CFG = 5.0

DOF_F_NUMBER = {"min": 1.4, "max": 22, "default": 2.8,
                "stops": [1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.5, 4.0, 4.5,
                          5.0, 5.6, 6.3, 7.1, 8.0, 9.0, 10.0, 11.0, 13.0, 14.0,
                          16.0, 18.0, 20.0, 22.0]}
# The delivered layers a dof blurs, all of them unless the request says so.
DOF_SCOPE = {"figure": True, "outline": True, "backdrop": True}
DOF_VIEWFINDER = {"values": ["off", "on", "both"], "default": "off"}
# A lens request's strength: positive swells the frame around its centre (a
# fisheye), negative shrinks it and stretches the rim (the inverse fisheye).
LENS_STRENGTH = {"min": -1.0, "max": 1.0, "default": 0.7}
LENS_CENTER_DEFAULT = (0.5, 0.5)
# `fisheye` bends the whole frame; `sphere` bends only a disc of `radius`
# (a share of the shorter side) around the centre, like a spherize filter.
LENS_SHAPE = {"values": ["fisheye", "sphere"], "default": "fisheye"}
LENS_RADIUS = {"min": 0.05, "max": 1.0, "default": 0.3}
# A photo request stands a delivery's figure and outline, one flat card like
# an acrylic stand, in a photo: the middle of the card's bottom edge on
# `place`, its height a share of the photo's, its depth the photo's where it
# stands pulled toward the camera by `nearer`. The f-number is the dof's.
PHOTO_PLACE_DEFAULT = (0.5, 0.95)
# The base may sit below the frame (y > 1), cropping the card's feet so it
# can be shown larger.
PHOTO_PLACE_MAX_Y = 1.5
PHOTO_HEIGHT = {"min": 0.05, "max": 1.5, "default": 0.7}
PHOTO_NEARER = {"min": 0.0, "max": 1.0, "default": 0.0}
# How much farther the card's top is than its base (negative: nearer), so a
# focus on the face blurs the body below it.
PHOTO_TILT = {"min": -1.0, "max": 1.0, "default": 0.0}


class Light(NamedTuple):
    scene: str
    direction: str


LIGHT_DENOISE = 0.40
LIGHT_FROM_DEFAULT = "nw"
LIGHT_ELEVATION = 0.15
LIGHT_REACH = 0.12
LIGHT_GRADIENT = 0.9
LIGHT_SHADE = 1.0
LIGHT_RIM = 1.0
LIGHT_CAST_STRENGTH = 0.45
LIGHT_BACKDROP_BRIGHTEN = (1.12, 0.3)
LIGHT_DIRECTION_WORDS = {
    "n": "light from above", "s": "light from below", "e": "light from right",
    "w": "light from left", "ne": "light from upper right",
    "nw": "light from upper left", "se": "light from lower right",
    "sw": "light from lower left",
}
LIGHT_WORDS = ("(rim lighting:1.2), (side lighting:1.2), ({direction}:1.1), "
               "(dramatic lighting:1.1), (cel shading:1.1), shadow")
# colour: the light; tint: how far the lit side drifts to it; shadow: figure
# shadow multiplier; backdrop: (rgb, amount, value); cast: sticker shadow
# multiplier; skin_keep: how much skin resists shade and tint.
LIGHT_SCENES = {
    "sunset": {"color": (255, 168, 96), "tint": 0.55,
               "shadow": (0.42, 0.38, 0.70),
               "backdrop": ((255, 168, 96), 0.45, 1.0),
               "cast": (0.42, 0.33, 0.62),
               "words": "(sunset:1.1), (orange light:1.1), evening",
               "skin_keep": 0.0},
    "moon": {"color": (184, 204, 255), "tint": 0.5,
             "shadow": (0.33, 0.37, 0.62),
             "backdrop": ((150, 170, 235), 0.5, 0.72),
             "cast": (0.22, 0.25, 0.48),
             "words": "(moonlight:1.15), night, (blue lighting:1.0)",
             "skin_keep": 0.5},
}

DELIVER_DEFAULTS = {"repin": True, "stroke_light": "n", "backdrop": "dots",
                    "outlines": [dict(outline) for outline in OUTLINES]}

# Replaces `STYLE`'s flat/cel-shaded finish with a rough, unfinished line
# for the redraw's different checkpoint.
ROUGH_STYLE = ("(sketch:1.45), (rough sketch:1.4), rough lines, sketchy "
              "lines, pencil sketch, (unfinished:1.2), construction lines, "
              "(colored pencil (medium):1.2), (soft shading:1.1)")

ROUGH_BAN = ("(clean lineart:1.3), (smooth lines:1.2), (cel shading:1.2), "
            "(flat color:1.2), ")
PAINT_BAN = ("(brown legwear:1.5), (brown pantyhose:1.4), "
            "(detailed shading:1.5), (heavy shading:1.5), (impasto:1.45), "
            "(painterly:1.45), ")
