# delivery_style

`src/chagama/domain/yukari/delivery_style.py` holds what happens to a
render after the model: the delivery identity and the redraw settings.
Every tool reads its values from there, and `scripts/delivery_check.py`
fingerprints them. The values live in the code; the reasons behind them are
in [`findings/delivery.md`](../findings/delivery.md).

## Constant groups

| Group | Read by |
|---|---|
| `BACKDROP`, `OUTLINES`, `OUTLINES_MAX`, `OUTLINE_WIDTH_MAX_PCT` | band drawing in `infrastructure/imaging/delivery.py`, the `outlines` deliver option |
| `STROKE_CUT_EPS_PCT`, `STROKE_EDGE_SMOOTH` | `band_alphas`: hand-cut outline and edge rounding |
| `STROKE_LIGHT_*`, `STROKE_LIGHTS`, `STROKE_CHOICES`, `STICKER_SHADOW_*` | the `stroke_light` deliver option |
| `STRIPES_*`, `WAVEFORM_*`, `EARS_*`, `BACKDROP_*` | named backdrops in `infrastructure/imaging/backdrops.py` |
| `SAT_BAND`, `BG_SAT_MAX`, `FIGURE_SAT_*`, `BACKDROP_SPREAD_MAX` | `measure`/`verdict` in `palette.py`: the gate at ingest and in `palette_check.py` |
| `FIGURE_LIGHT_*`, `PALETTE_WINDOWS`, `REPIN_*` | `repin` in `infrastructure/imaging/palette.py`, applied by the `saturation` request through `infrastructure/imaging/saturation.py` |
| `SATURATION_STRENGTH` | the `saturation` request's `strength` and the catalog's top-level `saturation` section |

| `MATTE_*`, `KEY_*`, `ENCLOSED_*`, `FRAME_WINDOW_*`, `DRAWN_OUTLINE_*` | matte, keyed edge, pocket cuts and the drawn outline in `delivery.py` |
| `REDRAW_*`, `ROUGH_STYLE`, `DELIVER_DEFAULTS` | the opt-in IL `canvas` redraw and the deliver defaults the catalog publishes |
| `DOF_*` | the `dof` request's options and the catalog's top-level `dof` section |
| `PHOTO_*` | the `photo` request's placement and depth options and the catalog's top-level `photo` section |
| `LENS_*` | the `lens` request's options (strength, centre, shape, sphere radius) and the catalog's top-level `lens` section |
| `FRAME_*` | the `stage` request's `frame` option and the catalog's `stage.frame` block |

## Contracts

- The delivered backdrop is repainted after the render; the render's own
  backdrop is not stable enough to keep. `BACKDROP` (white) is the flat
  colour when none is named, and the catalog publishes it as
  `deliver.backdrop_color` for the WebUI's solid-colour choice.
- Band widths are shares of the longest side; each band starts at the
  previous band's outer edge.
- `STROKE_CUT_EPS_PCT = 0` reproduces the smooth ramp exactly, and
  `band_alphas` branches on it. With `stroke_light`, the outermost band is
  extruded away from the light first and then simplified. Inner bands are
  never shaded.
- The `stroke_light` drop shadow falls straight away from the light, the
  same direction the outermost band is extruded. `even` throws none, and
  neither does a delivery without bands.
- The `stroke_light` drop shadow darkens only `clean_background`'s backdrop.
  `transparent` never draws it.
- `STROKE_EDGE_SMOOTH` counts 2x-supersample pixels, not band widths.
- A palette gate pass is not an approval; a FAIL never goes forward.
  `FIGURE_SAT_*` measures only pixels at V ≥ `FIGURE_MIDTONE_V`, so black
  tights and coat are exempt.
- The repin factor is at most 1.0, so a pale render is never pushed up.
  Palette windows do not overlap. `REPIN_SAT_ONLY_WINDOWS` borders the
  purple window and is compressed on the same curve with its hue left alone,
  only below `REPIN_SAT_ONLY_V_MAX`; where they meet, a pixel is compressed
  once. The dark band greys every hue
  except `REPIN_DARK_EXEMPT`. Light skin (from `REPIN_SKIN_CAP_V`) is
  capped on `REPIN_SKIN_CAP`; shadowed skin is not.
- The matte is cut from the raw render, never from skin-pinned or recoloured colour.
- Only the outermost `KEY_EDGE_RING_PX` of the figure is soft. Despill runs
  on the rim only, and only when the corner key's dominant channel clears the
  others by `KEY_DESPILL_MIN_EXCESS`.
- `enclosed_cut` runs only on a green key. When the corners are not green (a
  drawn frame line), the key comes from the figure's own green pixels once
  there are `ENCLOSED_POCKET_MIN_AREA` of them. The backdrop is then painted
  only inside the frame, and the frame line stays figure.
- `drawn_outline` stays figure and is painted pure white at full coverage,
  so the white band continues it. It applies only when it covers
  `DRAWN_OUTLINE_MIN_EDGE` of the figure's edge; otherwise the delivery is
  unchanged. Its tag ends in `-outline`.

## Stage frame

A `stage` request's `frame` sets the delivery in a square panel:

- `box` and `canvas` are `[x0, y0, x1, y1]` shares of the delivery's canvas
  (x of its width, y of its height, each within `FRAME_COORD_RANGE`);
  widths are percent of its longest side.
- The canvas is painted `paper`, the box filled with the request's
  `backdrop` rendered at the box's size, and the `line` straddles the box's
  sides (its `pattern` `solid`, or `slash`: `FRAME_SLASH` pairs of "/" in
  the line colour on the paper colour, the same direction on every side)
  with the `rim` along both its edges (either null or width 0 for
  none, an omitted color or width the default); together they are the
  border.
- A `cut` side ends the figure and outline at the border's inner edge, the
  half-plane running past the box; an `out` side leaves them over the
  border; a `behind` side keeps them on both sides of it and takes them out
  under its strip (the side's border out to the corners), so the border
  lies over the figure.
- `strokes` are round pens applied in order: an `erase` multiplies what is
  kept by one minus its anti-aliased coverage, a `restore` adds its
  coverage back, never past the figure's own alpha.
- Pixels count by their centres, so cuts, line and box are hard-edged.
- Each `canvas` entry may be null (or `canvas` itself omitted); that side
  is the box out to the border's outer edge with the visible figure and
  outline, `FRAME_MARGIN_PCT` around, rounded outward, and kept within the
  delivery's canvas where `expand` (one boolean, or `{top, right, bottom,
  left}`, `FRAME_EXPAND` (false) where omitted) is false.
  The worker records the frame with that canvas and reports it in
  `resolved_options`.
- `dof` lays the source depth where the frame put the delivery's canvas.
