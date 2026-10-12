# Queueing

> Yuzuki Yukari belongs to her original creators and rights holders — see
> [Derivative work](../README.md#derivative-work) in the README.

Every render is a row on chimera's `requests` queue, posted through the MCP
(`derive_request`, `redraw_generation`, `deliver_generation`,
`dof_generation`, `lens_generation`, `repair_generation`, `masked_redraw_generation`,
`create_request`) or `POST /api/v1/requests`.
`comfy-recipes work` on the GPU box claims a row, runs it, and ingests the
result back into chimera. Nothing else submits to ComfyUI.

## Where the details are

This page only covers rules that cannot be read from one place in the code.
Options, defaults and ranges live here:

| Question | Source |
|---|---|
| The wire contract (claim, heartbeat, done/failed) | chimera `docs/worker-protocol.md` |
| Every option of a kind, its default and range | `uv run comfy-recipes <generate\|work\|redraw\|deliver\|dof\|lens\|repair\|masked_redraw> --help`, `application/request_options.py` |
| Poses, costumes, parts, patch targets, dials, redraw, deliver, dof and lens defaults | `list_catalog` / `get_catalog_pose` on the MCP, or `uv run comfy-recipes catalog` |
| Patch targets and their ops | `domain/generation/patches.py` |
| What a request would render, without rendering | `uv run comfy-recipes generate --request r.json --dry-run` |

## Kinds

| Kind | Payload | Does |
|---|---|---|
| `generate` | a request.json body | render the recipe with its parameters and patches |
| `redraw` | `{generation_id, options}` | one pixel-changing `method` (`canvas`, `hires` or `light`) on a picture Generation; uploads one green-background picture |
| `repair` | `{generation_id, options}` | DWPose-masked crop-and-stitch reroll of hands/feet; uploads one picture per seed |
| `masked_redraw` | `{generation_id, options}` | the same reroll on caller rectangles with a free-text prompt patch |
| `deliver` | `{generation_id, options}` | cut and decorate a picture Generation: repin, backdrop, outline bands; never redraws |
| `dof` | `{generation_id, options}` | depth-of-field blur over a delivered picture's layers |
| `lens` | `{generation_id, options}` | fisheye / inverse fisheye over a delivered picture, or a spherize bulge / pinch of a disc in it |

A `finalize` row is refused. `redraw`, `repair` and `masked_redraw` leave a
picture on the green key; `deliver` makes the delivered picture, and `dof`
and `lens` make another delivered picture from it. They stack as separate
requests in one order: `dof` first, then `lens` (as often as wanted).

## Rules the code spreads across files

### Worker

- A row's `recipe_ref` must equal the worker's git branch, or the row fails.
- Idempotency keys come from the request id (`request:{id}`,
  `…:job:{i}`, `…:gen:{j}`, `…:asset:{role}`), so a re-claimed row resumes
  its own records. A ComfyUI prompt id the server forgot is resubmitted.
- The WorkerHub websocket claims rows as they are posted and relays
  progress. Polling at `--interval` always runs underneath it.
- The worker publishes its catalog at startup (best-effort). The catalog
  describes the worker's checkout, not your branch.

### Redraw options

- `options` is required and names a `method`; any key the method does not
  take is an error. One request changes the picture once and uploads one
  picture Generation (green background, no delivery, no `mask` asset) that
  refines the source, saved under `rdw-<generation id>`. The request's
  `parameters` are `kind: "redraw"`, `method`, `base_generation` and the
  resolved options; `seed` is the sampler seed that ran.
- `canvas` takes `denoise`, `size`, `route`, `finalizer`, `upscale`,
  `keep_regions` and `keep_strength`, with the defaults the catalog publishes
  as `redraw.defaults.canvas`.
  `denoise` accepts the `redraw` dial words. It re-samples the source on a
  bigger canvas and only redraws an Anima source.
- `hires` takes `hires` (the long side in px of the area a 1024x1640 canvas
  has) and `denoise` (0 < d <= 1, default 0.45). It re-samples the stored
  graph of the Anima generate output, so a `repair`, `masked_redraw` or
  `redraw` output and a stitched base are refused.
- `light` takes `scene` (`sunset` or `moon`) and `from` (any `stroke_light`
  direction, default `nw`; `redraw.light` in the catalog): the picture is painted with the
  scene's shade, tint and rim, then re-sampled at denoise 0.40 with the
  original graph's seed and negative and a positive that adds the scene's
  words. It works on any Anima picture, a `repair`, `masked_redraw` or
  `redraw` output included. The scene is cut with the source's `alpha` and
  `depth` assets under the same `cut` rule as deliver; assets that are
  missing or stale are cut in the graph and attached to the source.
- A source is refused when it is a delivered picture (the same set deliver
  refuses) or a LayerDiffuse picture. A `repair`, `masked_redraw` or
  `redraw` output is redrawn from its own pixels, with prompts and seed read
  from the original generate graph by following `base_generation`
  (at most 10 hops).
- Dial words are accepted for the `canvas` `denoise`. The row's result
  carries `resolved_options` with what actually ran.

### Deliver options

- The source must be a picture: a delivered picture and a LayerDiffuse
  picture are refused. Any other recipe or model is delivered as drawn.
- The options are `repin`, `recolor`, `skin`, `keep_legwear`, `keep_scene`,
  `transparent`, `backdrop`, `stroke_light`, `outlines`, `deliver_size` and
  `light`; any other key is an error. An absent `repin`, `stroke_light` or
  `backdrop` takes the recipe's `deliver.defaults`, the same values the WebUI
  sends. `light` only shades the stroke and tints the backdrop; it does not
  redraw. An explicit `backdrop: null` without `transparent` delivers a
  transparent picture; `transparent: false` keeps the white background.
- `outlines` lists the bands around the figure, innermost first, each
  `{"color": "#rrggbb", "width": w}` with `w` a percent of the longest side,
  0 < w <= 5, at most 6 bands. Absent or `null` takes
  `deliver.outlines.default` from the catalog; `[]` delivers without bands
  and without a drop shadow. Each band starts at the previous band's outer
  edge; the innermost also runs under the figure's soft edge.
- `stroke_light` is a light direction (`n` .. `nw`) or `even` (a uniform
  outermost band, the same as `null`); it shades and extrudes the outermost
  band only. Only a direction throws the drop shadow, straight away from the
  light, in the outermost band's shape. `light.scene`
  (`sunset` or `moon`) tints and brightens the backdrop toward the light and
  casts the sticker shadow in the scene's colour, both from `light.from`; an
  absent `stroke_light` follows `from`, `even` keeps its band and
  still gets the scene's shadow, and a different direction is an error.
  Transparent and `keep_scene` deliveries take the lit figure without the
  backdrop tint.
- Dial words (`"keep"`, `"on"`, …) are accepted wherever a number is. The
  row's result carries `resolved_options` with what actually ran.
- The cut is stored on the source Generation: `alpha` (ViTMatte alpha),
  `depth` (built by `dof`) and a json `cut` asset recording
  the matte model, matting model and revision, trimap and tile sizes, and
  the depth checkpoint and resolution. A later deliver loads an asset when
  its `cut` entry equals the worker's current values, and cuts again (and
  replaces the asset) when it differs or the asset is missing. Both are cut
  from the unrepinned source, so changing `repin` or `recolor` reuses
  them; the foreground colour is estimated again after the repin each time.
- The delivered Generation refines the source and carries its layers as
  PNG assets at the delivered size: `layer-figure` (RGBA, the repinned
  figure with the matte as alpha), `layer-outline` (RGBA, bands and drop
  shadow) and `layer-backdrop` (RGB; the kept scene for `keep_scene`, absent
  for a transparent delivery). Backdrop, outline, figure composited in that
  order give the delivered picture. No `mask` asset is attached.
- An absent `light` takes the `scene` and `from` of the nearest `light`
  redraw in the source's lineage (the source, then `base_generation` or what
  it refines, at most 10 hops). The inherited light is recorded in the
  request's `parameters`, and an absent `stroke_light` follows its `from`;
  an explicit `stroke_light` direction that differs is an error.

### Dof options

- The source must be a delivered picture carrying its layer assets; a dof
  output, a picture that was not delivered and a delivery without layers
  (delivered before layers existed: deliver it again) are refused.
- `focus` (`[x, y]`, fractions of the picture) is required. `f_number`
  1.4..22 (default 2.8), `scope` `{"figure", "outline", "backdrop"}` booleans
  (absent keys on; all off is an error) and `viewfinder` `off | on | both`
  follow the catalog's top-level `dof` section.
- Depth is the `depth` asset of the delivery's source (`base_generation`),
  reused under the `cut` rule, or built in the graph and attached to it.
  The figure takes its own depth, the outline the nearest figure depth, the
  backdrop the far plane; only the layers in `scope` are blurred, the rest
  are composited sharp. A transparent delivery stays transparent.
- The output Generation (`-dof`; `both` adds `-viewfinder`) refines the
  delivery, is itself a delivered picture, and records `kind: "dof"`,
  `base_generation` and the resolved options.

### Lens options

- The source must be a delivered picture: a delivery, a dof picture without
  the viewfinder overlay, or a lens picture. A picture that was not
  delivered and a viewfinder picture (`viewfinder: on`, or the second output
  of `both`) are refused. A dof on a lens picture is refused too: its depth
  and layers would no longer line up, so blur first and bend after.
- `center` (`[x, y]`, fractions of the picture, default `[0.5, 0.5]`) and
  `strength` (-1..1, default 0.7; 0 is an error) follow the catalog's
  top-level `lens` section. A positive strength is a fisheye: the picture is
  taken as rectilinear and remapped to an equidistant fisheye whose half
  angle at the farthest corner is `strength` × 75°, so what sits at the
  centre swells and the rim is squeezed while the farthest corner stays put.
  A negative strength is the exact inverse: the centre shrinks and the rim
  stretches, zoomed in just enough that no sample falls outside the picture.
  Put the centre on the feet to push the legs forward.
- `shape` `fisheye` (default) bends the whole frame as above. `shape`
  `sphere` is a spherize filter: only the disc of `radius` (0.05..1 of the
  shorter side, default 0.3; refused with `fisheye`) around `center` is
  wrapped over a hemisphere seen head-on, so a positive strength bulges it
  like a ball and a negative one pinches it; `|strength|` blends that with
  no bend, and everything outside the disc keeps its pixels.
- The delivered picture is bent, outline bands and backdrop with it; a
  transparent delivery stays transparent.
- The output Generation (`-lens`) refines its source, is itself a delivered
  picture, and records `kind: "lens"`, `base_generation` and the resolved
  options (`radius` only for a sphere).

### Repair and masked redraw

- The source must be a picture: a delivered picture is refused. Each seed
  uploads one picture Generation (plus a `repair-mask` asset); there is no
  delivered output.
- Redrawing or delivering a repair or masked_redraw output reads recipe,
  prompts, seed and loaders from `parameters.base_generation`. The repaired
  pixels are what gets redrawn.
- The repair prompt drops face, hair and framing tags. Part LoRAs (`lora`)
  are skipped on an Anima source.

### Generate requests

- `generation.parameters` is a closed set. Annotations go in
  `semantic.attributes`, and diffs go in `generation.patches`.
  `semantic.summary` is required.
- Patches apply in order after the recipe compiles, and each needs a
  `reason`. A `replace`/`remove` whose `old` is absent fails; it never
  silently skips.
- `prompt.positive.<part>` edits one component. A legacy group name
  resolves to the single member containing `old`.
- Patches are exclusive with `generation.graph` and with a full
  `prompt`/`negative_prompt`. `generation.graph` is the escape hatch for a
  graph the recipe cannot build. In graph mode, `parameters` only records
  what the graph really contains.
- After all edits, every identity tag of the unpatched prompt must still be
  in the positive, compared bare. Otherwise the request fails, unless
  `generation.identity_override` gives a reason, which is recorded on the
  generation.
- `experiment.overrides.patches` (from a chimera ExperimentRun) uses the
  same vocabulary and is exclusive with `generation.patches`.
