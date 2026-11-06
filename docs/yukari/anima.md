# yukari

> Yuzuki Yukari belongs to her original creators and rights holders -- see
> [Derivative work](../../README.md#derivative-work) in the README.

The first of the two recipes on SilvermoonMix Anima Turbo
(`silvermoonmixAnimaEvolved_v2329BTurbo.safetensors`); the other is
[akari](../akari/anima.md), drawn in the same style and poses.
The code is the description. This page maps where things are; the reasons
behind the values are in [`docs/findings/`](../findings/).

## Read the prompt, not the source

```bash
uv run chagama yukari prompt --pose bust                          # assembled positive/negative
uv run chagama yukari prompt --pose bust --costume standard --expression gao --json
uv run chagama catalog                                            # every pose, part, dial
```

On the MCP, `get_catalog_pose yukari <pose>` returns the same thing as the
production worker sees it.

## Where things are

A recipe is the shared Anima style drawing one `Character` in the shared
poses and expressions. The style, the poses and the expressions never name
a character; `Recipe.drawing` lays the character's own looks over them.

`src/chagama/domain/anima/` (the style, shared by every character):

| File | Holds |
|---|---|
| `style.py` | quality/count tags, artist tags, the green cut-out backdrop, `STYLE`, the structural negative bans (digits, detail, coloured line, thin body, shine, gradient, eye sparkle and motion lines, score, proportion, text), render constants (model, canvas, steps, cfg, sampler) |
| `character.py` | `Character` (character/series tags, identity and its guarded tag names, default costume, eye shape, eye quality per `EyeQuality`, nails, body, face, a style suffix, the character's own negative bans, persona pose name, its looks), `PoseLook` / `ExpressionLook` (a character's own blocks for one shared pose or expression) and `Wardrobe` (costume blocks, from-behind swaps, per-costume bans, hood ban, legwear blocks) |
| `recipe.py` | `Recipe(character, poses, expressions)` and `Recipe.drawing`, which dresses the shared poses and expressions in the character's looks: assembly, negative order, `render_spec`, identity tags, the persona pose |
| `poses.py` | `POSES`, every shared pose (action, mood, gesture, framing, angle, default expression, negative), `POSE_LABELS` (the Japanese display name every pose must have, published as the catalog pose's `label`), and the bangs-up and bare-face words a character's look builds on |
| `expressions.py` | `EXPRESSIONS`, every shared expression's mouth and character-neutral eyes |
| `pose.py` | the `Pose` record: action, mood, gesture, framing, angle (camera) and `angle_action` (action tags that sit beside it), default expression/costume/legwear, optional body/face/identity/nails/style/negative/canvas/loras |
| `expression.py` | the `Expression` record and the `EyeQuality` kinds |
| `legwear.py` | legwear kinds (`sheer` default, `sheer-gloss`, `opaque`) and states (`worn`, `removing`, `off`) with their shared negative edits |
| `angles.py` | `ANGLE_LABELS` (Japanese display names, published as `angle_labels`) and the `angle` request words (`from_below`, `from_above`, `from_front`, `from_side`, `from_behind`, `dutch`, `pov`), one axis each: height, facing, tilt, pov |
| `framing.py` | `FRAMING_LABELS` (published as `framing_labels`) and shot tags per `Framing` kind; `BUST` and `CLOSE` also carry their canvas |
| `components.py` | the component model: `(name, section, priority, text)`, `PART_NAMES` and the legacy `PART_GROUPS` |
| `dials.py` | words for `render.width`/`render.height` patches |

`src/chagama/domain/yukari/` (the character):

| File | Holds |
|---|---|
| `character.py` | `YUKARI`, the `Character` the recipe draws |
| `prompt_style.py` | her blocks: character/series tags, identity, eye shapes, cold eyes, face, body, nails, hair-ornament swaps, her negative bans (purple line, eye colour, vivid palette, hood, black garment, sheer tone, mechanical body) |
| `costumes.py` | garment block per costume, `LEGWEAR` per costume, the sheer blocks, assembled into `WARDROBE` |
| `looks.py` | her looks: the adult body and close face of `bust` / `gao` / `avatar`, the `nomakeup` poses (roomwear, no hair ornament, bare face), and the ジト目 eyes of `disdain` / `dubious` / `jitome` / `neutral`, the soft eye shape of `annoyed`, the face words `closed` drops |
| `poses.py` / `expressions.py` | `POSES` / `EXPRESSIONS` as she wears them; `yukari_persona` (`PERSONA_POSE`) is kept out of `POSES` |
| `recipe.py` | `RECIPE`, the style bound to `YUKARI`, and its `positive` / `negative` / `render_spec` |
| `reference.py` | a reference picture's tags onto `yukari_persona` |
| `delivery_style.py` | delivery identity, redraw settings and deliver defaults; see [delivery_style.md](delivery_style.md) |

The graph is built by `infrastructure/comfyui/anima_graph.py`.

`src/chagama/domain/recipes.py` registers the recipe under the name a
request gives in `generation.recipe`: its `Recipe`, pose labels, dials and
rejected parameters. The worker's graph build and identity guard,
`validate_request`, the catalog and `chagama <recipe> prompt` all read that
table; another character is one entry there. `DEFAULT_RECIPE` is what a
refining request records when its source names none, and the recipe the
`reference` tag rules are written for.

## Contracts that span files

- Components sort by Anima's card sections (`QUALITY`, `COUNT`,
  `CHARACTER`, `SERIES`, `ARTIST`, `GENERAL`). Inside `GENERAL`, the order is
  priority (`LEAD`, `MAIN`, `TAIL`), then declaration order. The joined
  components equal the positive byte for byte (`tests/test_yukari_components.py`).
- Each component name is a `prompt.positive.<name>` patch target. The 13
  legacy part names still resolve through `PART_GROUPS`.
- A request's `costume`, `expression`, `legwear`, `legwear_state`, `angle`
  and `framing` override the pose's own. A pose's `body`, `face`, `identity`
  and `nails` override the character's, its `style` and `background` the
  style's; `loras` belong to the pose. A shared pose names no costume and no
  character's body or face: those come from the character's `PoseLook`
  for that pose name, else the character's defaults. `pose.legwear=False` ignores both legwear parameters.
- `framing_tags` is the camera angle, then `pose.angle_action`, then the
  framing's shot tags. `angle` (comma-separated words or a list, at most
  one word per axis, joined in axis order) replaces only the camera angle;
  `angle_action` (gao's `looking back`, anyo's `foreshortening`) stays.
  `framing` takes a `Framing` value (`bust`, `upper`, `cowboy`, `full`,
  `lying`, `close`, `face`). A `framing` other than the pose's own also drops
  the shot tags (`FRAMING_TAG_NAMES`: every tag any framing text uses) from
  the pose's `negative` and `negative_tail`, so a full-body pose's
  `(cowboy shot)` ban does not fight a requested `cowboy`.
- No pose carries a place tag. Every pose draws on the green key the matte
  despills.
- Canvas precedence is `pose.canvas`, then the framing's canvas, then
  `1024x1640`. A request's `framing` skips `pose.canvas`. `hires` scales that canvas to the given long side (multiple
  of 8), and `denoise` needs `hires`.
- `redraw` takes only an Anima source (a `UNETLoader` in its base graph).
  Other sources are delivered as drawn.
