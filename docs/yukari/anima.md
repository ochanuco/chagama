# yukari

> Yuzuki Yukari belongs to her original creators and rights holders -- see
> [Derivative work](../../README.md#derivative-work) in the README.

The one live recipe, drawn on SilvermoonMix Anima Turbo (`silvermoonmixAnimaEvolved_v2329BTurbo.safetensors`).
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

## Where things are (`src/chagama/domain/yukari/`)

| File | Holds |
|---|---|
| `prompt_style.py` | blocks every pose wears (`QUALITY`, `CHARACTER`, `BODY`, `NAILS`, `BACKGROUND`, `FACE`, `STYLE`), the negative bans, render constants (model, canvas, steps, cfg, sampler) |
| `poses.py` | one `Pose` per pose: action, mood, gesture, framing, angle, default expression/costume/legwear, optional body/style/negative/canvas/loras |
| `costumes.py` | garment block per costume, `LEGWEAR` per costume, the legwear kinds (`sheer` default, `sheer-gloss`, `opaque`) and states (`worn`, `removing`, `off`) with their negative edits |
| `expressions.py` | mouth/eyes per expression and the `EyeQuality` routing |
| `framing.py` | shot tags per `Framing` kind; `BUST` also carries its canvas |
| `components.py` | the component model: `(name, section, priority, text)`, legacy part groups |
| `recipe.py` | assembly, `render_spec`, identity tags, the IL redraw prompt |
| `dials.py` | words for `render.width`/`render.height` patches |
| `delivery_style.py` | delivery identity, redraw settings and deliver defaults; see [delivery_style.md](delivery_style.md) |

The graph is built by `infrastructure/comfyui/anima_graph.py`.

## Contracts that span files

- Components sort by Anima's card sections (`QUALITY`, `COUNT`,
  `CHARACTER`, `SERIES`, `ARTIST`, `GENERAL`). Inside `GENERAL`, the order is
  priority (`LEAD`, `MAIN`, `TAIL`), then declaration order. The joined
  components equal the positive byte for byte (`tests/test_yukari_components.py`).
- Each component name is a `prompt.positive.<name>` patch target. The 13
  legacy part names still resolve through `PART_GROUPS`.
- A request's `costume`, `expression`, `legwear` and `legwear_state`
  override the pose's own. `body`, `style` and `loras` belong to the pose.
  `pose.legwear=False` ignores both legwear parameters.
- No pose carries a place tag. Every pose draws on the green key the matte
  despills.
- Canvas precedence is `pose.canvas`, then the framing's canvas, then
  `1024x1640`. `hires` scales that canvas to the given long side (multiple
  of 8), and `denoise` needs `hires`.
- `redraw` takes only an Anima source (a `UNETLoader` in its base graph).
  Other sources are delivered as drawn.
