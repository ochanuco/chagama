# Trail

A `trail` draws one stroke across a staged picture: a shooting star whose
glitter trail runs behind the figure on some stretches and crosses in front
of it on others. The workbench calls it 軌跡 and puts it between 背景
(`stage`) and 光 (`lighting`).

## Source and output

- The source must be a `stage` output; a trail of a trail is refused.
- The output (`-trailed`) refines the stage output and carries four layers:
  `layer-figure` and `layer-outline` as they came, `layer-backdrop` with the
  back stretches painted in, and `layer-front`, the stretches over the figure.
  It records `kind: "trail"`, `base_generation` and every resolved option,
  the seed included.
- `lighting` lights a trail like a stage output and lays `layer-front` back
  over the lit picture, unlit, and keeps it as an asset. `dof` blurs the
  layers as usual and lays `layer-front` over the blur, sharp. `lens` takes
  it as any delivered picture.

## Options

The catalog's top-level `trail` section lists the templates, the choices and
the width range; `recipes[].trail.color` is each character's default colour,
the outer outline band 12% darker.

- `template` (default `tsu`) supplies four points and a front flag per point.
  `points` (four `[x, y]`, shares of the picture, each within -1..2) and
  `front` (four booleans) replace the template's.
- `path`: `curve` runs a centripetal Catmull-Rom curve through the points,
  `line` straight segments.
- `rule` says which stretches cross in front: `both` (a segment is in front
  when both its points are), `split` (a segment whose ends differ changes
  over at its midpoint) or `start` (a segment follows its first point).
- `width` 1..15 (default 5), percent of the longest side: the trail's
  breadth near the head and the head's size. Particle counts and sizes are
  set for a 1440 px long side and scale with the picture, so a hires draws
  the same trail.
- `head` `light` (a white core, cross rays and a streak behind) or `sparkle`
  (a cluster of four-point stars); `head_at_end` puts it on the last point.
  The head goes in the layer of the stretch it ends.
- `in_frame` (default true) keeps the back stretches inside a framed stage's
  box; the front ones cross the frame like the figure does.
- `seed` fixes the scatter; without one the worker picks one, keeps it in its
  run state and reports it in `resolved_options`.

## Night backdrops

The light reads only on a dark ground. Every backdrop pattern has a night
variant, `<pattern>-night`: the pattern kept at 30% over 75% of a night blue
(`domain/yukari/trail_style.py`), painted under the frame so the frame's line
and paper stay as they are. The catalog lists them after the patterns with
`night: true`; pick one in `stage` for a picture that will get a trail.
On a light ground the head's light falls back to the trail colour.
