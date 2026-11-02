# Pose, composition and canvas

Tags and `a<n> §` pointers: see [`docs/README.md`](../README.md).

## Holds

- Canvas is a composition variable, not a resolution knob. Same seed and
  prompt on another canvas reframes the figure (and changes the pose, on
  Anima and IL alike). To go bigger without losing a pick: latent upscale +
  Anima's own d0.35–0.45 pass. `[all]` (a3 §「解像度を上げれば; a4 §The canvas)
- Anima renders 2048x2560 directly; 1536 and 2048 look the same on Anima, and
  delivery is 1536 (~54% of the 2048 cost). The 2048 need was an IL artefact.
  Re-test size dependencies after any checkpoint move. `[Anima]` (a6 §Why the delivery is 1536)
- The full-body pixel ceiling is a total pixel budget: 896x1728 fits a
  standing figure that 1024x1536 crops. `[all]` (a3 §The legs were never)
- Extra width beside a standing figure invites a second figure; a narrower
  canvas (832) fixed it where no tag did. `[all]` (a4 §The canvas, and the pick; a3 §`stand` is adopted)
- Orientation matters at equal pixels: prone kept its outline on 1536x1024;
  portrait gave a rear view, square cropped. `[all]` (a2 §`prone`)
- A close framing held only on a square canvas; the same prompt on a tall
  canvas filled the frame with more body. `[IL]` (a1 §The line is 1.91px)
- Square canvases printed at 2.0x staircase; no denoise fixes staircase and
  hard edges at once. `[IL]` (a4 §`tehe` is a square canvas; a4 §The other jaggedness)
- Floor contact is unreachable on a crop that excludes the floor. `[all]` (a2 §Feet at head height)
- Silhouette-read poses move by camera, not weight: `(from side:1.35)` fixed a
  sit-up three weight rounds could not. `[all]` (a4 §`situp` settles)
- A proportion complaint can be a structure tag: `sitting on floor` stretched
  the legs toward the camera; changing the seat fixed it. `[all]` (a2 §The thighs were a pose problem)
- Proportion tags fine from the front read as bulk from a rear or
  foreshortened camera; ease, do not delete. `[all]` (a2 §The lower body was)
- A framing or angle tag can hold a hidden second job (keeping hips out of
  centre); check what else moves when it goes. `[all]` (a1 §Making the view incidental)
- A framing outcome tag (`feet out of frame`) is not a guarantee; on some
  seeds it pushes the figure into the edge. `[all]` (a4 §`feet out of frame`)
- Pass 1 decides structure. A refine can delete an object pass 1 drew but not
  add length, a joint or an expression; change the pass-1 tag instead
  (yokozuwari → agura removed a hidden-leg joint). `[all]` (a5 §roomwear lounge; a4 §Where the second pass's reach; a3 §The pass was the ceiling)
- A figure can end at a hem or frame edge; that reads more finished than a
  redraw of the missing part. Garments close against backdrop, bare limbs do
  not end on a straight cut. `[all]` (a2 §"除去して完成"; a2 §The cut edge)
- If a masked region will not resolve, check the prompt names the pose the
  pixels show (`kneeling`, not `squatting`). `[all]` (a2 §Finishing an unfinished)
- A scene baked into a pose dominates every seed; a scene-free variant needs
  its own pose entry. No background is the default: no place tags, keep simple
  / grey background even in probes. `[all]` (a5 §roomwear costume)
- Seven heads is the proportion target on Anima; body tags alone do not move
  it and too many bans give stick legs. A block tuned on one checkpoint
  overshoots on another. `[all]` (a6 §The proportion block)
- Floor shadow: `no shadow` 1.3 + four floor-shadow negative words; the rest is
  the deliver cut's key. `flat lighting` / hatching bans blacken the tights. `[Anima]`
- `floor visible` calls a rug. Patterned tights leak into the leg on redraw.
  `[Anima]`
- `hires` holds the draft's look up to about 1.4x latent upscale (1024x1280
  → 1448x1808, 896x1440 → 1280x2048 at d0.45). A 640x800 draft at 2.26x
  leaves noise (background blotches, a smudged rim) and redraws the face
  cleaner (more catchlights, lashes, hair lines); d0.55 removes only the
  noise. `[Anima]`
- At equal pixels a tall pose wants its tall aspect: 1024x1280 for a
  1024x1640 pose fills the side space with floating faces or an extra leg,
  and crops a cowboy shot near the hip. 896x1440 frames like 1024x1640.
  `[Anima]`

## Levers away from the default (SilvermoonMix)

- A young-looking face is the canvas first: a square bust draws a short
  round face, 1280x2048 a longer one. A 896x1440 bust draft also reads
  younger (less face detail at that size) and face words do not fix it.
  `[Anima]`
- Even light on a bust: `(front lighting:1.3), (flat lighting:1.3), (even
  lighting:1.3), (no shadow:1.2), (bright lighting:1.1), (light purple hair
  shadows:1.2)`; not `id photo`. `[Anima]`
- A closer framing needs a square canvas (1152x1152) and the leg words gone
  (legwear and body-build legs, negative legs / thighs / pantyhose). `cropped`
  / `out of frame` call a small inset figure; `slightly from below` drifts to
  the crotch, so a chest close-up goes `from above`. `[Anima]`
- Foot close-up: `(extreme foreshortening:1.5), (foot in foreground:1.5),
  (close-up:1.2), (foot focus:1.5), (toes:1.3), (soles:1.2)` + a sentence that
  the toes hide part of the body. Some seeds add face stamps in the
  background that no ban removes. `[Anima]`
- A hand or object thrust at the camera comes out blurred at generation; no
  word or negative fixes it, so change the composition. Blur and focus come
  only from `dof`; dropping framing `depth of field` removes unwanted blur.
  `[Anima]`
- True side and rear views do not come from words (the figure looks back or
  stops at 3/4; bust + `from_behind` looks over the shoulder). A depth
  `render.pose` from a mesh of the figure at strength 1.0 sets the angle.
  `[Anima]`
- Reference poses hold at strength 1.0 with the head (and chest) freed; a
  lower strength adds faces and toys outside the figure, and an `end` cutoff
  does nothing on Turbo. `[Anima]`
- Lying: from `stand` add `lying` / `on back` / `knees up` and negative
  `standing`. `upside-down` mirrors a front view; use `top-down view`. Long
  English leg descriptions turn 3 of 4 into multi-view sheets; short tags
  (`kicking`, `legs up`, `motion lines`) do not. `[Anima]`
- Sitting with no furniture: `(sitting:1.4), (invisible chair:1.3)` +
  negative `chair` / `furniture` / `stool` / `bench` / `table` / `standing`.
  `[Anima]`
- Looking down at the viewer needs `(from below:1.6), (low angle:1.3)`; 1.4
  with `looking down at viewer` keeps the camera level. `[Anima]`
- Legs to one side, feet together: `(yokozuwari:1.5), (sitting on floor:1.2),
  (legs to the side:1.2)` + negative `wariza` / `kneeling`. `[Anima]`
- `bent over` stands and bends; a kneeling forward stretch needs a landscape
  canvas plus `kneeling`. `plank` draws lying down. `[Anima]`
- "Legs near, body far" by foreshortening detaches legs from hips;
  `from side` + `wide shot` keeps the depth. `[Anima]`
- Crossing a frame: pose `(outside border:1.5), (thin border:1.15),
  (square:1.1), (breaking the fourth wall:1.2), (stepping:1.3), (leaning
  forward:1.2)` + a sentence; a leg stepping over reads, leaning over the top
  does not. The frame's size and place do not follow the prompt. `[Anima]`

## Does not work

- Crops, cut-outs or repairs to deliver while the prompt is being tuned: the
  next arm is judged against a picture the recipe cannot make. `[all]` (a3 §Crops are banned)
- `wariza`: right on 1 of 3 seeds. `[all]` (a5 §roomwear lounge)
- `thick thighs` / `wide hips` for leg volume: they drag in a rear camera;
  `toned legs` gave volume without it. `[IL]` (a1 §Volume on the legs)
- Naming a framing defect in the negative. `[all]` (a1 §Volume on the legs)
- Text control of which hair lock a hand grips. `[all]` (a5 §hige —)
- `full body` as a front-view anchor: it is a distance tag. `[IL]` (a2 §Rendering it)
- Pixel upscale (decode, lanczos, encode) before `hires`: no noise, but the
  draft only gets bigger and the drawn line never comes back. `[Anima]`
