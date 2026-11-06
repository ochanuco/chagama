# Prompt and negative mechanics

Tags and `a<n> §` pointers: see [`docs/README.md`](../README.md).

## Holds

- Anima's prior dominates: at a fixed seed, removing FACE, style weights, the
  whole style block, the negative or the quality tags gave the same picture.
  What does move it: artist tags (recipe default since PR #192) and an eyes
  part whose head tag carries the weight (PR #217). `[Anima]` (a6 §Anima ignores)
- Quality vocabularies are per model and swap as a pair: Anima HEAD is
  `masterpiece, best quality, score_7, absurdres, 1girl, solo` with
  `score_1/2/3` negative. Never mix IL quality tags in. `[Anima]` (a6 §Why the quality)
- Append new tags at the tail. A mid-prompt insert re-encodes everything after
  it and re-rolls a settled composition; so can adding a tag beside a settled
  span, where swapping a word inside the span does not. `[all]` (a5 §hige —; a2 §「タイツになってないな)
- Substitute, never subtract, in a tuned block. Deleting one tag broke line and
  colour count as badly as a bad addition; deleting is not the midpoint of a
  tag's range. `[all]` (a1 §Substitute, never subtract; a2 §The eye tag had no job)
- An unweighted tag in a prompt that runs at 1.3+ is absent; weights under
  1.0 read as absent too. Raise to 1.25–1.4. `[all]` (a2 §An unweighted tag)
- Some tags are binary: `half-closed eyes` had no gradient between present and
  absent at 1.1–1.3 on IL. `[IL]` (a2 §Open the eyes)
- CLIP has no negation: `no ornament` in the positive asks for an ornament. `[all]` (a1 §Failure modes)
- Naming a body part raises its salience whatever the adjective. Naming a
  feature that should be out of frame pulls it back in. `[all]` (a1 §Two corrections; a3 §The face goes down)
- A literal object noun always draws the object (`mustache`). For a gesture,
  use two contact/action tags (hand-to-mouth + hair-to-lips). `[all]` (a5 §hige —)
- A held object needs the object tag and a verb that moves the hand
  (`holding cup` + `drinking`); naming the vessel twice pins what it is. `[all]` (a2 §The `sip` pose)
- A property tag (striped, skin tight, print) has no garment of its own and
  lands on whichever garment has room; guard the ones it must not touch. `[IL]` (a4 §Finalising `hoops`)
- Two adjacent weights with opposite complaints and nothing between them mean
  the wrong dial. Look for a construction tag elsewhere. `[all]` (a4 §Finalising `hoops`)
- A continuous proportion axis moves only by naming what is wanted; negative
  `long torso` / `long legs` measured zero. `[IL]` (a3 §`stand` settles its canvas)
- A character's identity lives in blocks nobody inventoried as hers. Porting
  class tags alone silently drops eyes, shading, body. `[all]` (a1 §Her design lived)
- A settled change lives in the recipe blocks, not in prose, memory or
  `.local/` scripts, or the next session does not get it. `[all]` (a2 §The one-garment leg)

## Negatives

- Every negative eventually fights a later request with no error, only a
  broken picture. `scripts/prompt_lint.py` checks positive/negative
  contradictions; audit the negative whenever the goal changes. `[all]` (a5 §根本原因; a1 §Failure modes)
- Listing a thing's parts in the negative draws a fancier thing. Ban the
  symptom's absence, not its parts. `[all]` (a1 §Failure modes)
- A negative that names a legitimate shape deletes it (`ragged` shrank a
  pointed hem). `[all]` (a1 §Failure modes)
- One guard per defect. Stacking two or more guards on the same defect broke
  the palette repeatedly and summoned backdrop intruders; raise the existing
  guard's weight instead. Guards on distinct defects stack fine. `[all]` (a3 §発色; a2 §`mature female` was also)
- Pushing 3+ body tags at once summoned an intruder; easing two was enough. `[all]` (a2 §The lower body was)
- A guard works in the pass that decides composition. The same guard in a
  refine negative cannot undo committed pixels, and a colour guard there
  greys the identity hues. `[all]` (a5 §ガードは pass 1)
- Never drop a whole named negative block to fix one symptom; it bundles other
  guards. `[all]` (a1 §Per-character structure)
- A shape with no negatable name is banned through the model's alternative
  reading (slouch: ban `arched back`, `bridge (pose)`). `[IL]` (a4 §「筋肉がなさすぎて)
- A property can be guarded when the model draws it as discrete strokes
  (`long eyelashes` worked), not when it is an undrawn proportion. `[IL]` (a4 §Correction: the length guard)
- Check the model's prior before a symmetric bracket: bare `skirt` already
  skews to miniskirt, so banning `long skirt` does nothing. `[IL]` (a4 §A guard pointing the wrong way)

## Seed and sampler

- Measure on 3+ seeds. Every one-seed fix in the log was later reversed. `[all]` (a2 §Correction: the ribbed-legwear)
- Seed dominates colour: five seeds of one prompt spanned saturation 21–51.
  Check the seed before tuning the prompt for colour. `[IL]` (a4 §The colour is a property)
- Seed decides raised-leg composition, which lock a hand grips, and hand
  quality below a threshold; weights did nothing. `[all]` (a2 §Feet at head height; a5 §hige —; a4 §指が正常化)
- Duplicate figures on IL were a sampler property (euler_ancestral fixed
  them); on a full-body stand, canvas width. `[IL]` (a1 §The clones; a4 §The canvas, and the pick)
- A load-bearing seed is a named constant in code, not a filename. `[all]` (a4 §The seed is now recorded)
- The same seed and prompt draw a different picture across a ComfyUI restart;
  A/B controls are re-rendered in the same process. `[all]` (observed 2026-09-21, after the archive)

## Words that draw something else (SilvermoonMix)

- `holding credit card` → a trading card; `coffee jar` / `instant coffee` →
  a large filled jar; `body pillow` / `plush` → a printed face or a plush (use
  `long cushion`, `seamless velvet bolster`); `pov headpat` → a selfie arm.
  `[Anima]`
- `hoodie` → a closed pullover; an open front needs `open jacket` /
  `hooded zip-up jacket` with `kangaroo pocket` / `pullover hoodie` banned,
  and `hand in pocket` → `hand in jacket side pocket`. `[Anima]`
- `track shorts` / `gym shorts` → dolphin shorts; `side stripes` moves onto
  the top. `light steam rising from body` → a manga steam cloud. `floating`
  reads as motion. `[Anima]`
- A pose's own negative can ban what a new request wants (the coffee pose
  bans mugs); remove the ban, not just add the word. `[Anima]`
- A legwear colour claim raises the saturation of hair and dress too: they
  share one purple band, so separating them belongs after the sampler. `[all]`
- The oshiki black fill (inside the hood, hem, back of the head) survives
  every negative, all-step CFG, the non-Turbo model and NAG; NAG also changes
  the style. `[Anima]`
- Skin sheen appears only with the shiny bans removed, and then the skin
  turns to a pink gradient with thin lines, off the bust style. `[Anima]`

## Does not work

- Duplicate-figure bans in the negative (`2girls`, `duplicate`, `clone`) —
  worse, and they break the palette. `(solo:1.5)` at the head of the pose
  block, or a narrower canvas, is what worked. `[all]` (a1 §Fixing a character; a2 §`sip` seed sweep)
- `character sheet` / `turnaround` bans against a turnaround sheet; changing
  the pose fixed it. `[IL]` (a2 §The `nape` pose)
- Relationship tags (`lap pillow`, `head on lap`) at any weight: they summon
  the other party. Delete them. `[all]` (a1 §A tag that names a relationship)
- Repeated-action verbs (`patting`): the hand multiplies. Delete the verb. `[all]` (a1 §`patting`)
- Stacked simultaneous moments (tripping + falling + fallen): one body per
  moment. Pick one; carry motion with motion lines. `[all]` (a3 §`flop` gets the skid)
- Vague action verbs: they hold a slot and draw nothing. `[all]` (a1 §Making the view incidental)
