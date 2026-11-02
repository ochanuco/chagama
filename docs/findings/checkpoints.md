# Checkpoints, LoRA and reference tools

Model files and provenance are in [`docs/models.md`](../models.md); the
Anima recipe itself in [`docs/yukari/anima.md`](../yukari/anima.md). Tags and
`a<n> §` pointers: see [`docs/README.md`](../README.md).

## Holds

- Stage 1 is SilvermoonMix-Anima-Evolved 2.9B Turbo: 10 steps, euler, CFG 2
  for the first 4 steps and CFG 1 after (~35 s a picture). CFG 1 switches
  the negative off; CFG 2 on all 10 steps draws nearly the same picture. `[Anima]`
- On SilvermoonMix the hair gloss comes from `masterpiece, best quality,
  score_7`: removing them alone gives a flat fill and an even green backdrop
  on bust; on full-body anyo the change is small and the soles lose toe detail.
  Not the default. `[Anima]`
- On SilvermoonMix `@oshiki hitoshi` carries the black line and shadow,
  `@yoshikawa hideaki` the round face, younger look and gloss. `[Anima]`
- A LoRA trained on 2B Anima barely moves the 2.9B layer expansion even at
  1.8 although ComfyUI attaches its patches; the same LoRA overpowers
  anima-turbo-v1.1. Check a LoRA's base (2B or 2.9B) before downloading. `[Anima]`
- Style on Anima moves with artist tags (`@oshiki hitoshi` 0.85,
  `@yoshikawa hideaki` 0.5), not with a sketch LoRA; the sketch LoRA route is
  withdrawn. An artist the user excluded is patched out even if a default
  still carries it. `[Anima]`
- On Turbo a weight under 1.0 still registers: `oshiki` at 0.85 keeps the
  thick black line, `yoshikawa` at 0.5 stops the face lengthening and the
  eyes shrinking, and also suppresses the handwritten text `oshiki` draws
  alone. `[Anima]`
- The AI look on IL was the recipe's own style block and texture bans, not the
  checkpoint: nine checkpoint swaps under the unchanged prompt all rated
  bad/neutral; removing the block let sketch LoRAs draw. Separate the two by
  one-variable blind A/B. `[IL]` (a6 §The AI look; a6 §On IL the style block)
- A checkpoint's best face and best palette can be different checkpoints; on
  IL, the checkpoint moved style more than any prompt change. `[IL]` (a1 §The base draws the face)
- The IL redraw needs an Anima source (a `UNETLoader` in its base graph);
  other sources are delivered as drawn, LayerDiffuse bases refused. `[all]`
- `/object_info/<node>` returns 200 with `{}` for a missing node; check the
  body, or let `/prompt` report `missing_node_type`. `[all]` (a3 §「解像度を上げれば)
- No upscale model is installed on the worker; Lanczos is the only purely
  bigger route. `[all]` (a4 §Purely bigger)

## Reference tools

- ControlNet moves the whole composition: a traced framing overrides the pose.
  A partial pose skeleton is worse than none. `[IL]` (a1 §Tracing a reference)
- A pose skeleton was holding composition, not just anatomy; do not remove it
  to isolate another variable. `[IL]` (a2 §Two garments on one leg)
- A lineart ControlNet strong enough to carry a hairstyle carries unwanted
  structure too; ~0.35 kept both. Area tags pass at any strength, outline
  tags lose above it. `[IL]` (a1 §Swapping the character)
- Region prompts (`ConditioningSetMask`) beat weights for two garments: each
  region carries the whole prompt, the base region is masked out,
  `set_cond_area: default`. Debug with an absurd colour first. Colour distance
  from neighbours, not weight, wins a region. `[IL]` (a2 §Regions beat tags; a2 §Both at once)
- LayerDiffuse removes redrawn furniture that negatives cannot; the LD alpha's
  white haze is cut. `[all]`

## Checking a new stage-1 model

Run each on bust (the style reference) and one full-body pose, two seeds,
control re-rendered in the same ComfyUI process:

1. Quality tags: render with and without `masterpiece, best quality, score_7`.
   They decided the gloss on SilvermoonMix when no style word did.
2. Artist tags: each artist alone and both at 1.2/0.8; note which carries line,
   face and gloss.
3. Hand-drawn line: does any line word move the main line at all? If not, the
   line has to come from a LoRA matched to the model's size, or the IL redraw.
4. LoRAs in use: confirm each one visibly moves the picture at 1.0.
5. Sampler and CFG: the model card's sampler under our guided-steps CFG 1
   tail, and the time per picture.
6. Green backdrop: flat and saturated enough for the key.

## Does not work

- IPAdapter for costume transfer: across 4 poses, no palette gain and a 7–15%
  thicker line. Earlier "it works" came from a confounded baseline. `[IL]` (a2 §2026-08-17 -- the generation test)
- Checkpoint swaps to fix a look the recipe's own blocks were causing. `[IL]` (a6 §The AI look)
- Qwen-Image edit fixes fingers, toes and soles and keeps her identity, but
  redraws the whole frame at its own pixel budget, so pasting a crop back
  leaves seams; its text-to-image does not draw her. `[all]`
- MOSS-Image draws her without a LoRA but in even CG paint; an image
  reference carries content, not line; artist tags do nothing. A style LoRA
  from our picks at 1.0 came nearest and still lost the hand-drawn line. `[all]`
- Krea2: tag prompts draw someone else, CG paint. `[all]`
- Detail Daemon: numbers rise, the picture does not improve. `[IL]` (a1 §Detail Daemon)
- SilvermoonMix gloss through words: flat color 1.6 or at the head, cel /
  minimal / two-tone shading, no highlights / matte hair, a stronger shiny-hair
  negative. `[Anima]`
- SilvermoonMix line through words: sketch 1.6, rough / loose / uneven
  linework, hand-drawn, oekaki, pencil underdrawing, traditional media. The
  main line stays a clean digital line. `[Anima]`
- er_sde under the CFG 1 tail: grain at 10 steps, speckle that grows with steps. `[Anima]`
- Repainting the SilvermoonMix latent with anima-turbo-v1.1: only d0.65 drops
  the gloss, ~3x the time. `[Anima]`
