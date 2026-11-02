# Evaluation and process

Tags and `a<n> §` pointers: see [`docs/README.md`](../README.md).

## Judging

- The user's eye decides; ratings are human-only on chimera. Image statistics
  have lost to the eye more than seven times. Metrics sort within one
  controlled sweep; they never overrule an accepted render. `[all]` (a4 §A fifth statistic)
- A number that agrees with a suspicion is the one to check by eye (a
  headcount tool "confirmed" two figures that were a sliver and a motion line).
  `[all]` (a3 §CORRECTION: `seiza`)
- A defect can inflate a metric and score "best" (intruders raise ink density).
  `[all]` (a1 §A note on the metric trap)
- Compare only at one canvas and framing: mark counts, colour counts and
  sharpness all move with size, crop and content. Downsample to a common side
  first. `[all]` (a2 §Measuring 「手書き感」; a4 §`winded` settled)
- Measure the region the defect lives in, not the frame mean; use top-decile
  or share-above-threshold for "too vivid". A fixed box is invalid when the
  composition moves. `[all]` (a1 §The clutter; a2 §Two garments on one leg)
- Use the mean, not the median, for small integer widths. `[all]` (a2 §Correction: the median)
- Retake a settings comparison on the render that will ship; a ladder on
  another seed ran the other way. `[all]` (a4 §Correction: the print is 1416)
- `palette_check.py` is a gate before showing: a FAIL is withheld, a pass is
  not an approval. Its corner-flatness check is invalid on head framings and
  scenes; check after the backdrop composite, not only raw. `[all]` (a5 §是正; a5 §2026-08-29 palette_check)
- Column-block headcounts work only on a flat backdrop and without motion
  lines; counts need an area share (≥2%), not a width. `[all]` (a3 §CORRECTION; a4 §`headcount.py` cannot read)
- A rising skin share can mean a faded figure; pair it with saturation. `[all]` (a5 §ガードは pass 1)
- Judge fine properties (gloss, line) on an enlarged crop, not a contact
  sheet. Judge the shipped pipeline output, not raw. `[all]` (a1 §How this was nearly missed; a4 §Two asks in one message)
- Do not open renders to browse; rating and semantic first, then
  `get_generation_image` on the short_id the user names. `[all]`

- In one ComfyUI process, the same seed and canvas with only the expression
  or action changed keeps the composition and moves only the face and hands.
  What is stable is prompt + canvas, not the seed. `[Anima]`
- Line weight can split by seed rather than prompt; pick the seed before
  tuning style words. `[Anima]`

## Process

- Render the control before adding a fix: the defect may already be gone.
  `[all]` (a3 §The ears were already off)
- Revert a dead hypothesis before the next test. `[all]` (a1 §The chair pose, brushed up)
- A combination that fails as pure noise: re-run on another seed before
  calling it dead. `[all]` (a1 §The gaming chair)
- Before adding a tag for a named thing, check whether the pose already draws
  it under another name. `[all]` (a4 §`(cola:1.4)`)
- Prompt edits go through splices that assert on a miss; a plain replace that
  misses shipped `closed mouth` and `open mouth` together. `[all]` (a4 §テヘペロ)
- Token order matters with an identical token set; reproduce a render's
  insertion order. `[all]` (a4 §The がおー family)
- After a block refactor, diff every pose × costume × pass against the prior
  commit; prompt snapshots and the two fingerprints (costume, delivery) cover
  different things. `[all]` (a4 §A second costume; a6 §What the fingerprint)
- Identify a render by its embedded graph, not its filename. Check what the
  server received (`/history`, chimera job) before diagnosing a picture. `[all]` (a1 §Matching a reference look; a2 §The lower body on)
- After three misreadings of one instruction, ask with a lettered annotated
  image. `[all]` (a2 §The cut edge)
- A new look starts from the recipe defaults; derive only when continuing a
  picture, and read the inherited patches first. `[all]`
- Every render is a chimera request with a semantic summary; A/B rounds are a
  chimera Experiment/Run with blind pairwise judging and `render_facts`. `[all]` (a6 §What chimera gained)

## Does not work

- Automatic clone detection by components or colour blobs. `[all]` (a1 §The clutter)
- Hough circles to count wheels; tuning a detector until its number looks
  right. `[all]` (a4 §Null: circles)
- Carrying a numeric threshold across characters or poses. `[all]` (a1 §What the parts block buys)
- Hand defects by detector: the WD tagger's `bad_hands` stays near 0 on
  broken hands; a hand crop judged by a small VLM catches finger count only;
  no model (Claude included) caught a wrong hand orientation. `[Anima]`
- Band/ringing checks that measure the deliberate white outline. `[all]` (a4 §Purely bigger)
