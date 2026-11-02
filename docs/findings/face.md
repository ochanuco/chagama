# Eyes and face

Tags and `a<n> §` pointers: see [`docs/README.md`](../README.md).

## Holds

- The identity is たれ目 + ジト目. What makes ジト目 is an expression tag that
  lowers the lid (`half-closed eyes`) plus an attitude word; direct shape tags
  (`jitome`, `narrowed eyes`, `tsurime`, `droopy`) lost every blind round on
  IL. `[IL]` (a5 §第3弾ピック; a5 §目の種類シート)
- On Anima, `jitome` works only at the head of the eyes part, where its weight
  counts; `half-closed eyes` is what removes the round eye. A high weight
  (1.8 and up) flattens a smile on expressions that also carry half-closed
  eyes. `[Anima]` (PR #217)
- On SilvermoonMix the jitome the user wants (wide eye, flat upper lid) comes
  from `large eyes` at 1.2 with `big eyes` gone, `jitome` 2.0 leading the
  eyes part, and no smile: `light smile` / `confident` keep the lid up
  whatever lid words are added. `thick eyelashes` stays — removing it only
  thins the lashes. `[Anima]`
- Eye colour moves only when `purple eyes` is replaced: adding
  `dark greyish purple eyes` next to it barely darkens the iris even at 1.6.
  `[Anima]`
- Bust eye size does not move with `large eyes` / `big eyes` weight or
  position on Anima; dropping `half-closed eyes` opens them a little. `[Anima]`
- Two settled states on IL: resting = `unamused` + `half-closed eyes` (1.3
  each); smug = `tareme` + `half-closed eyes`. 1.15 too little, 1.45 too much.
  `[IL]` (a5 §目の役割割当)
- Eye tags decide more than the face: swapping only the eye tag at a fixed
  seed moved the figure's skin-hue share from 81% to 2% (skin painted purple).
  `[all]` (a5 §2026-08-31 表情差分シート)
- Eye identity is not local. Crop-and-stitch on the eye region failed at every
  denoise up to a full repaint; change eyes by regenerating the face. `[all]` (a5 §目は局所プロパティ)
- Any path that repaints the face outside the prompt builder carries the face
  part and its eye bans (`EYE_BAN` in the IL recipe), or it falls back to the
  checkpoint's own eyes. `[all]` (a5 §目のアイデンティティ)
- Grimace comes from visible teeth, not from the tag that asked for them; use
  `open mouth` alone. A weight cannot rescue a tag that means the opposite;
  delete it. `[IL]` (a3 §The teeth; a3 §The gap is a different tag)
- When an expression will not land, compare with the reference render's
  actual prompt (`get_generation`) before re-weighting: two rounds went to
  `smug` while a shared `closed mouth` default silenced `open mouth`. `[all]` (a3 §The answer was in `/history`)
- An eye-lid guard banning half-closed and closed as a pair contradicts a
  wanted half-closed; ban only fully closed. `[IL]` (a3 §The answer was in)
- Expression and body/camera tags were independent axes: a 4-tag expression
  cluster swapped four times never moved body or canvas. `[all]` (a3 §2026-08-19 — `flop` takes 1.45)
- Lowering an expression tag's weight keeps the posture it drives (chin,
  spine) and drops only excess emotion; a synonym swap cost unrelated details.
  `[IL]` (a2 §Dialling the smirk down)
- Symbol tags draw literally and need low weight: `(@_@:1.0)`; `@_@` at 1.45
  overdraws. An additive symbol is not a drop-in for a subtractive tag at the
  same weight. `[IL]` (a4 §`dizzy` settles)
- Texture features (dark eyebags) need two different tags (`eyebags` 1.55 +
  `tired` 1.3), not one heavier tag. `[IL]` (a4 §`(eyebags:1.55)`)
- Below ~250 px face width the eyes stop matching and late expression edits
  stop reading. `[IL]` (a3 §Verbatim, and still)
- `portrait` means head and shoulders; it defeats a head-only crop whatever
  the negative bans. `[all]` (a4 §Dropped: a head-only)
- On IL, eye tags scale the face the checkpoint already draws; the checkpoint
  sets the face. Eye weights do not transfer across checkpoints or framings.
  `[IL]` (a1 §The base draws the face; a1 §A weight is only right)
- Face shape and eye choice are compared against the bust pick; a style probe
  is always shown next to bust. `[Anima]`

## Levers away from the default (SilvermoonMix)

- Only the word `jitome` flattens the upper lid. Shape words (`straight upper
  eyelids`) and English sentences draw a round, wide-open eye; dropping
  `jitome` gives a childish one. Narrow eyes come from `half-closed eyes`.
  `[Anima]`
- Closed eyes are expression `closed` (no `eye_base`, no `large eyes` /
  `big eyes` / `tareme` in the face, `closed eyes` 1.6). While jitome and
  large eyes stay, the eyes open whatever the negative says. `[Anima]`
- Strong emotion (annoyed, glaring, scolding) loses to the default jitome 2.0
  and lid words: expression `annoyed` lowers `eye_base` jitome to 1.2 and
  puts brow words (`furrowed brow`, `frown`) in the mouth part; bust damps
  emotion most. `[Anima]`
- Brow words are hidden by the bangs except in the mouth part at about 1.25;
  `v-shaped eyebrows` at 1.35 reads as anger. `[Anima]`
- Exasperation: mood `(exasperated:1.3), (sigh:1.1)` (draws a sigh mark).
  Putting exasperated in `eye_quality` opens the eyes. `[Anima]`
- Reluctant consent: eye_quality `(half-closed eyes:1.5), (exasperated:1.25),
  (resigned:1.1)`, mouth `(closed mouth:1.1), (wry smile:0.95), (light
  blush:1.1)`, mood `(reluctant:1.2), (sigh:1.1)`. A smile at 1.1 or more
  weakens the jitome. `[Anima]`
- Contempt: eye_quality `(half-closed eyes:1.45), (narrowed eyes:1.2), (cold
  eyes:1.4), (disgust:1.35), (contempt:1.3), (unamused:1.2)` + `as if looking
  at garbage` + `(shaded face:1.25)`. Cold and flat: `expressionless` /
  `unimpressed` / `slight frown` with `cold stare` 1.2. `[Anima]`
- Flustered: `(flustered:1.4), (nervous sweating:1.3), (sweatdrop:1.3), (wavy
  mouth:1.2)`. `blue face` draws a blue band. `[Anima]`
- Fine mouth shapes (・v・) are unreachable by words; offer `light smile`. A
  grin reads better as `clenched teeth` + `grin` than `sharp teeth`. `[Anima]`
- No makeup is pose `nomakeup` / `nomakeup-bangs-up`: the two eyelash words
  removed, negative `(thick eyelashes:1.2), (eyeliner:1.2)`; lowering their
  weight does nothing. The double eyelid does not move by any word. `[Anima]`
- No hair ornament (`NO_HAIR_ORNAMENT` / `HAIR_ORNAMENT_BAN`, worn by the
  nomakeup poses): the sidelock tube goes only when another tie (the orange
  hair tie) fills its place. On another pose it is an identity patch and
  needs `identity_override`. `[Anima]`
- One sidelock: identity `very long sidelocks` → `(single sidelock:1.4), very
  long sidelock`; seed-dependent. `[Anima]`
- Hair cut off the head: name the part plus `detached hair`; `cut-off hair`
  draws a wig on a head, `somen` draws real noodles. `[Anima]`
- Nails `(nail polish:1.3), (<colour> nails:1.4)` and inner hair colour
  `(colored inner hair:1.3), (<colour> inner hair:1.3)` work; inner colour is
  per look, not a default. `[Anima]`
- A childish full-body face came from an expression and an action together,
  not from either alone or face words; bisect by stacking parts on the bust
  base. `[Anima]`

## Does not work

- Pixel-counting irises or eye ratios on small crops — contaminated by crop
  edges and hair; the eye decided in one look. `[all]` (a3 §2026-08-19 — `flop` takes 1.15)
- Optimising eye-to-face ratio as the quality target; everything unmeasured
  drifted. `[all]` (a1 §Correction: eye ratio)
- `smirk` on Anima bust: it draws a duck mouth. `[Anima]`
- On SilvermoonMix, full-body floor poses carrying `unamused` / `bored` drew
  floating white jitome blobs in 4 of 8 renders; negative `floating heads` /
  `extra faces` / `mascot` / `ghost` did not remove them. `[Anima]`
