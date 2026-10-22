# docs index

Open the one file that matches the question, and stop. Sizes are rough
tokens (`uv run scripts/atlas.py docs` prints the live numbers).

## Findings — what the measurements concluded (current state)

| File | ~tok | Covers |
|---|---|---|
| [findings/prompt.md](findings/prompt.md) | 1.6k | tag order and weight, negatives, seed and sampler |
| [findings/face.md](findings/face.md) | 1k | ジト目, expressions, eye identity |
| [findings/legwear-costume.md](findings/legwear-costume.md) | 1k | canonical legwear, gloss, garment colour and fit |
| [findings/composition.md](findings/composition.md) | 1.1k | canvas, framing, pose structure |
| [findings/delivery.md](findings/delivery.md) | 1.7k | deliver, matte, stroke, backdrop, repin |
| [findings/checkpoints.md](findings/checkpoints.md) | 1.3k | Anima vs IL, LoRA, ControlNet / region tools, checking a new stage-1 model |
| [findings/evaluation.md](findings/evaluation.md) | 1.1k | metrics that failed, judging and process rules |

Each line carries a scope tag:

- `[all]` holds on the current pipeline.
- `[Anima]` was measured on Anima, stage 1 today.
- `[IL]` was measured on the retired Illustrious stage 1 and is kept for
  history.

`(a3 §The gradient direction)` points at archive file 3, at the heading
that starts with those words; `uv run scripts/atlas.py notes "The gradient
direction"` prints that section alone. A pointer with no `a<n>` came after
the archive and lives in `experiments/`, chimera or the named PR.

## Recipe

| File | ~tok | Covers |
|---|---|---|
| [yukari/anima.md](yukari/anima.md) | 1k | the live recipe: where each block lives, cross-file contracts |
| [yukari/delivery_style.md](yukari/delivery_style.md) | 0.9k | delivery constants: which code reads them, cross-file contracts |

## Operations

| File | ~tok | Covers |
|---|---|---|
| [queueing.md](queueing.md) | 2.7k | request kinds, where each option is defined, cross-file rules |
| [release.md](release.md) | 1.4k | manual deploy of main, worker deploy |
| [remote.md](remote.md) | 1.3k | the GPU box: setup, logon tasks, fetching models |
| [models.md](models.md) | 2.2k | model files and provenance |
| [architecture.md](architecture.md) | 1k | layer dependency rule |

## Archive

[archive/](archive/README.md) — the measurement log the findings came from,
frozen (~150k tokens). `a1` early / multi-character, `a2` 08-16–17,
`a3` 08-18–19, `a4` 08-19–25, `a5` 08-26–31, `a6` 09 (Anima move). Search
it, never open it whole.

## Writing here

New measurements go to `experiments/` (append-only). Docs are rewritten to
the current conclusion; see "Writing docs" in `AGENTS.md`.
`tests/test_docs_budget.py` enforces the caps and this index.
