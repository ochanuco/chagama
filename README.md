# chagama

A command-line front end for [ComfyUI](https://github.com/Comfy-Org/ComfyUI).
No web UI and no custom nodes of its own. The public entry point is
`chagama`; it builds a graph, records the run in Chimera, submits it to
ComfyUI, and pulls the result back.

Two things make it more than a thin API wrapper:

**The defaults are the recipe.** The Yukari domain owns the checkpoint,
sampler, resolution and prompt blocks that took many rounds of rendering to find.
`chagama yukari prompt --pose bust` inspects the exact prompt without
queueing; actual generation always uses a recorded request.

**The GPU box is the only executor.** `chagama work` runs there next to
ComfyUI, claims request rows from chimera, and renders them; a session on any
other machine queues rows and reads results, and never talks to ComfyUI.

```bash
uv run chagama work                # on the GPU box: serve chimera's queue
```

## Finding your way

There is no list of scripts here; the repository describes itself instead.
[`docs/README.md`](docs/README.md) is the one hand-kept index, and a test fails
when it misses a doc:

```bash
uv run scripts/atlas.py             # every script: its role, its size, its own first line
uv run scripts/atlas.py docs        # every doc with its size
uv run scripts/atlas.py notes       # findings and archive by heading, with line numbers
```

`.local/` is untracked and holds a session's throwaway scripts and logs.

[`docs/`](docs/) holds the operating notes — queueing, the GPU box,
releases, model provenance. [`docs/findings/`](docs/findings/) is what the
measurements concluded, one topic per file; `experiments/` is the measurements
themselves, and `docs/archive/` the original log they were distilled from. They
are the point of the repository; the scripts are how they were produced.

## What it runs on

Python 3.12 and [uv](https://github.com/astral-sh/uv). Nothing here imports
torch. The same environment runs the tests, as CI does:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . pillow numpy opencv-python scipy pytest websockets
PYTHONPATH=scripts .venv/bin/pytest -q
```

Models are not included and not downloaded automatically.
[`docs/models.md`](docs/models.md) lists the exact upstream and SHA256 of every
model these recipes were tuned against.

## Recording runs in chimera

A round is queued on [chimera](https://chimera.chanu.co) — through its MCP
tools (`derive_request`, `redraw_generation`, `deliver_generation`, `repair_generation`) or
`POST /api/v1/requests` — and `chagama work` on the GPU box is the one
command that executes it. Inside the worker, a `generate` row runs the same
code as `chagama generate --request`: it validates the request, submits
the graph, and records the request in the Management API on the way through —
the resolved values, Job registration, image ingest into R2, and a Discord notification
that carries the generation's canonical URL. Running `generate` by hand is a
worker-box operation for replaying a recorded request file, not a way to
queue work.

```bash
uv run chagama work                                        # serve the queue (GPU box)
uv run chagama generate --request request.json --dry-run   # validate a request file, nothing sent
```

`request.json` follows the contract in the chimera repository's
`docs/generation-request.md` (`schema_version` 1). The parts this CLI reads:
`generation.recipe` must be `yukari`; `generation.parameters` carries
the builder arguments (`pose` required; `costume`, `hires`, `denoise`
optional); `generation.prompt` / `negative_prompt`, when set, replace the
built node text wholesale. For attribution, `parameters.character` takes a
character *name* and resolves it against the API (creating it if new), while
`parameters.character_id` passes a known UUID straight through.

Authentication is a Cloudflare Access service token, read from
`$CHIMERA_CF_CLIENT_ID` / `$CHIMERA_CF_CLIENT_SECRET`, then from the untracked
cache `.local/chimera-token`, and only failing both from the 1Password item
`chimera-claude-agent` — which prompts Touch ID, so a successful fetch writes
the cache (mode 0600) and later runs stay unattended. The values are
credentials: they never go into a tracked file.

A run writes `<request>.state.json` next to the request. That file is what
makes re-running the same command safe — it holds the idempotency keys, request
and job ids, and the seeds, so a retry resumes from wherever the last run
stopped instead of creating duplicates.

`chagama import IMAGE... --recipe R --parameters JSON` registers existing image
files as an import request with one job, idempotent by `--idempotency-key`.

## What this is, and what it is not

This is personal tooling, published because the measurements in
[`docs/findings/`](docs/findings/) are worth more written down than
kept — a record of what was tried, what worked, and the several occasions where
the first conclusion turned out to be wrong. The recipe is tuned to a specific
Illustrious-family checkpoint and Yuzuki Yukari. It is installable only to
provide its local CLI; it is not a supported library or service.

Read it as a notebook rather than as a tool you are meant to adopt.

## Derivative work

The recipes draw characters from existing works. That matters more than anything
about the code, so it belongs above the licence and not in a footnote.

| character | source work |
|---|---|
| Yuzuki Yukari | VOICEROID / VOCALOID |

Every right in that character and its source works belongs to the original
creators and rights holders. Nothing here claims any part of it, and
nothing here could grant it.

Consult the original rights holders before redistributing derivative work.

## No licence

No licence is granted for the writing in this repository — the scripts, the
notes, the tag strings and the measurements. It is published to be read, not
used: every right is reserved, so it may not be copied, modified, or
redistributed. GitHub's terms of service still allow viewing and forking it
here, and that is the extent of it.

That is a deliberate choice rather than a missing file. If something in here is
useful to you, the ideas and the measurements are yours to take — take those and
write your own.

To be plain about the boundary: this covers only what was written here. It says
nothing about the characters above, which were never this repository's to
license either way.
