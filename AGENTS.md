# chagama — agent brief

Read this before the first tool call. It is the stuff that is not in the code
and costs a session an hour to rediscover.

## What this repository makes

Pictures of Yuzuki Yukari that are **hand-drawn in look, leg-forward in
composition, in tights**. Those four -- the drawn look, the legs, the tights,
the character -- are the user's whole requirement; a model, a LoRA or a
pipeline stage is kept for as long as it serves them and swapped when
something serves them better.

## What you are working on

The recipe and the worker behind chimera. `chagama` is the executor
that runs on the GPU box: `work` claims request rows from chimera, builds and
records a graph, submits it to ComfyUI, and ingests the result. Nothing renders
from this Mac. A session here drives rounds through chimera's MCP (see
"Renders reach the user through chimera") and edits the recipe in this repo.
**The defaults are the recipe** — every preset was arrived at by rendering, and
the exact prompt can be inspected with `get_catalog_pose` on the MCP or
`chagama <recipe> prompt` locally.

## The pipeline is two stages, and it is the user's

1. **Anima draws.** A new picture starts on `yukari` -- SilvermoonMix Anima
   Turbo (`silvermoonmixAnimaEvolved_v2329BTurbo`, 10 steps, CFG 2 for the
   first 4 then 1.0) with the artist tags
   `(@oshiki hitoshi:1.3), (@yoshikawa hideaki:0.5)` and no style LoRA:
   composition, proportion and hands obey the prompt there, and the artist
   tags set the face and line.
2. **Deliver cuts and decorates the raw pick.** `deliver` cuts a matte,
   composites the backdrop and the outline bands
   onto the Anima pick itself -- no redraw. The raw is drawn on a green key;
   ViTMatte cuts it out, and the bands (`outlines`: a white band, then the
   purple stroke) sit right outside it. It is the recipe default
   (`yukari/delivery_style.py`): the WebUI button and an
   option-less `deliver_generation` both deliver this way.

A `redraw` (method `hires`) is a per-request opt-in the user names, and it
makes a new picture that is delivered by a separate `deliver` request.

Which model draws, which model redraws, and whether a picture is redrawn
at all are the user's decisions. To change one: ask, then change the recipe default
and this section in the same PR.

A new picture starts from the recipe, not from yesterday's render. Render
the pose at its defaults (`plain_render`, or `derive_request` with
`replace_patches: true` and only the patches this request asks for) and
build the round from there.

A new picture an agent queues is a draft: patch `render.width` /
`render.height` to 896x1440 for a tall pose (one whose default canvas is
1024x1640) and to 1024x1280 for a bust or upper-body pose. Only a picture
the user picks is re-rendered with `redraw` method `hires` (2048,
denoise 0.45). A smaller draft is not a shortcut: the hires latent upscale
breaks past about 1.4x (see `docs/findings/composition.md`).

Derive from an existing generation when the user is continuing that
picture: they name the id, or this session already has it in context. Then
the parent's patches come along on purpose, and only then. A patch carries
the decision of the round that made it — a legwear rewrite, a ban, a
canvas — so inheriting one silently applies a look nobody asked for this
time. Read what comes along before you queue it.

Where inherited patches load `anima-sketch-style-chosen.safetensors`
(deleted from the worker), append `render.loras` at strength `0` (the patch
rejects an empty list; a zero-strength LoRA is left out of the graph) and
remove `(sketch style:1.2), ` from `prompt.positive.style`. The recipe's
plain `stand` still wears the pre-official costume.

One recipe is live, `yukari` under `src/chagama/domain/`: the Anima
Turbo checkpoint's identity, costumes, poses and prompt edit order.
`docs/yukari/anima.md` maps where each part lives. `domain/yukari/delivery_style.py`
holds both the delivery identity (backdrop, purple stroke, acceptance band)
every delivered picture wears, read by imaging, catalog, work and cli
alike, and the redraw settings (stage-2 model, sampler,
denoise); it is not itself a recipe.

`redraw` takes an Anima source only (a `UNETLoader` node in its base
graph). A source drawn by any other recipe is delivered as it is and never
redrawn, and a LayerDiffuse base is refused by both `redraw` and `deliver`.

The ComfyUI node encoding is under `infrastructure/comfyui/`.
`chagama yukari prompt --pose …` prints what the recipe sends.

Everything measured is recorded — see "Where information lives" below for
which store. The records are the point of the repository; the scripts are how
they were produced.

Run everything through `uv run` — the client env is pillow, numpy,
opencv-python and scipy, and nothing here imports torch.

## Where information lives (hard rule)

> Research logs are append-only. Code comments are current-state-only.

Four stores, one job each:

```text
src/          current recipe (What). Comments hold only the non-obvious
              constraints that guard today's values — a few lines per
              definition, no history.
experiments/  one observation per JSONL record (seed, render_id, parameter,
              value, outcome accepted|rejected|inconclusive, reason).
              Append-only: a refuted hypothesis gets a NEW record, the old
              one is never rewritten. Schema in experiments/README.md.
              An A/B round with arms and a chimera Request is recorded as a
              chimera Experiment/Run instead of a JSONL record; plain
              observations still append here. The JSONL stays the source
              of truth; chimera holds a derived index kept in sync by
              `scripts/observation_sync.py`.
docs/         conclusions, current state only. Lessons in
              docs/findings/<topic>.md, index in docs/README.md.
              docs/archive/ is the frozen pre-split log: grep, never edit.
tests/        invariants that must hold across models and seeds (prompt
              byte-stability is already pinned by the snapshot contract).
```

Rules that follow from it — enforced by `tests/test_comment_discipline.py`,
which bans dates, render IDs, `1.45 -> 1.3` walk-downs, 「」 quotes and
comment blocks over 8 lines in `src/**`:

- A new failure is an `experiments/` record first, never a code comment. If
  the code needs anything, it is one short line stating the constraint that
  is now in force.
- New knowledge REPLACES a comment; it never stacks on top of one. If old
  and new disagree, the old line is deleted.
- Before changing a pose or tag, grep `experiments/` for the same
  character/pose/parameter. Do not repeat a rejected trial without stating
  what is different this time.
- An experiment result is an observation under its conditions (pose, prompt,
  checkpoint, seed), not a universal rule.
- chimera stays the per-render channel (every render is recorded there);
  `experiments/` is the greppable ledger the comments can point to.
- The `PENDING_CLEANUP` list in the discipline test names the files written
  before this rule; it may only shrink.

## Branch strategy: task branches

Work on a task branch; do not commit directly to protected branches.

Use `dev/<topic>` branches for implementation and merge only after review.

## Look things up; do not read them

`docs/README.md` is the index: one line per doc, and it is short. Start
there, open the one finding that matches, and stop. `docs/archive/` (~150k
tokens) is where the findings came from; reach it only through
`atlas.py find` / `notes <pattern>` and a line range, when a finding's
`(a<n> §...)` pointer is not enough.

For what a recipe sends and what chimera holds, ask the chimera MCP first — it
is registered in this session as `chimera` and answers without touching the
repo:

```text
list_generations                      find a starting ID: published=true is every delivered
                                      look, each carrying its look:<pose> tag
list_catalog                          every recipe's pose / costume names + patch/dial vocabulary (~2k)
get_catalog_pose recipe pose          one pose: canvas, default costume, assembled prompts    (~1k)
get_generation <short_id>             rating, semantic, request prompt + parameters, seed, the request's other generations
get_generation_lineage <short_id>     what it was derived, redrawn or delivered from, and what came after
```

The catalog is what the worker published from its own checkout; it is the
production recipe, not this branch. For an uncommitted change, ask the code:

```bash
uv run scripts/atlas.py                    # every script: role, size, one line  (~1.5k)
uv run scripts/atlas.py docs               # every doc with its size              (~1k)
uv run scripts/atlas.py notes              # findings headings, archive file sizes (~0.5k)
uv run scripts/atlas.py notes <pattern>    # just the sections that match, archive included
uv run scripts/atlas.py find <regex>       # matching lines, each under its heading
uv run chagama yukari prompt --pose bust                    # ~0.6k, not the whole recipe
uv run chagama catalog                                     # what `work` would publish
uv run scripts/delivery_check.py                                  # the delivery identity, verified
```

`atlas.py` reads the tree every time it runs, so unlike a committed index it
cannot be stale. Use it first; then `Read` with `offset`/`limit` on the lines it
gave you. If you find yourself about to read a file over ~5k tokens to answer
something narrow, there is probably a command or a tool for it — and if there
is not, adding one to `atlas.py` is cheaper than the read you were about to do.

## Where the GPU is

**ComfyUI does not run on this Mac, and this Mac no longer talks to it.** The
GPU box runs ComfyUI and `chagama work`; chimera is the only thing a
session here submits to. `COMFYUI_HOST` / `COMFYUI_PORT` matter on the box
only.

The box's address, the ssh alias for a shell on it, and the checkpoint in use
are in `CLAUDE.local.md`, which is not tracked. Read it; do not copy what it
says into anything this repo commits. `docs/remote.md` covers the box itself
(setup, logon tasks, fetching models); `docs/release.md` covers deploys.

## Renders reach the user through chimera; Discord is a side channel

The user reviews renders on chimera (https://chimera.chanu.co), not Discord.
A render that reached Discord but not chimera **is not delivered** — never
close out a prompt on the strength of a Discord post alone. Every render,
including one-off probes and chained passes, is a chimera request row that the
worker executes; that is the only execution path. POSTing to ComfyUI `/prompt`
directly is forbidden. A probe whose graph the recipe cannot build goes in a
`create_request` payload as `generation.graph`.

A round starts from a delivered look and is three MCP calls with a human in
between:

```text
list_generations      the base: `published=true` lists every delivered look
                      with its `look:<pose>` tag. This is where a starting ID
                      comes from -- never from a session's memory.
derive_request        from a rated generation: same recipe, parameters and
                      patches, plus your diff (parameters override, patches
                      appended or replaced). A delivered pick resolves to the
                      raw render it came from. Semantic summary is required.
                      → human rates on chimera →
redraw_generation     optional: the pick re-rendered larger (`hires`); a new
                      picture on the green key.
deliver_generation    the pick, delivered: one ComfyUI graph that cuts a matte
                      and composites the backdrop and outline bands; recorded
                      as a request that refines the source, with its layers
                      (`layer-figure`, `layer-outline`, `layer-backdrop`)
                      as assets.
stage_generation      optional: set a (transparent) delivery's layers on a
                      backdrop pattern with its drop shadow, or in a photo
                      uploaded on chimera, unblurred (kind `stage`).
lighting_generation   optional: light a staged picture's figure and outline
                      by its backdrop: exposure, colour cast, optional side
                      shading (kind `lighting`).
dof_generation        optional: depth of field over a delivered, staged or
                      lit picture's layers (kind `dof`; a photo-set picture
                      blurs by the photo's depth); its output is a delivered
                      picture.
lens_generation       optional: a fisheye (strength > 0) or inverse fisheye
                      (< 0) over a delivered or dof picture, or with
                      shape `sphere` a bulge / pinch of a disc in it (kind `lens`);
                      its output is a delivered picture. Blur before bending:
                      dof refuses a lens picture.
photo_generation      optional: stand a delivery's figure and outline in a
                      photo uploaded on chimera, like an acrylic stand, and
                      blur the scene by depth (kind `photo`).
get_request           status of any of the above; list_requests for the queue.
```

`create_request` is the raw form of all of them (kind + payload) for anything
the dedicated tools do not cover. A round is not closed until the pick has
been delivered: what `generate` and `redraw` leave is the
pre-delivery SaveImage output, and the delivery identity is what `deliver`
adds on top of it. The cutout (`alpha`, and the `depth` a `dof` builds) is
stored as assets of the picture that was delivered, so a delivery is redone from the record without
cutting again.

Discord notification is the worker's job — every ingest and every delivery
posts to the webhook. There is no separate watcher daemon. The webhook is a
credential and lives in `C:\ProgramData\chagama\discord-webhook` on the box; never put it in a
tracked file.

## chimera 連携の不変条件

- 生成の記録は chimera（https://chimera.chanu.co）。実行経路は chimera の
  requests 行を worker（GPU 機の `chagama work`）が実行する一本のみ。
  Mac 側の入口は MCP（`derive_request` / `redraw_generation` /
  `deliver_generation` / `create_request`）か `POST /api/v1/requests`。
  semantic 判断（prompt 組み立て、reference の意味付け、検品）は呼び出し元
  エージェントの仕事で、chimera と worker は実行と記録だけを担う。例外は
  参考画像で、worker が `reference` でタグを提案し、GUI で選ばれたタグから
  `reference_render` が base（pose を持たないキャラクターだけの土台）への
  part patch を組んで（`domain/yukari/reference.py`）
  通常の generate として積む。GUI は prompt を書かず、候補から選ぶだけ。
- ComfyUI `/prompt` への直 POST は禁止。レシピが組めない graph は
  `create_request` の payload に `generation.graph` として入れて渡す（seed と
  SaveImage prefix は job ごとに worker が差し替える）。投稿した graph JSON は
  job に保存され、chimera のレコード単体で再投稿・再現できることがこの規則の
  目的。コードの置き場（`.local/` 含む）は provenance に関与しない。
- graph モードは生 `.replace` の抜け道ではない。recipe が組んだ prompt に
  `.replace` を当ててから `generation.graph` に載せるのは、needle 不在で
  黙って外れる splice の再導入 -- departure は request の `patches`
  （`generation.patches` / `experiment.overrides.patches`）にする。
- `generation.graph` 使用時、`generation.parameters` はビルドに使われず記録
  専用になる。graph に実在しない値を書くと chimera の記録だけが嘘になるので、
  graph に実際に入れた値だけを書く。
- 派生は `derive_request` で作る。親の recipe / parameters / patches を
  引き継ぎ、差分だけを渡す（派生は派生元の prompt + α、失敗の上に重ねず good へ戻る）。納品済みを親に
  渡すと raw まで自動で遡る。`.local/` に request 組み立てスクリプトを書く
  のは、この tool で表せない場合だけ。
- chimera への記録は生成の完了条件。画像だけでなく semantics（各 arm の狙い、
  base からの差分、何を検証する render か）も起票時に書く — `derive_request`
  の `semantic.summary` は必須で、ingest 直後に各 generation へ自動 PUT
  される。作業途中の評価はユーザーが chimera の semantics を見て行う。
  semantics/tag は AI が書いてよい（rating だけが人間専用）。
- 納品した generation は `chagama metadata publish <generation_id>`
  （MCP なら `record_publication`）で記録し、`look:<pose>` タグを付ける。次の
  セッションが起点 ID を引く索引は `list_generations published=true` と
  この `look:<pose>` タグで、prose とメモリは索引にならない。`look:` の値は
  レシピの pose 名（焼き込み前なら焼き込む予定の名前）。
- 事後の追記・上書きは `chagama metadata semantic <generation_id>
  <file.json>`、tag は `chagama metadata tag <generation_id> <name>`、
  納品記録は `chagama metadata publish <generation_id> [--url URL]
  [--idempotency-key KEY]`（key 省略時は呼び出しごとに新しい key）。
  semantic の API は `PUT /api/v1/generations/{id}/semantic`（schema_version:1、
  部分ペイロード可、再 PUT で全置換）。generation_id には short_id も使える。
- idempotency_key は呼び出し元が作る。同じ key の再送は同じ行を返す
  （`created: false`）。失敗した request を「もう一度」なら新しい key を使う。
  worker 側の `<request>.state.json` は worker の再開用で、Mac には無い。
- chimera への全リクエストに User-Agent の明示が必須（urllib のデフォルトは
  Cloudflare が 403/1010 で弾く）。MCP 経由は気にしなくてよい。
- rating（bad/neutral/good）を書くのは人間だけ。エージェントは人間の rating と
  semantic を `get_generation` / `list_generations` で読んで改善を進める。AI が
  画像を開く検品は、rating と semantic だけでは判断できない場合の最終手段
  （トークン消費が理由）。
- Service Token は 1Password `chimera-claude-agent`。取得後は untracked の
  `.local/chimera-token`（0600）にキャッシュされ、以後 Touch ID なしで動く。
  値をトラックされるファイルに書かない。
- request JSON のトップレベル `experiment` ブロック（`experiment_id` /
  `run_id` / `overrides.patches`）は chimera の ExperimentRun 由来の上書きを
  運ぶ経路。`overrides.patches` は `generation.patches` と同じ語彙・同じ
  `validate_request` の検証を通り、`generation.patches` /
  `generation.graph` / `generation.prompt`（`negative_prompt` も同様）とは
  併用できない — 二つの入口で適用順序が曖昧になるため。上書きの正本は
  chimera 側の ExperimentRun であり、worker は取り込んで既存の patch 機構に
  流すだけ。ブロック自体は chimera へ転送しない。worker を通らない実行
  （watch）では `run_id` だけを import Request の作成時に送り、Run との対応は
  chimera が requests.run_id で持つ。

## The delivery identity is a contract, not a preference

The delivery identity (backdrop `#c7e5e9`, the purple stroke, the acceptance
band, in `domain/yukari/delivery_style.py`) is worn by every delivered
picture. Editing it changes every render this repo has ever approved, which
is why `scripts/delivery_check.py` hashes it, from an explicit canonical
payload, and fails on any change it was not told about:

```bash
uv run scripts/delivery_check.py            # check the fingerprint
uv run scripts/delivery_check.py --accept   # record a change that is meant
```

When it fails, nothing is broken — something was changed. `--accept` writes
the new fingerprint into `assets/delivery-fingerprint.json`; rewrite
`docs/findings/delivery.md` to say what the look is now.

A rule that follows from this, learned the expensive way: **a settled design
decision that lives only in prose is a decision the next session does not
get.** The one-garment leg was agreed, written into the notes and into
memory, and applied by throwaway scripts in `.local/` — while the costume
block still built the retired two-layer garment, so another session got
tights under knee-highs straight out of the recipe. If a change is settled,
put it in the blocks (`yukari/costumes.py`'s `COSTUMES`/`LEGWEAR`).

## Working files

`.local/` is untracked (`.gitignore`) and is where analysis scripts, sweeps and
logs go — `uv run .local/foo.py`. It is not the repo: anything worth keeping
moves into `scripts/`, `experiments/` or `docs/findings/`. Request JSON no longer lives
here: a round is queued with `derive_request`, and the record on chimera is the
reproducible artefact.

### Writing docs: rewrite, never append

A measurement, null results included, is an `experiments/` record (append-only).
A doc holds only the conclusion that is true now:

- New knowledge rewrites the finding it touches. A contradicted line is
  replaced, not corrected underneath; the old version is in git.
- A result that must stop a retry is kept as one line under the finding's
  `Does not work` list, with its `(a<n> §...)` pointer.
- `tests/test_docs_budget.py` caps each finding at 150 lines and every other
  doc at 300, requires `docs/README.md` to list every doc, and freezes
  `docs/archive/`. Over the cap means rewrite or split by topic, not raise it.

## No crops while the prompt is being tuned

**Do not crop, cut out or repair a render to deliver it.** If a picture needs a
crop to be acceptable, the prompt has not solved the problem, and the next arm
gets judged against an image the recipe cannot produce.

This was learned on `stand`: a two-figure render had a good left half, and
cutting it would have shipped a pose that never draws one figure. What actually
fixed it was the canvas -- 768 wide leaves no room beside her -- and that was
only reached because the crop was off the table.

Post-processing that sets a value the prompt cannot hold is a different thing
and is still fine: the backdrop is repainted at delivery because the render's
own is unstable under any perturbation. Removing part of the picture is not that.

## Reading images

Rendered images are expensive in context. **Do not open renders to browse them.**
Queue, let the worker ingest, and let the user rate on chimera — they name the
short_id of the one they want, and that is when you look (`get_generation_image`
returns a reduced JPEG; the full PNG is `/g/<short_id>/image`). Measure with
numpy instead where a number will do, and keep in mind that four of this repo's
image statistics have already disagreed with the user's eye and lost.
