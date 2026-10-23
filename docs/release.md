# Release path

`main` is where PRs land, and a merge is a release: the push to `main`
builds that commit into a GitHub Release and the GPU box runs it. There is no
release branch and no git checkout on the box. A deploy stops for approval
only when its plan finds a reason to (below).

## Branches and rulesets

- `main`: deletion and force-push blocked, PR required (no approval), merge
  commits only.

## Workflows

- `deploy` runs on every push to `main` except one that touches only paths
  the worker never reads (`paths-ignore` in `deploy.yml`: docs, tests,
  experiments, Markdown, `.github/`), and as a `workflow_dispatch` with the
  input `ref` (`main`, the default, or a commit SHA on main; an older one is
  a rollback). Runs go one at a time, and of the runs waiting behind one only
  the newest is kept. Its jobs, in order:
  - `plan` (`deploy-plan.yml` in ochanuco/.github) pins the ref to a SHA
    (`main` to the commit the run started on) and writes to the Job Summary
    the PRs since the last deploy, what a rollback drops, the changed files,
    and a notice for a change that needs a step by hand (the `notices` list
    in `deploy.yml`). It asks for approval when there is no earlier deploy
    to compare with, when the run is a rollback or the last deployed commit
    is not an ancestor of this one, and when a `notices` path changed. The
    last deploy is the latest successful run's name, `deploy chagama @ <ref>`.
  - `approve` runs only when `plan` asked, in the Environment
    `production-approval`, whose required reviewer approves after reading
    that summary. The Environment holds no secrets.
  - `build` zips `src`, `comfy_nodes`, `manifests`, `assets`, `scripts`,
    `deploy`, `ops` and `pyproject.toml` at the SHA, plus a `release.json`
    (`scripts/release/build.sh`). The version is `r<run number>-<sha7>`;
    `release.json` holds `version`, `commit` and `ref: "production"`.
  - `deploy` runs in the Environment `production`, which has no reviewers
    and holds the 1Password service account token `OP_SA_DEPLOY`. It publishes
    the zip as the GitHub Release `<version>` with the asset `chagama.zip`
    (`scripts/release/publish.sh`: draft, then publish; releases are
    immutable), or reuses a published release of the same commit. Then
    `.github/actions/gpu-box-app` dispatches `gpu-box-app.yml` in
    ochanuco/chabatake (`app=chagama`, `action=deploy`, the version) with a
    GitHub App token read from 1Password, and waits for it. Only a
    conclusion of `success` passes. Last, `scripts/release/prune.sh 3`
    deletes every release but the 3 newest, tags included.
  The repository is public. Both Environments accept only `main`, and the
  main-only ref check in `plan` refuses any other commit; together they keep
  other code off the box. All box workflows share the concurrency group
  `gpu-box-chagama`.
- `fetch worker models` is a manual `workflow_dispatch` with the input
  `dry_run`, in the Environment `production`. Run it after a deploy that
  changes `manifests/worker-models.toml`. It runs the `fetch-models` op of
  `gpu-box-app` (`ops/fetch-models.ps1` of the running release;
  `fetch-models-dry-run` with `-DryRun`), which downloads the listed Hugging
  Face and Civitai models into the box's ComfyUI `models/`
  (`scripts/worker_models.py`) with `HF_TOKEN` and `CIVITAI_TOKEN` read from
  `C:\ProgramData\chagama\hf-token` and `civitai-token`. It restarts
  nothing.
- `restart worker comfyui` is a manual `workflow_dispatch` that runs the
  `restart` action of `gpu-box-app`: the running release's `stop.ps1`, the
  `comfyui` task, then its `check.ps1`.

## What the box does

`gpu-box-app` belongs to chabatake. For a deploy it wakes the box, checks that
the release is published and that the asset's sha256 equals the API digest,
and extracts it to `C:\apps\chagama\releases\<version>\`. Then:

1. `deploy\prepare.ps1` of the new release runs while ComfyUI keeps serving:
   it fetches the pinned ComfyUI tag and node commits (no checkout) and
   prefetches the wd-tagger and matting models.
2. `deploy\stop.ps1` of the running release drains the worker, stops the
   `comfyui` task and kills leftovers.
3. The junction `C:\apps\chagama\current` flips to the new release and the
   `comfyui` task starts.
4. `deploy\check.ps1` of the new release waits up to 9 minutes for
   `http://127.0.0.1:8188/system_stats` to answer 200 and for
   `data\worker\status.json` to report the release's version with a live pid.

The box keeps the 3 newest releases. The hooks receive `APP_ROOT`,
`APP_DATA`, `APP_CONFIG`, `APP_ACTION`, `APP_FROM_*` and `APP_TO_*`.

### A failed check

When `check.ps1` fails, the deployer stops the new release, flips `current`
back, starts the `comfyui` task and runs the previous release's `check.ps1`.
The dispatched run still ends as a failure, so the `deploy` job fails.

### Rollback

Run `deploy` by `workflow_dispatch` with `ref` set to an older commit on main.
Any commit on main works: a release that was pruned is built again. A
rollback always asks for approval.

## Draining before a deploy

`stop.ps1` asks the worker to stop before it stops the task: it writes
`C:\apps\chagama\data\worker\drain` and waits up to 300 seconds. The worker
stops claiming, finishes the request it is running, and deletes the file on
its way out, so the wait ends on the worker's own word rather than on a
clock. It skips the drain when ComfyUI is not running. Whatever is left after
the wait is killed. A drain that times out deletes the file itself -- one left
behind would drain the next worker the moment it started.

What the bound has to cover is one request, not the queue: the worker claims
nothing new once the file appears. A sketch seed takes about 45 seconds, a
redraw about 90, a four-seed request a few minutes.

The drain is what keeps the render in flight. The worker runs inside ComfyUI,
so a render still going when the task stops dies with the process and its
request has to be claimed and drawn again. A drain that times out leaves that
request `running` until it is handed back to the queue.

## Where the worker runs

The claim loop runs in ComfyUI's embedded Python, so what it imports has to be
installed there. `run-comfyui.ps1` adds only the gap -- `websockets`, and
`onnxruntime`, `pymatting` and `timm` (without its dependencies) when missing
-- because numpy, Pillow, OpenCV, SciPy and torch already come with the box and
reinstalling them into a working ComfyUI is how it stops working.

`scripts/worker/run-comfyui.ps1` is the action of the `comfyui` scheduled
task. On every start it:

- junctions each `comfy_nodes\<pack>` into ComfyUI's `custom_nodes` through
  `C:\apps\chagama\current\comfy_nodes\<pack>`;
- brings ComfyUI core and the third-party node packs to the pins in the
  release's `manifests/worker-nodes.toml` (`sync-comfyui.ps1`,
  `sync-nodes.ps1`) when the stamp `data\worker\comfyui-applied.txt` differs
  from them, so a ComfyUI upgrade is a PR that bumps the tag, and a rollback
  of `current` brings core and nodes back to the previous pins (packages pip
  added stay);
- runs ComfyUI with `CHAGAMA_WORKER=1`, `CHAGAMA_DATA` and `CHAGAMA_CONFIG`.

The claim loop is a thread inside ComfyUI's own process, started by
`comfy_nodes/yukari_worker/` when `CHAGAMA_WORKER` is set. The thread waits
for ComfyUI's own HTTP to answer before it claims anything -- custom nodes are
imported while the server is still coming up, and a request claimed then
would be submitted to a port with nothing behind it.

The worker inside ComfyUI is `<host>:comfyui` unless `CHAGAMA_WORKER_ID`
says otherwise, so the same ID comes back after a restart. On startup it
hands back to the queue whatever is still `running` under its ID -- a request
a killed ComfyUI was drawing -- instead of leaving it until chimera finds the
heartbeat stale.

## Where things live on the box

| What | Where |
| --- | --- |
| Releases, and `current` pointing at the running one | `C:\apps\chagama\releases\<version>\`, `C:\apps\chagama\current` |
| Data: render outputs, `models/wd-tagger`, worker logs, `drain`, `status.json` | `C:\apps\chagama\data` (`CHAGAMA_DATA`) |
| Config and secrets: `chimera-token`, `discord-webhook`, `hf-token`, `civitai-token`, `worker.json` (`comfyui_portable_root`) | `C:\ProgramData\chagama` (`CHAGAMA_CONFIG`) |

Outside the box the defaults are `<repo>/.local/_nogit` for data and
`<repo>/.local` for config (`src/chagama/infrastructure/locations.py`).

## Which rows the worker serves

Rows in chimera's requests queue default to `recipe_ref = production`. The
worker serves only rows whose `recipe_ref` equals its ref: the `ref` in
`release.json`, which is `production` for every release, or the git branch
when it runs from a checkout.
