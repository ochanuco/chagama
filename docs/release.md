# Release path

`main` is where PRs land, and a merge is a release: the push to `main`
deploys that commit to the GPU worker. There is no release branch. A deploy
stops for approval only when its plan finds a reason to (below).

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
  - `wake` asks the Wake-on-LAN service to wake the box
    (`.github/deploy/wake.sh`). It runs in the Environment `production`,
    which has no reviewers and holds the 1Password service account token
    `OP_SA_DEPLOY`; the service's address and Cloudflare Access token come
    from 1Password through `.github/deploy/wake.env.op`. It does not wait for
    the box to come up: the next job stays queued until the box's runner
    connects.
  - `deploy` runs on the self-hosted runner labelled `gpu-box`, in the same
    Environment. It deploys the SHA `plan` pinned, and shares its
    concurrency group with `restart worker comfyui`.
  The repository is public. Both Environments accept only `main`, and the
  main-only ref check (in `plan` and `scripts/worker/resolve-ref.ps1`)
  refuses any other commit; together they keep other code off the box.
  The `deploy` job calls `scripts/worker/deploy.ps1` in the standing checkout named by
  the repository variable `WORKER_CHECKOUT` with the resolved SHA. That
  moves the checkout to that commit on the local branch `production` -- the
  worker serves only requests whose `recipe_ref` equals the branch it is on,
  so the name stays -- and refreshes the venv. `deploy.ps1` also re-junctions our node packs
  into the ComfyUI install named by the repository variable `COMFYUI_ROOT`,
  moves that install to the ComfyUI tag pinned in
  `manifests/worker-nodes.toml` (`scripts/worker/sync-comfyui.ps1`: fetch,
  detached forced checkout, its `requirements.txt` into the portable Python
  when the tag moved -- so a ComfyUI upgrade is a PR that bumps the tag),
  brings the third-party node packs under its `custom_nodes/` to the commits
  pinned in the same file (`scripts/worker/sync-nodes.ps1`:
  clone, detached checkout, `requirements.txt` into the portable Python when
  an entry moved, then that entry's `pip` list -- packages a node needs
  that its own `requirements.txt` cannot supply, whether because
  `install_requirements` is off or because the list omits one), and
  restarts ComfyUI
  (`scripts/worker/restart-comfyui.ps1`) every time. That restart is not an
  optimisation to skip: the drain below ends the claim loop, and the loop is
  a thread in that process, so nothing else brings the worker back. A deploy
  that skipped it would leave the box with no worker at all.
- `fetch worker models` is a manual `workflow_dispatch` with the same `ref`
  input, in the Environment `production`. Run it after a deploy that
  changes `manifests/worker-models.toml`; it does not wake the box, so run
  it while the box is still up from that deploy. It downloads
  the listed Hugging Face models into the box's ComfyUI `models/`
  (`scripts/worker_models.py`) and does not restart anything.
- `pixal3d` is a manual `workflow_dispatch` (any ref) that runs
  `scripts/pixal3d_run.py` on the box: it splits a chimera turnaround sheet
  into its figures, sends four of them through the core Pixal3D multi-view
  graph on the local ComfyUI, and keeps the GLB, the views and the base-color
  texture as the run's artifact. It queues behind the worker's renders in the
  same ComfyUI.
- `restart worker comfyui` is a manual `workflow_dispatch` that runs the same
  restart on the box, for changes made outside a deploy.

## Draining before a deploy

`deploy.ps1` asks the worker to stop before it moves the checkout: it writes
`.local/_nogit/worker/drain` in the checkout and waits. The worker stops
claiming, finishes the request it is running, and deletes the file on its way
out, so the wait ends on the worker's own word rather than on a clock. Only
then does the checkout move; the `taskkill` that follows is for a worker
already gone or wedged. A drain that times out deletes the file itself -- one
left behind would drain the next worker the moment it started.

The bound is `-DrainSeconds`, 300 by default. What it has to cover is one
request, not the queue: the worker claims nothing new once the file appears.
A sketch seed takes about 45 seconds, a redraw about 90, a four-seed request a
few minutes. The ceiling above it is the deploy job's own `timeout-minutes:
15`, which also has to hold `git fetch` and the `uv pip install`.

The drain is what keeps the render in flight. The worker runs inside
ComfyUI, and `deploy.ps1` restarts ComfyUI on every deploy, so a render still
going when the restart comes dies with the process and its request has to be
claimed and drawn again. A drain that times out leaves that request `running`
until it is handed back to the queue.

## Where the worker runs

The claim loop runs in ComfyUI's embedded Python, not the checkout's
`.venv`, so what it imports has to be installed there. The deploy adds only
the gap -- `websockets` -- because numpy, Pillow, OpenCV and SciPy already
come with the box and reinstalling them into a working ComfyUI is how it
stops working.

The claim loop is a thread inside ComfyUI's own process, started by
`comfy_nodes/yukari_worker/` when `CHAGAMA_WORKER` is set. The
variable is a user environment variable on the box, written by
`register-comfyui.ps1`, because a scheduled task inherits the user
environment and not a shell's. The thread waits for ComfyUI's own HTTP to
answer before it claims anything -- custom nodes are imported while the
server is still coming up, and a request claimed then would be submitted to
a port with nothing behind it.

The worker inside ComfyUI is `<host>:comfyui` unless `CHAGAMA_WORKER_ID`
says otherwise, so the same ID comes back after a restart. On startup it
hands back to the queue whatever is still `running` under its ID -- a request
a killed ComfyUI was drawing -- instead of leaving it until chimera finds the
heartbeat stale.

`register-watch.ps1` and `watch.ps1` still register the standalone worker as
its own logon task. The deploy does not call them, and running both at once
means two workers claiming under one explicitly configured `worker_id`; the
standalone worker's default ID includes the process ID. They are there for
running the loop by hand, the way `register-comfyui.ps1` and
`register-runner.ps1` are.

## The box

`scripts/worker/register-runner.ps1 -Token <registration token> -Version <x.y.z>`
downloads the runner, registers it against this repository with the label
`gpu-box`, and installs `run.cmd` as the per-user logon task `actions-runner`
next to `comfyui` and `chagama-watch`. The registration token comes
from `gh api -X POST repos/ochanuco/chagama/actions/runners/registration-token`
and is valid for an hour.

Rows in chimera's requests queue default to `recipe_ref = production`; the
worker serves only the branch it is checked out on, so the box stays on its
local branch `production` between deploys.
