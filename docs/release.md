# Release path

`main` is where PRs land; `production` is what the GPU worker runs. The two
are joined by a promotion PR, the same shape as ochanuco/webull-trading.

## Branches and rulesets

- `main`: deletion and force-push blocked, PR required (no approval), merge
  commits only.
- `production`: PR required with one approval from a code owner and every
  thread resolved, the `production deploy preflight` check required, admins
  may bypass. Nothing pushes to it except the promotion PR's merge.

## Workflows

- `production release PR` runs on every push to `main`. It snapshots
  `main`'s tree onto `release/production` (a `read-tree`, not a merge) and
  opens or updates the PR into `production`. Merging that PR is the release.
  It authenticates as a GitHub App (secrets `APP_ID`, `APP_PRIVATE_KEY`) so
  the PR is not self-authored and its checks run.
- `production preflight` is the required check on that PR: tests with
  `PYTHONPATH=scripts`, then `scripts/delivery_check.py`.
- `deploy worker` runs on `push` to `production` only, on the self-hosted
  runner labelled `gpu-box`. It calls `scripts/worker/deploy.ps1` in the
  standing checkout named by the repository variable `WORKER_CHECKOUT`,
  which moves it to `origin/production` and refreshes the venv. The
  repository is public; limiting the runner to `production` pushes is what
  keeps fork PRs off the box. `deploy.ps1` also re-junctions our node packs
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
- `fetch worker models` runs on a `production` push that changes
  `manifests/worker-models.toml`, or by hand on `production`. It downloads
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

Today a killed worker was not losing renders -- ComfyUI is a separate process
that survives the deploy, and a re-claimed request rejoins the running job
through the `comfy_prompt_id` in its state file. What it lost was the wait.
Draining matters for its own sake once the worker runs inside ComfyUI, because
then the restart takes the render with it.

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

`register-watch.ps1` and `watch.ps1` still register the standalone worker as
its own logon task. The deploy does not call them, and running both at once
means two workers claiming under one explicitly configured `worker_id`; the
default ID includes the process ID. They are there for running
the loop by hand, the way `register-comfyui.ps1` and `register-runner.ps1`
are.

## The box

`scripts/worker/register-runner.ps1 -Token <registration token> -Version <x.y.z>`
downloads the runner, registers it against this repository with the label
`gpu-box`, and installs `run.cmd` as the per-user logon task `actions-runner`
next to `comfyui` and `chagama-watch`. The registration token comes
from `gh api -X POST repos/ochanuco/chagama/actions/runners/registration-token`
and is valid for an hour.

Rows in chimera's requests queue default to `recipe_ref = production`; the
worker serves only the branch it is checked out on, so the box stays on
`production` between releases.
