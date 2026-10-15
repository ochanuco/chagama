# The GPU box

ComfyUI and the worker run on the Windows GPU box; the Mac only edits recipes
and reads chimera. The box's address and ssh alias are in the untracked
`CLAUDE.local.md`. Deploys and restarts are in [release.md](release.md).

## Setting it up

The box runs releases under `C:\apps\chagama` (see [release.md](release.md)).
`scripts/worker/setup-apps.ps1`, run once by hand, elevated, with the worker
stopped, prepares it:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\worker\setup-apps.ps1 [-HfTokenFile <file>]
```

It creates the data directory `C:\apps\chagama\data` and the config
directory `C:\ProgramData\chagama` (Modify for the user, Full for
Administrators and SYSTEM, no inheritance), moves the old checkout's
`.local\_nogit\*` into data, copies `chimera-token` and `discord-webhook`
into config (plus `hf-token` when given), writes `worker.json` with the
portable ComfyUI root, and runs `register-comfyui.ps1`. The secrets are never
tracked.

`register-comfyui.ps1 -PortableRoot <dir>` registers `run-comfyui.ps1` as the
logon task `comfyui`, so the worker runs as a thread inside ComfyUI and a
reboot brings both back once the user logs on. The launch arguments live in
the script; changing one means editing it and re-running it (it also removes
the old `COMFYUI_RECIPES_WORKER` user variable; `src/chagama/envcompat.py`
still reads the pre-rename names until the box is migrated);
`run_nvidia_gpu.bat` is not used. The box serves the rows whose `recipe_ref`
equals the ref of the release it runs (see
[queueing.md](queueing.md#worker)).

Two things that only show up over `ssh comfyui-worker`:

- Reparse points do not resolve in that session. The WinGet `uv.exe` link
  fails with "no application is associated", and uv's
  `cpython-3.12-windows-x86_64-none` alias is a junction that fails with
  "untrusted mount point". Call the package's own `uv.exe` and point
  `uv venv` at the versioned `cpython-3.12.<patch>-...` directory instead.
- The locale encoding is cp932. Text I/O in this repo passes
  `encoding="utf-8"` explicitly, and the wrapper sets `PYTHONUTF8=1` for
  the CLI's stdout.

## Content rating

Every image the worker uploads to chimera is also rated by the WD tagger
(`wd-swinv2-tagger-v3`, ONNX on CPU) and the raw rating and general-tag
probabilities are PUT to `/api/v1/generations/{id}/safety`. Thresholds and
badges are chimera's. A failure -- no `onnxruntime`, no network for the first
download, chimera not yet serving the endpoint -- is only logged and never
fails the job. The model downloads on first use into
`models/wd-tagger` in the data directory (override with `CHAGAMA_MODEL_DIR`),
sha256-checked against the pinned revision. `chagama safety rate <short_id>...`
and `chagama safety backfill [--published] [--limit N]` rate existing
generations.

## Getting models onto it

A Hugging Face file listed in `manifests/worker-models.toml` gets there by
code: the `fetch worker models` workflow runs `scripts/worker_models.py` on
the box by hand (`workflow_dispatch`, with `dry_run` to only check). It
checks every file upstream before downloading any, resumes from `.part`,
keeps the box awake while it runs, and renames a file into place only when
its SHA256 matches the one Hugging Face publishes (and `sha256` in the
manifest, when pinned). `HF_TOKEN`, read from `hf-token` in the config directory, is needed only for a
gated repo. The rest of this section is for files outside that manifest.

They have to be on that machine's disk, and nothing in the ComfyUI HTTP API can
put them there. `/experiment/models` is read-only, there is no download
endpoint, and no standard node fetches a URL. Short of ComfyUI-Manager, that
leaves running a command over there.

[models.md](models.md) is the inventory to work from, with the Hugging Face
repo or Civitai version each model came from and its SHA256. `scripts/fetch-models-windows.ps1` automates that pull for a
Windows portable install, but only for hassaku-il-v22 and two LoRAs, so anything
else still needs a command by hand.

For a checkpoint mirrored to Hugging Face in diffusers layout:

```powershell
# on the Windows machine, from the portable install root
.\python_embeded\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('John6666/hassaku-xl-illustrious-v22-sdxl', local_dir=r'.\ComfyUI\models\diffusers\hassaku-il-v22')"
```

The `hf` CLI is a trap on a portable install. It imports `venv` on startup,
which the embedded Python does not ship, so it dies before downloading
anything; `snapshot_download` never touches that import.

`DiffusersLoader` lists a directory the moment its `model_index.json` lands,
which is long before the 5GB `unet/` does. A model showing up in
`/object_info/DiffusersLoader` is therefore not a signal that it can be loaded.
Loading a half-fetched folder fails with `'NoneType' object has no attribute
'lower'`, which is ComfyUI reporting a missing file badly.

### A download must outlive the ssh session

A process started over `ssh comfyui-worker` — `Start-Process` included — dies
the moment the session closes, and a `hf_hub_download` or `curl` that was
half-way through a 6GB file just stops without an error (measured three
times: 20s, 45s and 60s of session gave 1GB, 2.7GB and 4.7GB). Register the
fetch as a scheduled task and run it from there instead:

```powershell
schtasks /Create /TN fetch_ckpts /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\...\fetch_ckpts.ps1" /SC ONCE /ST 00:00 /F
schtasks /Run /TN fetch_ckpts
```

The script itself is a `curl.exe -L -C - --retry 5 --retry-all-errors` loop
per file into `<name>.part`, renamed on exit 0; `.local/ab2/fetch_ckpts.ps1`
is the one that fetched the 2026-09-05 checkpoints. Civitai's
`/api/download/models/<version>` worked anonymously for the Nova models; a
`CIVITAI_TOKEN` is only needed for gated ones. Watch the `.part` size from
here, not the log line — the log only moves when a file finishes.

## Checking whether the far end is up

Windows blocks ICMP by default, so `ping` fails against a machine that is up and
serving. Check the port, or ask `/system_stats`:

```bash
curl -s http://$COMFYUI_HOST:8188/system_stats
```

`/object_info` is also the honest answer to "does it have the models" — an empty
`CheckpointLoaderSimple` list means an empty `models/checkpoints`, whatever the
disk looks like from over here.

## Generation timings

Every claimed request ends with a `PUT /api/v1/requests/<id>/timings`
(version `v2`, source `worker`) after the final status PATCH, for `done` and
`failed` attempts. A failure to send is logged and never changes the
outcome. A worker killed mid-request sends nothing, and `released` is not
sent. Timestamps are epoch milliseconds on the box clock.

What one attempt records:

- `claimed_at` when the claim response returned, `finished_at` when the
  final PATCH returned.
- Per prompt: `submitted_at`, `outputs_ready_at` (the poll that saw the
  finished history entry) and `ingested_at` (outputs registered in chimera;
  null for the pose-detect prompt). `execution_start_at`, `execution_end_at`,
  the cached node ids and `status` come from the history entry's
  `status.messages`. A prompt picked up again after a restart is
  `resumed: true` with no `submitted_at`. `purpose` is `render`, or
  `pose_detect` for repair's DWPose pass.
- Per node: `started_at` at the `executing` event for it, `ended_at` at the
  next `executing`, `steps_total` and `step_ms` (gaps between consecutive
  `progress` events, the first measured from the node's start) from ComfyUI's
  `/ws` broadcast. Cached nodes carry no times. `cold_load` is true when a
  loader node (UNET, CLIP, VAE, checkpoint, LoRA, ControlNet, diffusers)
  ran uncached in any prompt.
- `env`: ComfyUI and PyTorch versions, launch `argv` and the `attention`
  derived from it, worker commit and dirtiness, GPU name and driver
  (`nvidia-smi`). Read once, and again after the `/ws` feed drops or while
  ComfyUI was unreachable.

The `/ws` feed runs whenever the worker runs, hub or not. It connects with
`?clientId=<id>` and every `POST /prompt` carries the same `client_id` (one
random id per worker process), because ComfyUI sends the `executing` family
only to the submitting client. `progress_state` node transitions set a node's
start and, with precedence, its end. Only `progress` events are relayed to the
hub. Binary preview frames are ignored.

### Role names

Every submitted graph is copied and each node without a `_meta.title`
gets one, so a role is the same across graph revisions:

| Node | Role |
| --- | --- |
| sampler fed from `EmptyLatentImage` | `base_sampler` |
| sampler fed through a latent upscale | `hires_sampler` |
| sampler fed from a `VAEEncode` | `redraw_sampler` |
| a further sampler on the same chain | `<group>_sampler_stage2`, `_stage3` |
| `LatentUpscale` / `LatentUpscaleBy` | `hires_upscale` |
| `CLIPTextEncode` on a sampler's `positive` / `negative` | `positive_prompt` / `negative_prompt` |
| `VAEDecode` feeding a save | `vae_decode` |
| loaders | `unet_loader`, `clip_loader`, `vae_loader`, `checkpoint_loader`, `lora_loader`, `controlnet_loader`, `diffusers_loader` |
| `SaveImage` | `save_image` |
| anything else | snake_case class name |

A name used by more than one node gets `_2`, `_3` in topological order.
