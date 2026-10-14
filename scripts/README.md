# Script surface

`scripts/` is not the application layer. The only public application entry
point is:

```bash
uv run comfy-recipes --help
```

The Python package under `src/comfyui_recipes/` owns the generation workflows,
Yukari domain, external adapters, and CLI wiring. Files here are grouped by why
they still need to exist:

- `delivery_check.py`, `palette_check.py`, `prompt_lint.py` — contract and
  request checks.
- `repin.py`, `glitch.py` — manual image utilities.
- `atlas.py` — where things are, without reading them.
- `observation_sync.py` — mirror `experiments/` into chimera.
- `worker/*.ps1`, `fetch-models-windows.ps1` — GPU box setup and deploy.
- `worker_models.py` — the Hugging Face models in
  `manifests/worker-models.toml`, fetched onto the GPU box by the
  `fetch worker models` workflow.
- `pixal3d_run.py` — a chimera turnaround sheet through the core Pixal3D
  multi-view graph into a GLB, run on the GPU box by the `pixal3d` workflow.

If a script starts coordinating a recorded batch, it belongs in
`application/` behind `comfy-recipes`; if it expresses Yukari policy, it
belongs in `domain/yukari/`.
