# Architecture

The repository uses four lightweight layers. This is a dependency rule, not a
full DDD framework.

```text
interfaces -> application -> domain
                    |            ^
                    v            |
              infrastructure ----+
```

The maintained application layout is:

```text
src/chagama/
├── interfaces/          # argparse and dependency wiring
├── application/         # generate, redraw, repair, deliver and metadata workflows
├── domain/
│   ├── generation/      # shared values and prompt checks
│   ├── repair/          # repair vocabulary: regions, prompt drops, models, LoRAs, ControlNet
│   └── yukari/          # the recipe that draws (identity, costumes, poses, prompt edit
│                        # order) and the delivery identity (backdrop, stroke, acceptance band)
└── infrastructure/
    ├── chimera/         # Management API
    ├── comfyui/         # HTTP client and graph encoders
    ├── imaging/         # delivery post-processing
    ├── notifications/   # Discord side channel
    └── persistence/     # crash-resume state
comfy_nodes/
├── yukari_finalize/      # ComfyUI custom node pack wrapping imaging/ for the redraw and deliver graphs
└── yukari_worker/        # ComfyUI custom node pack hosting the worker's claim loop
```

`infrastructure/imaging/` is shared: `comfy_nodes/yukari_finalize/` (named for
the junction already registered on the worker box) wraps its functions unchanged
as ComfyUI nodes rather than duplicating them.
`YukariMatting` turns the image and the coarse matte into the alpha, and
`YukariForeground` estimates the figure's colour under an alpha; the deliver
graph runs the first once per source picture and the second after every repin.
`YukariSeeThrough` scales the key showing through see-through clothing for a
delivery's `see_through` option, after the colour is solved.

## Domain

`src/chagama/domain/` owns pure generation values and the Yukari
recipe: identity, costumes, poses, prompt edit order and delivery policy. It
does not parse CLI arguments, read environment variables or files, make HTTP
requests, open images, or know ComfyUI node ids.

`RenderSpec` is the boundary between the recipe and a graph adapter. The recipe
decides prompts, sampling parameters, sizes and filenames; the ComfyUI adapter
decides how those values are encoded as nodes and links.

`yukari/` owns a `dials.py`: the named words (`{word: number}`) it
publishes for its own redraw/deliver/repair/patch option values, referencing the
same constants the recipe itself uses rather than duplicating a number.
`application/request_options.py`
and `application/generate.py` resolve a request's word against the source
generation's recipe; `application/catalog.py` publishes the vocabulary as
each recipe's `dials` block.

## Application

`src/chagama/application/` owns use cases: generate and record a request,
redraw, repair or deliver a selected generation, and manage its metadata. It coordinates domain
rules and concrete adapters without reimplementing either.

## Infrastructure

`src/chagama/infrastructure/` owns ComfyUI graphs and HTTP, Chimera,
Discord, image processing and crash-resume persistence. External payload shapes
stay here.

## Interfaces

`src/chagama/interfaces/` owns the single public `chagama` CLI.
Argument parsing stops at this boundary; commands call application use cases.

The worker's composition root -- wiring the generate/redraw/deliver/repair/
masked_redraw services, the hub and progress sockets and the drain sentinel
into a `WorkServices` -- lives in `interfaces/agent.py`, not in the CLI
itself. `comfy_nodes/yukari_worker/` calls the same `agent.run()` to host the
identical claim loop as a thread inside ComfyUI's own process, so a restart
becomes one process instead of two -- at the cost that stopping ComfyUI now
also stops the render in flight, which is why the drain sentinel
(`docs/release.md`) exists. This hosting is off unless
`CHAGAMA_WORKER` is set; a plain ComfyUI install is unaffected.

## Migration rule

`chagama` is the only public application entry point. Files under
`scripts/` are checks and manual utilities that import the package; none is
an alternate entry point.
