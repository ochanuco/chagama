"""Generate-and-record use case."""

from __future__ import annotations

import json
import secrets
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace as spec_replace
from pathlib import Path
from typing import Protocol

from ..domain.generation.models import PromptPair, RenderSpec
from ..domain.generation.patches import apply_patches, parse_patches
from ..domain.generation.prompt_lint import tags as prompt_tags
from ..domain.yukari.dials import DIALS
from ..infrastructure.comfyui.timings import note_ingested
from .ingest import create_job, open_request, rate_generation

PresetFetcher = Callable[[str, str, str, int], dict]

# `generation.recipe` -> its `dials.patches` vocabulary (target -> word ->
# number). Mirrored by request_options.py's _RECIPE_DIALS for the other request kinds.
PATCH_DIALS: dict[str, Mapping[str, Mapping[str, float]]] = {
    "yukari": DIALS.get("patches", {}),
}


def _patch_dials(recipe: str | None) -> Mapping[str, Mapping[str, float]]:
    return PATCH_DIALS.get(recipe, {})


class Management(Protocol):
    def request(self, method: str, path: str, payload: dict | None = None,
                multipart: tuple[dict, str, str, bytes, str] | None = None) -> dict: ...

    def resolve_character(self, name: str) -> str: ...

    def put_semantic(self, generation_id: str, semantic: dict) -> dict: ...


class ComfyUI(Protocol):
    def submit(self, graph: dict) -> str: ...

    def knows(self, prompt_id: str) -> bool: ...

    def wait_for(self, prompt_id: str) -> list[dict]: ...

    def fetch(self, image: dict) -> bytes: ...


class RunState(Protocol):
    def load(self, path: Path) -> dict: ...

    def save(self, path: Path, state: dict) -> None: ...


class Notifier(Protocol):
    def send(self, content: str, filename: str, image: bytes) -> None: ...


GraphBuilder = Callable[[dict, int, str], dict]
ConflictFinder = Callable[[str, str], list[tuple[str, str, str]]]

KNOWN_PARAMETERS = frozenset(
    {"pose", "costume", "hires", "denoise", "character", "character_id",
     "arm", "expression", "legwear", "legwear_state"})

# Read by validate_request and by the published catalog.
RECIPE_REJECTED_PARAMETERS: dict[str, frozenset[str]] = {
    "yukari": frozenset(),
}


@dataclass(frozen=True)
class GenerateServices:
    management: Management
    comfyui: ComfyUI
    state: RunState
    notifier: Notifier
    graph_builder: GraphBuilder
    git_metadata: Callable[[], dict]
    conflicts: ConflictFinder
    output_root: Path
    emit: Callable[[str], None] = print
    measure: Callable[[bytes], dict] | None = None
    presets: PresetFetcher | None = None
    pose_fingerprint: Callable[[str, str], str | None] | None = None
    # (recipe, pose, costume) -> identity tags, bare (weight/parens
    # stripped). None, or the field left unset, skips the identity guard.
    identity_tags: Callable[[str, str, str | None], frozenset[str] | None] | None = None


def _pins_pose(generation: Mapping) -> bool:
    presets = generation.get("presets")
    if not isinstance(presets, list):
        return False
    return any(isinstance(entry, Mapping) and entry.get("kind") == "pose"
               for entry in presets)


def validate_request(req: object) -> None:
    if not isinstance(req, Mapping):
        raise SystemExit("request root must be an object")
    if req.get("schema_version") != 1:
        raise SystemExit("schema_version must be 1")
    request = req.get("request")
    if not isinstance(request, Mapping):
        raise SystemExit("request must be an object")
    generation = req.get("generation")
    if not isinstance(generation, Mapping):
        raise SystemExit("generation must be an object")
    semantic = req.get("semantic")
    if not isinstance(semantic, Mapping):
        raise SystemExit("semantic must be an object")
    parameters = generation.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise SystemExit("generation.parameters must be an object")
    count = request.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise SystemExit("request.count must be an integer >= 1")
    seeds = request.get("seeds")
    if seeds is not None:
        if (not isinstance(seeds, list)
                or any(not isinstance(seed, int) or isinstance(seed, bool)
                       for seed in seeds)):
            raise SystemExit("request.seeds must be an array of integers")
        if len(seeds) != count:
            raise SystemExit(f"len(seeds)={len(seeds)} but count={count}")
    if generation.get("graph"):
        if not isinstance(generation["graph"], dict) or not generation["graph"]:
            raise SystemExit("generation.graph must be a non-empty graph dict")
        if not generation.get("recipe"):
            raise SystemExit("generation.recipe must name what this graph is")
    elif generation.get("recipe") not in ("yukari",):
        raise SystemExit(
            f"recipe {generation.get('recipe')} は使えません。使えるのは "
            "yukari です")
    elif not parameters.get("pose") and not _pins_pose(generation):
        raise SystemExit(
            f"generation.parameters.pose is required for {generation['recipe']} "
            "unless generation.presets pins one")
    elif set(parameters) - KNOWN_PARAMETERS:
        unknown = sorted(set(parameters) - KNOWN_PARAMETERS)
        raise SystemExit(
            f"unknown generation.parameters keys: {unknown} -- annotations "
            "belong in semantic.attributes, executable diffs in "
            "generation.patches"
        )
    elif set(parameters) & RECIPE_REJECTED_PARAMETERS.get(
            generation["recipe"], frozenset()):
        rejected = sorted(set(parameters)
                          & RECIPE_REJECTED_PARAMETERS[generation["recipe"]])
        raise SystemExit(
            f"generation.parameters {rejected} not supported for "
            f"{generation['recipe']}")
    if generation.get("patches") is not None:
        if generation.get("graph"):
            raise SystemExit(
                "generation.patches cannot combine with generation.graph -- "
                "the explicit graph is already the whole spec"
            )
        if generation.get("prompt") or generation.get("negative_prompt"):
            raise SystemExit(
                "generation.patches cannot combine with generation.prompt or "
                "generation.negative_prompt -- ordering between a full "
                "override and a patch would be ambiguous"
            )
        try:
            parse_patches(generation["patches"], _patch_dials(generation.get("recipe")))
        except ValueError as error:
            raise SystemExit(str(error))
    presets = generation.get("presets")
    if presets is not None:
        if not isinstance(presets, list):
            raise SystemExit("generation.presets must be an array")
        for entry in presets:
            if (not isinstance(entry, Mapping)
                    or set(entry) != {"kind", "name", "version"}):
                raise SystemExit(
                    f"generation.presets entry must have exactly kind, name "
                    f"and version: {entry!r}")
            if not isinstance(entry["kind"], str) or not entry["kind"]:
                raise SystemExit(
                    f"generation.presets kind must be a non-empty string: "
                    f"{entry!r}")
            if not isinstance(entry["name"], str) or not entry["name"]:
                raise SystemExit(
                    f"generation.presets name must be a non-empty string: "
                    f"{entry!r}")
            version = entry["version"]
            if (not isinstance(version, int) or isinstance(version, bool)
                    or version < 1):
                raise SystemExit(
                    f"generation.presets version must be an integer >= 1: "
                    f"{entry!r}")
        if generation.get("graph"):
            raise SystemExit(
                "generation.presets cannot combine with generation.graph -- "
                "the explicit graph is already the whole spec"
            )
        if generation.get("prompt") or generation.get("negative_prompt"):
            raise SystemExit(
                "generation.presets cannot combine with generation.prompt or "
                "generation.negative_prompt -- ordering between a full "
                "override and a preset would be ambiguous"
            )
    lint_waiver = generation.get("lint_waiver")
    if lint_waiver is not None and (
            not isinstance(lint_waiver, str) or not lint_waiver):
        raise SystemExit("generation.lint_waiver must be a non-empty string")
    identity_override = generation.get("identity_override")
    if identity_override is not None and (
            not isinstance(identity_override, str) or not identity_override):
        raise SystemExit(
            "generation.identity_override must be a non-empty string")
    experiment = req.get("experiment")
    if experiment is not None:
        if not isinstance(experiment, Mapping):
            raise SystemExit("experiment must be an object")
        experiment_id = experiment.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise SystemExit(
                "experiment.experiment_id must be a non-empty string")
        run_id = experiment.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise SystemExit("experiment.run_id must be a non-empty string")
        overrides = experiment.get("overrides")
        if overrides is not None and not isinstance(overrides, Mapping):
            raise SystemExit("experiment.overrides must be an object")
        override_patches = overrides.get("patches") if overrides else None
        if override_patches is not None:
            if generation.get("patches") is not None:
                raise SystemExit(
                    "experiment.overrides.patches cannot combine with "
                    "generation.patches -- two entrances to the same patch "
                    "mechanism would leave ordering ambiguous"
                )
            if generation.get("graph"):
                raise SystemExit(
                    "experiment.overrides.patches cannot combine with "
                    "generation.graph -- the explicit graph is already the "
                    "whole spec"
                )
            if generation.get("prompt") or generation.get("negative_prompt"):
                raise SystemExit(
                    "experiment.overrides.patches cannot combine with "
                    "generation.prompt or generation.negative_prompt -- "
                    "ordering between a full override and a patch would be "
                    "ambiguous"
                )
            try:
                parse_patches(override_patches, _patch_dials(generation.get("recipe")))
            except ValueError as error:
                raise SystemExit(str(error))
    if not semantic.get("summary"):
        raise SystemExit(
            "request.semantic.summary is required: state this arm's intent "
            "(what it tries, what it varies from the base) -- it is written "
            "to every generation at ingest"
        )


def request_graph(generation: dict, seed: int, prefix: str,
                  spec_builder: Callable[..., RenderSpec],
                  encode: Callable) -> dict:
    """Build a job graph while preserving explicit graphs verbatim.

    Explicit graph mode only rewrites seed and SaveImage prefix; recipe mode
    builds a RenderSpec, applies diffs, then encodes.
    """
    if generation.get("graph"):
        graph = json.loads(json.dumps(generation["graph"]))
        for node in graph.values():
            inputs = node.get("inputs", {})
            if "seed" in inputs:
                inputs["seed"] = seed
            if node.get("class_type") == "SaveImage":
                inputs["filename_prefix"] = prefix
        return graph
    params = generation.get("parameters", {})
    kwargs = {key: params[key] for key in
             ("hires", "denoise", "costume", "expression", "legwear",
              "legwear_state")
             if key in params}
    spec = spec_builder(params["pose"], seed, prefix, **kwargs)
    if generation.get("prompt") or generation.get("negative_prompt"):
        spec = spec_replace(spec, prompts=PromptPair(
            generation.get("prompt") or spec.prompts.positive,
            generation.get("negative_prompt") or spec.prompts.negative))
    if generation.get("patches"):
        spec = apply_patches(spec, parse_patches(
            generation["patches"], _patch_dials(generation.get("recipe"))))
    return encode(spec)


def request_generation(req: Mapping) -> dict:
    """Resolve the generation view graph_builder sees, experiment override included.

    experiment.overrides.patches and generation.patches are mutually
    exclusive; validate_request enforces it.
    """
    generation = dict(req["generation"])
    experiment = req.get("experiment")
    if experiment and generation.get("patches") is None:
        patches = (experiment.get("overrides") or {}).get("patches")
        if patches is not None:
            generation["patches"] = patches
    return generation


def apply_presets(generation: dict, fetch: PresetFetcher) -> dict:
    """Resolve generation.presets into parameters.pose and leading patches.

    Preset patches apply before the request's own generation.patches.
    """
    presets = generation.get("presets")
    if not presets:
        return dict(generation)
    resolved = dict(generation)
    parameters = dict(resolved.get("parameters", {}))
    preset_patches: list = []
    for entry in presets:
        label = f"{generation['recipe']}/{entry['kind']}/{entry['name']}/{entry['version']}"
        if entry["kind"] != "pose":
            raise SystemExit(
                f"preset {label}: kind {entry['kind']!r} not supported by this worker")
        body = fetch(generation["recipe"], entry["kind"], entry["name"], entry["version"])
        pose = (body.get("record") or {}).get("recipe_pose")
        if not isinstance(pose, str) or not pose:
            raise SystemExit(f"preset {label}: record.recipe_pose missing or empty")
        parameters["pose"] = pose
        patches = body.get("patches") or []
        if not isinstance(patches, list):
            raise SystemExit(f"preset {label}: patches must be an array")
        preset_patches.extend(patches)
    resolved["parameters"] = parameters
    combined = preset_patches + list(resolved.get("patches") or [])
    resolved.pop("patches", None)
    if combined:
        resolved["patches"] = combined
    return resolved


def graph_prompts(graph: dict) -> tuple[str | None, str | None]:
    """Read the prompt pair a graph actually carries, following the sampler.

    A hires graph has more than one CLIPTextEncode pair, so text is read
    from whatever the first sampler is wired to, not fixed node ids.
    """
    samplers = sorted(
        (key for key, node in graph.items()
         if "KSampler" in str(node.get("class_type"))),
        key=lambda key: (int("".join(ch for ch in key if ch.isdigit()) or 0), key),
    )
    if not samplers:
        return None, None
    inputs = graph[samplers[0]].get("inputs", {})

    def text(slot: str) -> str | None:
        link = inputs.get(slot)
        if not isinstance(link, list) or not link:
            return None
        return graph.get(link[0], {}).get("inputs", {}).get("text")

    return text("positive"), text("negative")


def resolution_payload(req: dict, git: dict, *,
                       generation: dict | None = None,
                       patches: list | None = None,
                       pose_fingerprint: str | None = None) -> dict:
    # A preset resolves recipe_pose after validation; req's raw generation
    # would misreport what actually rendered.
    generation = req["generation"] if generation is None else generation
    # Preset patches are recorded as the pin; counting them here would
    # apply them twice on anything derived from this Request.
    patches = generation.get("patches") if patches is None else patches
    payload = {
        "raw_instruction": req["request"]["instruction"],
        "recipe": generation["recipe"],
        "parameters": generation.get("parameters", {}),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
    }
    if patches:
        payload["patches"] = patches
    if pose_fingerprint is not None:
        payload["pose_fingerprint"] = pose_fingerprint
    if req.get("references"):
        payload["references"] = [
            {**{key: value for key, value in reference.items()
                if key != "generation_id"},
             "source_generation_id": reference["generation_id"]}
            for reference in req["references"]
        ]
    return payload


def _seeds(req: dict) -> list[int]:
    return (req["request"].get("seeds")
            or [secrets.randbelow(2 ** 32)
                for _ in range(req["request"]["count"])])


def request_file_path(output_root: Path, subdirectory: str, identifier: object,
                      *, label: str = "request id") -> Path:
    """<output_root>/<subdirectory>/<identifier>.json, containment-checked."""
    if (not isinstance(identifier, str) or not identifier
            or identifier in (".", "..")
            or "/" in identifier or "\\" in identifier):
        raise ValueError(f"invalid {label}: {identifier!r}")
    return output_root / subdirectory / f"{identifier}.json"


def _job_key(key_prefix: str | None, index: int) -> str:
    return f"{key_prefix}:job:{index}" if key_prefix is not None else str(uuid.uuid4())


def _generation_key(key_prefix: str | None, index: int, output_index: int) -> str:
    return (f"{key_prefix}:job:{index}:gen:{output_index}"
            if key_prefix is not None else str(uuid.uuid4()))


def _adopt_resend_jobs(state: dict, jobs: list | None, key_prefix: str | None) -> None:
    """Fold a resolution resend's jobs[] into state."""
    if not jobs:
        return
    ordered = sorted(jobs, key=lambda job: job.get("index", 0))
    if "seeds" not in state:
        seeds = [job.get("seed") for job in ordered]
        if seeds and all(isinstance(seed, int) for seed in seeds):
            state["seeds"] = seeds
    for job in ordered:
        index = job.get("index")
        if index is None:
            continue
        while len(state["jobs"]) <= index:
            state["jobs"].append(
                {"idempotency_key": _job_key(key_prefix, len(state["jobs"]))})
        if job.get("status") != "ingested":
            continue
        entry = state["jobs"][index]
        entry["status"] = "ingested"
        if job.get("id"):
            entry["job_id"] = job["id"]
        if job.get("comfy_prompt_id"):
            entry["comfy_prompt_id"] = job["comfy_prompt_id"]
        generations = entry.setdefault("generations", [])
        known = {gen.get("comfy_output_index") for gen in generations}
        for gen in job.get("generations") or []:
            output_index = gen.get("comfy_output_index")
            if not gen.get("id") or output_index is None or output_index in known:
                continue
            generations.append({
                "id": gen["id"], "comfy_output_index": output_index,
                "status": "registered",
                "idempotency_key": _generation_key(key_prefix, index, output_index),
            })


def _output_directory(root: Path, identifier: object) -> Path:
    if (not isinstance(identifier, str) or not identifier
            or identifier in (".", "..")
            or "/" in identifier or "\\" in identifier):
        raise ValueError(f"invalid output identifier: {identifier!r}")
    resolved_root = root.resolve()
    output_dir = (root / identifier).resolve()
    if output_dir.parent != resolved_root:
        raise ValueError(f"output escapes output root: {identifier!r}")
    return output_dir


def _image_output_path(output_dir: Path, filename: object) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"invalid ComfyUI output filename: {filename!r}")
    resolved_dir = output_dir.resolve()
    output_path = (output_dir / filename).resolve()
    if output_path == resolved_dir or not output_path.is_relative_to(resolved_dir):
        raise ValueError(f"ComfyUI output escapes output directory: {filename!r}")
    return output_path


def _check_identity(generation: dict, services: GenerateServices) -> list[str]:
    """Fail before rendering if a patch or a prompt override dropped an
    identity tag and `generation.identity_override` does not excuse it.

    Compares bare tags (weight/parens stripped) of the unpatched recipe
    prompt against the final positive, after presets and patches apply.
    """
    if services.identity_tags is None or generation.get("graph"):
        return []
    params = generation.get("parameters", {})
    pose = params.get("pose")
    if not pose:
        return []
    required = services.identity_tags(
        generation["recipe"], pose, params.get("costume"))
    if not required:
        return []
    probe = services.graph_builder(generation, 0, "chimera-identity-probe")
    positive_text, _ = graph_prompts(probe)
    present = set(prompt_tags(positive_text or ""))
    missing = sorted(required - present)
    if not missing:
        return []
    override = generation.get("identity_override")
    if not override:
        raise SystemExit(
            f"identity tags removed from prompt.positive: {missing} -- set "
            "generation.identity_override (a non-empty reason) to render "
            "anyway"
        )
    services.emit(
        f"  ! identity tags removed: {missing} (identity_override: {override})")
    return missing


def generate(request_path: Path, services: GenerateServices, *,
             dry_run: bool = False, force: bool = False,
             key_prefix: str | None = None,
             request_id: str | None = None) -> dict | None:
    req = json.loads(request_path.read_text(encoding="utf-8"))
    validate_request(req)
    generation = request_generation(req)
    if generation.get("presets") and services.presets is None:
        raise SystemExit(
            "generation.presets pins a preset but this worker has no preset "
            "source wired")
    request_patches = list(generation.get("patches") or [])
    if services.presets is not None:
        generation = apply_presets(generation, services.presets)
    if generation.get("prompt") and generation.get("negative_prompt"):
        hits = services.conflicts(
            generation["prompt"], generation["negative_prompt"])
        waiver = generation.get("lint_waiver")
        if hits and not force:
            for positive, negative, why in hits:
                services.emit(
                    f"positive asks ({positive}) while negative bans "
                    f"({negative})  [{why}]"
                )
            if waiver:
                services.emit(f"  lint waived: {waiver}")
            else:
                raise SystemExit(
                    "prompt contradicts its negative -- fix one side, or "
                    "--force if the pair is deliberate"
                )
    if generation.get("patches"):
        try:
            services.graph_builder(generation, 0, "chimera-probe")
        except ValueError as error:
            raise SystemExit(f"patch compile failed: {error}")
    identity_removed = _check_identity(generation, services)
    git = services.git_metadata()
    fingerprint = None
    if services.pose_fingerprint is not None and not generation.get("graph"):
        pose = generation.get("parameters", {}).get("pose")
        if pose:
            try:
                fingerprint = services.pose_fingerprint(generation["recipe"], pose)
            except Exception as error:
                services.emit(f"  ! pose fingerprint failed: {error}")
    # A Request carrying patches is promotion material, and a promoted version
    # without a fingerprint can never be checked against base drift.
    if request_patches and not fingerprint:
        raise SystemExit(
            "a Request with patches must carry pose_fingerprint, and this "
            "request's pose could not be fingerprinted")
    if dry_run:
        seeds = _seeds(req)
        graph = services.graph_builder(generation, seeds[0], "chimera-dryrun-0")
        services.emit("resolution payload:")
        services.emit(json.dumps(
            resolution_payload(req, git, generation=generation,
                               patches=request_patches,
                               pose_fingerprint=fingerprint),
            indent=2, ensure_ascii=False))
        services.emit(f"seeds: {seeds}")
        # A hires graph has suffixed node ids (6b, 7b); a plain int key dies.
        services.emit("graph nodes: " + str(sorted(
            graph, key=lambda key: (int("".join(
                ch for ch in key if ch.isdigit())), key))))
        positive = graph.get("6", {}).get("inputs", {}).get("text")
        if positive:
            services.emit(f"positive: {positive[:120]}...")
        if generation.get("patches"):
            services.emit(f"patches: {len(generation['patches'])} applied")
        return None

    state_path = request_path.with_suffix(request_path.suffix + ".state.json")
    state = services.state.load(state_path)
    for reference in req.get("references") or []:
        services.management.request(
            "GET", f"/api/v1/generations/{reference['generation_id']}/context")
    request = open_request(
        services.management, request_id=request_id,
        idempotency_key=state["idempotency_key"],
        resolution=resolution_payload(
            req, git, generation=generation, patches=request_patches,
            pose_fingerprint=fingerprint),
        run_id=(req.get("experiment") or {}).get("run_id"))
    state["request_id"] = request["id"]
    _adopt_resend_jobs(state, request.get("jobs"), key_prefix)
    state.setdefault("seeds", _seeds(req))
    services.state.save(state_path, state)
    short = request.get("short_id") or request["id"][:8]
    services.emit(f"request {request['id']} ({short})")

    output_dir = _output_directory(services.output_root, short)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = generation.get("parameters", {})
    character_id = params.get("character_id")
    if not character_id and params.get("character"):
        character_id = services.management.resolve_character(params["character"])

    for index, seed in enumerate(state["seeds"]):
        while len(state["jobs"]) <= index:
            state["jobs"].append(
                {"idempotency_key": _job_key(key_prefix, len(state["jobs"]))})
        job = state["jobs"][index]
        if job.get("status") == "ingested":
            continue
        try:
            if "job_id" not in job:
                created = create_job(
                    services.management, state["request_id"],
                    idempotency_key=job["idempotency_key"],
                    seed=seed, index=index)
                job["job_id"] = created["id"]
                services.state.save(state_path, state)
            if "comfy_prompt_id" in job:
                if services.comfyui.knows(job["comfy_prompt_id"]):
                    services.emit(
                        f"  job {index}: waiting on comfy job "
                        f"{job['comfy_prompt_id']}")
                else:
                    services.emit(
                        f"  job {index}: comfy job {job['comfy_prompt_id']} "
                        f"is unknown to ComfyUI, resubmitting")
                    del job["comfy_prompt_id"]
            if "comfy_prompt_id" not in job:
                graph = services.graph_builder(
                    generation, seed, f"chimera-{short}-{index}")
                job["comfy_prompt_id"] = services.comfyui.submit(graph)
                services.state.save(state_path, state)
                services.management.request(
                    "PATCH", f"/api/v1/jobs/{job['job_id']}",
                    {"status": "queued",
                     "comfy_prompt_id": job["comfy_prompt_id"],
                     "graph": graph},
                )
            services.management.request(
                "PATCH", f"/api/v1/jobs/{job['job_id']}", {"status": "running"})
            images = services.comfyui.wait_for(job["comfy_prompt_id"])
            services.management.request(
                "PATCH", f"/api/v1/jobs/{job['job_id']}", {"status": "completed"})
            job.setdefault("generations", [])
            for output_index, image in enumerate(images):
                output_path = _image_output_path(output_dir, image.get("filename"))
                output = next(
                    (item for item in job["generations"]
                     if item.get("comfy_output_index") == output_index),
                    None,
                )
                if (output is None and output_index < len(job["generations"])
                        and "comfy_output_index"
                        not in job["generations"][output_index]):
                    output = job["generations"][output_index]
                if output is None:
                    output = {
                        "idempotency_key": _generation_key(key_prefix, index, output_index),
                        "comfy_output_index": output_index,
                    }
                    job["generations"].append(output)
                if output.get("status") == "registered" or output.get("id"):
                    continue
                if "idempotency_key" not in output:
                    output["idempotency_key"] = _generation_key(
                        key_prefix, index, output_index)
                output.setdefault("comfy_output_index", output_index)
                services.state.save(state_path, state)
                data = services.comfyui.fetch(image)
                output_path.write_bytes(data)
                palette = None
                if services.measure is not None:
                    try:
                        summary = services.measure(data)
                    except Exception as error:
                        services.emit(f"  ! palette measure failed: {error}")
                    else:
                        palette = {key: round(value, 1) for key, value in summary.items()
                                   if isinstance(value, (int, float))
                                   and not isinstance(value, bool)}
                        palette["verdict"] = (
                            "FAIL: " + "; ".join(summary["fails"])
                            if summary["fails"] else "pass")
                        services.emit(
                            f"  palette {'FAIL' if summary['fails'] else 'pass'}: "
                            f"fig mid {summary['fig_sat_mean']:.1f} "
                            f"light {summary['light_sat']:.1f}")
                meta = {"seed": seed, "original_filename": image["filename"],
                        "comfy_output_index": output_index,
                        "idempotency_key": output["idempotency_key"]}
                if character_id:
                    meta["character_id"] = character_id
                rendered = services.management.request(
                    "POST", f"/api/v1/jobs/{job['job_id']}/generations",
                    multipart=(meta, "image", image["filename"], data, "image/png"),
                )
                rate_generation(services.management, services.emit,
                                rendered["id"], data)
                output.update(
                    {key: rendered[key]
                     for key in ("id", "short_id", "canonical_url")})
                output["status"] = "registered"
                services.state.save(state_path, state)
                semantic = json.loads(json.dumps(req["semantic"]))
                semantic.setdefault("attributes", {}).update(
                    {"seed": seed, **{key: value for key, value in params.items()
                                      if key in ("arm", "pose", "costume",
                                                 "legwear", "legwear_state")}})
                if generation.get("patches"):
                    semantic["attributes"]["patches"] = generation["patches"]
                if identity_removed:
                    semantic["attributes"]["identity_override"] = generation["identity_override"]
                    semantic["attributes"]["identity_removed"] = identity_removed
                if palette:
                    semantic["attributes"]["palette"] = palette
                try:
                    services.management.put_semantic(rendered["id"], semantic)
                except SystemExit as error:
                    services.emit(f"  ! semantic PUT failed: {error}")
                services.notifier.send(
                    f"**JOB ID** `{job['comfy_prompt_id']}`\n"
                    f"**file** `{image['filename']}`\n"
                    f"**chimera** {rendered['canonical_url']}",
                    image["filename"], data,
                )
            services.management.request(
                "PATCH", f"/api/v1/jobs/{job['job_id']}", {"status": "ingested"})
            note_ingested(services.comfyui, job["comfy_prompt_id"])
            job["status"] = "ingested"
        except (RuntimeError, OSError) as error:
            services.emit(f"  ! job {index} (seed {seed}): {error}")
            job["status"] = "failed"
            if "job_id" in job:
                services.management.request(
                    "PATCH", f"/api/v1/jobs/{job['job_id']}", {"status": "failed"})
        services.state.save(state_path, state)

    done = sum(1 for job in state["jobs"] if job.get("status") == "ingested")
    services.emit(f"{done}/{len(state['seeds'])} jobs ingested")
    generation_ids = [gen["id"] for job in state["jobs"]
                      for gen in job.get("generations", []) if gen.get("id")]
    return {"generation_ids": generation_ids}
