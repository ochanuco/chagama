"""The worker's composition root: wires a `WorkServices` from concrete adapters.

Shared by `interfaces/cli.py`'s `work` command and `comfy_nodes/yukari_worker`
(same claim loop inside ComfyUI's own process), so the wiring is written once.
"""

from __future__ import annotations

import os
import socket
import sys
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from ..application.deliver import DeliverServices
from ..application.dof import DofServices
from ..application.lens import LensServices
from ..application.photo import PhotoServices
from ..application.stage import StageServices
from ..application.redraw import RedrawServices
from ..application.reference import ReferenceServices
from ..application.generate import GenerateServices, request_graph
from ..application.work import WorkServices, work
from ..domain.generation.fingerprint import prompt_fingerprint
from ..domain.generation.prompt_lint import conflicts
from ..domain.yukari.recipe import identity_tags, render_spec
from ..infrastructure.chimera.client import USER_AGENT, ChimeraClient
from ..infrastructure.comfyui.anima_graph import build_graph as anima_build_graph
from ..infrastructure.comfyui.client import ComfyUIClient
from ..infrastructure.comfyui.progress import ProgressFeed
from ..infrastructure.comfyui.timings import AttemptTimings, EnvCollector
from ..infrastructure.imaging.delivery import graph_from_png_or_none, image_size
from ..infrastructure.imaging.palette import summarize
from ..infrastructure.notifications.discord import DiscordNotifier
from ..infrastructure.persistence.run_state import JsonRunState
from ..infrastructure.locations import data_dir
from ..infrastructure.repository import discover_repository, git_metadata
from ..infrastructure.worker_status import status_writer

# deploy writes this file to ask for a drain and waits for it to go; the
# worker removes it as its last act.
DRAIN_FILE = "worker/drain"

# Must track the `work` subparser's own defaults in interfaces/cli.py by
# hand -- no single source both can read.
DEFAULT_KINDS = ("generate", "redraw", "deliver", "dof", "lens", "photo", "stage",
                 "lighting", "saturation", "reference")


def default_worker_id() -> str:
    """Identify one worker process without conflating co-located workers."""
    return f"{socket.gethostname()}:{os.getpid()}"


# `generation.recipe` -> (RenderSpec builder, ComfyUI graph builder).
RECIPES = {
    "yukari": (render_spec, anima_build_graph),
}

# `generation.recipe` -> its identity_tags(pose, costume) function.
IDENTITY_TAGS = {
    "yukari": identity_tags,
}


def _build_generation_graph(generation: dict, seed: int, prefix: str) -> dict:
    if generation.get("graph"):
        return request_graph(generation, seed, prefix, None, None)
    spec_builder, encode = RECIPES[generation["recipe"]]
    return request_graph(generation, seed, prefix, spec_builder, encode)


def _identity_tags(recipe: str, pose: str, costume: str | None) -> frozenset[str] | None:
    fn = IDENTITY_TAGS.get(recipe)
    if fn is None:
        return None
    return fn(pose) if costume is None else fn(pose, costume)


def _pose_fingerprint(recipe: str, pose: str) -> str | None:
    if recipe not in RECIPES:
        return None
    spec_builder, _ = RECIPES[recipe]
    # Fixed at the default costume: an override is a legitimate request-time
    # choice, so folding it in would make it indistinguishable from drift.
    spec = spec_builder(pose, 0, "fingerprint")
    return prompt_fingerprint(recipe, pose, spec.prompts.positive, spec.prompts.negative)


def build_generate_services(chimera: ChimeraClient, comfyui: ComfyUIClient, notifier: object,
                       repository: Path, repository_metadata) -> GenerateServices:
    return GenerateServices(
        management=chimera,
        comfyui=comfyui,
        state=JsonRunState(),
        notifier=notifier,
        graph_builder=_build_generation_graph,
        git_metadata=repository_metadata,
        conflicts=conflicts,
        output_root=data_dir(repository) / "chimera",
        measure=summarize,
        presets=chimera.get_preset,
        pose_fingerprint=_pose_fingerprint,
        identity_tags=_identity_tags,
    )


def build_redraw_services(chimera: ChimeraClient, comfyui: ComfyUIClient, notifier: object,
                          repository: Path, repository_metadata) -> RedrawServices:
    return RedrawServices(
        management=chimera,
        comfyui=comfyui,
        graph_from_png=graph_from_png_or_none,
        git_metadata=repository_metadata,
        notifier=notifier,
        output_root=data_dir(repository) / "redraw",
    )


def build_deliver_services(chimera: ChimeraClient, comfyui: ComfyUIClient, notifier: object,
                           repository: Path, repository_metadata) -> DeliverServices:
    return DeliverServices(
        management=chimera,
        comfyui=comfyui,
        graph_from_png=graph_from_png_or_none,
        git_metadata=repository_metadata,
        notifier=notifier,
        output_root=data_dir(repository) / "deliver",
        measure=summarize,
    )


def build_dof_services(chimera: ChimeraClient, comfyui: ComfyUIClient, notifier: object,
                       repository: Path, repository_metadata) -> DofServices:
    return DofServices(
        management=chimera,
        comfyui=comfyui,
        git_metadata=repository_metadata,
        notifier=notifier,
        output_root=data_dir(repository) / "dof",
    )


def build_lens_services(chimera: ChimeraClient, comfyui: ComfyUIClient, notifier: object,
                        repository: Path, repository_metadata) -> LensServices:
    return LensServices(
        management=chimera,
        comfyui=comfyui,
        git_metadata=repository_metadata,
        notifier=notifier,
        output_root=data_dir(repository) / "lens",
    )


def build_photo_services(chimera: ChimeraClient, comfyui: ComfyUIClient, notifier: object,
                         repository: Path, repository_metadata) -> PhotoServices:
    return PhotoServices(
        management=chimera,
        comfyui=comfyui,
        git_metadata=repository_metadata,
        notifier=notifier,
        output_root=data_dir(repository) / "photo",
    )


def build_stage_services(chimera: ChimeraClient, comfyui: ComfyUIClient, notifier: object,
                         repository: Path, repository_metadata) -> StageServices:
    return StageServices(
        management=chimera,
        comfyui=comfyui,
        git_metadata=repository_metadata,
        notifier=notifier,
        output_root=data_dir(repository) / "stage",
    )


def build_reference_services(chimera: ChimeraClient, comfyui: ComfyUIClient,
                             emit: Callable[[str], None] = print) -> ReferenceServices:
    def load_tagger():
        from ..infrastructure.reference.anima_tagger import load_tagger
        return load_tagger()

    def load_verifier():
        from ..infrastructure.reference.d1 import load_verifier
        return load_verifier()

    return ReferenceServices(
        management=chimera,
        fetch_image=chimera.fetch_reference_image,
        delete_image=chimera.delete_reference_image,
        free_comfyui=comfyui.free,
        load_tagger=load_tagger,
        load_verifier=load_verifier,
        emit=emit,
    )


def wire_work_services(chimera: ChimeraClient, comfyui: ComfyUIClient, notifier: object,
                        repository: Path, repository_metadata, *, worker_id: str,
                        kinds: tuple[str, ...], hub: bool = True,
                        emit: Callable[[str], None] = print) -> WorkServices:
    """Assemble a `WorkServices` from already-built adapters.

    Shared by `build_work_services` and `interfaces/cli.py`, which already
    has its own `ChimeraClient`.
    """
    hub_factory = None
    if hub:
        from ..infrastructure.chimera.hub import HubConnection, hub_url

        def hub_factory() -> HubConnection:
            headers = {**chimera.credentials(), "User-Agent": USER_AGENT}
            return HubConnection(hub_url(chimera.base_url), headers).open()

    if comfyui.client_id is None:
        comfyui.client_id = uuid.uuid4().hex

    def progress_factory() -> ProgressFeed:
        return ProgressFeed(comfyui.base_url, client_id=comfyui.client_id).open()

    if comfyui.timings is None:
        comfyui.timings = AttemptTimings(
            env_provider=EnvCollector(comfyui, repository_metadata))

    drain_file = data_dir(repository) / DRAIN_FILE
    return WorkServices(
        management=chimera,
        generate_services=build_generate_services(
            chimera, comfyui, notifier, repository, repository_metadata),
        redraw_services=build_redraw_services(
            chimera, comfyui, notifier, repository, repository_metadata),
        deliver_services=build_deliver_services(
            chimera, comfyui, notifier, repository, repository_metadata),
        dof_services=build_dof_services(
            chimera, comfyui, notifier, repository, repository_metadata),
        lens_services=build_lens_services(
            chimera, comfyui, notifier, repository, repository_metadata),
        photo_services=build_photo_services(
            chimera, comfyui, notifier, repository, repository_metadata),
        stage_services=build_stage_services(
            chimera, comfyui, notifier, repository, repository_metadata),
        reference_services=build_reference_services(chimera, comfyui, emit),
        git_metadata=repository_metadata,
        worker_id=worker_id,
        kinds=kinds,
        hub=hub_factory,
        progress_feed=progress_factory,
        timings=comfyui.timings,
        status=status_writer(repository, repository_metadata),
        draining=drain_file.exists,
        drained=lambda: drain_file.unlink(missing_ok=True),
        emit=emit,
        keep_awake=_keep_awake(),
    )


def _keep_awake() -> Callable[[], AbstractContextManager]:
    if sys.platform != "win32":
        return nullcontext
    from ..infrastructure.power import windows

    return windows.keep_awake


def build_work_services(repository: Path, *, worker_id: str,
                        kinds: tuple[str, ...], hub: bool = True,
                        emit: Callable[[str], None] = print) -> WorkServices:
    """Build a complete `WorkServices` from just a repository checkout.

    The entry point for a host with no adapters of its own, such as
    `comfy_nodes/yukari_worker` running inside ComfyUI's process.
    """
    chimera = ChimeraClient(repository)
    comfyui = ComfyUIClient()
    notifier = DiscordNotifier(repository)

    def repository_metadata() -> dict:
        return git_metadata(repository)

    return wire_work_services(
        chimera, comfyui, notifier, repository, repository_metadata,
        worker_id=worker_id, kinds=kinds, hub=hub, emit=emit)


def run(repository: Path | None = None, *, worker_id: str | None = None,
        kinds: tuple[str, ...] | None = None, interval: float = 30,
        hub: bool = True, publish_catalog: bool = True,
        once: bool = False, emit: Callable[[str], None] = print) -> None:
    """Build a `WorkServices` and run the claim loop until stopped."""
    repository = repository or discover_repository()
    services = build_work_services(
        repository,
        worker_id=worker_id or default_worker_id(),
        kinds=kinds or DEFAULT_KINDS,
        hub=hub,
        emit=emit,
    )
    work(services, interval=interval, once=once, publish_catalog=publish_catalog)
