"""Turn chimera's pending ExperimentRuns into generations.

The bridge between a Cloudflare-hosted Agent (no render-host access) and a
local `chagama generate`. Evaluation, decision and promotion are out
of scope here.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .generate import GenerateServices, generate, request_file_path, validate_request

PENDING_RUNS_PATH = "/api/v1/experiment-runs?pending=true"


class Management(Protocol):
    def request(self, method: str, path: str, payload: dict | None = None,
                multipart: tuple[dict, str, str, bytes, str] | None = None) -> dict: ...


class SkipRun(ValueError):
    """A pending ExperimentRun cannot be turned into a request."""


@dataclass(frozen=True)
class WatchServices:
    management: Management
    generate_services: GenerateServices
    generate: Callable[..., None] = generate
    emit: Callable[[str], None] = print
    sleep: Callable[[float], None] = time.sleep


def _fallback_summary(item: Mapping) -> str:
    experiment = item.get("experiment") or {}
    name = experiment.get("name") or experiment.get("id") or "experiment"
    run_index = item.get("run_index")
    suffix = "" if run_index is None else f" run #{run_index}"
    return f"{name}{suffix}"


def build_request(item: Mapping) -> dict:
    """Build a chagama request.json body from one pending Run item.

    Raises SkipRun when the item's experiment names no base_recipe. Callers
    should also run the result through validate_request.
    """
    experiment = item.get("experiment") or {}
    base_recipe = experiment.get("base_recipe")
    if not base_recipe:
        raise SkipRun(f"run {item.get('id')}: experiment has no base_recipe")
    base_parameters = experiment.get("base_parameters")
    if base_parameters is not None and not isinstance(base_parameters, Mapping):
        raise SkipRun(
            f"run {item.get('id')}: base_parameters must be a mapping, got "
            f"{type(base_parameters).__name__}")
    parameters = dict(base_parameters or {})
    count = parameters.pop("count", 1)
    summary = item.get("objective") or _fallback_summary(item)
    return {
        "schema_version": 1,
        "request": {"instruction": summary, "count": count},
        "generation": {"recipe": base_recipe, "parameters": parameters},
        "semantic": {"summary": summary},
        "experiment": {
            "experiment_id": item.get("experiment_id"),
            "run_id": item.get("id"),
            "overrides": item.get("overrides"),
        },
    }


def _request_path(output_root: Path, run_id: object) -> Path:
    # A stable path per Run (not a fresh temp file each poll) is what makes
    # the CLI's <request>.state.json idempotency-key cache work on a resend.
    try:
        return request_file_path(output_root, "experiment-runs", run_id, label="run id")
    except ValueError as error:
        raise SkipRun(str(error)) from error


def poll_once(services: WatchServices, *, dry_run: bool = False) -> None:
    # Isolated per Run so one malformed/failing Run cannot stop the rest.
    # KeyboardInterrupt is a BaseException, so it is not caught here and
    # keeps propagating to watch()'s loop.
    response = services.management.request("GET", PENDING_RUNS_PATH)
    for item in response.get("items", []):
        run_id = item.get("id")
        try:
            request = build_request(item)
            validate_request(request)
            request_path = _request_path(
                services.generate_services.output_root, run_id)
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(
                json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")
            services.generate(
                request_path, services.generate_services, dry_run=dry_run)
        except (SystemExit, Exception) as error:
            services.emit(f"skip run {run_id}: {error}")


def watch(services: WatchServices, *, interval: float = 30, once: bool = False,
          dry_run: bool = False) -> None:
    try:
        while True:
            poll_once(services, dry_run=dry_run)
            if once:
                return
            services.sleep(interval)
    except KeyboardInterrupt:
        services.emit("watch stopped")
