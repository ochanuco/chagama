"""Claim, execute and report on chimera's `requests` queue.

The wire contract is chimera's docs/worker-protocol.md.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.parse import quote

from .catalog import publish_catalog as publish_catalog_document
from .deliver import DeliverServices, deliver
from .dof import DofServices, dof
from .generate import GenerateServices, generate, request_file_path
from .lens import LensServices, lens
from .masked_redraw import MaskedRedrawServices, masked_redraw
from .photo import PhotoServices, photo
from .stage import StageServices, lighting, saturation, stage
from .redraw import RedrawServices, redraw
from .repair import RepairServices, repair
from .request_options import (
    DELIVER_DIAL_KEYS,
    REDRAW_DIAL_KEYS,
    REPAIR_DIAL_KEYS,
    _MASKED_REDRAW_DIAL_KEYS,
    _resolved_options,
    deliver_arguments,
    dials_scope,
    dof_arguments,
    lens_arguments,
    lighting_arguments,
    saturation_arguments,
    masked_redraw_arguments,
    photo_arguments,
    stage_arguments,
    redraw_arguments,
    repair_arguments,
)
from .worker_channels import Connection, Heartbeat, HubListener, Management, ProgressRelay

if TYPE_CHECKING:
    from ..infrastructure.comfyui.timings import AttemptTimings

CLAIM_PATH = "/api/v1/requests/claim"
DRY_RUN_PATH = "/api/v1/requests?status=queued&limit=1"


def fetch_source(management: Management, generation_id: str) -> tuple[dict, str]:
    """The requested generation's context and its request's recipe."""
    context = management.request(
        "GET", f"/api/v1/generations/{generation_id}/context")
    return context, context["request"].get("recipe") or ""


@dataclass(frozen=True)
class WorkServices:
    management: Management
    generate_services: GenerateServices
    deliver_services: DeliverServices
    redraw_services: RedrawServices
    repair_services: RepairServices
    masked_redraw_services: MaskedRedrawServices
    dof_services: DofServices
    lens_services: LensServices
    photo_services: PhotoServices
    stage_services: StageServices
    git_metadata: Callable[[], dict]
    worker_id: str
    generate: Callable[..., dict | None] = generate
    deliver: Callable[..., dict] = deliver
    redraw: Callable[..., dict] = redraw
    repair: Callable[..., dict] = repair
    masked_redraw: Callable[..., dict] = masked_redraw
    dof: Callable[..., dict] = dof
    lens: Callable[..., dict] = lens
    photo: Callable[..., dict] = photo
    stage: Callable[..., dict] = stage
    lighting: Callable[..., dict] = lighting
    saturation: Callable[..., dict] = saturation
    emit: Callable[[str], None] = print
    sleep: Callable[[float], None] = time.sleep
    heartbeat_interval: float = 30
    kinds: tuple[str, ...] = ("generate", "redraw", "repair", "masked_redraw",
                              "deliver", "dof", "lens", "photo", "stage",
                              "lighting", "saturation")
    heartbeat: Callable[..., Heartbeat] = Heartbeat
    keep_awake: Callable[[], AbstractContextManager] = nullcontext
    hub: Callable[[], Connection] | None = None
    status: Callable[[], None] | None = None
    draining: Callable[[], bool] | None = None
    drained: Callable[[], None] | None = None
    progress_feed: Callable[[], Connection] | None = None
    backoff_max: float = 60
    ping_interval: float = 30
    clock: Callable[[], float] = time.monotonic
    timings: AttemptTimings | None = None


def _request_path(output_root: Path, request_id: object) -> Path:
    return request_file_path(output_root, "requests", request_id)


def _execute_generate(services: WorkServices, row: Mapping) -> dict:
    request_id = row["id"]
    path = _request_path(services.generate_services.output_root, request_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(row.get("payload") or {}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    result = services.generate(
        path, services.generate_services, key_prefix=f"request:{request_id}",
        request_id=request_id)
    return result or {"generation_ids": []}


def _execute_redraw(services: WorkServices, row: Mapping) -> dict:
    payload = row.get("payload") or {}
    generation_id = payload.get("generation_id")
    if not generation_id:
        raise SystemExit("redraw payload.generation_id is required")
    context, recipe = fetch_source(services.management, generation_id)
    dials = dials_scope(recipe, "redraw")
    options = payload.get("options")
    try:
        arguments = redraw_arguments(options, dials)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = services.redraw(generation_id, services.redraw_services,
                             key_prefix=f"request:{row['id']}",
                             request_id=row["id"], context=context,
                             **arguments)
    result["resolved_options"] = _resolved_options(options, arguments, REDRAW_DIAL_KEYS)
    return result


def _execute_deliver(services: WorkServices, row: Mapping) -> dict:
    payload = row.get("payload") or {}
    generation_id = payload.get("generation_id")
    if not generation_id:
        raise SystemExit("deliver payload.generation_id is required")
    context, recipe = fetch_source(services.management, generation_id)
    dials = dials_scope(recipe, "deliver")
    options = payload.get("options") or {}
    try:
        arguments = deliver_arguments(options, dials)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = services.deliver(generation_id, services.deliver_services,
                              key_prefix=f"request:{row['id']}",
                              request_id=row["id"], context=context,
                              **arguments)
    result["resolved_options"] = {
        **_resolved_options(options, arguments, DELIVER_DIAL_KEYS),
        "outlines": arguments["outlines"]}
    return result


def _execute_dof(services: WorkServices, row: Mapping) -> dict:
    payload = row.get("payload") or {}
    generation_id = payload.get("generation_id")
    if not generation_id:
        raise SystemExit("dof payload.generation_id is required")
    context, _ = fetch_source(services.management, generation_id)
    try:
        arguments = dof_arguments(payload.get("options") or {})
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = services.dof(generation_id, services.dof_services,
                          key_prefix=f"request:{row['id']}",
                          request_id=row["id"], context=context, **arguments)
    resolved = {**arguments, "focus": list(arguments["focus"])}
    photo_blur = resolved.pop("photo_blur")
    if photo_blur is not None:
        stand_at = photo_blur["stand_at"]
        resolved.update({**photo_blur,
                         "stand_at": list(stand_at) if stand_at is not None else None})
    result["resolved_options"] = resolved
    return result


def _execute_lens(services: WorkServices, row: Mapping) -> dict:
    payload = row.get("payload") or {}
    generation_id = payload.get("generation_id")
    if not generation_id:
        raise SystemExit("lens payload.generation_id is required")
    context, _ = fetch_source(services.management, generation_id)
    try:
        arguments = lens_arguments(payload.get("options") or {})
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = services.lens(generation_id, services.lens_services,
                           key_prefix=f"request:{row['id']}",
                           request_id=row["id"], context=context, **arguments)
    resolved = {**arguments, "center": list(arguments["center"])}
    if resolved["radius"] is None:
        del resolved["radius"]
    result["resolved_options"] = resolved
    return result


def _execute_photo(services: WorkServices, row: Mapping) -> dict:
    payload = row.get("payload") or {}
    generation_id = payload.get("generation_id")
    if not generation_id:
        raise SystemExit("photo payload.generation_id is required")
    context, _ = fetch_source(services.management, generation_id)
    try:
        arguments = photo_arguments(payload.get("options") or {})
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = services.photo(generation_id, services.photo_services,
                            key_prefix=f"request:{row['id']}",
                            request_id=row["id"], context=context, **arguments)
    resolved = {key: list(value) if isinstance(value, tuple) else value
                for key, value in arguments.items() if key != "photo_id"}
    result["resolved_options"] = {"photo": arguments["photo_id"], **resolved}
    return result


def _execute_stage(services: WorkServices, row: Mapping) -> dict:
    payload = row.get("payload") or {}
    generation_id = payload.get("generation_id")
    if not generation_id:
        raise SystemExit("stage payload.generation_id is required")
    context, _ = fetch_source(services.management, generation_id)
    try:
        arguments = stage_arguments(payload.get("options") or {})
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = services.stage(generation_id, services.stage_services,
                            key_prefix=f"request:{row['id']}",
                            request_id=row["id"], context=context, **arguments)
    if "photo_id" in arguments:
        result["resolved_options"] = {"photo": arguments["photo_id"],
                                      "place": list(arguments["place"]),
                                      "height": arguments["height"]}
    else:
        result["resolved_options"] = {"backdrop": arguments["backdrop"]}
        if not arguments["inherit_shadow"]:
            result["resolved_options"]["shadow_from"] = arguments["shadow_from"]
        if not arguments["inherit_light"]:
            light = arguments["light"]
            result["resolved_options"]["light"] = (
                {"scene": light.scene, "from": light.direction} if light else None)
        if arguments["frame"] is not None:
            result["resolved_options"]["frame"] = result.pop("frame", arguments["frame"])
    return result


def _execute_lighting(services: WorkServices, row: Mapping) -> dict:
    payload = row.get("payload") or {}
    generation_id = payload.get("generation_id")
    if not generation_id:
        raise SystemExit("lighting payload.generation_id is required")
    context, _ = fetch_source(services.management, generation_id)
    try:
        arguments = lighting_arguments(payload.get("options") or {})
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = services.lighting(generation_id, services.stage_services,
                               key_prefix=f"request:{row['id']}",
                               request_id=row["id"], context=context, **arguments)
    result["resolved_options"] = dict(arguments)
    return result


def _execute_saturation(services: WorkServices, row: Mapping) -> dict:
    payload = row.get("payload") or {}
    generation_id = payload.get("generation_id")
    if not generation_id:
        raise SystemExit("saturation payload.generation_id is required")
    context, _ = fetch_source(services.management, generation_id)
    try:
        arguments = saturation_arguments(payload.get("options") or {})
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = services.saturation(generation_id, services.stage_services,
                                 key_prefix=f"request:{row['id']}",
                                 request_id=row["id"], context=context, **arguments)
    result["resolved_options"] = dict(arguments)
    return result


def _execute_repair(services: WorkServices, row: Mapping) -> dict:
    payload = row.get("payload") or {}
    generation_id = payload.get("generation_id")
    if not generation_id:
        raise SystemExit("repair payload.generation_id is required")
    context, recipe = fetch_source(services.management, generation_id)
    dials = dials_scope(recipe, "repair")
    options = payload.get("options") or {}
    try:
        arguments = repair_arguments(options, dials)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = services.repair(generation_id, services.repair_services,
                             key_prefix=f"request:{row['id']}",
                             request_id=row["id"], context=context,
                             **arguments)
    result["resolved_options"] = _resolved_options(options, arguments, REPAIR_DIAL_KEYS)
    return result


def _execute_masked_redraw(services: WorkServices, row: Mapping) -> dict:
    payload = row.get("payload") or {}
    generation_id = payload.get("generation_id")
    if not generation_id:
        raise SystemExit("masked_redraw payload.generation_id is required")
    context, recipe = fetch_source(services.management, generation_id)
    dials = dials_scope(recipe, "repair")
    options = payload.get("options") or {}
    try:
        arguments = masked_redraw_arguments(options, dials)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = services.masked_redraw(generation_id, services.masked_redraw_services,
                                    key_prefix=f"request:{row['id']}",
                                    request_id=row["id"], context=context,
                                    **arguments)
    result["resolved_options"] = _resolved_options(
        options, arguments, _MASKED_REDRAW_DIAL_KEYS)
    return result


def execute(services: WorkServices, row: Mapping) -> dict:
    ref = row.get("recipe_ref")
    if ref != services.git_metadata().get("branch"):
        raise SystemExit(f"recipe_ref not served: {ref}")
    kind = row.get("kind")
    if kind == "generate":
        return _execute_generate(services, row)
    if kind == "redraw":
        return _execute_redraw(services, row)
    if kind == "deliver":
        return _execute_deliver(services, row)
    if kind == "repair":
        return _execute_repair(services, row)
    if kind == "masked_redraw":
        return _execute_masked_redraw(services, row)
    if kind == "dof":
        return _execute_dof(services, row)
    if kind == "lens":
        return _execute_lens(services, row)
    if kind == "photo":
        return _execute_photo(services, row)
    if kind == "stage":
        return _execute_stage(services, row)
    if kind == "lighting":
        return _execute_lighting(services, row)
    if kind == "saturation":
        return _execute_saturation(services, row)
    if kind == "finalize":
        raise SystemExit(
            "finalize requests are no longer supported; use redraw and deliver")
    raise SystemExit(f"unsupported request kind: {kind!r}")


def _report(services: WorkServices, row_id: str, payload: dict) -> None:
    try:
        services.management.request(
            "PATCH", f"/api/v1/requests/{row_id}",
            {**payload, "worker_id": services.worker_id})
    except SystemExit as error:
        services.emit(f"report failed for {row_id} (worker moved on?): {error}")


def _send_timings(services: WorkServices, row_id: str, status: str) -> None:
    """Best-effort: a timing failure never changes the request's outcome."""
    if services.timings is None:
        return
    try:
        payload = services.timings.payload(services.worker_id, status)
        if payload is None:
            services.emit(f"timings skipped for {row_id}: claim row has no attempt")
            return
        services.management.request(
            "PUT", f"/api/v1/requests/{row_id}/timings", payload)
    except (SystemExit, Exception) as error:
        services.emit(f"timings failed for {row_id}: {error}")


def work_once(services: WorkServices, *, dry_run: bool = False,
             listener: HubListener | None = None,
             relay: ProgressRelay | None = None) -> bool:
    if dry_run:
        response = services.management.request("GET", DRY_RUN_PATH)
        items = (response or {}).get("items", [])
        if not items:
            services.emit("no queued requests")
        else:
            services.emit("would execute:")
            services.emit(json.dumps(items[0], indent=2, ensure_ascii=False))
        return False
    try:
        row = services.management.request(
            "POST", CLAIM_PATH,
            {"worker_id": services.worker_id, "kinds": list(services.kinds)})
    except SystemExit as error:
        services.emit(f"claim failed: {error}")
        return False
    if row is None:
        return False
    if services.timings is not None:
        services.timings.begin(row.get("attempt"))
    if listener is not None:
        listener.send_progress(row["id"], "submit")
    failure: BaseException | None = None
    with services.keep_awake(), services.heartbeat(
            services.management, row["id"], services.worker_id,
            interval=services.heartbeat_interval, emit=services.emit):
        if relay is not None:
            relay.current = row["id"]
        try:
            result = execute(services, row)
        except (SystemExit, Exception) as error:
            failure = error
        finally:
            if relay is not None:
                relay.current = None
    if failure is not None:
        _report(services, row["id"], {"status": "failed", "error": str(failure)})
        _send_timings(services, row["id"], "failed")
        services.emit(f"request {row['id']} failed: {failure}")
        return True
    _report(services, row["id"], {"status": "done", "result": result})
    _send_timings(services, row["id"], "done")
    services.emit(f"request {row['id']} done")
    return True


def release_claims(services: WorkServices) -> None:
    """Hand back the rows this worker still holds from a killed process.

    Otherwise they sit running until the heartbeat goes stale, and a worker
    that came back in seconds waits minutes for its own queue.
    """
    worker = quote(services.worker_id, safe="")
    try:
        response = services.management.request(
            "GET", f"/api/v1/requests?status=running&worker_id={worker}")
    except (SystemExit, Exception) as error:
        services.emit(f"! release query failed: {error}")
        return
    for row in (response or {}).get("items", []):
        if not row.get("id"):
            continue
        try:
            released = services.management.request(
                "PATCH", f"/api/v1/requests/{row['id']}",
                {"status": "queued", "worker_id": services.worker_id})
        except (SystemExit, Exception) as error:
            services.emit(f"! release failed for {row['id']}: {error}")
            continue
        status = (released or {}).get("status", "queued")
        services.emit(f"released {row['id']}: {status}")


def _draining(services: WorkServices) -> bool:
    """Never let a stop signal that cannot be read stop the worker."""
    if services.draining is None:
        return False
    try:
        return bool(services.draining())
    except (SystemExit, Exception) as error:
        services.emit(f"! drain check failed: {error}")
        return False


def _report_status(services: WorkServices) -> None:
    """Write the status file; report a failure instead of raising it."""
    if services.status is None:
        return
    try:
        services.status()
    except (SystemExit, Exception) as error:
        services.emit(f"! status write failed: {error}")


def _idle(services: WorkServices, wake: threading.Event | None,
          interval: float) -> None:
    """Sleep out the idle interval, cut short once a drain is asked for.

    Sliced rather than waited whole so an idle worker does not hold up a
    deploy waiting on it to leave. With no drain wired, waits whole.
    """
    if services.draining is None:
        if wake is not None:
            wake.wait(interval)
        else:
            services.sleep(interval)
        return
    remaining = interval
    while remaining > 0:
        slice_ = min(1.0, remaining)
        if wake is not None:
            if wake.wait(slice_):
                return
        else:
            services.sleep(slice_)
        if _draining(services):
            return
        remaining -= slice_


def work(services: WorkServices, *, interval: float = 30, once: bool = False,
         dry_run: bool = False, publish_catalog: bool = True) -> None:
    listener: HubListener | None = None
    relay: ProgressRelay | None = None
    wake = threading.Event()
    interrupted = False
    if not dry_run:
        release_claims(services)
    if publish_catalog and not dry_run:
        try:
            publish_catalog_document(services.management, services.git_metadata())
        except (SystemExit, Exception) as error:
            services.emit(f"! catalog publish failed: {error}")
    try:
        if services.hub is not None:
            listener = HubListener(services, wake).start()
        if services.progress_feed is not None:
            relay = ProgressRelay(services, listener).start()
        while True:
            _report_status(services)
            if _draining(services):
                services.emit("draining: no new work claimed")
                break
            if listener is not None:
                wake.clear()
            did_something = work_once(services, dry_run=dry_run,
                                      listener=listener, relay=relay)
            if once:
                return
            if not did_something:
                _idle(services, wake if listener is not None else None, interval)
    except KeyboardInterrupt:
        interrupted = True
        services.emit("work stopped")
    finally:
        if relay is not None:
            relay.stop()
        if listener is not None:
            listener.stop()
        if interrupted and not dry_run:
            release_claims(services)
        if services.drained is not None and _draining(services):
            # The deploy waits on this acknowledgement rather than on a clock.
            try:
                services.drained()
            except (SystemExit, Exception) as error:
                services.emit(f"! drain acknowledgement failed: {error}")
