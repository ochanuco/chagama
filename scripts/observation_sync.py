#!/usr/bin/env python3
"""Push experiments/**/*.jsonl records into chimera's Observation index.

experiments/ stays the source of truth and the write path; this syncs a
derived, queryable copy to chimera.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chagama.infrastructure.chimera.client import ChimeraClient
from chagama.infrastructure.repository import discover_repository

# Manual overrides for records whose character/component can't be derived
# from the file path alone.
ENVELOPE: dict[tuple[str, int], dict[str, str]] = {
    ("experiments/yukari/prompt_style.jsonl", 33):
        {"character": "yukari", "component": "prompt_style"},
    ("experiments/yukari/prompt_style.jsonl", 34):
        {"character": "yukari", "component": "prompt_style"},
    ("experiments/yukari/prompt_style.jsonl", 35):
        {"character": "yukari", "component": "prompt_style"},
    ("experiments/yukari/prompt_style.jsonl", 36):
        {"character": "yukari", "component": "prompt_style"},
    ("experiments/yukari/delivery_style.jsonl", 48):
        {"character": "yukari", "component": "prompt_style"},
}


def build_files(root: Path) -> list[dict]:
    """One entry per experiments JSONL file, records in line order."""
    files = []
    experiments = root / "experiments"
    for path in sorted(experiments.rglob("*.jsonl"),
                        key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        records = []
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_no, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{rel}:{line_no}: invalid JSON: {error}")
            entry: dict = {"line": line_no, "record": record}
            for key, value in ENVELOPE.get((rel, line_no), {}).items():
                if key not in record:
                    entry[key] = value
            records.append(entry)
        files.append({"path": rel, "records": records})
    return files


def unlabelled(files: list[dict]) -> list[tuple[str, int]]:
    """(path, line) for records carrying no pose, component or envelope."""
    result = []
    for file in files:
        for entry in file["records"]:
            record = entry["record"]
            if "pose" in record or "component" in record or "component" in entry:
                continue
            result.append((file["path"], entry["line"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None,
                         help="repository root (default: discovered from git)")
    parser.add_argument("--dry-run", action="store_true",
                         help="print counts and unlabelled records; post nothing")
    args = parser.parse_args()

    root = args.root or discover_repository()
    files = build_files(root)

    if args.dry_run:
        total = 0
        for file in files:
            count = len(file["records"])
            total += count
            print(f"{file['path']}: {count}")
        print(f"total: {total} records across {len(files)} files")
        for path, line in unlabelled(files):
            print(f"unlabelled: {path}:{line}")
        return

    chimera = ChimeraClient(root)
    response = chimera.request(
        "POST", "/api/v1/observations/sync", {"files": files})
    print(f"inserted: {response['inserted']}")
    print(f"unchanged: {response['unchanged']}")
    for item in response.get("skipped", []):
        print(f"skipped: {item}")


if __name__ == "__main__":
    main()
