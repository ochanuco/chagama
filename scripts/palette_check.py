"""Measure a render against the look spec before a human sees it.

    uv run scripts/palette_check.py <short_id> [<short_id> ...]
    uv run scripts/palette_check.py --file <png> [...]

This is the acceptance gate: run it on every arm the moment it is ingested,
and do not present a FAIL as a candidate. A pass is not approval -- the
human still judges -- but a FAIL never goes forward. `measure`/`verdict`
live in `chagama.infrastructure.imaging.palette`; this is the CLI.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from chagama.infrastructure.chimera.client import ChimeraClient
from chagama.infrastructure.imaging.palette import measure, verdict
from chagama.infrastructure.repository import discover_repository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", help="chimera short ids")
    parser.add_argument("--file", action="append", type=Path, default=[])
    args = parser.parse_args()
    failed = False
    sources: list[tuple[str, bytes]] = []
    if args.ids:
        chimera = ChimeraClient(discover_repository())
        for sid in args.ids:
            sources.append((sid, chimera.fetch_generation_image(sid)))
    for path in args.file:
        sources.append((str(path), path.read_bytes()))
    for name, data in sources:
        m = measure(data)
        fails = verdict(m)
        status = "FAIL" if fails else "pass"
        failed |= bool(fails)
        print(f"{name}: {status} | bg #%02x%02x%02x sat {m['bg_sat']:.0f} "
              f"| mean sat {m['sat']:.1f} | fig mid sat {m['fig_sat_mean']:.1f} "
              f"p90 {m['fig_sat_p90']:.0f} | light {m['light_sat']:.1f} "
              f"(drift x{m['norm_factor']:.2f})" % m["bg"]
              + ("".join(f"\n  - {f}" for f in fails)))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
