"""Pick one pre-deliver image per picture from good_chains.json and download it.

PYTHONPATH=src uv run -q --no-project --with pillow --with numpy python research/moss-lora/dataset_extract.py <since-iso> <outdir>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from chagama.infrastructure.chimera.client import ChimeraClient

DECOR = {"deliver", "stage", "dof", "lighting", "photo", "saturation", "lens", "finalize"}
RANK = {"redraw": 0, "masked_redraw": 1, "repair": 1, "generate": 2, "import": 3}

here = Path(__file__).parent
since, out = sys.argv[1], Path(sys.argv[2])
(out / "images").mkdir(parents=True, exist_ok=True)
chains = [ch for ch in json.loads((here / "good_chains.json").read_text()) if ch[-1]["created_at"] >= since]
candidates = defaultdict(dict)
for ch in chains:
    for n in ch:
        if n["kind"] not in DECOR:
            candidates[ch[-1]["short_id"]][n["short_id"]] = n

c = ChimeraClient(Path("."))
manifest = []
for root, nodes in sorted(candidates.items()):
    pick = min(nodes.values(), key=lambda n: (RANK.get(n["kind"], 9), n["rating"] != "good"))
    latest = [n for n in nodes.values() if n["kind"] == pick["kind"] and (n["rating"] == "good") == (pick["rating"] == "good")]
    pick = max(latest, key=lambda n: n["created_at"])
    rg = c.request("GET", f"/api/v1/generations/{root}?include_prompts=true")
    g = c.request("GET", f"/api/v1/generations/{pick['short_id']}")
    path = out / "images" / f"{pick['short_id']}.png"
    if not path.exists():
        path.write_bytes(c.fetch_generation_image(g["id"]))
    tags = (g.get("safety") or {}).get("tags") or {}
    manifest.append({"short_id": pick["short_id"], "root": root, "kind": pick["kind"], "rating": pick["rating"],
                     "size": pick["size"], "created_at": rg["created_at"],
                     "root_prompt": (rg["request"] or {}).get("prompt"),
                     "tags": {k: v for k, v in sorted(tags.items(), key=lambda kv: -kv[1]) if v >= 0.35}})
    print(pick["short_id"], pick["kind"], root, flush=True)
(out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
print(len(manifest), "images")
