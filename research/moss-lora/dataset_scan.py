"""Walk every rating=good Generation down its refines chain and record each step.

PYTHONPATH=src uv run -q --no-project --with pillow --with numpy python research/moss-lora/dataset_scan.py <out.json>
"""
import json
import sys
from pathlib import Path

from chagama.infrastructure.chimera.client import ChimeraClient

c = ChimeraClient(Path("."))
goods, offset = [], 0
while True:
    page = c.request("GET", f"/api/v1/generations?rating=good&limit=200&offset={offset}")
    goods += page["items"]
    offset += len(page["items"])
    if not page["items"] or offset >= page["total"]:
        break

cache = {}


def node(gid):
    if gid not in cache:
        g = c.request("GET", f"/api/v1/generations/{gid}")
        req = g["request"] or {}
        params = req.get("parameters") or {}
        ref = g.get("refines_generation")
        cache[gid] = {"short_id": g["short_id"], "kind": req.get("kind"), "recipe": req.get("recipe"),
                      "op": params.get("kind") or params.get("operation"), "rating": g.get("rating"),
                      "published": bool(g.get("publications")), "created_at": g["created_at"],
                      "size": [g["generations"][0]["image_width"], g["generations"][0]["image_height"]] if g.get("generations") else None,
                      "refines": ref["short_id"] if ref else None,
                      "assets": [a["role"] for a in g.get("assets", [])]}
    return cache[gid]


chains = []
for item in goods:
    chain, gid = [], item["short_id"]
    while gid and len(chain) < 20:
        n = node(gid)
        chain.append(n)
        gid = n["refines"]
    chains.append(chain)
    print(" <- ".join(f"{n['short_id']}:{n['kind']}" for n in chain), flush=True)
Path(sys.argv[1]).write_text(json.dumps(chains, ensure_ascii=False, indent=1))
