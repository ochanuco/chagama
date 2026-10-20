"""Register a MOSS trial's outputs in chimera as import requests, one per arm/seed.

PYTHONPATH=src uv run -q --no-project --with pillow --with numpy python research/moss-lora/ingest.py <arms.json> <outdir> <tag>
"""
import json
import sys
from pathlib import Path

from chagama.application.ingest import import_images
from chagama.infrastructure.chimera.client import ChimeraClient

cfg = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
tag = sys.argv[3]
ref = cfg.get("ref_id")
c = ChimeraClient(Path("."))
ids = []
for seed in cfg["seeds"]:
    for arm in cfg["arms"]:
        path = out / f"{arm['name']}_{seed}.png"
        refs = [{"source_generation_id": ref, "purpose": "style", "aspect": "vision-reference",
                 "instruction": "Qwen3.5 TE に画像として入力"}] if arm.get("ref") else []
        params = {"kind": "moss-trial", "model": "moss_image_sft_bf16" + (" + 4step LoRA v2.0" if arm.get("lora", 1.0) else ""),
                  "te": cfg.get("te", "qwen3.5_9b_w4a8_gptq"), "steps": arm.get("steps", cfg["steps"]),
                  "cfg": arm.get("cfg", 1.0), "sampler": "euler/simple", "canvas": [cfg["width"], cfg["height"]],
                  "style_lora": arm.get("style"), "style_strength": arm.get("style_strength", 1.0) if arm.get("style") else None, "arm": arm["name"], "prompt": arm["prompt"], "vision_ref": bool(arm.get("ref"))}
        r = import_images(c, print, images=[path], seed=seed, idempotency_key=f"moss:{tag}:{arm['name']}:{seed}",
                          resolution={"recipe": None, "raw_instruction": f"MOSS-Image {tag} {arm['name']} seed {seed}",
                                      "parameters": params, "references": refs})
        gid = r["generation_ids"][0]
        c.put_semantic(gid, {"schema_version": 1, "summary": arm["summary"],
                             "attributes": {"trial": f"moss-{tag}", "arm": arm["name"], "seed": seed}})
        c.add_tag(gid, f"moss-{tag}")
        ids.append(c.request("GET", f"/api/v1/generations/{gid}")["short_id"])
print("https://chimera.chanu.co/compare?ids=" + ",".join(ids))
