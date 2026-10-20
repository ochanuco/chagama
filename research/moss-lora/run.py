"""MOSS-Image 4-step LoRA + Qwen3.5 vision reference trial against the box ComfyUI.

usage: COMFYUI_HOST=<host> python3 -I research/moss-lora/run.py <arms.json> <outdir>
arms.json: {"ref": "ref.jpg", "seeds": [..], "width": 896, "height": 1440, "steps": 6,
            "arms": [{"name": "a", "prompt": "...", "ref": true}]}
"""
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request
import uuid

HOST = f"http://{os.environ.get('COMFYUI_HOST', '127.0.0.1')}:8188"
VISION = "<|vision_start|><|image_pad|><|vision_end|>"


def post(path, data, headers=None):
    req = urllib.request.Request(HOST + path, data=data, headers=headers or {})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def upload(path):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    return post("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})["name"]


def graph(cfg, arm, seed, ref_name):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "moss_image_sft_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "moss_image_sft_4step_v2.0.safetensors", "strength_model": arm.get("lora", 1.0)}},
        "12": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": arm.get("shift", 5.0)}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": cfg.get("te", "qwen3.5_9b_w4a8_gptq.safetensors"), "type": "moss_image", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}},
        "7": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": arm.get("width", cfg["width"]), "height": arm.get("height", cfg["height"]), "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["2", 0] if arm.get("lora", 1.0) else ["12", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0],
                                                    "seed": seed, "steps": arm.get("steps", cfg["steps"]), "cfg": arm.get("cfg", 1.0),
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": f"moss/{arm['name']}_{seed}"}},
    }
    if arm.get("style"):
        g["13"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": g["8"]["inputs"]["model"], "lora_name": arm["style"], "strength_model": arm.get("style_strength", 1.0)}}
        g["8"]["inputs"]["model"] = ["13", 0]
    if arm.get("hires"):
        hw, hh = arm["hires"]
        g["14"] = {"class_type": "ImageScale", "inputs": {"image": ["9", 0], "upscale_method": "lanczos", "width": hw, "height": hh, "crop": "disabled"}}
        g["15"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["14", 0], "vae": ["4", 0]}}
        g["16"] = dict(g["8"], inputs=dict(g["8"]["inputs"], latent_image=["15", 0], denoise=arm.get("hires_denoise", 0.45)))
        if "hires_style_strength" in arm:
            g["18"] = dict(g["13"], inputs=dict(g["13"]["inputs"], strength_model=arm["hires_style_strength"]))
            g["16"]["inputs"]["model"] = ["18", 0]
        g["17"] = {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["4", 0]}}
        g["10"]["inputs"]["images"] = ["17", 0]
    if arm.get("ref"):
        g["11"] = {"class_type": "LoadImage", "inputs": {"image": ref_name}}
        g["5"] = {"class_type": "TextEncodeQwenImageEdit", "inputs": {"clip": ["3", 0], "prompt": VISION + arm["prompt"], "image": ["11", 0]}}
    else:
        g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": arm["prompt"]}}
    return g


def main():
    cfg_path = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(cfg_path.read_text())
    ref_name = upload(cfg_path.parent / cfg["ref"]) if cfg.get("ref") else None
    jobs = []
    for seed in cfg["seeds"]:
        for arm in cfg["arms"]:
            pid = post("/prompt", json.dumps({"prompt": graph(cfg, arm, seed, ref_name)}).encode(), {"Content-Type": "application/json"})["prompt_id"]
            jobs.append((arm["name"], seed, pid))
    t0 = time.time()
    for name, seed, pid in jobs:
        while True:
            h = json.loads(urllib.request.urlopen(f"{HOST}/history/{pid}", timeout=60).read())
            if pid in h and h[pid].get("status", {}).get("completed") is not None:
                break
            if pid in h and h[pid].get("status", {}).get("status_str") == "error":
                print(name, seed, "ERROR", json.dumps(h[pid]["status"])[:2000], flush=True)
                break
            time.sleep(3)
        entry = h.get(pid, {})
        for node in entry.get("outputs", {}).values():
            for img in node.get("images", []):
                q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
                (out / f"{name}_{seed}.png").write_bytes(urllib.request.urlopen(f"{HOST}/view?{q}", timeout=120).read())
        msgs = entry.get("status", {}).get("messages", [])
        ts = {m[0]: m[1].get("timestamp") for m in msgs}
        dur = (ts.get("execution_success", 0) - ts.get("execution_start", 0)) / 1000 if ts.get("execution_success") else None
        print(name, seed, entry.get("status", {}).get("status_str"), f"{dur}s" if dur else "", f"elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
