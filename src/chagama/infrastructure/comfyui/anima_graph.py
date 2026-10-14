"""Translate a pure Yukari-anima render specification into a ComfyUI graph."""

from __future__ import annotations

from ...domain.generation.models import RenderSpec

CLIP_NAME = "qwen_3_06b_base.safetensors"
VAE_NAME = "qwen_image_vae.safetensors"


def build_graph(spec: RenderSpec) -> dict[str, dict]:
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": spec.model_path, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": CLIP_NAME, "type": "qwen_image"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {
            "width": spec.width, "height": spec.height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["2", 0], "text": spec.prompts.positive}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["2", 0], "text": spec.prompts.negative}},
        "3": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0], "seed": spec.seed, "steps": spec.steps,
            "cfg": spec.cfg, "sampler_name": spec.sampler_name,
            "scheduler": spec.scheduler, "denoise": spec.denoise}},
        "8": {"class_type": "VAEDecode", "inputs": {
            "samples": ["3", 0], "vae": ["4", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {
            "images": ["8", 0], "filename_prefix": spec.filename_prefix}},
    }
    loader_id = 10
    loras = [(name, weight) for name, weight in spec.loras if weight]
    if loras:
        model_ref = ["1", 0]
        for lora_name, weight in loras:
            node_id = str(loader_id)
            graph[node_id] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": model_ref, "lora_name": lora_name,
                "strength_model": weight}}
            model_ref = [node_id, 0]
            loader_id += 1
        graph["3"]["inputs"]["model"] = model_ref
    if spec.guided_steps and 0 < spec.guided_steps < spec.steps:
        model_ref = graph["3"]["inputs"]["model"]
        stage1_id = str(loader_id)
        loader_id += 1
        graph[stage1_id] = {"class_type": "KSamplerAdvanced", "inputs": {
            "model": model_ref, "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0], "add_noise": "enable",
            "noise_seed": spec.seed, "steps": spec.steps, "cfg": spec.cfg,
            "sampler_name": spec.sampler_name, "scheduler": spec.scheduler,
            "start_at_step": 0, "end_at_step": spec.guided_steps,
            "return_with_leftover_noise": "enable"}}
        graph["3"] = {"class_type": "KSamplerAdvanced", "inputs": {
            "model": model_ref, "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": [stage1_id, 0], "add_noise": "disable",
            "noise_seed": spec.seed, "steps": spec.steps, "cfg": 1.0,
            "sampler_name": spec.sampler_name, "scheduler": spec.scheduler,
            "start_at_step": spec.guided_steps, "end_at_step": 10000,
            "return_with_leftover_noise": "disable"}}
    if spec.hires is not None:
        model_ref = graph["3"]["inputs"]["model"]
        upscale_id, sampler_id = str(loader_id), str(loader_id + 1)
        graph[upscale_id] = {"class_type": "LatentUpscale", "inputs": {
            "samples": ["3", 0], "upscale_method": "bicubic",
            "width": spec.hires.width, "height": spec.hires.height,
            "crop": "disabled"}}
        graph[sampler_id] = {"class_type": "KSampler", "inputs": {
            "model": model_ref, "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": [upscale_id, 0], "seed": spec.seed,
            "steps": spec.steps, "cfg": spec.cfg,
            "sampler_name": spec.sampler_name, "scheduler": spec.scheduler,
            "denoise": spec.hires.denoise}}
        graph["8"]["inputs"]["samples"] = [sampler_id, 0]
    return graph
