"""Translate a pure Yukari-anima render specification into a ComfyUI graph."""

from __future__ import annotations

import json

from ...domain.generation.models import PoseControl, RenderSpec

CLIP_NAME = "qwen_3_06b_base.safetensors"
VAE_NAME = "qwen_image_vae.safetensors"
# One file carries both the adapter's LoRA deltas and its control embedder.
POSE_ADAPTER = "anima_pose_preview2.safetensors"
OUTLINE_ADAPTER = "anima-lllite-any-test-like-v2.safetensors"


def _pose_nodes(graph: dict, model_ref: list, pose: PoseControl, width: int,
                height: int, first_id: int) -> tuple[list, int]:
    """Add the skeleton, its latent and the adapter after `model_ref`;
    return the controlled model and the next free node id."""
    lora_id, skeleton_id, encode_id, apply_id = (
        str(first_id + offset) for offset in range(4))
    graph[lora_id] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": model_ref, "lora_name": POSE_ADAPTER, "strength_model": 1.0}}
    graph[skeleton_id] = {"class_type": "YukariPoseSkeleton", "inputs": {
        "points": json.dumps([list(point) for point in pose.points]),
        "aspect": pose.aspect, "width": width, "height": height}}
    graph[encode_id] = {"class_type": "VAEEncode", "inputs": {
        "pixels": [skeleton_id, 0], "vae": ["4", 0]}}
    graph[apply_id] = {"class_type": "AnimaControlApply", "inputs": {
        "model": [lora_id, 0], "control_latent": [encode_id, 0],
        "control_embedder_path": POSE_ADAPTER, "strength": pose.strength}}
    return [apply_id, 0], first_id + 4


def _outline_nodes(graph: dict, model_ref: list, pose: PoseControl, width: int,
                   height: int, first_id: int) -> tuple[list, int]:
    """Add the silhouette or lineart picture and the any-test LLLite after
    `model_ref`; return the controlled model and the next free node id."""
    picture_id, loader_id, apply_id = (str(first_id + offset) for offset in range(3))
    graph[picture_id] = {"class_type": "YukariPoseOutline", "inputs": {
        "picture": pose.picture, "aspect": pose.aspect,
        "width": width, "height": height}}
    graph[loader_id] = {"class_type": "ModelPatchLoader", "inputs": {
        "name": OUTLINE_ADAPTER}}
    graph[apply_id] = {"class_type": "AnimaLLLiteApply", "inputs": {
        "model": model_ref, "model_patch": [loader_id, 0],
        "image": [picture_id, 0], "strength": pose.strength,
        "start_percent": 0.0, "end_percent": 1.0}}
    return [apply_id, 0], first_id + 3


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
    # The control is the first pass's size; the hires pass samples a larger
    # latent, so it draws on the model without the adapter.
    free_model = graph["3"]["inputs"]["model"]
    if spec.pose is not None and spec.pose.strength:
        nodes = _pose_nodes if spec.pose.control == "skeleton" else _outline_nodes
        graph["3"]["inputs"]["model"], loader_id = nodes(
            graph, free_model, spec.pose, spec.width, spec.height, loader_id)
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
        model_ref = free_model
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
