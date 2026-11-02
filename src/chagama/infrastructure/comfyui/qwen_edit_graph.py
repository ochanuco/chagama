"""Qwen-Image 2.1 edit nodes: one image re-made from a prompt in 8 steps."""

from __future__ import annotations

from collections.abc import Callable

UNET_NAME = "qwen_image_2.1_int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_8b_int8_convrot.safetensors"
VAE_NAME = "qwen_image_2.1_vae_bf16.safetensors"
LORA_NAME = "p_q_image_2.1_8_steps_v0.1_comfyui_bf16.safetensors"
# The 8-step LoRA's schedule; it is trained for these sigmas at CFG 1.
SIGMAS = "1.0, 0.93333, 0.85714, 0.76923, 0.66667, 0.54545, 0.4, 0.22222, 0.0"
# The edit's pixel budget, as the long-side-equivalent of a square.
RESOLUTION = 1536


def edit_nodes(graph: dict, allocate: Callable[[], str], image_ref: list,
               prompt: str, seed: int) -> list:
    """Add the edit of `image_ref` by `prompt` to `graph`; returns the ref of
    the decoded picture, at the edit's own size."""
    def node(class_type: str, inputs: dict) -> str:
        node_id = allocate()
        graph[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    unet = node("UNETLoader", {"unet_name": UNET_NAME, "weight_dtype": "default"})
    lora = node("LoraLoaderModelOnly", {"model": [unet, 0], "lora_name": LORA_NAME,
                                        "strength_model": 1.0})
    model = node("QwenImage21Cache", {"model": [lora, 0], "device": "auto",
                                      "dtype": "default"})
    clip = node("CLIPLoader", {"clip_name": CLIP_NAME, "type": "qwen_image"})
    vae = node("VAELoader", {"vae_name": VAE_NAME})
    encoded = node("TextEncodeQwenImage21", {
        "clip": [clip, 0], "vae": [vae, 0], "prompt": prompt, "negative_prompt": "",
        "resolution": RESOLUTION, "images.image_1": image_ref})
    sampler = node("KSamplerSelect", {"sampler_name": "euler"})
    sigmas = node("ManualSigmas", {"sigmas": SIGMAS})
    sampled = node("SamplerCustom", {
        "model": [model, 0], "add_noise": True, "noise_seed": seed, "cfg": 1.0,
        "positive": [encoded, 0], "negative": [encoded, 1], "sampler": [sampler, 0],
        "sigmas": [sigmas, 0], "latent_image": [encoded, 2]})
    decoded = node("VAEDecode", {"samples": [sampled, 0], "vae": [vae, 0]})
    return [decoded, 0]
