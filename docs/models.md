# Models

No models are included in this repository. They live on the GPU box. The
Hugging Face files listed in `manifests/worker-models.toml` are fetched there
by the `fetch worker models` workflow; everything else is fetched by hand
([remote.md](remote.md) covers both).

## In use now

The code names the files; this is where to look.

| Role | File | Named in |
|---|---|---|
| Stage 1 | `diffusion_models/silvermoonmixAnimaEvolved_v2329BTurbo.safetensors` (SilvermoonMix-Anima-Evolved v2.3_2.9B_Turbo) + `qwen_3_06b_base` + `qwen_image_vae` | `domain/yukari/prompt_style.py` `MODEL`, `GUIDED_STEPS` |
| Opt-in canvas redraw | `diffusers/hassaku-il-v22` | `domain/yukari/delivery_style.py` `REDRAW_MODEL` |
| Stage 1 reserve, opt-in by `render.model` patch | `diffusion_models/fnMomentAnimaTurbo_v20.safetensors` | — |
| Pixal3D / TRELLIS.2 image-to-3D (ComfyUI core template, run by hand) | Comfy-Org Pixal3D, TRELLIS.2, MoGe and BiRefNet files | `manifests/worker-models.toml` |

`silvermoonmixAnimaEvolved_v2329BTurbo` is sampled with `render.guided_steps`:
the leading steps at the recipe's own CFG, the rest at CFG 1.0 off the same
noise. It needs a ComfyUI build that loads a 2.9B UNET; the pinned v0.40.0
does. `anima-turbo-v1.1`, the previous stage-1 model, stays on the box. It
is not in the verified inventory below. Everything else in the
inventory was used by retired recipes and is kept only for provenance.

Each has its own upstream license — the Illustrious-family checkpoints are
mostly [FAIPL-1.0-SD](https://freedevproject.org/faipl-1.0-sd/) — which this
list does not alter. Check the linked page before redistributing any of them.

## Where every file came from

Verified 2026-08-17: each SHA256 below was computed from a local copy and
matched against the upstream it is listed under, so every one of these is
re-downloadable bit-for-bit. Civitai rows were resolved through
`/api/v1/model-versions/by-hash/<sha256>`, which is also the way to re-check
them later. `manifests/models-sha256.txt` carries the full hashes in
`shasum -a 256 -c` format.

`loras/feet-xl-ill.safetensors` and `loras/hands-xl-ill.safetensors` are not
on any local machine yet, so their SHA256 rows carry the upstream Civitai API
hash and have not been verified against a local copy.

`diffusion_models/fnMomentAnimaTurbo_v20.safetensors` was hashed on the GPU
box (2026-10-02) and matches the Civitai API hash.

`diffusion_models/silvermoonmixAnimaEvolved_v2329BTurbo.safetensors` was
hashed on the GPU box (2026-10-02) and matches the Civitai API hash.

### Hugging Face

| file | size | repo :: path | sha256 |
|---|---|---|---|
| `checkpoints/NoobAI-XL-v1.1.safetensors` | 6.62 GB | `Laxhar/noobai-XL-1.1` | `6681e8e4b134c81f…` |
| `clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | 2.35 GB | `h94/IP-Adapter :: models/image_encoder/model.safetensors` | `6ca9667da1ca9e0b…` |
| `controlnet/ill-softedge-fp16.safetensors` | 2.33 GB | `r3gm/controlnet-Illustrious-softedge-hed-sdxl-fp16 :: diffusion_pytorch_model.fp16.safetensors` | `1fc0db474df39485…` |
| `controlnet/noob-canny-fp16.safetensors` | 2.33 GB | `Eugeoter/noob-sdxl-controlnet-canny :: noob_sdxl_controlnet_canny.fp16.safetensors` | `e37bcdb2f4a6d178…` |
| `controlnet/noob-lineart-anime-fp16.safetensors` | 2.33 GB | `Eugeoter/noob-sdxl-controlnet-lineart_anime :: diffusion_pytorch_model.fp16.safetensors` | `44eae6a514a60ae4…` |
| `controlnet/noob-openpose-fp16.safetensors` | 2.33 GB | `r3gm/controlnet-noobai-openpose-sdxl-fp16 :: diffusion_pytorch_model.fp16.safetensors` | `bf0f438479182848…` |
| `diffusion_models/anima-preview3-base.safetensors` | 3.89 GB | `circlestone-labs/Anima :: split_files/diffusion_models/anima-preview3-base.safetensors` | `14fffe8ad5116cd7…` |
| `ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors` | 808 MB | `h94/IP-Adapter :: sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors` | `3f5062b8400c94b7…` |
| `text_encoders/qwen_3_06b_base.safetensors` | 1.11 GB | `circlestone-labs/Anima :: split_files/text_encoders/qwen_3_06b_base.safetensors` | `cd2a512003e2f9f3…` |
| `vae/qwen_image_vae.safetensors` | 242 MB | `circlestone-labs/Anima :: split_files/vae/qwen_image_vae.safetensors` | `a70580f0213e6796…` |

### Hugging Face — diffusers folders

Whole-folder copies of a Civitai model converted to diffusers by John6666.
`CheckpointLoaderSimple` cannot read this layout; `DiffusersLoader` can.

| folder | size | repo | files verified |
|---|---|---|---|
| `diffusers/amanatsu-il-v11` | 6.46 GB | `John6666/amanatsu-illustrious-v11-sdxl` | 4/4 |
| `diffusers/hassaku-il-v22` | 6.46 GB | `John6666/hassaku-xl-illustrious-v22-sdxl` | 4/4 |
| `diffusers/moe-vpred-v2` | 6.46 GB | `John6666/moe-is-all-you-need-noobai-illustrious-vpred-v2-sdxl` | 4/4 |
| `diffusers/sweet-mix-v14` | 6.46 GB | `John6666/sweet-mix-illustriousxl-v14-sdxl` | 4/4 |

### Civitai

| file | size | model / version | link |
|---|---|---|---|
| `checkpoints/miaomiaoPixel_vPred11.safetensors` | 6.46 GB | MiaoMiao Pixel / V-Pred_1.1 | [1180112/2316419](https://civitai.com/models/1180112?modelVersionId=2316419) |
| `checkpoints/novaAnimeXL_ilV170.safetensors` | 6.46 GB | Nova Anime XL / IL V17.0 | [376130/2741698](https://civitai.com/models/376130?modelVersionId=2741698) |
| `diffusion_models/fnMomentAnimaTurbo_v20.safetensors` | 3.90 GB | Fn-Moment Anima-Turbo / v2.0 | [2733842/3078189](https://civitai.com/models/2733842?modelVersionId=3078189) |
| `diffusion_models/silvermoonmixAnimaEvolved_v2329BTurbo.safetensors` | 5.44 GB | SilvermoonMix-Anima-Evolved [2.9B/2B] / v2.3_2.9B_Turbo | [2639339/3335132](https://civitai.com/models/2639339?modelVersionId=3335132) |
| `loras/add-micro-details-ill-v6.safetensors` | 218 MB | Add Micro Details - Concept (Illustrious \| Pony \| NoobAI) / v6.0_Illustrious | [1377820/2832991](https://civitai.com/models/1377820?modelVersionId=2832991) |
| `loras/anima-handdrawn-feel-chosen.safetensors` | 175 MB | 手绘质感增强/Hand-drawn feel, trigger `Hand-drawn feel` / v2.0 anima | [2529114/3267407](https://civitai.com/models/2529114?modelVersionId=3267407) |
| `loras/anima-sketch-style-chosen.safetensors` | 175 MB | 草稿线条风格/sketch style, trigger `sketch style` / v2.0 anima | [2061990/3218039](https://civitai.com/models/2061990?modelVersionId=3218039) |
| `loras/detail-slider-ill.safetensors` | 8 MB | Detail Slider LoRA \| Illustrious-XL / v1.0 - Initial Release | [1001945/1122976](https://civitai.com/models/1001945?modelVersionId=1122976) |
| `loras/detailed-perfection-ill.safetensors` | 435 MB | Detailed Perfection style (Hands + Feet + Face + Body + All in one) XL + F1D + SD1.5 + Pony + Illu + zit + zib / Detailed Illu v0.9 | [411088/1506333](https://civitai.com/models/411088?modelVersionId=1506333) |
| `loras/feet-xl-ill.safetensors` | 435 MB | Feet XL + SD 1.5 + F1D + Pony + Illustrious + zit / Feet Illu v0.1 | [200251/1464471](https://civitai.com/models/200251?modelVersionId=1464471) |
| `loras/glossy-eyedetail-ill.safetensors` | 218 MB | Eye detail LoRA [Illustrious+FLUX+Z Image Turbo] / Glossy EyeDetail | [1300857/2552350](https://civitai.com/models/1300857?modelVersionId=2552350) |
| `loras/hands-xl-ill.safetensors` | 435 MB | Hands XL + SD 1.5 + F1D + Pony + Illustrious + zit + ZIB / Hands Illu v1.1 | [200255/2212079](https://civitai.com/models/200255?modelVersionId=2212079) |
| `loras/miru-tights-ill.safetensors` | 218 MB | Miru Tights Pack (Characters and style) - NatMontero / illustrious v1.0 | [997012/1680610](https://civitai.com/models/997012?modelVersionId=1680610) |
| `loras/moe-2000s-a.safetensors` | 244 MB | moe style / v1.0 | [2030057/2297509](https://civitai.com/models/2030057?modelVersionId=2297509) |
| `loras/moe-2000s-b.safetensors` | 37 MB | moe style 2000s / v1.0 | [2031779/2299478](https://civitai.com/models/2031779?modelVersionId=2299478) |
| `loras/mozudoll.safetensors` | 325 MB | Mozudoll-style — Anime Soft-body Moe Doll Aesthetic / v1.0 | [2461974/2784868](https://civitai.com/models/2461974?modelVersionId=2784868) |
| `loras/outlined-ill.safetensors` | 177 MB | Outlined / Illustrious | [2741162/3085331](https://civitai.com/models/2741162?modelVersionId=3085331) |
| `loras/perfect-eyes-ill.safetensors` | 218 MB | Eyes for Illustrious Lora (Perfect anime eyes) / V1 | [1826240/2066663](https://civitai.com/models/1826240?modelVersionId=2066663) |
| `loras/re4lity.safetensors` | 218 MB | Anime X Reality \| Shiiro's Styles / reality_sync | [1163407/1308731](https://civitai.com/models/1163407?modelVersionId=1308731) |
| `loras/shiny-legwear-ill.safetensors` | 163 MB | Shiny Legwear / illustr v0.11111 | [1142238/1284558](https://civitai.com/models/1142238?modelVersionId=1284558) |
| `loras/smooth-booster-v5.safetensors` | 188 MB | Smooth Detailer Booster (NoobAI/Illustrious/Pony) / Smooth Booster v5 | [1145743/2746768](https://civitai.com/models/1145743?modelVersionId=2746768) |
| `loras/usnr-style-ill-v1.safetensors` | 874 MB | 薄塗り / USNR STYLE / USNR_STYLE_ILL_V1.0 | [176554/1552087](https://civitai.com/models/176554?modelVersionId=1552087) |

`checkpoints/NoobAI-XL-v1.1.safetensors` is on Civitai as well
(833294/1116447) but Hugging Face is the better source for it.

## Removed from the worker (2026-10-02)

Fetched straight onto the Windows box for the checkpoint and LoRA A/B rounds
(`docs/archive/render-notes/6-2026-09.md`), then deleted. No local copy and no
SHA256 recorded; re-fetch from the source if they are ever needed again. The
checkpoints rated bad against hassaku-il-v22, and the sketch LoRAs served the
retired `yukari-sketch` recipe.

| file | source |
|---|---|
| `checkpoints/Illustrious-XL-v0.1.safetensors` | `OnomaAIResearch/Illustrious-xl-early-release-v0` |
| `checkpoints/blue_pencil-XL-v7.0.0.safetensors` | `bluepen5805/blue_pencil-XL` |
| `checkpoints/animagine-xl-3.1.safetensors` | `cagliostrolab/animagine-xl-3.1` |
| `checkpoints/animagine-xl-4.0-opt.safetensors` | `cagliostrolab/animagine-xl-4.0` |
| `checkpoints/XL_caulkinumACA.safetensors` | `Shiyaku/XL_caulkinumAnimeLine :: model/XL_caulkinumACA.safetensors` |
| `checkpoints/AAM_XL_Anime_Mix.safetensors` | `Lykon/AAM_XL_AnimeMix` |
| `checkpoints/novaMoeXL_v10.safetensors` | Civitai Nova Moe XL / v1.0 (version 2242867) |
| `checkpoints/novaRetroXL_v10.safetensors` | Civitai Nova Retro XL / v1.0 (version 2229136) |
| `loras/sketch-style-xl-linaqruf.safetensors` | `Linaqruf/sketch-style-xl-lora :: sketch-style-xl.safetensors` — used by the retired `yukari-sketch` at 0.8 |
| `loras/sketch-worthyhuman.safetensors` | `WorthyHuman1/Sketch_LoRA :: Sketch_LoRA.safetensors` — no visible effect |
| `loras/anime-sketch-muapi.safetensors` | `Muapi/anime-sketch-style-sdxl-sd1.5` — good raw, collapses in the redraw |
| `diffusion_models/qwen_image_layered_fp8mixed.safetensors`, `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`, `vae/qwen_image_layered_vae.safetensors` | not recorded — Qwen-Image-Layered for ComfyUI |
| `background_removal/birefnet.safetensors` | unknown |
| `loras/anima-sketch-style-chosen.safetensors`, `loras/anima-handdrawn-feel-chosen.safetensors` | Civitai, see the inventory above — a `render.loras` patch at strength `0` drops the loader, so inherited patches still render |

The worker's NoobAI-XL, IP-Adapter, CLIP-ViT-H and `noob-openpose-fp16` copies
were deleted the same day; their upstreams are in the inventory above.
