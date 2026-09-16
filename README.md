# ComfyUI RunPod Template (FLUX.2 Klein 9B)

Docker image: `promptalchemist/comfyui-runpod-jnkprod`

Boots straight into ComfyUI with the FLUX.2 Klein 9B model set and a fixed set
of custom nodes. Models are downloaded to the pod's volume on the first start
and reused on every later start.

| What | Version |
| --- | --- |
| ComfyUI | 0.36.0 |
| PyTorch | 2.13.0+cu130 (CUDA 13.0) |
| Python | 3.12 |
| SageAttention | 2.2.0 (on by default) |
| triton | 3.7.1 |

## Ports

| Port | Service |
| --- | --- |
| 8188 | ComfyUI |
| 8888 | JupyterLab (no token unless `JUPYTER_TOKEN` is set) |

Open them from the pod's **Connect** menu. Read [Security](#security) before
sharing a pod.

## Pod setup

- **Volume:** provision at least 60 GB at `/workspace`. The models take about
  30 GB; the rest is room for outputs and for downloads in progress.
- **Image tag:** every build is pushed as `:latest` and as a dated
  `:YYYYMMDD` tag, for example
  `promptalchemist/comfyui-runpod-jnkprod:20260917`. Pin the RunPod template
  to a dated tag so a new build never changes a working setup under you.

## Environment variables

Set these on the RunPod template or pod before the first start.

| Variable | Required | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | **Yes** | Hugging Face token with *read* access. `black-forest-labs/FLUX.2-klein-9B` is gated: open its page, accept the license, then create a token at https://huggingface.co/settings/tokens |
| `CIVITAI_TOKEN` | **Yes** for the Consistence Edit LoRA | Civitai API key from https://civitai.com/user/account (the other two Civitai LoRAs download without it) |
| `MODELS_CONFIG_URL` | No | URL of your own `models_config.json`; used on the first start only |
| `SKIP_MODEL_DOWNLOAD` | No | `true` skips the download step |
| `USE_SAGE_ATTENTION` | No | Default `true`. `false`, `0`, `no` or `off` (any letter case) start ComfyUI without `--use-sage-attention` |
| `COMFYUI_EXTRA_ARGS` | No | Extra ComfyUI flags, for example `--fast` |
| `COMFYUI_RESTART_DELAY` | No | Seconds to wait before restarting ComfyUI after a crash. Default `10` |
| `JUPYTER_TOKEN` | No | Token JupyterLab asks for when you open it. Empty (the default) means no authentication |
| `MAX_CONCURRENT_DOWNLOADS` | No | Files downloaded at the same time. Default `5` |
| `USE_HF_XET` | No | Default `true`. `false` downloads Hugging Face files with aria2c instead of the official client |
| `DOWNLOAD_HEARTBEAT_SECONDS` | No | How often the log lists the files still downloading. Default `60` |

Tokens are only sent to their own site and never written to the logs.

## First start

1. JupyterLab is up within seconds on port 8888.
2. The missing models are downloaded to `/workspace/models` (about 30 GB in
   total). Watch progress in a JupyterLab terminal:
   `tail -f /workspace/logs/comfyui.log`. aria2c downloads log a progress
   summary every 30 seconds, and a `Still downloading` line lists the files
   in flight every `DOWNLOAD_HEARTBEAT_SECONDS` (default 60).
3. ComfyUI starts on port 8188 once the downloads have finished. Its startup
   command line is written to `/workspace/logs/comfyui.log`, so you can check
   there to confirm `--use-sage-attention` is active. If ComfyUI crashes, it
   is restarted automatically after `COMFYUI_RESTART_DELAY` seconds
   (default 10).

Later starts skip files that are already on the volume, so ComfyUI is up in
under a minute. If the pod stops mid-download, interrupted downloads resume
automatically on the next start.

## What is on the volume

| Path | Contents |
| --- | --- |
| `/workspace/models/<type>/` | all models (`diffusion_models`, `text_encoders`, `vae`, `loras`, ...) |
| `/workspace/output` | generated images |
| `/workspace/input` | uploaded inputs |
| `/workspace/user` | saved workflows, ComfyUI settings, ComfyUI-Manager config |
| `/workspace/logs/comfyui.log` | ComfyUI and downloader log |
| `/workspace/models_config.json` | the model list this pod downloads |

ComfyUI itself and the custom nodes live in the image, not on the volume.
Nodes installed through ComfyUI-Manager work until the pod is recreated.
Python packages installed inside the pod (by pip, uv or ComfyUI-Manager) are
constrained to the image's torch/numpy versions, so a node that needs other
versions fails to install instead of breaking ComfyUI.

## Included models

| Type | File | Source |
| --- | --- | --- |
| diffusion_models | `flux-2-klein-9b.safetensors` | black-forest-labs/FLUX.2-klein-9B |
| text_encoders | `qwen3vl_8b_fp8_scaled.safetensors` | kiwidebringue/QwenTextencoder |
| vae | `flux2-vae.safetensors` | Comfy-Org/flux2-dev |
| loras | `Klein_realistic_I2I.safetensors` | kiwidebringue/kleini2i |
| loras | `HighResolution9B.safetensors` | Civitai 2436859 |
| loras | `Samsung_fluxklein9b.safetensors` | Civitai 1551668 |
| loras | `f2k_9B_lcs_consist_20260415.safetensors` | Civitai 1939453 |

## Adding models

Either drop files into the matching folder under `/workspace/models/`, or edit
`/workspace/models_config.json` and restart the pod. Each entry is a URL, or
an object with `url` and `filename` when the URL does not end in a filename:

```json
{
  "loras": [
    "https://huggingface.co/org/repo/resolve/main/my_lora.safetensors",
    {"url": "https://civitai.com/api/download/models/123456", "filename": "my_lora.safetensors"}
  ]
}
```

Hugging Face URLs are downloaded with the official client (Xet); everything
else with aria2c.

`MODELS_CONFIG_URL` and the built-in list are only read when
`/workspace/models_config.json` does not exist yet. To pick up a changed URL
or a newer built-in list, delete `/workspace/models_config.json` and restart
the pod: it is re-fetched from `MODELS_CONFIG_URL`, or re-copied from the
image when that is unset or cannot be fetched.

## Security

Ports 8188 and 8888 have no authentication unless you set JUPYTER_TOKEN.
Anyone with your pod URL can use ComfyUI and open a root terminal in
JupyterLab, which can read HF_TOKEN and CIVITAI_TOKEN. Do not share the URL.

`JUPYTER_TOKEN` protects JupyterLab only; ComfyUI itself has no login.

## Custom nodes

ComfyUI-Manager, rgthree-comfy, CRT-Nodes, ComfyUI-FameGridColorFinish,
ComfyUi-TextEncodeEditAdvanced, comfyui_controlnet_aux, Onyx_Custom_Nodes.

## Building the image yourself

```bash
docker build --secret id=github_token,src=/path/to/github_token.txt \
  -t promptalchemist/comfyui-runpod-jnkprod .
```

The secret is a GitHub token with read access to the private
`onyxaipro/Onyx_Custom_Nodes` repository. `scripts/build-sageattention.sh`
rebuilds the SageAttention wheel on a RunPod pod should the torch version
change.
