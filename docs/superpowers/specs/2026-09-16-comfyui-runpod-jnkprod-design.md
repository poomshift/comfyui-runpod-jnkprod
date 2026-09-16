# ComfyUI RunPod template for JNK Prod — design

Date: 2026-09-16

## Goal

A RunPod template (Docker image) that boots straight into ComfyUI 0.34.2 on port
8188 and JupyterLab on port 8888, with the customer's custom nodes baked into
the image and the customer's FLUX.2 Klein 9B model set downloaded onto the
persistent volume on first start via Hugging Face Xet.

Derived from `../comfyui-docker-new` (same author). Everything dashboard-related
from that project (port 8189, FastAPI log viewer, downloader UI, `static/`,
`templates/`, `workers/`, `dto/`, `constants/`) is dropped.

## Non-goals

- No web dashboard, no port 8189.
- ComfyUI itself is not copied to the volume. Nodes installed at runtime via
  ComfyUI-Manager do not survive a pod rebuild. (ComfyUI-Manager is not
  installed; the customer did not ask for it.)
- No SageAttention 3, no CUDA 12 variant, no Python other than 3.12.

## Pinned versions

| Component | Version | Source |
| --- | --- | --- |
| Base image | `nvidia/cuda:13.0.3-base-ubuntu24.04` | Docker Hub. Ubuntu 24.04 ships Python 3.12 natively, no deadsnakes |
| Python | 3.12 | apt `python3.12`, venv at `/opt/venv` |
| PyTorch | `torch==2.13.0+cu130`, `torchvision==0.28.0+cu130`, `torchaudio==2.11.0+cu130` | `https://download.pytorch.org/whl/cu130`. torchaudio has no 2.13 release; 2.11.0 is the last one and declares no torch pin |
| triton | 3.7.1 | pulled in by torch 2.13.0 (`triton==3.7.1` on Linux). Matches the customer's `triton_windows 3.7.1.post27` |
| ComfyUI | tag `v0.34.2` | `git clone --branch v0.34.2` into `/opt/ComfyUI` |
| comfy-kitchen / comfy-aimdo | 0.2.31 / 0.4.15 | pinned by ComfyUI v0.34.2 `requirements.txt` |
| SageAttention | `2.2.0+cu130.torch2.13.0` | prebuilt wheel produced by `scripts/build-sageattention.sh` on a RunPod pod (2026-09-17, arches 8.0/8.6/8.9/9.0/12.0, verified against SDPA on the GPU). Hosted at `https://huggingface.co/Patarapoom/sageattention-wheels/resolve/main/sageattention-2.2.0+cu130.torch2.13.0-cp312-cp312-linux_x86_64.whl`, which is the default of build arg `SAGEATTENTION_WHEEL_URL` |
| huggingface_hub | latest 1.x with `[hf_xet]` | PyPI |
| JupyterLab | latest 4.x | PyPI |

A `constraints.txt` pins torch, torchvision, torchaudio, triton and numpy and is
passed as `-c` to every `pip install` in the Dockerfile, so no custom node
requirement (ultralytics, openai-whisper, mediapipe, ...) can move torch.

## Repository layout

```
Dockerfile
.dockerignore
start.sh                      boot script (CMD)
constraints.txt               torch/triton pins used by every pip install
extra_model_paths.yaml        points ComfyUI at /workspace/models
models_config.json            default model list, baked into the image
download_models.py            boot-time downloader (from reference, trimmed)
utils/hfDownload.py           HF resolve-URL parser + hf_hub_download wrapper (from reference)
utils/hfAuth.py               HF token lookup + redaction (from reference)
utils/civitai.py              Civitai token header + filename resolution (new)
scripts/build-sageattention.sh  one-off wheel build, run on a pod
.github/workflows/docker-build.yml
README.md                     customer-facing
docs/superpowers/specs/       this document
```

## Dockerfile

Single stage. In order:

1. apt: `python3.12 python3.12-venv python3.12-dev git build-essential
   libgl1 libglib2.0-0 ffmpeg aria2 curl ca-certificates`. Clean apt lists.
2. `python3.12 -m venv /opt/venv`, `PATH=/opt/venv/bin:$PATH`, install `uv`
   into the venv and use `uv pip install` for speed.
3. `uv pip install -c constraints.txt torch torchvision torchaudio
   --index-url https://download.pytorch.org/whl/cu130`.
4. `git clone --depth 1 --branch v0.34.2 https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI`
   and `uv pip install -c constraints.txt -r /opt/ComfyUI/requirements.txt`.
5. `uv pip install $SAGEATTENTION_WHEEL_URL` (build arg, required).
6. Custom nodes into `/opt/ComfyUI/custom_nodes`, each `git clone --depth 1`:
   - `rgthree/rgthree-comfy`
   - `PGCRT/CRT-Nodes`
   - `Elevenheights/ComfyUI-FameGridColorFinish`
   - `BigStationW/ComfyUi-TextEncodeEditAdvanced`
   - `Fannovel16/comfyui_controlnet_aux`
   - `onyxaipro/Onyx_Custom_Nodes` — private. Cloned with
     `RUN --mount=type=secret,id=github_token` using
     `git -c http.extraheader="AUTHORIZATION: bearer $(cat /run/secrets/github_token)"`,
     so the token never lands in a layer or in `.git/config`.
   Then one `uv pip install -c constraints.txt -r` per `requirements.txt`
   found under `custom_nodes/`, plus any `install.py` a node ships. A failing
   node install fails the build (no `|| true`) so a broken node is noticed at
   build time, not on the customer's pod.
7. `uv pip install -c constraints.txt jupyterlab "huggingface_hub[hf_xet]" aiohttp`.
8. Copy `start.sh`, `download_models.py`, `utils/`, `models_config.json`,
   `extra_model_paths.yaml`. Copy `extra_model_paths.yaml` into `/opt/ComfyUI/`
   where ComfyUI auto-loads it.
9. Build-time smoke test as a `RUN`:
   `python -c "import torch, torchvision, torchaudio, triton, sageattention, comfy_kitchen, comfy_aimdo"`
   and `python /opt/ComfyUI/main.py --cpu --quick-test-for-ci` (loads every
   custom node once). Fails the build if any node cannot import.
10. `EXPOSE 8188 8888`, `CMD ["/start.sh"]`.

Version labels (`org.opencontainers.image.*`) record ComfyUI tag, torch and
SageAttention versions.

## Runtime layout and persistence

| Path | Lives in | Purpose |
| --- | --- | --- |
| `/opt/ComfyUI` | image | ComfyUI + custom nodes, read-mostly |
| `/opt/venv` | image | Python |
| `/workspace/models/<category>` | volume | all models; `extra_model_paths.yaml` sets `is_default: true` so ComfyUI reads and writes here |
| `/workspace/output`, `/workspace/input` | volume | `--output-directory`, `--input-directory` |
| `/workspace/user` | volume | `--user-directory`: saved workflows, settings |
| `/workspace/logs/comfyui.log` | volume | ComfyUI + downloader log |
| `/workspace/.cache/huggingface` | volume | `HF_HOME`, holds the Xet chunk cache |
| `/workspace/models_config.json` | volume | the effective model list (see below) |

`extra_model_paths.yaml` lists every ComfyUI model folder name (checkpoints,
vae, diffusion_models, text_encoders, loras, clip, clip_vision, controlnet,
upscale_models, unet, ipadapter, style_models, model_patches, embeddings) under
`base_path: /workspace/models`. `start.sh` creates all of them.

## start.sh

```
set defaults (env table below)
mkdir -p /workspace/{models/*,output,input,user,logs,.cache/huggingface}
start JupyterLab on 8888 immediately, root /workspace, no token (CUDA_VISIBLE_DEVICES="")
resolve models_config.json:
   - if /workspace/models_config.json missing: fetch MODELS_CONFIG_URL if set,
     else copy the baked-in /models_config.json
python /download_models.py           # skips files already present; log to comfyui.log
   - SKIP_MODEL_DOWNLOAD=true skips this step entirely
   - a failed download logs an error and does NOT abort boot
start ComfyUI on 8188 with the flags in the table, tee to comfyui.log
wait
```

ComfyUI is started after the downloads finish, so the first thing the customer
sees in ComfyUI already has every model. JupyterLab is up from the first seconds
so the customer can `tail -f /workspace/logs/comfyui.log`.

ComfyUI command:

```
python main.py --listen 0.0.0.0 --port 8188 \
  --output-directory /workspace/output --input-directory /workspace/input \
  --user-directory /workspace/user \
  [--use-sage-attention]   # default on
  $COMFYUI_EXTRA_ARGS
```

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | empty | Required: `black-forest-labs/FLUX.2-klein-9B` is gated. Customer must accept the license on HF first |
| `CIVITAI_TOKEN` | empty | Required for `Consistence Edit Lora` (Civitai returns 401 without it). The other two Civitai LoRAs download anonymously |
| `MODELS_CONFIG_URL` | empty | Optional URL of a `models_config.json` to use instead of the baked-in one, on first boot only |
| `SKIP_MODEL_DOWNLOAD` | `false` | Skip the download step |
| `USE_SAGE_ATTENTION` | `true` | Adds `--use-sage-attention` |
| `COMFYUI_EXTRA_ARGS` | empty | Appended verbatim to the ComfyUI command |
| `HF_HOME` | `/workspace/.cache/huggingface` | Xet chunk cache location |
| `HF_XET_CHUNK_CACHE_SIZE_BYTES` | `8589934592` (8 GiB) | Chunk cache cap. Smaller than the reference project because this model set has little cross-model overlap |
| `HF_XET_HIGH_PERFORMANCE` | `1` | More concurrency in the Xet client |
| `HF_HUB_DISABLE_PROGRESS_BARS` | `1` | Keeps the log readable |

Tokens are only ever sent to their own host (`huggingface.co` / `civitai.com`)
and are redacted from log lines (reuse `utils/hfAuth.redact_token`, extended to
cover `CIVITAI_TOKEN`).

## Model download

`models_config.json` maps ComfyUI model folder → list of entries. An entry is
either a URL string (filename = last path segment) or an object
`{"url": ..., "filename": ...}` for URLs whose last segment is not a filename
(Civitai). Baked-in default:

```json
{
  "diffusion_models": [
    "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors"
  ],
  "text_encoders": [
    "https://huggingface.co/kiwidebringue/QwenTextencoder/resolve/main/qwen3vl_8b_fp8_scaled.safetensors"
  ],
  "vae": [
    "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"
  ],
  "loras": [
    "https://huggingface.co/kiwidebringue/kleini2i/resolve/main/Klein_realistic_I2I.safetensors",
    {"url": "https://civitai.com/api/download/models/2760799?fileId=2647078", "filename": "HighResolution9B.safetensors"},
    {"url": "https://civitai.com/api/download/models/2777498?fileId=2663630", "filename": "Samsung_fluxklein9b.safetensors"},
    {"url": "https://civitai.com/api/download/models/2863285?fileId=2747224", "filename": "f2k_9B_lcs_consist_20260415.safetensors"}
  ]
}
```

`f2k_9B_lcs_consist_20260415` is the newest Klein 9B version of "Consistence
Edit Lora"; the customer's link named no version.

Backend per URL, as in the reference project:

- `huggingface.co/<repo>/resolve/<rev>/<path>` → `hf_hub_download` with
  `HF_TOKEN`, which uses Xet where the repo has it. Staged under
  `/workspace/.hf_staging/<name>` then moved into place so `.cache` metadata
  never lands in `models/`. On any exception, fall back to aria2c.
- everything else → `aria2c -x 4 -s 4 -c`, with
  `Authorization: Bearer $CIVITAI_TOKEN` added for `civitai.com` hosts and
  `Authorization: Bearer $HF_TOKEN` for `huggingface.co` hosts.

Downloads run concurrently, at most 5 at a time. A file is skipped when
`<category>/<filename>` already exists. Partial aria2c downloads resume; the HF
client restarts the file. Failures are logged and the script exits 0 so ComfyUI
still starts.

## CI

`.github/workflows/docker-build.yml`, copied from the reference project:

- on push to `main` (and manual dispatch): build with Buildx, push
  `promptalchemist/comfyui-runpod-jnkprod:latest` and `:<YYYYMMDD>`.
- on pull request: build only.
- secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `ONYX_GITHUB_TOKEN`
  (passed as Buildx secret `github_token`).
- build arg `SAGEATTENTION_WHEEL_URL` set from a repository variable so the
  wheel can be swapped without editing the Dockerfile.
- GHA layer cache (`type=gha,mode=max`). Free runner disk is tight for a
  ~12 GB image, so the workflow removes `/usr/share/dotnet`,
  `/opt/ghc` and `/usr/local/lib/android` before building, as the reference's
  disabled GHCR workflow does.

GitHub repository: `poomshift/comfyui-runpod-jnkprod`.

## Testing

- **Unit (local, no GPU, `pytest`)**: `utils/hfDownload.parse_hf_url`,
  `utils/civitai` header/filename logic, `download_models` config parsing
  (string vs object entries, skip-existing, unknown category warning),
  token redaction. Network calls mocked.
- **Build-time**: the import smoke test and `--quick-test-for-ci` run inside
  `docker build`, so a green CI build proves every custom node imports against
  torch 2.13.
- **Pod acceptance (manual, on RunPod)**: start the template with `HF_TOKEN`
  and `CIVITAI_TOKEN`; expect JupyterLab within ~30 s, all 7 files under
  `/workspace/models` after the download, ComfyUI on 8188 with
  `--use-sage-attention` in its startup log, a Klein 9B workflow running
  end-to-end. Restart the pod and confirm nothing is re-downloaded.

## Risks and open items

- **torchaudio 2.11.0 with torch 2.13.0**: no 2.13 build exists. The
  build-time import test catches an ABI break; fallback is a torchaudio built
  from source or dropping it if ComfyUI tolerates its absence.
- **Onyx_Custom_Nodes**: private repo, contents not yet inspected. Its
  requirements may need extra apt packages; discovered at first build.
- **CRT-Nodes** pulls heavy deps (ultralytics, openai-whisper, faster-whisper,
  librosa, pedalboard). Adds several GB to the image. Accepted; the customer
  asked for it.
- **mediapipe** (comfyui_controlnet_aux) latest 1.0.x has no cp312 Linux wheel;
  pip will resolve an older 0.10.x. Only affects a few preprocessors.
- **SageAttention wheel**: built and uploaded (see version table). Upstream
  v2.2.0 `setup.py` compiles the Hopper-only sm90 extension for every arch;
  the build script patches it to sm_90a only. Rebuild needed only if torch
  changes minor version.

## Customer-facing README outline

Ports, required env vars with where to get each token (accept the Klein
license first), what is on the volume, how to add models
(`/workspace/models_config.json` or drop files under `/workspace/models`),
how to see logs, how to disable SageAttention.
