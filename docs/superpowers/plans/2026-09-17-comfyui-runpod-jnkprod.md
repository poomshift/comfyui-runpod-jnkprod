# ComfyUI RunPod template (JNK Prod) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Docker image for RunPod that boots ComfyUI 0.34.2 (port 8188) and JupyterLab (port 8888) with the customer's custom nodes baked in and the FLUX.2 Klein 9B model set downloaded to the persistent volume on first start.

**Architecture:** Single-stage Dockerfile on `nvidia/cuda:13.0.3-base-ubuntu24.04` with a Python 3.12 venv, ComfyUI and custom nodes under `/opt`, the boot script and downloader under `/app`. `start.sh` starts JupyterLab, runs `download_models.py` (Hugging Face client with Xet for HF URLs, aria2c for everything else), then starts ComfyUI pointed at `/workspace` for models, input, output and user data. GitHub Actions builds and pushes to Docker Hub.

**Tech Stack:** Docker (BuildKit secrets), bash, Python 3.12 (asyncio, huggingface_hub[hf_xet], aria2c), pytest, GitHub Actions, RunPod.

**Spec:** `docs/superpowers/specs/2026-09-16-comfyui-runpod-jnkprod-design.md`

## Global Constraints

- Python 3.12 only. Base image `nvidia/cuda:13.0.3-base-ubuntu24.04`.
- `torch==2.13.0+cu130`, `torchvision==0.28.0+cu130`, `torchaudio==2.11.0+cu130`, `triton==3.7.1` from `https://download.pytorch.org/whl/cu130`. Every `pip install` in the Dockerfile passes `-c /app/constraints.txt`.
- ComfyUI at git tag `v0.34.2` in `/opt/ComfyUI` (brings `comfy-kitchen==0.2.31`, `comfy-aimdo==0.4.15`).
- SageAttention wheel: `https://huggingface.co/Patarapoom/sageattention-wheels/resolve/main/sageattention-2.2.0+cu130.torch2.13.0-cp312-cp312-linux_x86_64.whl` (default of build arg `SAGEATTENTION_WHEEL_URL`).
- Custom nodes (7): `Comfy-Org/ComfyUI-Manager`, `rgthree/rgthree-comfy`, `PGCRT/CRT-Nodes`, `Elevenheights/ComfyUI-FameGridColorFinish`, `BigStationW/ComfyUi-TextEncodeEditAdvanced`, `Fannovel16/comfyui_controlnet_aux`, `onyxaipro/Onyx_Custom_Nodes` (private, cloned with BuildKit secret `github_token`).
- Ports: 8188 ComfyUI, 8888 JupyterLab. No dashboard, no 8189.
- Image name `promptalchemist/comfyui-runpod-jnkprod`. GitHub repo `poomshift/comfyui-runpod-jnkprod`.
- Tokens (`HF_TOKEN`, `CIVITAI_TOKEN`) are only sent to their own host and never appear in logs.
- Commits: `git -c user.name="patarapoom" -c user.email="police35region4@gmail.com" commit ...` (no global git identity is configured on this machine). End commit messages with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Local test environment: `uv venv .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt`, run tests with `.venv/bin/python -m pytest -q`. Local Python is 3.11; the code must not use 3.12-only syntax.

## File structure

| File | Responsibility |
| --- | --- |
| `constraints.txt` | torch/triton pins for every pip install |
| `extra_model_paths.yaml` | tells ComfyUI every model folder lives under `/workspace/models` |
| `models_config.json` | default model list baked into the image |
| `utils/__init__.py` | package marker |
| `utils/hfAuth.py` | HF token lookup, HF host check, aria2c auth header, token redaction (HF + Civitai) |
| `utils/civitai.py` | Civitai token lookup, host check, aria2c auth header |
| `utils/hfDownload.py` | HF resolve-URL parser and `hf_hub_download` wrapper with staging dir |
| `download_models.py` | reads config, plans jobs, downloads concurrently, logs |
| `start.sh` | boot: dirs, config, JupyterLab, downloads, ComfyUI |
| `Dockerfile`, `.dockerignore` | image |
| `.github/workflows/docker-build.yml` | build + push on `main`, build-only on PR |
| `.github/workflows/tests.yml` | pytest on every push |
| `README.md` | customer-facing |
| `tests/test_*.py` | unit tests, no network |
| `requirements-dev.txt` | pytest, pyyaml, huggingface_hub |

---

### Task 1: Dev environment, constraints, model paths and default model config

**Files:**
- Create: `requirements-dev.txt`, `constraints.txt`, `extra_model_paths.yaml`, `models_config.json`, `tests/__init__.py`, `tests/test_config_files.py`

**Interfaces:**
- Produces: `models_config.json` schema used by Task 5: top-level object, key = ComfyUI model folder name, value = list of entries; an entry is a URL string or `{"url": str, "filename": str}`.

- [ ] **Step 1: Create the dev requirements and venv**

`requirements-dev.txt`:
```
pytest>=8
pyyaml>=6
huggingface_hub>=1.0
```

Run: `cd /Users/patarapoomsmacpro/Project/comfyui-runpod-jnkprod && uv venv .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt`
Expected: packages install without error. `.venv/` is already in `.gitignore`.

- [ ] **Step 2: Write the failing tests**

`tests/__init__.py`: empty file.

`tests/test_config_files.py`:
```python
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Every folder name ComfyUI v0.34.2 knows, minus custom_nodes (stays in the image)
COMFY_MODEL_FOLDERS = {
    "audio_encoders", "background_removal", "checkpoints", "classifiers",
    "clip_vision", "configs", "controlnet", "datasets", "detection",
    "diffusers", "diffusion_models", "embeddings", "frame_interpolation",
    "geometry_estimation", "gligen", "hypernetworks", "latent_upscale_models",
    "loras", "model_patches", "optical_flow", "photomaker", "style_models",
    "text_encoders", "upscale_models", "vae", "vae_approx",
}


def test_constraints_pin_torch_stack():
    text = (ROOT / "constraints.txt").read_text()
    for line in (
        "torch==2.13.0+cu130",
        "torchvision==0.28.0+cu130",
        "torchaudio==2.11.0+cu130",
        "triton==3.7.1",
    ):
        assert line in text.splitlines(), line


def test_extra_model_paths_cover_every_comfy_folder():
    cfg = yaml.safe_load((ROOT / "extra_model_paths.yaml").read_text())
    section = cfg["runpod"]
    assert section["base_path"] == "/workspace/models"
    assert section["is_default"] is True
    folders = {k for k in section if k not in ("base_path", "is_default")}
    assert folders == COMFY_MODEL_FOLDERS
    for name in folders:
        assert section[name] == name, f"{name} must map to a folder of the same name"


def test_models_config_entries_are_well_formed():
    cfg = json.loads((ROOT / "models_config.json").read_text())
    assert set(cfg) <= COMFY_MODEL_FOLDERS
    for category, entries in cfg.items():
        assert isinstance(entries, list), category
        for entry in entries:
            if isinstance(entry, str):
                assert entry.startswith("https://")
                assert entry.rsplit("/", 1)[-1].endswith(".safetensors")
            else:
                assert set(entry) == {"url", "filename"}, entry
                assert entry["url"].startswith("https://")
                assert entry["filename"].endswith(".safetensors")


def test_models_config_lists_the_customer_files():
    cfg = json.loads((ROOT / "models_config.json").read_text())
    names = set()
    for entries in cfg.values():
        for entry in entries:
            names.add(entry if isinstance(entry, str) else entry["filename"])
    for expected in (
        "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors",
        "https://huggingface.co/kiwidebringue/QwenTextencoder/resolve/main/qwen3vl_8b_fp8_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
        "https://huggingface.co/kiwidebringue/kleini2i/resolve/main/Klein_realistic_I2I.safetensors",
        "HighResolution9B.safetensors",
        "Samsung_fluxklein9b.safetensors",
        "f2k_9B_lcs_consist_20260415.safetensors",
    ):
        assert expected in names, expected
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config_files.py -q`
Expected: 4 failures, `FileNotFoundError` for `constraints.txt`.

- [ ] **Step 4: Create the three config files**

`constraints.txt`:
```
# Pins passed as `-c` to every pip install in the Dockerfile so that no custom
# node requirement can replace the CUDA 13 torch stack.
torch==2.13.0+cu130
torchvision==0.28.0+cu130
torchaudio==2.11.0+cu130
triton==3.7.1
```

`extra_model_paths.yaml`:
```yaml
# Loaded by ComfyUI from /opt/ComfyUI/extra_model_paths.yaml.
# Every model folder lives on the RunPod volume so downloads survive restarts.
# is_default makes these folders the first search path and the download target.
runpod:
  base_path: /workspace/models
  is_default: true
  audio_encoders: audio_encoders
  background_removal: background_removal
  checkpoints: checkpoints
  classifiers: classifiers
  clip_vision: clip_vision
  configs: configs
  controlnet: controlnet
  datasets: datasets
  detection: detection
  diffusers: diffusers
  diffusion_models: diffusion_models
  embeddings: embeddings
  frame_interpolation: frame_interpolation
  geometry_estimation: geometry_estimation
  gligen: gligen
  hypernetworks: hypernetworks
  latent_upscale_models: latent_upscale_models
  loras: loras
  model_patches: model_patches
  optical_flow: optical_flow
  photomaker: photomaker
  style_models: style_models
  text_encoders: text_encoders
  upscale_models: upscale_models
  vae: vae
  vae_approx: vae_approx
```

`models_config.json`:
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

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config_files.py -q`
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add requirements-dev.txt constraints.txt extra_model_paths.yaml models_config.json tests/
git -c user.name="patarapoom" -c user.email="police35region4@gmail.com" commit -m "Add torch constraints, model path config and default model list

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Token helpers (`utils/hfAuth.py`, `utils/civitai.py`)

**Files:**
- Create: `utils/__init__.py`, `utils/hfAuth.py`, `utils/civitai.py`, `tests/test_auth.py`

**Interfaces:**
- Produces (used by Tasks 3 and 5):
  - `utils.hfAuth.get_hf_token(explicit_token=None) -> str | None`
  - `utils.hfAuth.is_huggingface_url(url: str) -> bool`
  - `utils.hfAuth.hf_auth_args(url: str) -> list[str]` (aria2c args, `[]` when not applicable)
  - `utils.hfAuth.redact_token(text: str) -> str` (masks HF and Civitai tokens with `***`)
  - `utils.civitai.get_civitai_token() -> str | None`
  - `utils.civitai.is_civitai_url(url: str) -> bool`
  - `utils.civitai.civitai_auth_args(url: str) -> list[str]`

- [ ] **Step 1: Write the failing tests**

`tests/test_auth.py`:
```python
import pytest

from utils import civitai, hfAuth


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "CIVITAI_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_hf_token_prefers_explicit_then_env(monkeypatch):
    assert hfAuth.get_hf_token() is None
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_env")
    assert hfAuth.get_hf_token() == "hf_env"
    monkeypatch.setenv("HF_TOKEN", " hf_first ")
    assert hfAuth.get_hf_token() == "hf_first"
    assert hfAuth.get_hf_token("hf_explicit") == "hf_explicit"
    assert hfAuth.get_hf_token("   ") == "hf_first"


@pytest.mark.parametrize("url,expected", [
    ("https://huggingface.co/a/b/resolve/main/x.safetensors", True),
    ("https://hf.co/a/b/resolve/main/x.safetensors", True),
    ("https://cdn-lfs.huggingface.co/x", True),
    ("https://civitai.com/api/download/models/1", False),
    ("https://evilhuggingface.co/x", False),
    ("not a url", False),
])
def test_is_huggingface_url(url, expected):
    assert hfAuth.is_huggingface_url(url) is expected


def test_hf_auth_args_only_for_hf_hosts_with_token(monkeypatch):
    hf = "https://huggingface.co/a/b/resolve/main/x.safetensors"
    assert hfAuth.hf_auth_args(hf) == []
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    assert hfAuth.hf_auth_args(hf) == ["--header=Authorization: Bearer hf_abc"]
    assert hfAuth.hf_auth_args("https://civitai.com/api/download/models/1") == []


def test_civitai_token_and_host(monkeypatch):
    assert civitai.get_civitai_token() is None
    monkeypatch.setenv("CIVITAI_TOKEN", " civ_1 ")
    assert civitai.get_civitai_token() == "civ_1"
    assert civitai.is_civitai_url("https://civitai.com/api/download/models/1?fileId=2") is True
    assert civitai.is_civitai_url("https://civitai.red/models/1") is True
    assert civitai.is_civitai_url("https://huggingface.co/x") is False


def test_civitai_auth_args(monkeypatch):
    url = "https://civitai.com/api/download/models/1"
    assert civitai.civitai_auth_args(url) == []
    monkeypatch.setenv("CIVITAI_TOKEN", "civ_1")
    assert civitai.civitai_auth_args(url) == ["--header=Authorization: Bearer civ_1"]
    assert civitai.civitai_auth_args("https://huggingface.co/x") == []


def test_redact_token_masks_every_configured_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("CIVITAI_TOKEN", "civ_secret")
    text = "GET ?token=civ_secret Authorization: Bearer hf_secret done"
    assert hfAuth.redact_token(text) == "GET ?token=*** Authorization: Bearer *** done"
    assert hfAuth.redact_token("") == ""
    assert hfAuth.redact_token(None) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_auth.py -q`
Expected: `ModuleNotFoundError: No module named 'utils'`

- [ ] **Step 3: Implement the two modules**

`utils/__init__.py`: empty file.

`utils/hfAuth.py`:
```python
"""Hugging Face token handling, shared by the boot-time downloader."""
import os
from urllib.parse import urlparse

# Checked in order for a Hugging Face access token.
_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")

_HF_HOSTS = ("huggingface.co", "hf.co")

# Every env var whose value must never reach a log line.
_ALL_SECRET_ENV_VARS = _TOKEN_ENV_VARS + ("CIVITAI_TOKEN",)


def get_hf_token(explicit_token=None):
    """Return the token to use, preferring one passed in over the environment."""
    if explicit_token and explicit_token.strip():
        return explicit_token.strip()

    for name in _TOKEN_ENV_VARS:
        value = (os.getenv(name) or "").strip()
        if value:
            return value

    return None


def is_huggingface_url(url):
    """True only for Hugging Face hosts, so the token is never sent anywhere else."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False

    return any(host == h or host.endswith("." + h) for h in _HF_HOSTS)


def hf_auth_args(url, explicit_token=None):
    """aria2c args that authenticate a Hugging Face download, or [] when not applicable."""
    token = get_hf_token(explicit_token)

    if not token or not is_huggingface_url(url):
        return []

    return ["--header=Authorization: Bearer " + token]


def redact_token(text, explicit_token=None):
    """Strip every configured token value out of text before it is logged."""
    if not text:
        return text

    candidates = [explicit_token] + [os.getenv(name) for name in _ALL_SECRET_ENV_VARS]

    for candidate in candidates:
        if candidate and candidate.strip():
            text = text.replace(candidate.strip(), "***")

    return text
```

`utils/civitai.py`:
```python
"""Civitai token handling for the boot-time downloader."""
import os
from urllib.parse import urlparse

_CIVITAI_HOSTS = ("civitai.com", "civitai.red")


def get_civitai_token():
    """Return CIVITAI_TOKEN from the environment, or None."""
    value = (os.getenv("CIVITAI_TOKEN") or "").strip()
    return value or None


def is_civitai_url(url):
    """True only for Civitai hosts, so the token is never sent anywhere else."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False

    return any(host == h or host.endswith("." + h) for h in _CIVITAI_HOSTS)


def civitai_auth_args(url):
    """aria2c args that authenticate a Civitai download, or [] when not applicable."""
    token = get_civitai_token()

    if not token or not is_civitai_url(url):
        return []

    return ["--header=Authorization: Bearer " + token]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_auth.py -q`
Expected: `11 passed` (parametrized cases count individually)

- [ ] **Step 5: Commit**

```bash
git add utils/ tests/test_auth.py
git -c user.name="patarapoom" -c user.email="police35region4@gmail.com" commit -m "Add Hugging Face and Civitai token helpers with log redaction

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Hugging Face client download (`utils/hfDownload.py`)

**Files:**
- Create: `utils/hfDownload.py`, `tests/test_hf_download.py`

**Interfaces:**
- Consumes: `utils.hfAuth.get_hf_token`
- Produces (used by Task 5):
  - `parse_hf_url(url) -> dict | None` with keys `repo_id`, `repo_type`, `revision`, `path`
  - `hf_client_enabled() -> bool` (env `USE_HF_XET`, default true)
  - `download_via_hf(url, output_dir, filename, staging_root="/workspace/.hf_staging") -> str` (blocking; returns final path; raises on failure)

- [ ] **Step 1: Write the failing tests**

`tests/test_hf_download.py`:
```python
import os

import pytest

from utils import hfDownload


@pytest.mark.parametrize("url,expected", [
    (
        "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
        {"repo_id": "Comfy-Org/flux2-dev", "repo_type": "model", "revision": "main",
         "path": "split_files/vae/flux2-vae.safetensors"},
    ),
    (
        "https://huggingface.co/datasets/org/data/resolve/v1/a%20b.bin",
        {"repo_id": "org/data", "repo_type": "dataset", "revision": "v1", "path": "a b.bin"},
    ),
    ("https://huggingface.co/org/repo/blob/main/x.safetensors", None),
    ("https://cdn-lfs.huggingface.co/repos/abc", None),
    ("https://civitai.com/api/download/models/1", None),
    ("https://huggingface.co/a/b/c/resolve/main/x", None),
])
def test_parse_hf_url(url, expected):
    assert hfDownload.parse_hf_url(url) == expected


def test_hf_client_enabled_env(monkeypatch):
    monkeypatch.delenv("USE_HF_XET", raising=False)
    assert hfDownload.hf_client_enabled() is True
    monkeypatch.setenv("USE_HF_XET", "false")
    assert hfDownload.hf_client_enabled() is False
    monkeypatch.setenv("USE_HF_XET", "0")
    assert hfDownload.hf_client_enabled() is False


def test_download_via_hf_moves_file_into_place_and_cleans_staging(tmp_path, monkeypatch):
    calls = {}

    def fake_hf_hub_download(**kwargs):
        calls.update(kwargs)
        staged = os.path.join(kwargs["local_dir"], os.path.basename(kwargs["filename"]))
        os.makedirs(os.path.dirname(staged), exist_ok=True)
        with open(staged, "wb") as f:
            f.write(b"weights")
        return staged

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setenv("HF_TOKEN", "hf_x")

    out_dir = tmp_path / "models" / "vae"
    staging = tmp_path / "staging"
    result = hfDownload.download_via_hf(
        "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
        str(out_dir), "flux2-vae.safetensors", staging_root=str(staging),
    )

    assert result == str(out_dir / "flux2-vae.safetensors")
    assert (out_dir / "flux2-vae.safetensors").read_bytes() == b"weights"
    assert calls["repo_id"] == "Comfy-Org/flux2-dev"
    assert calls["filename"] == "split_files/vae/flux2-vae.safetensors"
    assert calls["token"] == "hf_x"
    assert not (staging / "flux2-vae.safetensors").exists()


def test_download_via_hf_rejects_non_hf_url(tmp_path):
    with pytest.raises(ValueError):
        hfDownload.download_via_hf("https://civitai.com/x", str(tmp_path), "x.bin", staging_root=str(tmp_path / "s"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hf_download.py -q`
Expected: `ImportError: cannot import name 'hfDownload'`

- [ ] **Step 3: Implement the module**

`utils/hfDownload.py`:
```python
"""Fetch Hugging Face files with the official client so Xet dedup applies."""
import os
import re
import shutil
from urllib.parse import unquote, urlparse

from utils.hfAuth import get_hf_token

# https://huggingface.co/<repo>/resolve/<revision>/<path within the repo>
_RESOLVE_RE = re.compile(r"^/(?P<repo>.+?)/resolve/(?P<revision>[^/]+)/(?P<path>.+)$")

# Only the site itself serves repos; the CDN hosts it redirects to do not.
_REPO_HOSTS = ("huggingface.co", "hf.co")

_REPO_TYPE_PREFIXES = {"datasets": "dataset", "spaces": "space"}


def parse_hf_url(url):
    """Split a Hugging Face resolve URL into the parts hf_hub_download needs.

    Returns None for anything else, including CDN links, which have to keep
    going through aria2c.
    """
    try:
        parts = urlparse(url)
    except ValueError:
        return None

    if (parts.hostname or "").lower() not in _REPO_HOSTS:
        return None

    match = _RESOLVE_RE.match(parts.path)
    if not match:
        return None

    repo = match.group("repo")
    repo_type = "model"

    head, _, rest = repo.partition("/")
    if head in _REPO_TYPE_PREFIXES and rest:
        repo_type = _REPO_TYPE_PREFIXES[head]
        repo = rest

    # A repo id is "name" or "org/name"; anything deeper is not one.
    if repo.count("/") > 1:
        return None

    return {
        "repo_id": repo,
        "repo_type": repo_type,
        "revision": unquote(match.group("revision")),
        "path": unquote(match.group("path")),
    }


def hf_client_enabled():
    """Whether to route Hugging Face URLs through the official client."""
    return (os.getenv("USE_HF_XET", "true") or "").strip().lower() not in ("false", "0", "no")


def download_via_hf(url, output_dir, filename, staging_root="/workspace/.hf_staging"):
    """Download one file with huggingface_hub, which uses Xet where the repo has it.

    Blocking; call it from a thread. Returns the final path. Raises on failure
    so the caller can fall back to aria2c.
    """
    import huggingface_hub

    info = parse_hf_url(url)
    if info is None:
        raise ValueError("not a Hugging Face resolve URL: " + url)

    # Staged separately so the client's .cache metadata never lands in models/
    staging = os.path.join(staging_root, re.sub(r"[^\w.-]", "_", filename))
    os.makedirs(staging, exist_ok=True)

    try:
        staged_path = huggingface_hub.hf_hub_download(
            repo_id=info["repo_id"],
            repo_type=info["repo_type"],
            revision=info["revision"],
            filename=info["path"],
            local_dir=staging,
            token=get_hf_token(),
        )

        os.makedirs(output_dir, exist_ok=True)
        target = os.path.join(output_dir, filename)
        shutil.move(staged_path, target)
        return target
    finally:
        shutil.rmtree(staging, ignore_errors=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hf_download.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add utils/hfDownload.py tests/test_hf_download.py
git -c user.name="patarapoom" -c user.email="police35region4@gmail.com" commit -m "Add Hugging Face client download with Xet and staging

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Boot-time downloader (`download_models.py`)

**Files:**
- Create: `download_models.py`, `tests/test_download_models.py`

**Interfaces:**
- Consumes: `utils.hfAuth.hf_auth_args`, `utils.hfAuth.redact_token`, `utils.civitai.civitai_auth_args`, `utils.hfDownload.parse_hf_url`, `utils.hfDownload.hf_client_enabled`, `utils.hfDownload.download_via_hf`
- Produces (used by `start.sh`, Task 5): CLI `python /app/download_models.py --config /workspace/models_config.json --models-dir /workspace/models [--force]`. Exit code 0 always unless the config cannot be read (exit 1). Env: `SKIP_MODEL_DOWNLOAD`, `MAX_CONCURRENT_DOWNLOADS` (default 5), `LOG_PATH` (default `/workspace/logs/comfyui.log`).
- Pure functions tested directly: `parse_entry`, `load_config`, `plan_downloads`, `aria2c_command`.

- [ ] **Step 1: Write the failing tests**

`tests/test_download_models.py`:
```python
import asyncio
import json
from pathlib import Path

import pytest

import download_models as dm


def test_parse_entry_string_uses_last_path_segment():
    url = "https://huggingface.co/a/b/resolve/main/dir/My%20Model.safetensors"
    assert dm.parse_entry(url) == (url, "My Model.safetensors")


def test_parse_entry_object_requires_url_and_filename():
    entry = {"url": "https://civitai.com/api/download/models/1?fileId=2", "filename": "x.safetensors"}
    assert dm.parse_entry(entry) == (entry["url"], "x.safetensors")
    with pytest.raises(ValueError):
        dm.parse_entry({"url": "https://civitai.com/api/download/models/1"})
    with pytest.raises(ValueError):
        dm.parse_entry({"url": "https://civitai.com/x", "filename": "../evil"})
    with pytest.raises(ValueError):
        dm.parse_entry(42)


def test_load_config_from_file(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"loras": ["https://x/y.safetensors"], "note": "ignored"}))
    assert dm.load_config(str(p)) == {"loras": ["https://x/y.safetensors"]}


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dm.load_config(str(tmp_path / "nope.json"))


def test_plan_downloads_skips_existing_unless_forced(tmp_path):
    (tmp_path / "loras").mkdir()
    (tmp_path / "loras" / "have.safetensors").write_bytes(b"x")
    cfg = {
        "loras": [
            "https://h/have.safetensors",
            {"url": "https://civitai.com/api/download/models/1", "filename": "need.safetensors"},
        ],
        "vae": ["https://h/vae.safetensors"],
        "bogus": "not a list",
    }
    jobs = dm.plan_downloads(cfg, tmp_path)
    assert [(j.category, j.filename) for j in jobs] == [("loras", "need.safetensors"), ("vae", "vae.safetensors")]
    assert jobs[1].dest_dir == tmp_path / "vae"
    assert (tmp_path / "vae").is_dir()

    forced = dm.plan_downloads(cfg, tmp_path, force=True)
    assert [j.filename for j in forced] == ["have.safetensors", "need.safetensors", "vae.safetensors"]


def test_aria2c_command_adds_only_the_matching_auth_header(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_1")
    monkeypatch.setenv("CIVITAI_TOKEN", "civ_1")
    dest = Path("/tmp/models/loras")

    hf = dm.aria2c_command("https://huggingface.co/a/b/resolve/main/x.safetensors", dest, "x.safetensors")
    assert hf[0] == "aria2c"
    assert "--header=Authorization: Bearer hf_1" in hf
    assert "--header=Authorization: Bearer civ_1" not in hf
    assert hf[-4:] == ["-d", str(dest), "-o", "x.safetensors"]

    civ = dm.aria2c_command("https://civitai.com/api/download/models/1", dest, "y.safetensors")
    assert "--header=Authorization: Bearer civ_1" in civ
    assert "--header=Authorization: Bearer hf_1" not in civ

    other = dm.aria2c_command("https://example.com/z.bin", dest, "z.bin")
    assert not any(a.startswith("--header=Authorization") for a in other)
    assert "-c" in other and "-x" in other


def test_download_job_prefers_hf_client_then_falls_back(monkeypatch, tmp_path):
    job = dm.Job("vae", "https://huggingface.co/a/b/resolve/main/v.safetensors", "v.safetensors", tmp_path)
    monkeypatch.setenv("USE_HF_XET", "true")
    seen = []

    def failing_hf(url, output_dir, filename, staging_root):
        seen.append("hf")
        raise RuntimeError("xet down")

    async def fake_aria(url, dest_dir, filename):
        seen.append("aria2c")
        return True

    monkeypatch.setattr(dm, "download_via_hf", failing_hf)
    monkeypatch.setattr(dm, "download_with_aria2c", fake_aria)
    ok = asyncio.run(dm.download_job(job, asyncio.Semaphore(1)))
    assert ok is True
    assert seen == ["hf", "aria2c"]


def test_download_job_uses_aria2c_directly_for_non_hf(monkeypatch, tmp_path):
    job = dm.Job("loras", "https://civitai.com/api/download/models/1", "l.safetensors", tmp_path)
    seen = []

    def hf_should_not_run(*a, **k):
        seen.append("hf")
        raise AssertionError("hf client must not be used for civitai")

    async def fake_aria(url, dest_dir, filename):
        seen.append("aria2c")
        return False

    monkeypatch.setattr(dm, "download_via_hf", hf_should_not_run)
    monkeypatch.setattr(dm, "download_with_aria2c", fake_aria)
    assert asyncio.run(dm.download_job(job, asyncio.Semaphore(1))) is False
    assert seen == ["aria2c"]


def test_main_skip_env_returns_zero_without_touching_config(monkeypatch, tmp_path):
    monkeypatch.setenv("SKIP_MODEL_DOWNLOAD", "true")
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "log"))
    assert dm.main(["--config", str(tmp_path / "missing.json"), "--models-dir", str(tmp_path)]) == 0


def test_main_missing_config_returns_one(monkeypatch, tmp_path):
    monkeypatch.delenv("SKIP_MODEL_DOWNLOAD", raising=False)
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "log"))
    assert dm.main(["--config", str(tmp_path / "missing.json"), "--models-dir", str(tmp_path)]) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_download_models.py -q`
Expected: `ModuleNotFoundError: No module named 'download_models'`

- [ ] **Step 3: Implement the downloader**

`download_models.py`:
```python
#!/usr/bin/env python3
"""Boot-time model downloader.

Reads a models_config.json (category -> list of entries), downloads every file
that is not already under <models-dir>/<category>/, and logs to LOG_PATH and
stdout. Hugging Face resolve URLs go through the official client (Xet where
the repo has it) and fall back to aria2c; everything else uses aria2c.

Exit code is 0 even when downloads fail, so ComfyUI still starts; 1 only when
the config cannot be read.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.civitai import civitai_auth_args  # noqa: E402
from utils.hfAuth import hf_auth_args, redact_token  # noqa: E402
from utils.hfDownload import download_via_hf, hf_client_enabled, parse_hf_url  # noqa: E402

logger = logging.getLogger("download_models")


@dataclass
class Job:
    category: str
    url: str
    filename: str
    dest_dir: Path


def parse_entry(entry):
    """Return (url, filename) for a config entry.

    A string entry is a URL whose last path segment is the filename. An object
    entry carries an explicit filename, for URLs like Civitai's whose path ends
    in an id rather than a name.
    """
    if isinstance(entry, str):
        url = entry
        filename = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    elif isinstance(entry, dict):
        url = entry.get("url")
        filename = entry.get("filename")
        if not url or not filename:
            raise ValueError("object entries need both 'url' and 'filename': %r" % (entry,))
    else:
        raise ValueError("entry must be a URL string or an object: %r" % (entry,))

    if not filename or "/" in filename or filename in (".", ".."):
        raise ValueError("invalid filename %r for %s" % (filename, url))

    return url, filename


def load_config(path_or_url):
    """Load the config from a local path or an http(s) URL. Keeps only list values."""
    if path_or_url.startswith(("http://", "https://")):
        with urlopen(path_or_url, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    else:
        with open(path_or_url, "r", encoding="utf-8") as f:
            raw = json.load(f)

    return {k: v for k, v in raw.items() if isinstance(v, list)}


def plan_downloads(config, models_dir, force=False):
    """Turn the config into Jobs, creating category dirs and skipping present files."""
    models_dir = Path(models_dir)
    jobs = []

    for category, entries in config.items():
        if not isinstance(entries, list):
            logger.warning("Skipping '%s': not a list", category)
            continue

        dest_dir = models_dir / category
        dest_dir.mkdir(parents=True, exist_ok=True)

        for entry in entries:
            try:
                url, filename = parse_entry(entry)
            except ValueError as e:
                logger.error("Skipping bad entry in '%s': %s", category, e)
                continue

            if (dest_dir / filename).exists() and not force:
                logger.info("Skipping %s, already present in %s", filename, category)
                continue

            jobs.append(Job(category, url, filename, dest_dir))

    return jobs


def aria2c_command(url, dest_dir, filename):
    """Build the aria2c command, adding the auth header that matches the host."""
    return [
        "aria2c",
        *hf_auth_args(url),
        *civitai_auth_args(url),
        "--console-log-level=warn",
        "-c",
        "-x", "4",
        "-s", "4",
        "-k", "1M",
        "--file-allocation=none",
        "--max-tries=5",
        "--retry-wait=10",
        "--connect-timeout=30",
        "--timeout=600",
        "--summary-interval=30",
        url,
        "-d", str(dest_dir),
        "-o", filename,
    ]


async def download_with_aria2c(url, dest_dir, filename):
    """Run aria2c for one file. Returns True on success."""
    cmd = aria2c_command(url, dest_dir, filename)
    logger.info("aria2c: %s -> %s/%s", redact_token(url), dest_dir, filename)

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
    except Exception as e:  # aria2c missing, etc.
        logger.error("aria2c could not run for %s: %s", filename, redact_token(str(e)))
        return False

    if process.returncode == 0:
        return True

    output = (stderr or stdout or b"").decode("utf-8", errors="replace")
    logger.error("aria2c failed for %s (exit %s): %s", filename, process.returncode, redact_token(output.strip()[-2000:]))
    return False


async def download_job(job, semaphore):
    """Download one Job, preferring the Hugging Face client where it applies."""
    async with semaphore:
        logger.info("Starting %s (%s)", job.filename, job.category)

        if hf_client_enabled() and parse_hf_url(job.url):
            try:
                await asyncio.to_thread(
                    download_via_hf, job.url, str(job.dest_dir), job.filename, "/workspace/.hf_staging"
                )
                logger.info("Downloaded %s via the Hugging Face client", job.filename)
                return True
            except Exception as e:
                logger.warning("Hugging Face client failed for %s: %s; falling back to aria2c",
                               job.filename, redact_token(str(e)))

        ok = await download_with_aria2c(job.url, job.dest_dir, job.filename)
        if ok:
            logger.info("Downloaded %s via aria2c", job.filename)
        else:
            logger.error("FAILED %s (%s)", job.filename, job.category)
        return ok


async def run_jobs(jobs, max_concurrent):
    semaphore = asyncio.Semaphore(max_concurrent)
    results = await asyncio.gather(*(download_job(j, semaphore) for j in jobs))
    return sum(1 for r in results if r), sum(1 for r in results if not r)


def _configure_logging(log_path):
    logging.getLogger().handlers = []
    fmt = logging.Formatter("[download] %(message)s")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)

    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="models_config.json path or URL")
    parser.add_argument("--models-dir", required=True, help="ComfyUI models root, e.g. /workspace/models")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args(argv)

    _configure_logging(os.getenv("LOG_PATH", "/workspace/logs/comfyui.log"))

    if (os.getenv("SKIP_MODEL_DOWNLOAD") or "").strip().lower() == "true":
        logger.info("SKIP_MODEL_DOWNLOAD=true, not downloading anything")
        return 0

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error("Cannot read config %s: %s", args.config, redact_token(str(e)))
        return 1

    jobs = plan_downloads(config, args.models_dir, force=args.force)
    if not jobs:
        logger.info("All models present, nothing to download")
        return 0

    max_concurrent = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "5"))
    logger.info("Downloading %d file(s), up to %d at a time", len(jobs), max_concurrent)
    ok, failed = asyncio.run(run_jobs(jobs, max_concurrent))
    logger.info("Done: %d succeeded, %d failed", ok, failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_download_models.py -q`
Expected: `10 passed`

- [ ] **Step 5: Run the whole suite and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

```bash
git add download_models.py tests/test_download_models.py
git -c user.name="patarapoom" -c user.email="police35region4@gmail.com" commit -m "Add boot-time model downloader with HF client and aria2c fallback

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Boot script (`start.sh`)

**Files:**
- Create: `start.sh`, `tests/test_start_sh.py`

**Interfaces:**
- Consumes: `/app/download_models.py` CLI (Task 4), `/app/models_config.json` (Task 1), `/opt/ComfyUI/main.py`, `/opt/venv/bin/python`.
- Produces: functions `ensure_dirs`, `resolve_models_config`, `seed_manager_config`, `comfy_args`, `main`. The script only runs `main` when executed, not when sourced, so tests can source it with overridden roots (`WORKSPACE`, `APP_DIR`, `COMFY_DIR`).

- [ ] **Step 1: Write the failing tests**

`tests/test_start_sh.py`:
```python
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")


def run_fn(tmp_path, snippet, env=None):
    """Source start.sh with fake roots and run a bash snippet."""
    app = tmp_path / "app"
    app.mkdir(exist_ok=True)
    shutil.copy(ROOT / "models_config.json", app / "models_config.json")
    full_env = {
        "PATH": os.environ["PATH"],
        "WORKSPACE": str(tmp_path / "workspace"),
        "APP_DIR": str(app),
        "COMFY_DIR": str(tmp_path / "comfy"),
    }
    full_env.update(env or {})
    return subprocess.run(
        ["bash", "-c", f"source '{START}'; {snippet}"],
        env=full_env, capture_output=True, text=True, check=False,
    )


def test_script_parses():
    subprocess.run(["bash", "-n", str(START)], check=True)


def test_ensure_dirs_creates_workspace_layout(tmp_path):
    r = run_fn(tmp_path, "ensure_dirs")
    assert r.returncode == 0, r.stderr
    ws = tmp_path / "workspace"
    for sub in ("models/loras", "models/diffusion_models", "models/vae", "models/text_encoders",
                "output", "input", "user", "logs", ".cache/huggingface"):
        assert (ws / sub).is_dir(), sub


def test_resolve_models_config_copies_baked_default(tmp_path):
    r = run_fn(tmp_path, "ensure_dirs; resolve_models_config")
    assert r.returncode == 0, r.stderr
    target = tmp_path / "workspace" / "models_config.json"
    assert target.read_text() == (ROOT / "models_config.json").read_text()


def test_resolve_models_config_keeps_existing_file(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "models_config.json").write_text('{"loras": []}')
    r = run_fn(tmp_path, "ensure_dirs; resolve_models_config")
    assert r.returncode == 0, r.stderr
    assert (ws / "models_config.json").read_text() == '{"loras": []}'


def test_resolve_models_config_falls_back_when_url_fails(tmp_path):
    r = run_fn(tmp_path, "ensure_dirs; resolve_models_config",
               env={"MODELS_CONFIG_URL": "http://127.0.0.1:9/nope.json"})
    assert r.returncode == 0, r.stderr
    target = tmp_path / "workspace" / "models_config.json"
    assert target.read_text() == (ROOT / "models_config.json").read_text()


def test_seed_manager_config_only_when_missing(tmp_path):
    r = run_fn(tmp_path, "ensure_dirs; seed_manager_config")
    assert r.returncode == 0, r.stderr
    ini = tmp_path / "workspace" / "user" / "default" / "ComfyUI-Manager" / "config.ini"
    assert "use_uv = True" in ini.read_text()
    ini.write_text("[default]\ncustom = yes\n")
    run_fn(tmp_path, "seed_manager_config")
    assert ini.read_text() == "[default]\ncustom = yes\n"


def test_comfy_args_default_and_disabled_sage(tmp_path):
    r = run_fn(tmp_path, "comfy_args")
    assert r.returncode == 0, r.stderr
    args = r.stdout.split()
    ws = str(tmp_path / "workspace")
    assert args[:4] == ["--listen", "0.0.0.0", "--port", "8188"]
    assert "--use-sage-attention" in args
    assert args[args.index("--output-directory") + 1] == f"{ws}/output"
    assert args[args.index("--input-directory") + 1] == f"{ws}/input"
    assert args[args.index("--user-directory") + 1] == f"{ws}/user"

    r = run_fn(tmp_path, "comfy_args", env={"USE_SAGE_ATTENTION": "false", "COMFYUI_EXTRA_ARGS": "--fast --lowvram"})
    args = r.stdout.split()
    assert "--use-sage-attention" not in args
    assert args[-2:] == ["--fast", "--lowvram"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_start_sh.py -q`
Expected: failures, `start.sh: No such file or directory`.

- [ ] **Step 3: Write the boot script**

`start.sh`:
```bash
#!/usr/bin/env bash
# Boot script for the ComfyUI RunPod template.
#
# Order: create the /workspace layout -> start JupyterLab (8888) -> resolve
# models_config.json -> download missing models -> start ComfyUI (8188).
#
# Sourcing this file defines the functions without running anything, which is
# how the tests exercise it.

WORKSPACE="${WORKSPACE:-/workspace}"
APP_DIR="${APP_DIR:-/app}"
COMFY_DIR="${COMFY_DIR:-/opt/ComfyUI}"

export HF_TOKEN="${HF_TOKEN:-}"
export CIVITAI_TOKEN="${CIVITAI_TOKEN:-}"
export MODELS_CONFIG_URL="${MODELS_CONFIG_URL:-}"
export SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-false}"
export USE_SAGE_ATTENTION="${USE_SAGE_ATTENTION:-true}"
export COMFYUI_EXTRA_ARGS="${COMFYUI_EXTRA_ARGS:-}"
export LOG_PATH="${LOG_PATH:-$WORKSPACE/logs/comfyui.log}"

# Hugging Face client. HF_HOME lives on the volume so the Xet chunk cache
# survives restarts. The chunk cache only helps when files share content, which
# this model set mostly does not, hence the modest default.
export USE_HF_XET="${USE_HF_XET:-true}"
export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
export HF_XET_CHUNK_CACHE_SIZE_BYTES="${HF_XET_CHUNK_CACHE_SIZE_BYTES:-8589934592}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

MODEL_FOLDERS=(audio_encoders background_removal checkpoints classifiers clip_vision configs
    controlnet datasets detection diffusers diffusion_models embeddings frame_interpolation
    geometry_estimation gligen hypernetworks latent_upscale_models loras model_patches
    optical_flow photomaker style_models text_encoders upscale_models vae vae_approx)

log() { echo "[start] $*" | tee -a "$LOG_PATH"; }

ensure_dirs() {
    mkdir -p "$WORKSPACE/logs" "$WORKSPACE/output" "$WORKSPACE/input" "$WORKSPACE/user" \
        "$WORKSPACE/.cache/huggingface" "$WORKSPACE/.hf_staging"
    for f in "${MODEL_FOLDERS[@]}"; do
        mkdir -p "$WORKSPACE/models/$f"
    done
    touch "$LOG_PATH"
}

# The effective model list lives at $WORKSPACE/models_config.json. On first
# boot it comes from MODELS_CONFIG_URL when set and reachable, otherwise from
# the copy baked into the image. An existing file is never overwritten, so the
# customer can edit it on the volume.
resolve_models_config() {
    local target="$WORKSPACE/models_config.json"
    if [ -f "$target" ]; then
        log "Using existing $target"
        return 0
    fi
    if [ -n "$MODELS_CONFIG_URL" ]; then
        log "Fetching models config from $MODELS_CONFIG_URL"
        if curl -fsSL --retry 3 --retry-delay 5 --max-time 60 -o "$target.tmp" "$MODELS_CONFIG_URL" \
            && python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$target.tmp" 2>/dev/null; then
            mv "$target.tmp" "$target"
            return 0
        fi
        rm -f "$target.tmp"
        log "WARNING: could not fetch a valid config from MODELS_CONFIG_URL, using the built-in list"
    fi
    cp "$APP_DIR/models_config.json" "$target"
    log "Wrote default models config to $target"
}

# ComfyUI-Manager reads its config from the user directory. Pre-seed it once so
# it installs node dependencies with uv (fast) instead of pip.
seed_manager_config() {
    local dir="$WORKSPACE/user/default/ComfyUI-Manager"
    local ini="$dir/config.ini"
    [ -f "$ini" ] && return 0
    mkdir -p "$dir"
    cat >"$ini" <<'INI'
[default]
preview_method = auto
git_exe =
use_uv = True
channel_url = https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main
share_option = all
bypass_ssl = False
file_logging = True
component_policy = workflow
update_policy = stable-comfyui
windows_selector_event_loop_policy = False
model_download_by_agent = False
downgrade_blacklist =
security_level = weak
skip_migration_check = True
always_lazy_install = False
network_mode = public
db_mode = cache
INI
}

start_jupyter() {
    log "Starting JupyterLab on port 8888"
    CUDA_VISIBLE_DEVICES="" nohup jupyter lab \
        --allow-root --no-browser --ip=0.0.0.0 --port=8888 \
        --ServerApp.token='' --ServerApp.password='' \
        --ServerApp.allow_origin='*' --ServerApp.root_dir="$WORKSPACE" \
        >"$WORKSPACE/logs/jupyter.log" 2>&1 &
}

download_models() {
    log "Checking models (SKIP_MODEL_DOWNLOAD=$SKIP_MODEL_DOWNLOAD)"
    python "$APP_DIR/download_models.py" \
        --config "$WORKSPACE/models_config.json" \
        --models-dir "$WORKSPACE/models" \
        || log "WARNING: model download step reported an error, starting ComfyUI anyway"
}

# Prints the ComfyUI argument list, one per line.
comfy_args() {
    local args=(--listen 0.0.0.0 --port 8188
        --output-directory "$WORKSPACE/output"
        --input-directory "$WORKSPACE/input"
        --user-directory "$WORKSPACE/user")
    if [ "$USE_SAGE_ATTENTION" = "true" ]; then
        args+=(--use-sage-attention)
    fi
    if [ -n "$COMFYUI_EXTRA_ARGS" ]; then
        # shellcheck disable=SC2206
        args+=($COMFYUI_EXTRA_ARGS)
    fi
    printf '%s\n' "${args[@]}"
}

start_comfyui() {
    local args=()
    while IFS= read -r line; do args+=("$line"); done < <(comfy_args)
    cd "$COMFY_DIR" || exit 1
    log "==================== ComfyUI starting $(date -u +%FT%TZ) ===================="
    log "python main.py ${args[*]}"
    python main.py "${args[@]}" 2>&1 | tee -a "$LOG_PATH" &
}

main() {
    ensure_dirs
    log "ComfyUI RunPod template starting (workspace: $WORKSPACE)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read -r gpu; do log "GPU: $gpu"; done
    start_jupyter
    resolve_models_config
    seed_manager_config
    download_models
    start_comfyui
    wait
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
```

Run: `chmod +x start.sh`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_start_sh.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add start.sh tests/test_start_sh.py
git -c user.name="patarapoom" -c user.email="police35region4@gmail.com" commit -m "Add boot script: workspace layout, JupyterLab, downloads, ComfyUI

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Dockerfile and `.dockerignore`

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `tests/test_dockerfile.py`

**Interfaces:**
- Consumes: every file from Tasks 1-5 (copied to `/app`, `extra_model_paths.yaml` also to `/opt/ComfyUI/`).
- Produces: image with `CMD ["/app/start.sh"]`, build arg `SAGEATTENTION_WHEEL_URL`, BuildKit secret id `github_token` (used by Task 7's workflow).

- [ ] **Step 1: Write the failing test**

A full build needs an x86_64 machine with ~30 GB of disk and runs in CI (Task 7). Locally we check the Dockerfile's contract with a text test.

`tests/test_dockerfile.py`:
```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NODES = [
    "https://github.com/Comfy-Org/ComfyUI-Manager",
    "https://github.com/rgthree/rgthree-comfy",
    "https://github.com/PGCRT/CRT-Nodes",
    "https://github.com/Elevenheights/ComfyUI-FameGridColorFinish",
    "https://github.com/BigStationW/ComfyUi-TextEncodeEditAdvanced",
    "https://github.com/Fannovel16/comfyui_controlnet_aux",
    "https://github.com/onyxaipro/Onyx_Custom_Nodes",
]


def test_dockerfile_contract():
    text = (ROOT / "Dockerfile").read_text()
    assert text.startswith("# syntax=docker/dockerfile:1")
    assert "FROM nvidia/cuda:13.0.3-base-ubuntu24.04" in text
    assert "python3.12" in text
    assert "--index-url https://download.pytorch.org/whl/cu130" in text
    assert "--branch v0.34.2" in text
    assert "ARG SAGEATTENTION_WHEEL_URL=https://huggingface.co/Patarapoom/sageattention-wheels/resolve/main/sageattention-2.2.0+cu130.torch2.13.0-cp312-cp312-linux_x86_64.whl" in text
    assert "--mount=type=secret,id=github_token" in text
    for node in NODES:
        assert node in text, node
    assert "opencv-contrib-python-headless" in text
    assert "--quick-test-for-ci" in text
    assert "EXPOSE 8188 8888" in text
    assert 'CMD ["/app/start.sh"]' in text
    # every pip install in the image is constrained
    for line in text.splitlines():
        if "uv pip install" in line and "constraints.txt" not in line and "--find-links" not in line:
            assert "-c /app/constraints.txt" in line, line


def test_dockerignore_excludes_dev_files():
    text = (ROOT / ".dockerignore").read_text().splitlines()
    for pattern in (".git", ".venv", "tests", "docs", "scripts", ".github", "__pycache__", "*.pyc"):
        assert pattern in text, pattern
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dockerfile.py -q`
Expected: `FileNotFoundError: Dockerfile`

- [ ] **Step 3: Write `.dockerignore` and the Dockerfile**

`.dockerignore`:
```
.git
.gitignore
.venv
.pytest_cache
__pycache__
*.pyc
tests
docs
scripts
.github
.DS_Store
README.md
requirements-dev.txt
```

`Dockerfile`:
```dockerfile
# syntax=docker/dockerfile:1
# ComfyUI RunPod template for JNK Prod.
# ComfyUI v0.34.2 + custom nodes live in the image; models, input, output and
# user data live on the /workspace volume (see start.sh).
FROM nvidia/cuda:13.0.3-base-ubuntu24.04

ARG SAGEATTENTION_WHEEL_URL=https://huggingface.co/Patarapoom/sageattention-wheels/resolve/main/sageattention-2.2.0+cu130.torch2.13.0-cp312-cp312-linux_x86_64.whl
ARG COMFYUI_TAG=v0.34.2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# Ubuntu 24.04 ships Python 3.12; no third-party PPA needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        git git-lfs build-essential curl ca-certificates \
        libgl1 libglib2.0-0 libgomp1 ffmpeg aria2 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip uv

COPY constraints.txt /app/constraints.txt

# 1. CUDA 13 torch stack. The constraints file keeps every later install on it.
RUN uv pip install -c /app/constraints.txt \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu130

# 2. ComfyUI at a fixed tag (its requirements pin comfy-kitchen / comfy-aimdo).
RUN git clone --depth 1 --branch ${COMFYUI_TAG} https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
    && uv pip install -c /app/constraints.txt -r /opt/ComfyUI/requirements.txt

# 3. SageAttention, prebuilt for this exact torch/CUDA (see scripts/build-sageattention.sh).
RUN uv pip install -c /app/constraints.txt "${SAGEATTENTION_WHEEL_URL}"

# 4. Public custom nodes.
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/Comfy-Org/ComfyUI-Manager \
    && git clone --depth 1 https://github.com/rgthree/rgthree-comfy \
    && git clone --depth 1 https://github.com/PGCRT/CRT-Nodes \
    && git clone --depth 1 https://github.com/Elevenheights/ComfyUI-FameGridColorFinish \
    && git clone --depth 1 https://github.com/BigStationW/ComfyUi-TextEncodeEditAdvanced \
    && git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux

# 5. Private custom node. The token is read from a BuildKit secret for this
#    single command and never written to a layer or to .git/config.
RUN --mount=type=secret,id=github_token \
    git -c http.extraheader="AUTHORIZATION: bearer $(cat /run/secrets/github_token)" \
        clone --depth 1 https://github.com/onyxaipro/Onyx_Custom_Nodes \
    && rm -rf Onyx_Custom_Nodes/.git

# 6. Node dependencies, one install per requirements.txt so a failure names the node.
#    Then collapse the three OpenCV variants the nodes ask for into one package:
#    they all unpack into the same cv2/ directory and pip does not de-duplicate.
RUN set -e; for req in */requirements.txt; do \
        echo "==> $req"; uv pip install -c /app/constraints.txt -r "$req"; \
    done \
    && for inst in */install.py; do \
        [ -f "$inst" ] || continue; echo "==> $inst"; (cd "$(dirname "$inst")" && python install.py); \
    done \
    && uv pip uninstall $(uv pip freeze | grep -iE '^opencv' | cut -d= -f1) \
    && uv pip install -c /app/constraints.txt opencv-contrib-python-headless

# 7. Services and downloader dependencies.
RUN uv pip install -c /app/constraints.txt jupyterlab "huggingface_hub[hf_xet]"

# 8. App files.
WORKDIR /app
COPY start.sh download_models.py models_config.json extra_model_paths.yaml ./
COPY utils/ ./utils/
RUN chmod +x /app/start.sh \
    && cp /app/extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml

# 9. Build-time smoke test: the torch stack imports and every custom node loads.
#    Runs on CPU so it works on a GPU-less CI runner.
RUN python -c "import torch, torchvision, torchaudio, triton, sageattention, comfy_kitchen, comfy_aimdo; \
        print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)" \
    && cd /opt/ComfyUI \
    && (python main.py --cpu --quick-test-for-ci --user-directory /tmp/ci-user --output-directory /tmp/ci-out \
        > /tmp/quick-test.log 2>&1 || (cat /tmp/quick-test.log; echo "quick test exited non-zero"; exit 1)) \
    && cat /tmp/quick-test.log \
    && ! grep -E "IMPORT FAILED|Cannot import" /tmp/quick-test.log \
    && rm -rf /tmp/ci-user /tmp/ci-out /tmp/quick-test.log

LABEL org.opencontainers.image.title="comfyui-runpod-jnkprod" \
      org.opencontainers.image.description="ComfyUI ${COMFYUI_TAG}, torch 2.13.0+cu130, SageAttention 2.2.0, Python 3.12" \
      org.opencontainers.image.source="https://github.com/poomshift/comfyui-runpod-jnkprod"

EXPOSE 8188 8888
CMD ["/app/start.sh"]
```

Notes for the implementer:
- `uv pip uninstall` with an empty argument list errors; every node here does pull some OpenCV variant, so the list is never empty. If it ever is, the build fails loudly, which is preferable to silently skipping.
- The quick test is written to a file rather than piped through `tee`, because `/bin/sh` has no `pipefail` and a crash of `main.py` would otherwise be masked. The grep then fails the build when ComfyUI logs a node import failure (`IMPORT FAILED`, or `Cannot import` with the traceback).

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dockerfile.py -q`
Expected: `2 passed`

- [ ] **Step 5: Lint the Dockerfile if a Docker daemon is running (optional)**

Run: `docker build --check . 2>&1 | tail -20`
Expected: `Check complete, no warnings found.` If the daemon is not running, skip; CI is the real check.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore tests/test_dockerfile.py
git -c user.name="patarapoom" -c user.email="police35region4@gmail.com" commit -m "Add Dockerfile: CUDA 13 torch 2.13, ComfyUI v0.34.2, custom nodes, SageAttention

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: GitHub Actions (tests + Docker build/push) and repository secrets

**Files:**
- Create: `.github/workflows/tests.yml`, `.github/workflows/docker-build.yml`, `tests/test_workflows.py`

**Interfaces:**
- Consumes: Dockerfile build arg `SAGEATTENTION_WHEEL_URL`, secret id `github_token`.
- Produces: images `promptalchemist/comfyui-runpod-jnkprod:latest` and `:<YYYYMMDD>` on push to `main`. Repository secrets `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `ONYX_GITHUB_TOKEN`; optional repository variable `SAGEATTENTION_WHEEL_URL`.

- [ ] **Step 1: Write the failing test**

`tests/test_workflows.py`:
```python
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())


def test_tests_workflow_runs_pytest_on_push_and_pr():
    wf = _load("tests.yml")
    on = wf.get("on") or wf.get(True)  # PyYAML parses bare `on:` as True
    assert "push" in on and "pull_request" in on
    steps = wf["jobs"]["pytest"]["steps"]
    assert any("pytest" in (s.get("run") or "") for s in steps)


def test_docker_workflow_pushes_only_on_main():
    wf = _load("docker-build.yml")
    on = wf.get("on") or wf.get(True)
    assert on["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in on
    build = wf["jobs"]["build"]
    step = next(s for s in build["steps"] if s.get("uses", "").startswith("docker/build-push-action"))
    w = step["with"]
    assert "promptalchemist/comfyui-runpod-jnkprod:latest" in w["tags"]
    assert "github_token=${{ secrets.ONYX_GITHUB_TOKEN }}" in w["secrets"]
    assert "SAGEATTENTION_WHEEL_URL=" in w["build-args"]
    assert w["push"] == "${{ github.event_name != 'pull_request' }}"
    assert any("rm -rf /usr/share/dotnet" in (s.get("run") or "") for s in build["steps"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_workflows.py -q`
Expected: `FileNotFoundError`

- [ ] **Step 3: Write the workflows**

`.github/workflows/tests.yml`:
```yaml
name: Tests

on:
  push:
  pull_request:

jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements-dev.txt
      - run: python -m pytest -q
```

`.github/workflows/docker-build.yml`:
```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      # The image is ~12 GB; the free runner needs the space these take.
      - name: Free disk space
        run: |
          sudo rm -rf /usr/share/dotnet /opt/ghc /usr/local/lib/android /usr/local/.ghcup "$AGENT_TOOLSDIRECTORY"
          sudo docker image prune -af
          df -h /

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to DockerHub
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Get current date
        id: date
        run: echo "date=$(date +'%Y%m%d')" >> $GITHUB_OUTPUT

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: |
            promptalchemist/comfyui-runpod-jnkprod:latest
            promptalchemist/comfyui-runpod-jnkprod:${{ steps.date.outputs.date }}
          build-args: |
            SAGEATTENTION_WHEEL_URL=${{ vars.SAGEATTENTION_WHEEL_URL || 'https://huggingface.co/Patarapoom/sageattention-wheels/resolve/main/sageattention-2.2.0+cu130.torch2.13.0-cp312-cp312-linux_x86_64.whl' }}
          secrets: |
            github_token=${{ secrets.ONYX_GITHUB_TOKEN }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_workflows.py -q`
Expected: `2 passed`

- [ ] **Step 5: Set the repository secrets**

The Docker Hub credentials are the same account as `comfyui-docker-new`, but GitHub never exposes secret values, so they must be entered again. `ONYX_GITHUB_TOKEN` must be a GitHub personal access token of the account that was added to `onyxaipro/Onyx_Custom_Nodes` (fine-grained: that repository, Contents: read; or classic with `repo`). These commands prompt for the value, so the user runs them, not the agent:

```bash
cd /Users/patarapoomsmacpro/Project/comfyui-runpod-jnkprod
gh secret set DOCKERHUB_USERNAME
gh secret set DOCKERHUB_TOKEN
gh secret set ONYX_GITHUB_TOKEN
gh secret list
```
Expected: `gh secret list` shows the three names.

- [ ] **Step 6: Commit and push, then watch the build**

```bash
git add .github/ tests/test_workflows.py
git -c user.name="patarapoom" -c user.email="police35region4@gmail.com" commit -m "Add GitHub Actions: pytest and Docker Hub build

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin main
gh run list --limit 3
gh run watch --exit-status $(gh run list --workflow docker-build.yml --limit 1 --json databaseId -q '.[0].databaseId')
```
Expected: both workflows green. If the Docker build fails, read `gh run view --log-failed`, fix the Dockerfile, commit, push, repeat. Typical first-build failures: a node's `requirements.txt` pulling an incompatible package (add a pin to the install line for that node), the quick test reporting `IMPORT FAILED` for a node (read the traceback in the log and add the missing apt or pip package), or disk exhaustion (drop `cache-to`).

---

### Task 8: Customer-facing README

**Files:**
- Create: `README.md`
- Test: `tests/test_readme.py`

- [ ] **Step 1: Write the failing test**

`tests/test_readme.py`:
```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_ports_env_and_layout():
    text = (ROOT / "README.md").read_text()
    for needle in (
        "8188", "8888",
        "HF_TOKEN", "CIVITAI_TOKEN", "MODELS_CONFIG_URL", "SKIP_MODEL_DOWNLOAD",
        "USE_SAGE_ATTENTION", "COMFYUI_EXTRA_ARGS",
        "/workspace/models", "/workspace/output", "/workspace/input", "/workspace/user",
        "/workspace/logs/comfyui.log", "/workspace/models_config.json",
        "black-forest-labs/FLUX.2-klein-9B",
        "promptalchemist/comfyui-runpod-jnkprod",
        "ComfyUI-Manager",
    ):
        assert needle in text, needle
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_readme.py -q`
Expected: `FileNotFoundError`

- [ ] **Step 3: Write the README**

`README.md`:
```markdown
# ComfyUI RunPod Template (FLUX.2 Klein 9B)

Docker image: `promptalchemist/comfyui-runpod-jnkprod`

Boots straight into ComfyUI with the FLUX.2 Klein 9B model set and a fixed set
of custom nodes. Models are downloaded to the pod's volume on the first start
and reused on every later start.

| What | Version |
| --- | --- |
| ComfyUI | 0.34.2 |
| PyTorch | 2.13.0+cu130 (CUDA 13.0) |
| Python | 3.12 |
| SageAttention | 2.2.0 (on by default) |
| triton | 3.7.1 |

## Ports

| Port | Service |
| --- | --- |
| 8188 | ComfyUI |
| 8888 | JupyterLab (no password) |

Open them from the pod's **Connect** menu.

## Environment variables

Set these on the RunPod template or pod before the first start.

| Variable | Required | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | **Yes** | Hugging Face token with *read* access. `black-forest-labs/FLUX.2-klein-9B` is gated: open its page, accept the license, then create a token at https://huggingface.co/settings/tokens |
| `CIVITAI_TOKEN` | **Yes** for the Consistence Edit LoRA | Civitai API key from https://civitai.com/user/account (the other two Civitai LoRAs download without it) |
| `MODELS_CONFIG_URL` | No | URL of your own `models_config.json`; used on the first start only |
| `SKIP_MODEL_DOWNLOAD` | No | `true` skips the download step |
| `USE_SAGE_ATTENTION` | No | `false` starts ComfyUI without `--use-sage-attention` |
| `COMFYUI_EXTRA_ARGS` | No | Extra ComfyUI flags, for example `--fast` |

Tokens are only sent to their own site and never written to the logs.

## First start

1. JupyterLab is up within seconds on port 8888.
2. The missing models are downloaded to `/workspace/models` (about 30 GB in
   total). Watch progress in a JupyterLab terminal:
   `tail -f /workspace/logs/comfyui.log`
3. ComfyUI starts on port 8188 once the downloads have finished.

Later starts skip files that are already on the volume, so ComfyUI is up in
under a minute.

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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] **Step 5: Commit and push**

```bash
git add README.md tests/test_readme.py
git -c user.name="patarapoom" -c user.email="police35region4@gmail.com" commit -m "Add customer-facing README

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin main
```

---

### Task 9: Pod acceptance test (manual, with the user)

**Files:** none. This task verifies the pushed image on RunPod.

- [ ] **Step 1: Create the RunPod template**

In RunPod: Templates > New Template. Container image `promptalchemist/comfyui-runpod-jnkprod:latest`, container disk 30 GB, volume 60 GB mounted at `/workspace`, expose HTTP ports `8188,8888`, env `HF_TOKEN` and `CIVITAI_TOKEN`. Start command left empty (the image's CMD runs).

- [ ] **Step 2: First boot checks**

Deploy a pod (24 GB VRAM or more). Expected within ~30 s: JupyterLab reachable on 8888. In a JupyterLab terminal:
```bash
tail -f /workspace/logs/comfyui.log
```
Expected: `[download] Downloading 7 file(s)`, then seven `Downloaded ...` lines, then `ComfyUI starting`, and the ComfyUI startup log mentioning `Using sage attention`. Then:
```bash
ls -la /workspace/models/{diffusion_models,text_encoders,vae,loras}
python -c "import sageattention; print('sage ok')"
```
Expected: the seven files with realistic sizes (Klein 9B ~18 GB, Qwen ~9 GB, VAE ~330 MB, LoRAs 80 MB-340 MB), `sage ok`.

- [ ] **Step 3: Workflow check**

Open ComfyUI on 8188, build or load a FLUX.2 Klein 9B image-to-image workflow with the Qwen text encoder, the flux2 VAE and one LoRA, and run it. Expected: an image in `/workspace/output` and no red node on the canvas. Check the Onyx, CRT and rgthree nodes appear in the node search.

- [ ] **Step 4: Restart check**

Stop and start the pod. Expected in the log: `All models present, nothing to download` and ComfyUI up within a minute. Saved workflows are still listed in ComfyUI.

- [ ] **Step 5: Record the result**

Add a "Verified on RunPod" line with the date and GPU to the README, commit, push.
