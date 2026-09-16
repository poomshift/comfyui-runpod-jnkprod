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
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip uv \
    && git lfs install --system

COPY constraints.txt /app/constraints.txt
# Constrain *every* later install, including plain `pip install` run by a node's
# install.py. Set after the COPY: pip errors on a missing constraints file, which
# would break the `pip install --upgrade pip uv` above.
ENV PIP_CONSTRAINT=/app/constraints.txt UV_CONSTRAINT=/app/constraints.txt

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
#    `uv pip uninstall` with no names is an error, hence the guard.
RUN set -e; for req in */requirements.txt; do \
        echo "==> $req"; uv pip install -c /app/constraints.txt -r "$req"; \
    done; \
    for inst in */install.py; do \
        [ -f "$inst" ] || continue; echo "==> $inst"; (cd "$(dirname "$inst")" && python install.py); \
    done; \
    pkgs=$(uv pip freeze | grep -iE '^opencv' | cut -d= -f1 || true); \
    if [ -n "$pkgs" ]; then uv pip uninstall $pkgs; fi; \
    uv pip install -c /app/constraints.txt opencv-contrib-python-headless

# 7. Services and downloader dependencies.
RUN uv pip install -c /app/constraints.txt jupyterlab "huggingface_hub[hf_xet]"

# 8. Build-time smoke test: the torch stack imports, is still the cu130 build a
#    node installer could have replaced, and every custom node loads. Runs on CPU
#    so it works on a GPU-less CI runner. Kept above the app COPY so editing
#    start.sh or download_models.py does not re-run the quick test.
WORKDIR /opt/ComfyUI
RUN python -c "import torch, torchvision, torchaudio, triton, sageattention, comfy_kitchen, comfy_aimdo; \
        print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__); \
        assert torch.__version__.startswith('2.13.0+cu130'), torch.__version__" \
    && (python main.py --cpu --quick-test-for-ci --user-directory /tmp/ci-user --output-directory /tmp/ci-out \
        > /tmp/quick-test.log 2>&1 || (cat /tmp/quick-test.log; echo "quick test exited non-zero"; exit 1)) \
    && cat /tmp/quick-test.log \
    && ! grep -E "IMPORT FAILED|Cannot import" /tmp/quick-test.log \
    && rm -rf /tmp/ci-user /tmp/ci-out /tmp/quick-test.log

# 9. App files.
WORKDIR /app
COPY start.sh download_models.py models_config.json extra_model_paths.yaml ./
COPY utils/ ./utils/
RUN chmod +x /app/start.sh \
    && cp /app/extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml

LABEL org.opencontainers.image.title="comfyui-runpod-jnkprod" \
      org.opencontainers.image.description="ComfyUI ${COMFYUI_TAG}, torch 2.13.0+cu130, SageAttention 2.2.0, Python 3.12" \
      org.opencontainers.image.source="https://github.com/poomshift/comfyui-runpod-jnkprod"

EXPOSE 8188 8888
CMD ["/app/start.sh"]
