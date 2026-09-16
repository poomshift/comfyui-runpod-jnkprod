#!/usr/bin/env bash
# Build a SageAttention 2.2.0 wheel for CUDA 13.0 + PyTorch 2.13.0 + Python 3.12
# on a RunPod pod started from runpod/pytorch:1.3.1-cu1300-torch2130-ubuntu2404
# (or any Ubuntu 24.04 image; the CUDA toolkit is installed if nvcc is missing).
#
# Usage on the pod:
#   bash build-sageattention.sh
#
# Optional env:
#   HF_TOKEN        upload the wheel to Hugging Face when set
#   HF_WHEEL_REPO   target repo (default: Patarapoom/sageattention-wheels)
#   ARCH_LIST       override TORCH_CUDA_ARCH_LIST (default covers A100, 3090/A40, 4090/L40, H100, 5090/RTX PRO 6000)
#   SKIP_GPU_TEST=1 skip the numerical check (only when the pod has no GPU)
set -euo pipefail

TORCH_VERSION="2.13.0"
TORCHVISION_VERSION="0.28.0"
TORCHAUDIO_VERSION="2.11.0"
SAGE_TAG="v2.2.0"
LOCAL_VERSION="cu130.torch${TORCH_VERSION}"
ARCH_LIST="${ARCH_LIST:-8.0;8.6;8.9;9.0;12.0}"
HF_WHEEL_REPO="${HF_WHEEL_REPO:-Patarapoom/sageattention-wheels}"

WORK=/workspace/sage-build
VENV=$WORK/venv
OUT=$WORK/wheels
mkdir -p "$WORK" "$OUT"

log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

log "System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev git build-essential ninja-build ca-certificates >/dev/null

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
if ! command -v nvcc >/dev/null 2>&1; then
    log "nvcc not found, installing CUDA toolkit 13.0 (a few minutes)"
    if ! ls /etc/apt/sources.list.d/ 2>/dev/null | grep -qi cuda; then
        curl -fsSL -o /tmp/cuda-keyring.deb \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i /tmp/cuda-keyring.deb
        apt-get update -qq
    fi
    apt-get install -y -qq --no-install-recommends cuda-toolkit-13-0 >/dev/null
fi

log "nvcc / GPU"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,compute_cap --format=csv || echo "no GPU visible"

if [ ! -x "$VENV/bin/python" ]; then
    log "Creating venv (Python 3.12)"
    python3.12 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip

log "Installing torch ${TORCH_VERSION}+cu130"
pip install -q torch=="${TORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" torchaudio=="${TORCHAUDIO_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu130
# Build deps pinned by SageAttention's pyproject; installed here because we build with --no-build-isolation
pip install -q "setuptools>=62,<75" "wheel>=0.38,<0.44" "packaging>=21,<24" ninja
python - <<'PY'
import torch, triton
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| triton", triton.__version__)
assert torch.__version__.startswith("2.13.0"), torch.__version__
PY

log "Cloning SageAttention ${SAGE_TAG}"
rm -rf "$WORK/SageAttention"
git clone -q --depth 1 --branch "$SAGE_TAG" https://github.com/thu-ml/SageAttention "$WORK/SageAttention"
cd "$WORK/SageAttention"
# Tag the wheel with the CUDA/torch it was built against so it is never confused with other builds
sed -i "s/version='2.2.0'/version='2.2.0+${LOCAL_VERSION}'/" setup.py
grep -q "2.2.0+${LOCAL_VERSION}" setup.py

log "Building wheel for arches: ${ARCH_LIST} (this takes 10-30 min)"
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
export MAX_JOBS="$(nproc)"
export EXT_PARALLEL=4
rm -f "$OUT"/sageattention-*.whl
time pip wheel . --no-build-isolation --no-deps -w "$OUT" 2>&1 | grep -vE 'ptxas info|bytes stack frame|bytes spill|Compiling entry function|Function properties|Used [0-9]+ registers' | tail -40

WHEEL=$(ls "$OUT"/sageattention-*.whl)
log "Built: $WHEEL"
ls -lh "$WHEEL"
sha256sum "$WHEEL" | tee "$WHEEL.sha256"

log "Installing wheel and checking import"
pip install -q --force-reinstall --no-deps "$WHEEL"
cd /
python - <<'PY'
import sageattention, importlib
print("sageattention", getattr(sageattention, "__version__", "?"), "from", sageattention.__file__)
for m in ("_fused", "_qattn_sm80", "_qattn_sm89", "_qattn_sm90"):
    try:
        importlib.import_module(f"sageattention.{m}")
        print("  ok  ", m)
    except Exception as e:
        print("  FAIL", m, "->", e)
PY

if [ "${SKIP_GPU_TEST:-0}" != "1" ]; then
    log "Numerical check against torch SDPA on the GPU"
    python - <<'PY'
import torch, torch.nn.functional as F
from sageattention import sageattn
assert torch.cuda.is_available(), "no GPU; rerun with SKIP_GPU_TEST=1 to skip"
print("GPU:", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
torch.manual_seed(0)
q, k, v = (torch.randn(2, 16, 4096, 128, device="cuda", dtype=torch.float16) for _ in range(3))
ref = F.scaled_dot_product_attention(q, k, v)
out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
cos = F.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0).item()
err = (out - ref).abs().max().item()
print(f"cosine similarity {cos:.6f} | max abs err {err:.4f}")
assert cos > 0.99, "SageAttention output does not match SDPA"
print("PASS")
PY
fi

if [ -n "${HF_TOKEN:-}" ]; then
    log "Uploading to https://huggingface.co/${HF_WHEEL_REPO}"
    pip install -q "huggingface_hub[cli]"
    NAME=$(basename "$WHEEL")
    hf upload "$HF_WHEEL_REPO" "$WHEEL" "$NAME" --repo-type model >/dev/null
    hf upload "$HF_WHEEL_REPO" "$WHEEL.sha256" "$NAME.sha256" --repo-type model >/dev/null
    echo
    echo "Wheel URL:"
    echo "  https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/${NAME}"
else
    echo
    echo "HF_TOKEN not set, wheel left at: $WHEEL"
    echo "Upload later with:"
    echo "  HF_TOKEN=hf_xxx hf upload ${HF_WHEEL_REPO} $WHEEL $(basename "$WHEEL") --repo-type model"
fi
