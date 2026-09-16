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
    assert "ARG COMFYUI_TAG=v0.34.2" in text
    assert "--branch ${COMFYUI_TAG}" in text
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
