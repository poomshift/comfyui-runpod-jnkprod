import shutil
import subprocess
from pathlib import Path

import pytest

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
    assert "&& git lfs install --system" in text
    assert "--index-url https://download.pytorch.org/whl/cu130" in text
    assert "ARG COMFYUI_TAG=v0.34.2" in text
    assert "--branch ${COMFYUI_TAG}" in text
    assert "ARG SAGEATTENTION_WHEEL_URL=https://huggingface.co/Patarapoom/sageattention-wheels/resolve/main/sageattention-2.2.0+cu130.torch2.13.0-cp312-cp312-linux_x86_64.whl" in text
    assert "--mount=type=secret,id=github_token" in text
    assert "ENV PIP_CONSTRAINT=/app/constraints.txt UV_CONSTRAINT=/app/constraints.txt" in text
    assert "assert torch.__version__.startswith('2.13.0+cu130')" in text
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


def _run_bodies():
    """Shell bodies of every RUN instruction, with continuations joined and flags dropped."""
    logical, current = [], ""
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        if not current and (not line.strip() or line.lstrip().startswith("#")):
            continue
        if line.endswith("\\"):
            current += line[:-1]
            continue
        logical.append(current + line)
        current = ""
    bodies = []
    for instruction in logical:
        if not instruction.startswith("RUN "):
            continue
        body = instruction[len("RUN "):].lstrip()
        while body.startswith("--"):
            body = body.split(None, 1)[1]
        bodies.append(body)
    return bodies


@pytest.mark.skipif(shutil.which("dash") is None and shutil.which("sh") is None, reason="needs a POSIX sh")
def test_run_bodies_are_valid_posix_sh(tmp_path):
    shell = shutil.which("dash") or shutil.which("sh")
    bodies = _run_bodies()
    assert len(bodies) >= 8
    for i, body in enumerate(bodies):
        script = tmp_path / f"run{i}.sh"
        script.write_text(body + "\n")
        r = subprocess.run([shell, "-n", str(script)], capture_output=True, text=True, check=False)
        assert r.returncode == 0, (body, r.stderr)


def test_opencv_uninstall_is_skipped_when_nothing_matches():
    body = next(b for b in _run_bodies() if "opencv-contrib-python-headless" in b)
    assert "pkgs=$(uv pip freeze | grep -iE '^opencv' | cut -d= -f1 || true)" in body
    assert 'if [ -n "$pkgs" ]; then uv pip uninstall $pkgs; fi' in body
    assert "uv pip uninstall $(" not in body
