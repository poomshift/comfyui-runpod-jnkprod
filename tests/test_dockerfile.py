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

# Packs Onyx_Custom_Nodes imports at run time (OnyxDetailer, onyx_rife_batched)
# or expects next to itself.
ONYX_DEPENDENCY_NODES = [
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
    "https://github.com/ltdrdata/ComfyUI-Impact-Subpack",
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation",
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
]


def test_dockerfile_contract():
    text = (ROOT / "Dockerfile").read_text()
    assert text.startswith("# syntax=docker/dockerfile:1")
    assert "FROM nvidia/cuda:13.0.3-base-ubuntu24.04" in text
    assert "python3.12" in text
    assert "&& git lfs install --system" in text
    assert "--index-url https://download.pytorch.org/whl/cu130" in text
    assert "ARG COMFYUI_TAG=v0.36.0" in text
    assert "--branch ${COMFYUI_TAG}" in text
    assert "ARG SAGEATTENTION_WHEEL_URL=https://huggingface.co/Patarapoom/sageattention-wheels/resolve/main/sageattention-2.2.0+cu130.torch2.13.0-cp312-cp312-linux_x86_64.whl" in text
    assert "--mount=type=secret,id=github_token" in text
    assert "AUTHORIZATION: basic" in text
    assert "AUTHORIZATION: bearer" not in text
    assert "ENV PIP_CONSTRAINT=/app/constraints.txt UV_CONSTRAINT=/app/constraints.txt" in text
    assert "assert torch.__version__.startswith('2.13.0+cu130')" in text
    for node in NODES:
        assert node in text, node
    for node in ONYX_DEPENDENCY_NODES:
        assert f"git clone --depth 1 {node} " in text or f"git clone --depth 1 {node}\n" in text, node
    assert "opencv-contrib-python-headless" in text
    # The smoke test fetches /object_info; --quick-test-for-ci only imported nodes.
    assert "--quick-test-for-ci" not in text
    assert "COPY docker/smoke_test.sh /app/smoke_test.sh" in text
    assert "EXPOSE 8188 8888" in text
    assert 'CMD ["/app/start.sh"]' in text
    # start.sh passes --models-directory; the yaml it replaced must not come back.
    assert "extra_model_paths" not in text
    # every pip install in the image is constrained
    for line in text.splitlines():
        if "uv pip install" in line and "constraints.txt" not in line and "--find-links" not in line:
            assert "-c /app/constraints.txt" in line, line


def test_dockerignore_excludes_dev_files():
    text = (ROOT / ".dockerignore").read_text().splitlines()
    for pattern in (".git", ".venv", "tests", "docs", "scripts", ".github", "__pycache__", "*.pyc"):
        assert pattern in text, pattern
    # docker/smoke_test.sh is part of the build context.
    for line in text:
        assert not line.strip().lstrip("/").startswith("docker"), line
        assert line.strip() not in ("*", "**"), line


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


def _node_deps_body():
    return next(b for b in _run_bodies() if "*/requirements.txt" in b)


def test_onyx_dependency_packs_are_cloned_with_the_public_nodes():
    body = next(b for b in _run_bodies() if "rgthree-comfy" in b)
    for node in ONYX_DEPENDENCY_NODES:
        assert f"git clone --depth 1 {node}" in body, node


def test_node_deps_step_contract():
    body = _node_deps_body()
    loop = body.index("for req in")
    # Impact-Pack and Impact-Subpack install.py download models unless this file exists.
    assert body.index("touch /opt/ComfyUI/custom_nodes/skip_download_model") < loop
    # SAM 2 builds against the image's torch, without CUDA kernels.
    assert body.index("export SAM2_BUILD_CUDA=0") < loop
    assert body.index("uv pip install -c /app/constraints.txt setuptools wheel") < loop
    assert "ComfyUI-Frame-Interpolation/requirements-no-cupy.txt" in body
    assert "ComfyUI-Frame-Interpolation/install.py" in body or "ComfyUI-Frame-Interpolation/*" in body
    # uv reads no such variable; setting it would only look like it works.
    assert "UV_NO_BUILD_ISOLATION_PACKAGE" not in (ROOT / "Dockerfile").read_text()


@pytest.mark.skipif(shutil.which("dash") is None and shutil.which("sh") is None, reason="needs a POSIX sh")
def test_node_deps_step_runs_the_expected_installs(tmp_path):
    """Run the node dependency RUN body under sh with uv and python stubbed out."""
    shell = shutil.which("dash") or shutil.which("sh")
    nodes = tmp_path / "custom_nodes"
    for name, files in {
        "ComfyUI-Impact-Pack": ["requirements.txt", "install.py"],
        "ComfyUI-Impact-Subpack": ["requirements.txt", "install.py"],
        "ComfyUI-Frame-Interpolation": ["requirements-no-cupy.txt", "requirements-with-cupy.txt", "install.py"],
        "ComfyUI-Custom-Scripts": [],
        "rgthree-comfy": ["requirements.txt"],
    }.items():
        (nodes / name).mkdir(parents=True)
        for f in files:
            (nodes / name / f).write_text("x\n")

    bindir = tmp_path / "bin"
    bindir.mkdir()
    record = tmp_path / "calls.txt"
    (bindir / "uv").write_text(
        "#!/bin/sh\n"
        'skip=no; [ -f "$NODES/skip_download_model" ] && skip=yes\n'
        'printf \'uv %s|SAM2_BUILD_CUDA=%s|skip=%s\\n\' "$*" "${SAM2_BUILD_CUDA-unset}" "$skip" >> "$RECORD"\n'
        'if [ "$*" = "pip freeze" ]; then printf \'numpy==2.1.0\\nopencv-contrib-python==4.12.0\\nopencv-python==4.12.0\\n\'; fi\n'
    )
    (bindir / "python").write_text(
        "#!/bin/sh\n"
        'printf \'python %s|dir=%s\\n\' "$*" "$(basename "$PWD")" >> "$RECORD"\n'
    )
    for stub in ("uv", "python"):
        (bindir / stub).chmod(0o755)

    script = tmp_path / "deps.sh"
    script.write_text(_node_deps_body().replace("/opt/ComfyUI/custom_nodes", str(nodes)) + "\n")
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "RECORD": str(record), "NODES": str(nodes)}
    r = subprocess.run([shell, str(script)], cwd=nodes, env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (nodes / "skip_download_model").is_file()

    calls = record.read_text().splitlines()
    uv_calls = [c for c in calls if c.startswith("uv ")]
    assert uv_calls[0].startswith("uv pip install -c /app/constraints.txt setuptools wheel|")
    assert all("|skip=yes" in c for c in uv_calls), calls
    installs = {c.split("-r ", 1)[1].split("|")[0]: c for c in uv_calls if " -r " in c}
    assert set(installs) == {
        "ComfyUI-Impact-Pack/requirements.txt",
        "ComfyUI-Impact-Subpack/requirements.txt",
        "ComfyUI-Frame-Interpolation/requirements-no-cupy.txt",
        "rgthree-comfy/requirements.txt",
    }
    for req, call in installs.items():
        assert call.startswith("uv pip install -c /app/constraints.txt "), call
        assert "|SAM2_BUILD_CUDA=0|" in call, call
        # Only Impact-Pack's file (sam2 named by URL alone) builds against the venv.
        assert ("--no-build-isolation" in call) is req.startswith("ComfyUI-Impact-Pack/"), call

    pythons = [c for c in calls if c.startswith("python ")]
    assert "python install.py|dir=ComfyUI-Impact-Pack" in pythons
    assert "python install.py|dir=ComfyUI-Impact-Subpack" in pythons
    assert not any("ComfyUI-Frame-Interpolation" in c for c in pythons), pythons

    assert "uv pip uninstall opencv-contrib-python opencv-python|SAM2_BUILD_CUDA=0|skip=yes" in calls
    assert calls[-1].startswith("uv pip install -c /app/constraints.txt opencv-contrib-python-headless|")


def test_smoke_test_script_is_copied_right_before_the_run_that_uses_it():
    lines = [ln for ln in (ROOT / "Dockerfile").read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    copy = lines.index("COPY docker/smoke_test.sh /app/smoke_test.sh")
    # Late in the file, so editing the script re-runs only the smoke test.
    node_deps = next(i for i, ln in enumerate(lines) if "*/requirements.txt" in ln)
    services = next(i for i, ln in enumerate(lines) if "jupyterlab" in ln)
    assert node_deps < services < copy
    assert lines[copy + 1].startswith("RUN ")
    body = next(b for b in _run_bodies() if "/app/smoke_test.sh" in b)
    assert "assert torch.__version__.startswith('2.13.0+cu130')" in body
    assert body.index("torch.__version__") < body.index("bash /app/smoke_test.sh")
