import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.constants import COMFY_MODEL_FOLDERS, CUSTOM_NODE_MODEL_FOLDERS

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
                "models/sams", "models/onnx", "models/ultralytics/bbox", "models/ultralytics/segm",
                "output", "input", "user", "logs", ".cache/huggingface"):
        assert (ws / sub).is_dir(), sub


def test_ensure_dirs_copies_bundled_model_configs_without_overwriting(tmp_path):
    # --models-directory hides /opt/ComfyUI/models/configs, so its bundled
    # yaml files are copied to the volume, never over a file already there.
    src = tmp_path / "comfy" / "models" / "configs"
    src.mkdir(parents=True)
    (src / "a.yaml").write_text("bundled a")
    (src / "c.yaml").write_text("bundled c")
    dst = tmp_path / "workspace" / "models" / "configs"
    dst.mkdir(parents=True)
    (dst / "b.yaml").write_text("customer b")
    (dst / "c.yaml").write_text("customer c")

    r = run_fn(tmp_path, "ensure_dirs")
    assert r.returncode == 0, r.stderr
    assert (dst / "a.yaml").read_text() == "bundled a"
    assert (dst / "b.yaml").read_text() == "customer b"
    assert (dst / "c.yaml").read_text() == "customer c"

    (dst / "a.yaml").write_text("edited a")
    r = run_fn(tmp_path, "ensure_dirs; echo done")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "done"
    assert (dst / "a.yaml").read_text() == "edited a"


def test_ensure_dirs_succeeds_without_bundled_model_configs(tmp_path):
    r = run_fn(tmp_path, "ensure_dirs; echo done")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "done"
    assert r.stderr == ""
    assert (tmp_path / "workspace" / "models" / "configs").is_dir()
    assert list((tmp_path / "workspace" / "models" / "configs").iterdir()) == []


def test_ensure_dirs_wipes_orphaned_hf_staging(tmp_path):
    stale = tmp_path / "workspace" / ".hf_staging" / "old"
    stale.mkdir(parents=True)
    (stale / "flux.safetensors.part").write_bytes(b"x" * 16)
    r = run_fn(tmp_path, "ensure_dirs")
    assert r.returncode == 0, r.stderr
    staging = tmp_path / "workspace" / ".hf_staging"
    assert staging.is_dir()
    assert list(staging.iterdir()) == []


def test_ensure_dirs_honours_hf_staging_dir_but_never_wipes_the_workspace(tmp_path):
    ws = tmp_path / "workspace"
    (ws / "models" / "loras").mkdir(parents=True)
    (ws / "models" / "loras" / "keep.safetensors").write_bytes(b"x")
    custom = tmp_path / "staging"
    (custom / "old").mkdir(parents=True)
    r = run_fn(tmp_path, 'ensure_dirs; echo "$HF_STAGING_DIR"', env={"HF_STAGING_DIR": str(custom)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == str(custom)
    assert custom.is_dir() and list(custom.iterdir()) == []

    r = run_fn(tmp_path, "ensure_dirs", env={"HF_STAGING_DIR": str(ws)})
    assert r.returncode == 0, r.stderr
    assert (ws / "models" / "loras" / "keep.safetensors").exists()


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


def test_resolve_models_config_logs_an_error_when_the_copy_fails(tmp_path):
    r = run_fn(tmp_path, "ensure_dirs; resolve_models_config",
               env={"APP_DIR": str(tmp_path / "missing-app")})
    assert r.returncode == 0, r.stderr
    target = tmp_path / "workspace" / "models_config.json"
    assert not target.exists()
    log = (tmp_path / "workspace" / "logs" / "comfyui.log").read_text()
    assert f"ERROR: could not write {target}" in log
    assert "Wrote default models config" not in log


def test_seed_manager_config_only_when_missing(tmp_path):
    r = run_fn(tmp_path, "ensure_dirs; seed_manager_config")
    assert r.returncode == 0, r.stderr
    # ComfyUI v0.36.0 has the System User API, so Manager reads user/__manager.
    ini = tmp_path / "workspace" / "user" / "__manager" / "config.ini"
    # A legacy-path file would make Manager re-run its migration on every boot.
    assert not (tmp_path / "workspace" / "user" / "default" / "ComfyUI-Manager").exists()
    assert "use_uv = True" in ini.read_text().splitlines()
    assert "security_level = normal" in ini.read_text().splitlines()
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
    # Custom nodes (Impact-Pack, Impact-Subpack) register their model folders
    # under models_dir at import, so it must be the volume, not the image.
    assert args[args.index("--models-directory") + 1] == f"{ws}/models"

    r = run_fn(tmp_path, "comfy_args", env={"USE_SAGE_ATTENTION": "false", "COMFYUI_EXTRA_ARGS": "--fast --lowvram"})
    args = r.stdout.split()
    assert "--use-sage-attention" not in args
    assert args[-2:] == ["--fast", "--lowvram"]


@pytest.mark.parametrize("token", [None, "abc"])
def test_start_jupyter_arguments(tmp_path, token):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_file = tmp_path / "jupyter-argv.txt"
    stub = bindir / "jupyter"
    stub.write_text('#!/bin/sh\nfor a in "$@"; do printf \'%s\\n\' "$a"; done > "$ARGV_FILE"\n')
    stub.chmod(0o755)
    env = {"PATH": f"{bindir}:{os.environ['PATH']}", "ARGV_FILE": str(argv_file)}
    if token is not None:
        env["JUPYTER_TOKEN"] = token

    # `wait` lets the backgrounded jupyter stub finish writing before we read.
    r = run_fn(tmp_path, "ensure_dirs; start_jupyter; wait", env=env)
    assert r.returncode == 0, r.stderr
    argv = argv_file.read_text().splitlines()
    assert argv[0] == "lab"
    assert not any("allow_origin" in a for a in argv), argv
    assert f"--ServerApp.token={token or ''}" in argv
    assert "--ServerApp.password=" in argv
    assert f"--ServerApp.root_dir={tmp_path / 'workspace'}" in argv
    if token:
        assert token not in (tmp_path / "workspace" / "logs" / "comfyui.log").read_text()
        assert token not in r.stdout + r.stderr


@pytest.mark.parametrize("value", [None, "", "true", "True", "TRUE", "1", "yes"])
def test_comfy_args_sage_attention_is_on_unless_disabled(tmp_path, value):
    env = {} if value is None else {"USE_SAGE_ATTENTION": value}
    r = run_fn(tmp_path, "comfy_args", env=env)
    assert r.returncode == 0, r.stderr
    assert "--use-sage-attention" in r.stdout.split()


@pytest.mark.parametrize("value", ["false", "FALSE", "False", "0", "no", "No", "off", "OFF"])
def test_comfy_args_sage_attention_disabled_case_insensitively(tmp_path, value):
    r = run_fn(tmp_path, "comfy_args", env={"USE_SAGE_ATTENTION": value})
    assert r.returncode == 0, r.stderr
    assert "--use-sage-attention" not in r.stdout.split()


def test_resolve_models_config_fetches_valid_url(tmp_path):
    remote = tmp_path / "remote.json"
    remote.write_text('{"loras": ["https://example.invalid/a.safetensors"]}')
    r = run_fn(tmp_path, "ensure_dirs; resolve_models_config",
               env={"MODELS_CONFIG_URL": remote.as_uri()})
    assert r.returncode == 0, r.stderr
    target = tmp_path / "workspace" / "models_config.json"
    assert target.read_text() == remote.read_text()
    assert target.read_text() != (ROOT / "models_config.json").read_text()


def test_resolve_models_config_rejects_invalid_json_url(tmp_path):
    remote = tmp_path / "remote.html"
    remote.write_text("<html>not json</html>")
    r = run_fn(tmp_path, "ensure_dirs; resolve_models_config",
               env={"MODELS_CONFIG_URL": remote.as_uri()})
    assert r.returncode == 0, r.stderr
    target = tmp_path / "workspace" / "models_config.json"
    assert target.read_text() == (ROOT / "models_config.json").read_text()
    assert not (tmp_path / "workspace" / "models_config.json.tmp").exists()


def test_model_folders_cover_comfy_and_custom_node_folders(tmp_path):
    r = run_fn(tmp_path, 'printf "%s\\n" "${MODEL_FOLDERS[@]}"')
    assert r.returncode == 0, r.stderr
    printed = set(r.stdout.split())
    assert printed == COMFY_MODEL_FOLDERS | CUSTOM_NODE_MODEL_FOLDERS


def test_supervise_comfyui_restarts_after_crash(tmp_path):
    app = tmp_path / "app"
    app.mkdir(exist_ok=True)
    shutil.copy(ROOT / "models_config.json", app / "models_config.json")
    comfy = tmp_path / "comfy"
    comfy.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    record = tmp_path / "invocations.txt"
    stub = bindir / "python"
    stub.write_text('#!/bin/sh\necho "$*" >> "$RECORD"\nexit 3\n')
    stub.chmod(0o755)
    env = {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "WORKSPACE": str(tmp_path / "workspace"),
        "APP_DIR": str(app),
        "COMFY_DIR": str(comfy),
        "COMFYUI_RESTART_DELAY": "0.2",
        "RECORD": str(record),
    }
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            ["bash", "-c", f"source '{START}'; ensure_dirs; supervise_comfyui"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2,
        )
    calls = [c for c in record.read_text().splitlines()
             if c.startswith("main.py --listen 0.0.0.0 --port 8188")]
    assert len(calls) >= 2, record.read_text()
    log = (tmp_path / "workspace" / "logs" / "comfyui.log").read_text()
    crashes = [ln for ln in log.splitlines() if "ComfyUI exited with status 3" in ln]
    assert len(crashes) >= 2, log
