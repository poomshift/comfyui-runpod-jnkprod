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
