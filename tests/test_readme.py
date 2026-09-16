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
