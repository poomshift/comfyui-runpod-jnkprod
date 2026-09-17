from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_ports_env_and_layout():
    text = (ROOT / "README.md").read_text()
    for needle in (
        "8188", "8888",
        "HF_TOKEN", "CIVITAI_TOKEN", "MODELS_CONFIG_URL", "SKIP_MODEL_DOWNLOAD",
        "USE_SAGE_ATTENTION", "COMFYUI_EXTRA_ARGS", "COMFYUI_RESTART_DELAY",
        "/workspace/models", "/workspace/output", "/workspace/input", "/workspace/user",
        "/workspace/logs/comfyui.log", "/workspace/models_config.json",
        "black-forest-labs/FLUX.2-klein-9B",
        "promptalchemist/comfyui-runpod-jnkprod",
        "ComfyUI-Manager", "--use-sage-attention",
        "JUPYTER_TOKEN", "60 GB", "Security",
        "MAX_CONCURRENT_DOWNLOADS", "USE_HF_XET", "DOWNLOAD_HEARTBEAT_SECONDS",
        "ComfyUI-Impact-Pack", "face_yolov8m.pt",
    ):
        assert needle in text, needle


def test_readme_tells_existing_pods_how_to_get_the_new_models():
    # start.sh never overwrites /workspace/models_config.json, so an older
    # volume keeps its list; the README has to say how to pick up new entries.
    text = " ".join((ROOT / "README.md").read_text().split())
    assert "already has `/workspace/models_config.json`" in text
    notice = text[text.index("already has `/workspace/models_config.json`"):][:400]
    assert "delete it" in notice
    assert "restart the pod" in notice
