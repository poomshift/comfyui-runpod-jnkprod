import json
from pathlib import Path

from tests.constants import COMFY_MODEL_FOLDERS

ROOT = Path(__file__).resolve().parents[1]


def test_constraints_pin_torch_stack():
    text = (ROOT / "constraints.txt").read_text()
    for line in (
        "torch==2.13.0+cu130",
        "torchvision==0.28.0+cu130",
        "torchaudio==2.11.0+cu130",
        "triton==3.7.1",
        "numpy>=2,<3",
    ):
        assert line in text.splitlines(), line


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
