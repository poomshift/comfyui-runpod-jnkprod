import os

import pytest

from utils import hfDownload


@pytest.mark.parametrize("url,expected", [
    (
        "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
        {"repo_id": "Comfy-Org/flux2-dev", "repo_type": "model", "revision": "main",
         "path": "split_files/vae/flux2-vae.safetensors"},
    ),
    (
        "https://huggingface.co/datasets/org/data/resolve/v1/a%20b.bin",
        {"repo_id": "org/data", "repo_type": "dataset", "revision": "v1", "path": "a b.bin"},
    ),
    ("https://huggingface.co/org/repo/blob/main/x.safetensors", None),
    ("https://cdn-lfs.huggingface.co/repos/abc", None),
    ("https://civitai.com/api/download/models/1", None),
    ("https://huggingface.co/a/b/c/resolve/main/x", None),
])
def test_parse_hf_url(url, expected):
    assert hfDownload.parse_hf_url(url) == expected


def test_hf_client_enabled_env(monkeypatch):
    monkeypatch.delenv("USE_HF_XET", raising=False)
    assert hfDownload.hf_client_enabled() is True
    monkeypatch.setenv("USE_HF_XET", "false")
    assert hfDownload.hf_client_enabled() is False
    monkeypatch.setenv("USE_HF_XET", "0")
    assert hfDownload.hf_client_enabled() is False


def test_download_via_hf_moves_file_into_place_and_cleans_staging(tmp_path, monkeypatch):
    calls = {}

    def fake_hf_hub_download(**kwargs):
        calls.update(kwargs)
        staged = os.path.join(kwargs["local_dir"], os.path.basename(kwargs["filename"]))
        os.makedirs(os.path.dirname(staged), exist_ok=True)
        with open(staged, "wb") as f:
            f.write(b"weights")
        return staged

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setenv("HF_TOKEN", "hf_x")

    out_dir = tmp_path / "models" / "vae"
    staging = tmp_path / "staging"
    result = hfDownload.download_via_hf(
        "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
        str(out_dir), "flux2-vae.safetensors", staging_root=str(staging),
    )

    assert result == str(out_dir / "flux2-vae.safetensors")
    assert (out_dir / "flux2-vae.safetensors").read_bytes() == b"weights"
    assert calls["repo_id"] == "Comfy-Org/flux2-dev"
    assert calls["filename"] == "split_files/vae/flux2-vae.safetensors"
    assert calls["token"] == "hf_x"
    assert list(staging.iterdir()) == []


def test_download_via_hf_rejects_non_hf_url(tmp_path):
    with pytest.raises(ValueError):
        hfDownload.download_via_hf("https://civitai.com/x", str(tmp_path), "x.bin", staging_root=str(tmp_path / "s"))
