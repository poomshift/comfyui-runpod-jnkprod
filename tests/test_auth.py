import pytest

from utils import civitai, hfAuth


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "CIVITAI_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_hf_token_prefers_explicit_then_env(monkeypatch):
    assert hfAuth.get_hf_token() is None
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_env")
    assert hfAuth.get_hf_token() == "hf_env"
    monkeypatch.setenv("HF_TOKEN", " hf_first ")
    assert hfAuth.get_hf_token() == "hf_first"
    assert hfAuth.get_hf_token("hf_explicit") == "hf_explicit"
    assert hfAuth.get_hf_token("   ") == "hf_first"


@pytest.mark.parametrize("url,expected", [
    ("https://huggingface.co/a/b/resolve/main/x.safetensors", True),
    ("https://hf.co/a/b/resolve/main/x.safetensors", True),
    ("https://cdn-lfs.huggingface.co/x", True),
    ("https://civitai.com/api/download/models/1", False),
    ("https://evilhuggingface.co/x", False),
    ("not a url", False),
])
def test_is_huggingface_url(url, expected):
    assert hfAuth.is_huggingface_url(url) is expected


def test_hf_auth_args_only_for_hf_hosts_with_token(monkeypatch):
    hf = "https://huggingface.co/a/b/resolve/main/x.safetensors"
    assert hfAuth.hf_auth_args(hf) == []
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    assert hfAuth.hf_auth_args(hf) == ["--header=Authorization: Bearer hf_abc"]
    assert hfAuth.hf_auth_args("https://civitai.com/api/download/models/1") == []


def test_civitai_token_and_host(monkeypatch):
    assert civitai.get_civitai_token() is None
    monkeypatch.setenv("CIVITAI_TOKEN", " civ_1 ")
    assert civitai.get_civitai_token() == "civ_1"
    assert civitai.is_civitai_url("https://civitai.com/api/download/models/1?fileId=2") is True
    assert civitai.is_civitai_url("https://civitai.red/models/1") is True
    assert civitai.is_civitai_url("https://huggingface.co/x") is False


def test_civitai_auth_args(monkeypatch):
    url = "https://civitai.com/api/download/models/1"
    assert civitai.civitai_auth_args(url) == []
    monkeypatch.setenv("CIVITAI_TOKEN", "civ_1")
    assert civitai.civitai_auth_args(url) == ["--header=Authorization: Bearer civ_1"]
    assert civitai.civitai_auth_args("https://huggingface.co/x") == []


def test_redact_token_masks_every_configured_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("CIVITAI_TOKEN", "civ_secret")
    text = "GET ?token=civ_secret Authorization: Bearer hf_secret done"
    assert hfAuth.redact_token(text) == "GET ?token=*** Authorization: Bearer *** done"
    assert hfAuth.redact_token("") == ""
    assert hfAuth.redact_token(None) is None
