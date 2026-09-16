import asyncio
import json
from pathlib import Path

import pytest

import download_models as dm


def test_parse_entry_string_uses_last_path_segment():
    url = "https://huggingface.co/a/b/resolve/main/dir/My%20Model.safetensors"
    assert dm.parse_entry(url) == (url, "My Model.safetensors")


def test_parse_entry_object_requires_url_and_filename():
    entry = {"url": "https://civitai.com/api/download/models/1?fileId=2", "filename": "x.safetensors"}
    assert dm.parse_entry(entry) == (entry["url"], "x.safetensors")
    with pytest.raises(ValueError):
        dm.parse_entry({"url": "https://civitai.com/api/download/models/1"})
    with pytest.raises(ValueError):
        dm.parse_entry({"url": "https://civitai.com/x", "filename": "../evil"})
    with pytest.raises(ValueError):
        dm.parse_entry(42)


def test_load_config_from_file(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"loras": ["https://x/y.safetensors"], "note": "ignored"}))
    assert dm.load_config(str(p)) == {"loras": ["https://x/y.safetensors"]}


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dm.load_config(str(tmp_path / "nope.json"))


def test_plan_downloads_skips_existing_unless_forced(tmp_path):
    (tmp_path / "loras").mkdir()
    (tmp_path / "loras" / "have.safetensors").write_bytes(b"x")
    cfg = {
        "loras": [
            "https://h/have.safetensors",
            {"url": "https://civitai.com/api/download/models/1", "filename": "need.safetensors"},
        ],
        "vae": ["https://h/vae.safetensors"],
        "bogus": "not a list",
    }
    jobs = dm.plan_downloads(cfg, tmp_path)
    assert [(j.category, j.filename) for j in jobs] == [("loras", "need.safetensors"), ("vae", "vae.safetensors")]
    assert jobs[1].dest_dir == tmp_path / "vae"
    assert (tmp_path / "vae").is_dir()

    forced = dm.plan_downloads(cfg, tmp_path, force=True)
    assert [j.filename for j in forced] == ["have.safetensors", "need.safetensors", "vae.safetensors"]


def test_aria2c_command_adds_only_the_matching_auth_header(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_1")
    monkeypatch.setenv("CIVITAI_TOKEN", "civ_1")
    dest = Path("/tmp/models/loras")

    hf = dm.aria2c_command("https://huggingface.co/a/b/resolve/main/x.safetensors", dest, "x.safetensors")
    assert hf[0] == "aria2c"
    assert "--header=Authorization: Bearer hf_1" in hf
    assert "--header=Authorization: Bearer civ_1" not in hf
    assert hf[-4:] == ["-d", str(dest), "-o", "x.safetensors"]

    civ = dm.aria2c_command("https://civitai.com/api/download/models/1", dest, "y.safetensors")
    assert "--header=Authorization: Bearer civ_1" in civ
    assert "--header=Authorization: Bearer hf_1" not in civ

    other = dm.aria2c_command("https://example.com/z.bin", dest, "z.bin")
    assert not any(a.startswith("--header=Authorization") for a in other)
    assert "-c" in other and "-x" in other


def test_download_job_prefers_hf_client_then_falls_back(monkeypatch, tmp_path):
    job = dm.Job("vae", "https://huggingface.co/a/b/resolve/main/v.safetensors", "v.safetensors", tmp_path)
    monkeypatch.setenv("USE_HF_XET", "true")
    seen = []

    def failing_hf(url, output_dir, filename, staging_root):
        seen.append("hf")
        raise RuntimeError("xet down")

    async def fake_aria(url, dest_dir, filename):
        seen.append("aria2c")
        return True

    monkeypatch.setattr(dm, "download_via_hf", failing_hf)
    monkeypatch.setattr(dm, "download_with_aria2c", fake_aria)
    ok = asyncio.run(dm.download_job(job, asyncio.Semaphore(1)))
    assert ok is True
    assert seen == ["hf", "aria2c"]


def test_download_job_uses_aria2c_directly_for_non_hf(monkeypatch, tmp_path):
    job = dm.Job("loras", "https://civitai.com/api/download/models/1", "l.safetensors", tmp_path)
    seen = []

    def hf_should_not_run(*a, **k):
        seen.append("hf")
        raise AssertionError("hf client must not be used for civitai")

    async def fake_aria(url, dest_dir, filename):
        seen.append("aria2c")
        return False

    monkeypatch.setattr(dm, "download_via_hf", hf_should_not_run)
    monkeypatch.setattr(dm, "download_with_aria2c", fake_aria)
    assert asyncio.run(dm.download_job(job, asyncio.Semaphore(1))) is False
    assert seen == ["aria2c"]


def test_main_skip_env_returns_zero_without_touching_config(monkeypatch, tmp_path):
    monkeypatch.setenv("SKIP_MODEL_DOWNLOAD", "true")
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "log"))
    assert dm.main(["--config", str(tmp_path / "missing.json"), "--models-dir", str(tmp_path)]) == 0


def test_main_missing_config_returns_one(monkeypatch, tmp_path):
    monkeypatch.delenv("SKIP_MODEL_DOWNLOAD", raising=False)
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "log"))
    assert dm.main(["--config", str(tmp_path / "missing.json"), "--models-dir", str(tmp_path)]) == 1
