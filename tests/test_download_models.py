import asyncio
import json
import logging
import os
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


def test_plan_downloads_supports_nested_categories(tmp_path):
    url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt"
    jobs = dm.plan_downloads({"ultralytics/bbox": [url]}, tmp_path)
    assert (tmp_path / "ultralytics" / "bbox").is_dir()
    assert [(j.category, j.url, j.filename, j.dest_dir) for j in jobs] == [
        ("ultralytics/bbox", url, "face_yolov8m.pt", tmp_path / "ultralytics" / "bbox")
    ]


def test_plan_downloads_rejects_categories_that_escape_the_models_dir(tmp_path, caplog):
    models = tmp_path / "models"
    models.mkdir()
    outside = tmp_path / "outside"
    cfg = {
        str(outside): ["https://h/abs.safetensors"],
        "../escape": ["https://h/up.safetensors"],
        "loras/../../escape2": ["https://h/up2.safetensors"],
        "sams": ["https://h/ok.pth"],
    }

    caplog.set_level(logging.INFO, logger="download_models")
    dm.logger.addHandler(caplog.handler)
    try:
        jobs = dm.plan_downloads(cfg, models)
    finally:
        dm.logger.removeHandler(caplog.handler)

    assert [(j.category, j.filename) for j in jobs] == [("sams", "ok.pth")]
    assert not outside.exists()
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "escape2").exists()
    assert not (models / "loras").exists()
    # A set: the record can reach caplog twice, through the root logger and the handler.
    errors = {r.getMessage() for r in caplog.records if r.levelno == logging.ERROR}
    assert len(errors) == 3, caplog.text
    for category in (str(outside), "../escape", "loras/../../escape2"):
        assert category in caplog.text


def test_plan_downloads_resumes_a_partial_aria2c_download(tmp_path, caplog):
    loras = tmp_path / "loras"
    loras.mkdir()
    (loras / "x.safetensors").write_bytes(b"partial")
    control = loras / "x.safetensors.aria2"
    control.write_bytes(b"ctl")
    cfg = {"loras": ["https://h/x.safetensors"]}

    caplog.set_level(logging.INFO, logger="download_models")
    dm.logger.addHandler(caplog.handler)
    try:
        jobs = dm.plan_downloads(cfg, tmp_path)
    finally:
        dm.logger.removeHandler(caplog.handler)
    assert [(j.category, j.url, j.filename, j.dest_dir) for j in jobs] == [
        ("loras", "https://h/x.safetensors", "x.safetensors", loras)
    ]
    assert "Resuming partial download of x.safetensors" in caplog.text

    control.unlink()
    assert dm.plan_downloads(cfg, tmp_path) == []


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
    # The per-second readout would flood the log; only the periodic summaries remain.
    assert "--show-console-readout=false" in other
    assert "--summary-interval=30" in other


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


def test_download_job_hf_success_drops_a_stale_aria2c_control_file(monkeypatch, tmp_path):
    # Otherwise the partial left by an earlier aria2c attempt would make every
    # later boot re-plan (and re-download) a file the HF client already finished.
    job = dm.Job("vae", "https://huggingface.co/a/b/resolve/main/v.safetensors", "v.safetensors", tmp_path)
    monkeypatch.setenv("USE_HF_XET", "true")
    (tmp_path / "v.safetensors.aria2").write_bytes(b"ctl")

    def fake_hf(url, output_dir, filename, staging_root):
        (Path(output_dir) / filename).write_bytes(b"model")
        return str(Path(output_dir) / filename)

    monkeypatch.setattr(dm, "download_via_hf", fake_hf)
    assert asyncio.run(dm.download_job(job, asyncio.Semaphore(1))) is True
    assert not (tmp_path / "v.safetensors.aria2").exists()
    assert dm.plan_downloads({tmp_path.name: [job.url]}, tmp_path.parent) == []


def test_download_job_stages_hf_downloads_in_hf_staging_dir(monkeypatch, tmp_path):
    job = dm.Job("vae", "https://huggingface.co/a/b/resolve/main/v.safetensors", "v.safetensors", tmp_path)
    monkeypatch.setenv("USE_HF_XET", "true")
    monkeypatch.setenv("HF_STAGING_DIR", str(tmp_path / "staging"))
    seen = []

    def fake_hf(url, output_dir, filename, staging_root):
        seen.append(staging_root)
        return str(Path(output_dir) / filename)

    monkeypatch.setattr(dm, "download_via_hf", fake_hf)
    assert asyncio.run(dm.download_job(job, asyncio.Semaphore(1))) is True
    assert seen == [str(tmp_path / "staging")]


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


def test_load_config_warns_about_non_list_values(tmp_path, caplog):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"loras": ["https://x/y.safetensors"], "note": "ignored"}))
    caplog.set_level(logging.INFO, logger="download_models")
    dm.logger.addHandler(caplog.handler)
    try:
        assert dm.load_config(str(p)) == {"loras": ["https://x/y.safetensors"]}
    finally:
        dm.logger.removeHandler(caplog.handler)
    assert "Ignoring 'note': not a list" in caplog.text


def test_log_sites_redact_every_token(monkeypatch, tmp_path, caplog):
    hf_token = "hf_tokenvalue1"
    civ_token = "civ_tokenvalue2"
    monkeypatch.setenv("HF_TOKEN", hf_token)
    monkeypatch.setenv("CIVITAI_TOKEN", civ_token)
    monkeypatch.delenv("SKIP_MODEL_DOWNLOAD", raising=False)
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "logs" / "comfyui.log"))

    cfg = {
        "loras": [
            {"url": "https://civitai.com/api/download/models/1?token=" + civ_token},
            "https://huggingface.co/a/b/resolve/main/?token=" + hf_token,
        ]
    }

    caplog.set_level(logging.INFO, logger="download_models")
    dm.logger.addHandler(caplog.handler)
    try:
        assert dm.plan_downloads(cfg, tmp_path / "models") == []
        planned = caplog.text
    finally:
        dm.logger.removeHandler(caplog.handler)

    missing = tmp_path / ("missing-" + hf_token + ".json")
    assert dm.main(["--config", str(missing), "--models-dir", str(tmp_path / "models")]) == 1

    logged = planned + (tmp_path / "logs" / "comfyui.log").read_text(encoding="utf-8")
    assert hf_token not in logged
    assert civ_token not in logged
    assert "***" in planned and "***" in logged


def test_max_concurrent_downloads_ignores_blank_and_junk(monkeypatch):
    monkeypatch.delenv("MAX_CONCURRENT_DOWNLOADS", raising=False)
    assert dm._max_concurrent_downloads() == 5
    for bad in ("", "   ", "0", "-3", "lots"):
        monkeypatch.setenv("MAX_CONCURRENT_DOWNLOADS", bad)
        assert dm._max_concurrent_downloads() == 5
    monkeypatch.setenv("MAX_CONCURRENT_DOWNLOADS", "3")
    assert dm._max_concurrent_downloads() == 3


def test_main_returns_zero_when_the_download_stage_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("SKIP_MODEL_DOWNLOAD", raising=False)
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "log"))
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"loras": ["https://h/x.safetensors"]}))

    def boom(*a, **k):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(dm, "plan_downloads", boom)
    assert dm.main(["--config", str(config), "--models-dir", str(tmp_path / "models")]) == 0


def test_configure_logging_survives_an_unwritable_log_path(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    dm._configure_logging(str(blocker / "comfyui.log"))
    try:
        assert [type(h) for h in dm.logger.handlers] == [logging.StreamHandler]
        dm.logger.info("still logging")
    finally:
        dm._configure_logging(None)


@pytest.fixture
def dm_log(caplog):
    """Capture download_models log records even after _configure_logging stopped propagation."""
    caplog.set_level(logging.INFO, logger="download_models")
    dm.logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        dm.logger.removeHandler(caplog.handler)


def _stub_aria2c(monkeypatch, tmp_path, body):
    """Put a fake aria2c first on PATH. It sees the real argv; $dir/$out are -d/-o."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "aria2c"
    stub.write_text(
        "#!/bin/sh\n"
        'dir=""; out=""\n'
        'while [ $# -gt 0 ]; do\n'
        '  case "$1" in -d) dir="$2"; shift ;; -o) out="$2"; shift ;; esac\n'
        "  shift\n"
        "done\n" + body
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])


def test_download_with_aria2c_streams_output_to_the_log(monkeypatch, tmp_path, dm_log):
    _stub_aria2c(monkeypatch, tmp_path, (
        'echo "[#a1b2c3 512MiB/1.0GiB(50%) CN:4 DL:120MiB ETA:4s]"\n'
        'echo "[#a1b2c3 1.0GiB/1.0GiB(100%) CN:4 DL:121MiB]"\n'
        'printf model > "$dir/$out"\n'
        "exit 0\n"
    ))
    dest = tmp_path / "models" / "loras"
    dest.mkdir(parents=True)

    ok = asyncio.run(dm.download_with_aria2c("https://example.com/x.safetensors", dest, "x.safetensors"))

    assert ok is True
    assert (dest / "x.safetensors").read_text() == "model"
    assert "aria2c[x.safetensors]: [#a1b2c3 512MiB/1.0GiB(50%) CN:4 DL:120MiB ETA:4s]" in dm_log.text
    assert "aria2c[x.safetensors]: [#a1b2c3 1.0GiB/1.0GiB(100%) CN:4 DL:121MiB]" in dm_log.text


def test_download_with_aria2c_failure_output_is_redacted(monkeypatch, tmp_path, dm_log):
    token = "civ_streamsecret42"
    monkeypatch.setenv("CIVITAI_TOKEN", token)
    _stub_aria2c(monkeypatch, tmp_path, (
        'echo "errorCode=24 Authorization failed for header Bearer ' + token + '" >&2\n'
        "exit 1\n"
    ))

    ok = asyncio.run(dm.download_with_aria2c(
        "https://civitai.com/api/download/models/1", tmp_path, "y.safetensors"))

    assert ok is False
    assert token not in dm_log.text
    assert "aria2c[y.safetensors]: errorCode=24 Authorization failed for header Bearer ***" in dm_log.text
    assert "aria2c failed for y.safetensors (exit 1)" in dm_log.text


def test_download_job_tracks_in_flight_names(monkeypatch, tmp_path):
    job = dm.Job("loras", "https://civitai.com/api/download/models/1", "l.safetensors", tmp_path)
    in_flight = set()
    seen = []

    async def fake_aria(url, dest_dir, filename):
        seen.append(set(in_flight))
        raise RuntimeError("boom")

    monkeypatch.setattr(dm, "download_with_aria2c", fake_aria)
    with pytest.raises(RuntimeError):
        asyncio.run(dm.download_job(job, asyncio.Semaphore(1), in_flight))
    assert seen == [{"loras/l.safetensors"}]
    assert in_flight == set()


def test_run_jobs_logs_a_heartbeat_while_downloads_run(monkeypatch, tmp_path, dm_log):
    monkeypatch.setenv("DOWNLOAD_HEARTBEAT_SECONDS", "0.05")
    jobs = [dm.Job("loras", "https://h/a.safetensors", "a.safetensors", tmp_path),
            dm.Job("vae", "https://h/b.safetensors", "b.safetensors", tmp_path)]

    async def slow_job(job, semaphore, in_flight):
        # Registers itself the way the real download_job does, then just waits.
        async with semaphore:
            in_flight.add(job.category + "/" + job.filename)
            try:
                await asyncio.sleep(0.2)
                return True
            finally:
                in_flight.discard(job.category + "/" + job.filename)

    monkeypatch.setattr(dm, "download_job", slow_job)
    assert asyncio.run(dm.run_jobs(jobs, 5)) == (2, 0)
    beats = [r.getMessage() for r in dm_log.records if r.getMessage().startswith("Still downloading")]
    assert beats, dm_log.text
    assert beats[0] == "Still downloading 2 file(s): loras/a.safetensors, vae/b.safetensors"


def test_run_jobs_counts_a_raising_job_as_failed(monkeypatch, tmp_path, dm_log):
    token = "hf_gathersecret7"
    monkeypatch.setenv("HF_TOKEN", token)
    jobs = [dm.Job("loras", "https://h/bad.safetensors", "bad.safetensors", tmp_path),
            dm.Job("loras", "https://h/good.safetensors", "good.safetensors", tmp_path)]

    async def fake_job(job, semaphore, in_flight):
        if job.filename == "bad.safetensors":
            raise RuntimeError("exploded with " + token)
        return True

    monkeypatch.setattr(dm, "download_job", fake_job)
    assert asyncio.run(dm.run_jobs(jobs, 5)) == (1, 1)
    assert "bad.safetensors" in dm_log.text and "exploded with ***" in dm_log.text
    assert token not in dm_log.text


def test_heartbeat_seconds_ignores_blank_and_junk(monkeypatch):
    monkeypatch.delenv("DOWNLOAD_HEARTBEAT_SECONDS", raising=False)
    assert dm._heartbeat_seconds() == 60
    for bad in ("", "  ", "0", "-1", "soon", "nan", "inf"):
        monkeypatch.setenv("DOWNLOAD_HEARTBEAT_SECONDS", bad)
        assert dm._heartbeat_seconds() == 60, bad
    monkeypatch.setenv("DOWNLOAD_HEARTBEAT_SECONDS", "30")
    assert dm._heartbeat_seconds() == 30
    monkeypatch.setenv("DOWNLOAD_HEARTBEAT_SECONDS", "0.05")
    assert dm._heartbeat_seconds() == 0.05
