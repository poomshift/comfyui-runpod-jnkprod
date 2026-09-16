#!/usr/bin/env python3
"""Boot-time model downloader.

Reads a models_config.json (category -> list of entries), downloads every file
that is not already under <models-dir>/<category>/, and logs to LOG_PATH and
stdout. Hugging Face resolve URLs go through the official client (Xet where
the repo has it) and fall back to aria2c; everything else uses aria2c.

Exit code is 0 even when downloads fail, so ComfyUI still starts; 1 only when
the config cannot be read.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.civitai import civitai_auth_args  # noqa: E402
from utils.hfAuth import hf_auth_args, redact_token  # noqa: E402
from utils.hfDownload import download_via_hf, hf_client_enabled, parse_hf_url  # noqa: E402

logger = logging.getLogger("download_models")


@dataclass
class Job:
    category: str
    url: str
    filename: str
    dest_dir: Path


def parse_entry(entry):
    """Return (url, filename) for a config entry.

    A string entry is a URL whose last path segment is the filename. An object
    entry carries an explicit filename, for URLs like Civitai's whose path ends
    in an id rather than a name.
    """
    if isinstance(entry, str):
        url = entry
        filename = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    elif isinstance(entry, dict):
        url = entry.get("url")
        filename = entry.get("filename")
        if not url or not filename:
            raise ValueError("object entries need both 'url' and 'filename': %r" % (entry,))
    else:
        raise ValueError("entry must be a URL string or an object: %r" % (entry,))

    if not filename or "/" in filename or filename in (".", ".."):
        raise ValueError("invalid filename %r for %s" % (filename, url))

    return url, filename


def load_config(path_or_url):
    """Load the config from a local path or an http(s) URL. Keeps only list values."""
    if path_or_url.startswith(("http://", "https://")):
        with urlopen(path_or_url, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    else:
        with open(path_or_url, "r", encoding="utf-8") as f:
            raw = json.load(f)

    return {k: v for k, v in raw.items() if isinstance(v, list)}


def plan_downloads(config, models_dir, force=False):
    """Turn the config into Jobs, creating category dirs and skipping present files."""
    models_dir = Path(models_dir)
    jobs = []

    for category, entries in config.items():
        if not isinstance(entries, list):
            logger.warning("Skipping '%s': not a list", category)
            continue

        dest_dir = models_dir / category
        dest_dir.mkdir(parents=True, exist_ok=True)

        for entry in entries:
            try:
                url, filename = parse_entry(entry)
            except ValueError as e:
                logger.error("Skipping bad entry in '%s': %s", category, e)
                continue

            if (dest_dir / filename).exists() and not force:
                logger.info("Skipping %s, already present in %s", filename, category)
                continue

            jobs.append(Job(category, url, filename, dest_dir))

    return jobs


def aria2c_command(url, dest_dir, filename):
    """Build the aria2c command, adding the auth header that matches the host."""
    return [
        "aria2c",
        *hf_auth_args(url),
        *civitai_auth_args(url),
        "--console-log-level=warn",
        "-c",
        "-x", "4",
        "-s", "4",
        "-k", "1M",
        "--file-allocation=none",
        "--max-tries=5",
        "--retry-wait=10",
        "--connect-timeout=30",
        "--timeout=600",
        "--summary-interval=30",
        url,
        "-d", str(dest_dir),
        "-o", filename,
    ]


async def download_with_aria2c(url, dest_dir, filename):
    """Run aria2c for one file. Returns True on success."""
    cmd = aria2c_command(url, dest_dir, filename)
    logger.info("aria2c: %s -> %s/%s", redact_token(url), dest_dir, filename)

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
    except Exception as e:  # aria2c missing, etc.
        logger.error("aria2c could not run for %s: %s", filename, redact_token(str(e)))
        return False

    if process.returncode == 0:
        return True

    output = (stderr or stdout or b"").decode("utf-8", errors="replace")
    logger.error("aria2c failed for %s (exit %s): %s", filename, process.returncode, redact_token(output.strip()[-2000:]))
    return False


async def download_job(job, semaphore):
    """Download one Job, preferring the Hugging Face client where it applies."""
    async with semaphore:
        logger.info("Starting %s (%s)", job.filename, job.category)

        if hf_client_enabled() and parse_hf_url(job.url):
            try:
                await asyncio.to_thread(
                    download_via_hf, job.url, str(job.dest_dir), job.filename, "/workspace/.hf_staging"
                )
                logger.info("Downloaded %s via the Hugging Face client", job.filename)
                return True
            except Exception as e:
                logger.warning("Hugging Face client failed for %s: %s; falling back to aria2c",
                               job.filename, redact_token(str(e)))

        ok = await download_with_aria2c(job.url, job.dest_dir, job.filename)
        if ok:
            logger.info("Downloaded %s via aria2c", job.filename)
        else:
            logger.error("FAILED %s (%s)", job.filename, job.category)
        return ok


async def run_jobs(jobs, max_concurrent):
    semaphore = asyncio.Semaphore(max_concurrent)
    results = await asyncio.gather(*(download_job(j, semaphore) for j in jobs))
    return sum(1 for r in results if r), sum(1 for r in results if not r)


def _configure_logging(log_path):
    """Log to stdout and LOG_PATH, exactly once per line and only from our logger."""
    fmt = logging.Formatter("[download] %(message)s")
    logger.setLevel(logging.INFO)
    # Our records never reach the root handlers, so nothing is ever logged twice.
    logger.propagate = False

    # Closed before being dropped, so a second call cannot leak the open log file.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)

    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="models_config.json path or URL")
    parser.add_argument("--models-dir", required=True, help="ComfyUI models root, e.g. /workspace/models")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args(argv)

    _configure_logging(os.getenv("LOG_PATH", "/workspace/logs/comfyui.log"))

    if (os.getenv("SKIP_MODEL_DOWNLOAD") or "").strip().lower() == "true":
        logger.info("SKIP_MODEL_DOWNLOAD=true, not downloading anything")
        return 0

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error("Cannot read config %s: %s", args.config, redact_token(str(e)))
        return 1

    jobs = plan_downloads(config, args.models_dir, force=args.force)
    if not jobs:
        logger.info("All models present, nothing to download")
        return 0

    max_concurrent = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "5"))
    logger.info("Downloading %d file(s), up to %d at a time", len(jobs), max_concurrent)
    ok, failed = asyncio.run(run_jobs(jobs, max_concurrent))
    logger.info("Done: %d succeeded, %d failed", ok, failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
