"""Civitai token handling for the boot-time downloader."""
import os
from urllib.parse import urlparse

_CIVITAI_HOSTS = ("civitai.com", "civitai.red")


def get_civitai_token():
    """Return CIVITAI_TOKEN from the environment, or None."""
    value = (os.getenv("CIVITAI_TOKEN") or "").strip()
    return value or None


def is_civitai_url(url):
    """True only for Civitai hosts, so the token is never sent anywhere else."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False

    return any(host == h or host.endswith("." + h) for h in _CIVITAI_HOSTS)


def civitai_auth_args(url):
    """aria2c args that authenticate a Civitai download, or [] when not applicable."""
    token = get_civitai_token()

    if not token or not is_civitai_url(url):
        return []

    return ["--header=Authorization: Bearer " + token]
