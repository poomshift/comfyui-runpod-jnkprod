from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())


def test_tests_workflow_runs_pytest_on_push_and_pr():
    wf = _load("tests.yml")
    on = wf.get("on") or wf.get(True)  # PyYAML parses bare `on:` as True
    assert "push" in on and "pull_request" in on
    steps = wf["jobs"]["pytest"]["steps"]
    assert any("pytest" in (s.get("run") or "") for s in steps)


def test_docker_workflow_pushes_only_on_main():
    wf = _load("docker-build.yml")
    on = wf.get("on") or wf.get(True)
    assert on["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in on
    build = wf["jobs"]["build"]
    step = next(s for s in build["steps"] if s.get("uses", "").startswith("docker/build-push-action"))
    w = step["with"]
    assert "promptalchemist/comfyui-runpod-jnkprod:latest" in w["tags"]
    assert "github_token=${{ secrets.ONYX_GITHUB_TOKEN }}" in w["secrets"]
    assert "SAGEATTENTION_WHEEL_URL=" in w["build-args"]
    assert w["push"] == "${{ github.event_name != 'pull_request' }}"
    assert any("rm -rf /usr/share/dotnet" in (s.get("run") or "") for s in build["steps"])


def test_docker_workflow_frees_disk_before_buildx_and_uses_registry_cache():
    raw = (ROOT / ".github" / "workflows" / "docker-build.yml").read_text()
    assert "type=gha" not in raw
    assert "AGENT_TOOLSDIRECTORY" not in raw

    steps = _load("docker-build.yml")["jobs"]["build"]["steps"]
    cleanup = next(i for i, s in enumerate(steps) if "rm -rf /usr/share/dotnet" in (s.get("run") or ""))
    buildx = next(i for i, s in enumerate(steps) if s.get("uses", "").startswith("docker/setup-buildx-action"))
    assert cleanup < buildx
    run = steps[cleanup]["run"]
    assert "/opt/hostedtoolcache" in run
    assert "data-root" in run and "/mnt/docker" in run

    w = next(s for s in steps if s.get("uses", "").startswith("docker/build-push-action"))["with"]
    assert w["cache-from"] == "type=registry,ref=promptalchemist/comfyui-runpod-jnkprod:buildcache"
    assert w["cache-to"] == (
        "${{ github.event_name != 'pull_request' && "
        "'type=registry,ref=promptalchemist/comfyui-runpod-jnkprod:buildcache,mode=max' || '' }}"
    )
