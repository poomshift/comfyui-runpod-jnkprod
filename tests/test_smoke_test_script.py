import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docker" / "smoke_test.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")

# One node from every pack in the image, plus the ones Onyx relies on.
REQUIRED_NODE_IDS = [
    "OnyxDetailer",
    "FaceDetailer",
    "SAMLoader",
    "UltralyticsDetectorProvider",
    "RIFE VFI",
    "ShowText|pysssss",
    "Power Lora Loader (rgthree)",
    "CRT Post-Process Suite",
    "FameGridColorFinish",
    "TextEncodeEditAdvanced",
    "CannyEdgePreprocessor",
]

FAILURE_STRINGS = ["IMPORT FAILED", "Cannot import", "An error occurred while retrieving information"]


def test_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_script_lists_every_required_node_id():
    text = SCRIPT.read_text()
    for node in REQUIRED_NODE_IDS:
        assert f'"{node}"' in text, node


def test_script_greps_the_log_for_every_failure_string():
    text = SCRIPT.read_text()
    for needle in FAILURE_STRINGS:
        assert needle in text, needle


def test_script_starts_comfyui_the_way_the_brief_requires():
    text = SCRIPT.read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    for arg in ("--cpu", "--listen 127.0.0.1", "--disable-auto-launch", '--models-directory "$ci/models"'):
        assert arg in text, arg
    assert "port=8199" in text
    assert "timeout_seconds=300" in text
    assert 'rm -rf "$ci"' in text


def _run(tmp_path, nodes, server_log="", server_exits=False, curl_fails=False):
    """Run the script with stub python (for main.py only) and curl."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "main.py" ]; then\n'
        '    echo "$*" > "$STUB_DIR/server-argv.txt"\n'
        '    printf "%s\\n" "Starting server" "$STUB_SERVER_LOG"\n'
        '    [ "$STUB_SERVER_EXITS" = 1 ] && exit 3\n'
        # Like ComfyUI, answer HTTP only once startup (and its log) is done.
        '    touch "$STUB_DIR/ready"\n'
        "    exec sleep 60\n"
        "fi\n"
        'exec "$REAL_PYTHON" "$@"\n'
    )
    (bindir / "curl").write_text(
        "#!/bin/sh\n"
        '[ "$STUB_CURL_FAILS" = 1 ] && exit 7\n'
        '[ -f "$STUB_DIR/ready" ] || exit 7\n'
        'out=""; while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done\n'
        'cp "$STUB_DIR/object_info.json" "$out"\n'
    )
    for stub in ("python", "curl"):
        (bindir / stub).chmod(0o755)
    (tmp_path / "object_info.json").write_text(json.dumps({n: {"name": n} for n in nodes}))

    ci = tmp_path / "ci"
    env = {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "STUB_DIR": str(tmp_path),
        "STUB_SERVER_LOG": server_log,
        "STUB_SERVER_EXITS": "1" if server_exits else "0",
        "STUB_CURL_FAILS": "1" if curl_fails else "0",
        "SMOKE_TEST_DIR": str(ci),
    }
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
                       check=False, timeout=60)
    return r, ci


def test_passes_when_every_node_is_listed_and_the_log_is_clean(tmp_path):
    r, ci = _run(tmp_path, REQUIRED_NODE_IDS + ["KSampler"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "including all 11 required" in r.stdout
    argv = (tmp_path / "server-argv.txt").read_text().split()
    assert argv[argv.index("--models-directory") + 1] == f"{ci}/models"
    assert argv[argv.index("--port") + 1] == "8199"
    assert not ci.exists()


def test_fails_and_names_every_missing_node(tmp_path):
    present = [n for n in REQUIRED_NODE_IDS if n not in ("OnyxDetailer", "RIFE VFI")]
    r, _ = _run(tmp_path, present)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "missing from /object_info: OnyxDetailer, RIFE VFI" in r.stdout


@pytest.mark.parametrize("line", [
    "Cannot import /opt/ComfyUI/custom_nodes/x module for custom nodes: boom",
    "   0.1 seconds (IMPORT FAILED): /opt/ComfyUI/custom_nodes/ComfyUI-Impact-Pack",
    "[ERROR] An error occurred while retrieving information for the 'OnyxDetailer' node.",
])
def test_fails_when_the_log_reports_a_broken_node(tmp_path, line):
    r, _ = _run(tmp_path, REQUIRED_NODE_IDS, server_log=line)
    assert r.returncode == 1, r.stdout + r.stderr
    assert line in r.stdout


def test_fails_when_comfyui_exits_before_answering(tmp_path):
    r, _ = _run(tmp_path, REQUIRED_NODE_IDS, server_log="Traceback: boom", server_exits=True, curl_fails=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "Traceback: boom" in r.stdout
    assert "ComfyUI exited before /object_info answered" in r.stdout
