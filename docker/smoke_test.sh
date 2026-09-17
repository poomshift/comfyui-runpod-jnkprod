#!/usr/bin/env bash
# Build-time smoke test, run by the Dockerfile from /opt/ComfyUI.
#
# --quick-test-for-ci only imports the custom nodes, and some nodes import
# their dependencies lazily inside INPUT_TYPES (OnyxDetailer imports
# Impact-Pack there). So this starts ComfyUI on CPU, fetches /object_info,
# which calls INPUT_TYPES on every node, and then checks the server log and
# that the node list has one node from every pack.
set -euo pipefail

ci="${SMOKE_TEST_DIR:-/tmp/ci}"
port=8199
timeout_seconds=300

# ComfyUI-Manager is not listed: its NODE_CLASS_MAPPINGS is empty.
required_nodes=(
    "OnyxDetailer"                 # Onyx_Custom_Nodes
    "FaceDetailer"                 # ComfyUI-Impact-Pack
    "SAMLoader"                    # ComfyUI-Impact-Pack
    "UltralyticsDetectorProvider"  # ComfyUI-Impact-Subpack
    "RIFE VFI"                     # ComfyUI-Frame-Interpolation
    "ShowText|pysssss"             # ComfyUI-Custom-Scripts
    "Power Lora Loader (rgthree)"  # rgthree-comfy
    "CRT Post-Process Suite"       # CRT-Nodes
    "FameGridColorFinish"          # ComfyUI-FameGridColorFinish
    "TextEncodeEditAdvanced"       # ComfyUi-TextEncodeEditAdvanced
    "CannyEdgePreprocessor"        # comfyui_controlnet_aux
)

log() { echo "[smoke test] $*"; }

server=""
stop_server() {
    [ -n "$server" ] || return 0
    log "Stopping ComfyUI"
    kill "$server" 2>/dev/null || true
    local i=0
    while kill -0 "$server" 2>/dev/null && [ "$i" -lt 30 ]; do
        sleep 1
        i=$((i + 1))
    done
    kill -9 "$server" 2>/dev/null || true
    wait "$server" 2>/dev/null || true
    server=""
}
trap stop_server EXIT

mkdir -p "$ci"/{user,out,in,models}

python main.py --cpu --listen 127.0.0.1 --port "$port" --disable-auto-launch \
    --user-directory "$ci/user" --output-directory "$ci/out" \
    --input-directory "$ci/in" --models-directory "$ci/models" \
    >"$ci/server.log" 2>&1 &
server=$!

deadline=$((SECONDS + timeout_seconds))
until curl -sf --max-time 60 "http://127.0.0.1:$port/object_info" -o "$ci/object_info.json"; do
    if ! kill -0 "$server" 2>/dev/null; then
        cat "$ci/server.log"
        log "FAILED: ComfyUI exited before /object_info answered"
        exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        cat "$ci/server.log"
        log "FAILED: /object_info did not answer within ${timeout_seconds}s"
        exit 1
    fi
    sleep 2
done

stop_server
cat "$ci/server.log"

if grep -nE "IMPORT FAILED|Cannot import|An error occurred while retrieving information" "$ci/server.log"; then
    log "FAILED: the server log reports a node that did not load (lines above)"
    exit 1
fi

python -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    info = json.load(f)
missing = [node for node in sys.argv[2:] if node not in info]
if missing:
    print("[smoke test] FAILED: missing from /object_info: " + ", ".join(missing))
    sys.exit(1)
print("[smoke test] /object_info lists %d nodes, including all %d required" % (len(info), len(sys.argv) - 2))
' "$ci/object_info.json" "${required_nodes[@]}"

rm -rf "$ci"
log "OK"
