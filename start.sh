#!/usr/bin/env bash
# Boot script for the ComfyUI RunPod template.
#
# Order: create the /workspace layout -> start JupyterLab (8888) -> resolve
# models_config.json -> download missing models -> start ComfyUI (8188).
#
# Sourcing this file defines the functions without running anything, which is
# how the tests exercise it.

WORKSPACE="${WORKSPACE:-/workspace}"
APP_DIR="${APP_DIR:-/app}"
COMFY_DIR="${COMFY_DIR:-/opt/ComfyUI}"

export HF_TOKEN="${HF_TOKEN:-}"
export CIVITAI_TOKEN="${CIVITAI_TOKEN:-}"
# Empty means JupyterLab asks for no token.
export JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"
export MODELS_CONFIG_URL="${MODELS_CONFIG_URL:-}"
export SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-false}"
export USE_SAGE_ATTENTION="${USE_SAGE_ATTENTION:-true}"
export COMFYUI_EXTRA_ARGS="${COMFYUI_EXTRA_ARGS:-}"
export COMFYUI_RESTART_DELAY="${COMFYUI_RESTART_DELAY:-10}"
export LOG_PATH="${LOG_PATH:-$WORKSPACE/logs/comfyui.log}"

# Hugging Face client. HF_HOME lives on the volume so the Xet chunk cache
# survives restarts. The chunk cache only helps when files share content, which
# this model set mostly does not, hence the modest default.
export USE_HF_XET="${USE_HF_XET:-true}"
export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
export HF_XET_CHUNK_CACHE_SIZE_BYTES="${HF_XET_CHUNK_CACHE_SIZE_BYTES:-8589934592}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
# Per-download temp dirs for the Hugging Face client; read by download_models.py.
export HF_STAGING_DIR="${HF_STAGING_DIR:-$WORKSPACE/.hf_staging}"

# ComfyUI's own model folders, then the ones custom nodes register under
# --models-directory at import: sams and onnx (Impact-Pack), ultralytics/bbox
# and ultralytics/segm (Impact-Subpack).
MODEL_FOLDERS=(audio_encoders background_removal checkpoints classifiers clip_vision configs
    controlnet datasets detection diffusers diffusion_models embeddings frame_interpolation
    geometry_estimation gligen hypernetworks latent_upscale_models loras model_patches
    optical_flow photomaker style_models text_encoders upscale_models vae vae_approx
    sams onnx ultralytics/bbox ultralytics/segm)

log() { echo "[start] $*" | tee -a "$LOG_PATH"; }

ensure_dirs() {
    # Nothing downloads before this runs, so anything left in the staging dir is
    # an orphan of an interrupted download. The guard keeps a mistyped
    # HF_STAGING_DIR from wiping the volume.
    case "$HF_STAGING_DIR" in
        "" | / | "$WORKSPACE" | "$WORKSPACE"/) ;;
        *) rm -rf "$HF_STAGING_DIR" ;;
    esac
    mkdir -p "$WORKSPACE/logs" "$WORKSPACE/output" "$WORKSPACE/input" "$WORKSPACE/user" \
        "$WORKSPACE/.cache/huggingface" "$HF_STAGING_DIR"
    for f in "${MODEL_FOLDERS[@]}"; do
        mkdir -p "$WORKSPACE/models/$f"
    done
    touch "$LOG_PATH"
}

# The effective model list lives at $WORKSPACE/models_config.json. On first
# boot it comes from MODELS_CONFIG_URL when set and reachable, otherwise from
# the copy baked into the image. An existing file is never overwritten, so the
# customer can edit it on the volume.
resolve_models_config() {
    local target="$WORKSPACE/models_config.json"
    if [ -f "$target" ]; then
        log "Using existing $target"
        return 0
    fi
    if [ -n "$MODELS_CONFIG_URL" ]; then
        log "Fetching models config from $MODELS_CONFIG_URL"
        if curl -fsSL --retry 3 --retry-delay 5 --max-time 60 -o "$target.tmp" "$MODELS_CONFIG_URL" \
            && python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$target.tmp" 2>/dev/null; then
            mv "$target.tmp" "$target"
            return 0
        fi
        rm -f "$target.tmp"
        log "WARNING: could not fetch a valid config from MODELS_CONFIG_URL, using the built-in list"
    fi
    if cp "$APP_DIR/models_config.json" "$target"; then
        log "Wrote default models config to $target"
    else
        log "ERROR: could not write $target"
    fi
}

# ComfyUI-Manager reads its config from the user directory. Pre-seed it once so
# it installs node dependencies with uv (fast) instead of pip. ComfyUI v0.36.0
# has the System User API, so Manager uses user/__manager; a file at the legacy
# user/default/ComfyUI-Manager path would be migrated away and re-seeded on
# every boot.
seed_manager_config() {
    local dir="$WORKSPACE/user/__manager"
    local ini="$dir/config.ini"
    [ -f "$ini" ] && return 0
    mkdir -p "$dir"
    cat >"$ini" <<'INI'
[default]
preview_method = auto
git_exe =
use_uv = True
channel_url = https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main
share_option = all
bypass_ssl = False
file_logging = True
component_policy = workflow
update_policy = stable-comfyui
windows_selector_event_loop_policy = False
model_download_by_agent = False
downgrade_blacklist =
security_level = normal
skip_migration_check = True
always_lazy_install = False
network_mode = public
db_mode = cache
INI
}

# The token is never logged.
start_jupyter() {
    log "Starting JupyterLab on port 8888"
    CUDA_VISIBLE_DEVICES="" nohup jupyter lab \
        --allow-root --no-browser --ip=0.0.0.0 --port=8888 \
        --ServerApp.token="$JUPYTER_TOKEN" --ServerApp.password='' \
        --ServerApp.root_dir="$WORKSPACE" \
        >"$WORKSPACE/logs/jupyter.log" 2>&1 &
}

download_models() {
    log "Checking models (SKIP_MODEL_DOWNLOAD=$SKIP_MODEL_DOWNLOAD)"
    python "$APP_DIR/download_models.py" \
        --config "$WORKSPACE/models_config.json" \
        --models-dir "$WORKSPACE/models" \
        || log "WARNING: model download step reported an error, starting ComfyUI anyway"
}

# Prints the ComfyUI argument list, one per line.
comfy_args() {
    local args=(--listen 0.0.0.0 --port 8188
        --output-directory "$WORKSPACE/output"
        --input-directory "$WORKSPACE/input"
        --user-directory "$WORKSPACE/user"
        --models-directory "$WORKSPACE/models")
    # On unless explicitly disabled, in any letter case.
    local sage
    sage=$(printf '%s' "$USE_SAGE_ATTENTION" | tr '[:upper:]' '[:lower:]')
    case "$sage" in
        false | 0 | no | off) ;;
        *) args+=(--use-sage-attention) ;;
    esac
    if [ -n "$COMFYUI_EXTRA_ARGS" ]; then
        # shellcheck disable=SC2206
        args+=($COMFYUI_EXTRA_ARGS)
    fi
    printf '%s\n' "${args[@]}"
}

# Runs ComfyUI in the foreground of whatever calls it and restarts it whenever
# it dies, so a crash cannot leave the pod running with port 8188 dead.
# PIPESTATUS[0] is ComfyUI's own status; the pipeline's status is tee's.
supervise_comfyui() {
    local args=()
    while IFS= read -r line; do args+=("$line"); done < <(comfy_args)
    cd "$COMFY_DIR" || { log "ERROR: $COMFY_DIR missing"; return 1; }
    while true; do
        log "==================== ComfyUI starting $(date -u +%FT%TZ) ===================="
        log "python main.py ${args[*]}"
        python main.py "${args[@]}" 2>&1 | tee -a "$LOG_PATH"
        local status=${PIPESTATUS[0]}
        log "ComfyUI exited with status $status, restarting in ${COMFYUI_RESTART_DELAY}s"
        sleep "$COMFYUI_RESTART_DELAY"
    done
}

start_comfyui() {
    supervise_comfyui &
}

main() {
    # PID 1 ignores signals it has no handler for, so forward a pod stop to
    # ComfyUI and JupyterLab instead of dropping it.
    trap 'log "Stopping"; kill -TERM 0 2>/dev/null; exit 143' TERM INT
    ensure_dirs
    log "ComfyUI RunPod template starting (workspace: $WORKSPACE)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read -r gpu; do log "GPU: $gpu"; done
    start_jupyter
    resolve_models_config
    seed_manager_config
    download_models
    start_comfyui
    wait
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
