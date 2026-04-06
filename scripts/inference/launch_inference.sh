#!/bin/bash

# Unified inference launcher for MP + DP
# Usage:
#   ./scripts/inference/launch_inference.sh --algorithm mp
#   ./scripts/inference/launch_inference.sh --algorithm syncps
#   ./scripts/inference/launch_inference.sh --algorithm classicdp
#   ./scripts/inference/launch_inference.sh --algorithm mp --dry-run
#   ./scripts/inference/launch_inference.sh --cleanup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/inference/cluster_config_inference.yaml"

# Load environment variables and log in to HuggingFace Hub (for gated models).
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi
# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/lib/node_helpers.sh"
# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/lib/logging_helpers.sh"
NODE_HELPERS_PROJECT_DIR="$PROJECT_DIR"
ensure_hf_login_local
ensure_redis_running

ALGORITHM="mp"
DRY_RUN=false
CLEANUP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithm|-a)
            ALGORITHM="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --help|-h)
            echo "Unified inference launcher"
            echo ""
            echo "Options:"
            echo "  --algorithm, -a   mp | syncps | classicdp (default: mp)"
            echo "  --dry-run         Print commands without executing"
            echo "  --cleanup         Kill ALL smolcluster inference sessions and free ports (syncps + classicdp + mp)"
            echo "  --help, -h        Show help"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# --cleanup: kill every smolcluster inference session across all algorithms
# and free the shared API/frontend ports
# ---------------------------------------------------------------------------
do_cleanup() {
    echo ""
    echo "🧹 Cleaning up ALL smolcluster sessions (training + inference)..."
    echo ""

    # All smolcluster tmux session name patterns (training + inference + api/frontend)
    local SMOL_PATTERN='^(server|worker[0-9]*|classicdp_worker[0-9]*|fsdp_worker[0-9]*|ep_worker[0-9]*|mp_pipeline_worker[0-9]*|syncps_inf_.*|classicdp_inf_.*|mp_inference_.*|mp_tablet_proxy[0-9]*|syncps_api|syncps_frontend|classicdp_api|classicdp_frontend|mp_api|mp_frontend)$'

    # Read ports from config (fall back to defaults if yq not available)
    local API_PORT=8000
    local FRONTEND_PORT=5050
    if [[ -f "$CONFIG_FILE" ]] && command -v yq >/dev/null 2>&1; then
        API_PORT=$(yq '.web_interface.api_port'    "$CONFIG_FILE" 2>/dev/null) || API_PORT=8000
        FRONTEND_PORT=$(yq '.web_interface.frontend_port' "$CONFIG_FILE" 2>/dev/null) || FRONTEND_PORT=5050
    fi

    # --- LOCAL tmux sessions ---
    echo "   Killing local tmux sessions..."
    local _killed=0
    while IFS= read -r _s; do
        [[ -z "$_s" ]] && continue
        tmux kill-session -t "$_s" 2>/dev/null && echo "   [OK] Killed: $_s" && (( _killed++ )) || true
    done < <(tmux ls 2>/dev/null | cut -d: -f1 | grep -E "$SMOL_PATTERN" || true)
    [[ $_killed -eq 0 ]] && echo "   (no matching local sessions)"

    # --- LOCAL ports ---
    echo ""
    echo "   Freeing local ports ($API_PORT, $FRONTEND_PORT)..."
    for PORT in "$API_PORT" "$FRONTEND_PORT"; do
        local PIDS
        PIDS=$(lsof -ti tcp:"$PORT" 2>/dev/null | sort -u)
        if [[ -n "$PIDS" ]]; then
            echo "$PIDS" | xargs kill 2>/dev/null || true
            sleep 1
            PIDS=$(lsof -ti tcp:"$PORT" 2>/dev/null | sort -u)
            [[ -n "$PIDS" ]] && { echo "$PIDS" | xargs kill -9 2>/dev/null || true; }
            echo "   [OK] Freed port $PORT"
        else
            echo "   [OK] Port $PORT already free"
        fi
    done

    # --- REMOTE tmux sessions via SSH ---
    echo ""
    echo "   Attempting remote cleanup on cluster nodes..."

    # Collect SSH hosts: union of inference config + training configs
    local SSH_HOSTS=()
    local _seen_hosts=()
    _add_hosts_from_config() {
        local cfg="$1"
        [[ -f "$cfg" ]] || return
        command -v yq >/dev/null 2>&1 || return
        while IFS= read -r host; do
            [[ -z "$host" || "$host" == "null" ]] && continue
            # skip duplicates
            local _dup=0
            for _h in "${_seen_hosts[@]}"; do [[ "$_h" == "$host" ]] && _dup=1 && break; done
            [[ $_dup -eq 1 ]] && continue
            _seen_hosts+=("$host")
            SSH_HOSTS+=("$host")
        done < <(yq '.host_ip | keys | .[]' "$cfg" 2>/dev/null || true)
    }
    _add_hosts_from_config "$CONFIG_FILE"
    _add_hosts_from_config "$PROJECT_DIR/src/smolcluster/configs/cluster_config_syncps.yaml"
    _add_hosts_from_config "$PROJECT_DIR/src/smolcluster/configs/cluster_config_mp.yaml"
    _add_hosts_from_config "$PROJECT_DIR/src/smolcluster/configs/cluster_config_classicdp.yaml"

    if [[ ${#SSH_HOSTS[@]} -eq 0 ]]; then
        echo "   [SKIP] No SSH hosts found in any config"
    else
        local _remote_kill_cmd="tmux ls 2>/dev/null | cut -d: -f1 | grep -E '$SMOL_PATTERN' | while IFS= read -r _s; do tmux kill-session -t \"\$_s\" 2>/dev/null; done; echo ok"
        for host in "${SSH_HOSTS[@]}"; do
            [[ "$host" == "ipad" ]] && continue
            if ssh -o ConnectTimeout=3 -o BatchMode=yes "$host" "$_remote_kill_cmd" >/dev/null 2>&1; then
                echo "   [OK] Remote cleanup done on: $host"
            else
                echo "   [SKIP] $host unreachable or SSH failed"
            fi
        done
    fi

    echo ""
    echo "✅ All smolcluster sessions terminated and ports freed."
    echo ""
}

if [[ "$CLEANUP" == "true" ]]; then
    do_cleanup
    exit 0
fi

do_cleanup

case "$ALGORITHM" in
    mp)
        TARGET_SCRIPT="$SCRIPT_DIR/model_parallelism/launch_mp_inference.sh"
        ;;
    syncps)
        TARGET_SCRIPT="$SCRIPT_DIR/data_parallelism/launch_syncps_inference.sh"
        ;;
    classicdp)
        TARGET_SCRIPT="$SCRIPT_DIR/data_parallelism/launch_classicdp_inference.sh"
        ;;
    *)
        echo "❌ Unsupported algorithm: $ALGORITHM"
        echo "Supported: mp, syncps, classicdp"
        exit 1
        ;;
esac

if [[ ! -f "$TARGET_SCRIPT" ]]; then
    echo "❌ Missing target script: $TARGET_SCRIPT"
    exit 1
fi

if [[ "$DRY_RUN" != "true" ]]; then
    do_cleanup
fi

echo "🚀 Launching inference with algorithm: $ALGORITHM"

if [[ "$DRY_RUN" == "true" ]]; then
    bash "$TARGET_SCRIPT" --dry-run
else
    bash "$TARGET_SCRIPT"
fi
