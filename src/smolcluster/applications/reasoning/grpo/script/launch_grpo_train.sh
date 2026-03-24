#!/bin/bash

# GRPO training launcher with preflight checks and vLLM worker health checks.
# Usage:
#   ./scripts/training/launch_grpo_train.sh
#   ./scripts/training/launch_grpo_train.sh --dry-run
#   ./scripts/training/launch_grpo_train.sh --cleanup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find project root by searching upward for pyproject.toml or .git
find_project_root() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/pyproject.toml" ]] || [[ -d "$dir/.git" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

PROJECT_DIR=$(find_project_root "$SCRIPT_DIR")
if [[ -z "$PROJECT_DIR" ]]; then
    echo "Error: could not find project root (looking for pyproject.toml or .git)"
    exit 1
fi

GRPO_CONFIG="$PROJECT_DIR/src/smolcluster/configs/inference/reasoning/grpo/config.yaml"
CLUSTER_CONFIG="$PROJECT_DIR/src/smolcluster/configs/inference/cluster_config_inference.yaml"
SESSION_NAME="grpo_train"

DRY_RUN=false
CLEANUP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --help|-h)
            echo "GRPO training launcher"
            echo ""
            echo "Options:"
            echo "  --dry-run    Print commands without executing"
            echo "  --cleanup    Kill the local tmux session used for GRPO training"
            echo "  --help, -h   Show help"
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

if [[ "$CLEANUP" == "true" ]]; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux kill-session -t "$SESSION_NAME"
        echo "Killed tmux session: $SESSION_NAME"
    else
        echo "No tmux session found: $SESSION_NAME"
    fi
    exit 0
fi

require_cmd() {
    local cmd="$1"
    echo -n "Checking for required command: $cmd ... "
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command not found: $cmd"
        exit 1
    fi
    echo "OK"
}

echo "Verifying required commands:"
require_cmd yq
require_cmd curl
require_cmd tmux
require_cmd ssh
require_cmd uv
echo "All required commands found."

if [[ ! -f "$GRPO_CONFIG" ]]; then
    echo "Error: GRPO config not found: $GRPO_CONFIG"
    exit 1
fi
if [[ ! -f "$CLUSTER_CONFIG" ]]; then
    echo "Error: cluster config not found: $CLUSTER_CONFIG"
    exit 1
fi

VLLM_ENABLED=$(yq '.vllm' "$GRPO_CONFIG")
if [[ "$VLLM_ENABLED" != "true" ]]; then
    echo "Error: vllm must be enabled in GRPO config to use this launcher"
    exit 1
fi

# Cluster topology comes from cluster_config_inference.yaml
NUM_WORKERS=$(yq '.num_workers' "$CLUSTER_CONFIG")
TOTAL_NUM_NODES=$(yq '.total_num_nodes' "$CLUSTER_CONFIG")
SERVER_HOST=$(yq '.server' "$CLUSTER_CONFIG")
# vLLM endpoint details stay in grpo config
VLLM_PORT=$(yq '.vllm_cluster.port' "$GRPO_CONFIG")
COMPLETION_PATH=$(yq '.vllm_cluster.completion_path' "$GRPO_CONFIG")

WORKER_ENTRIES=()
while IFS= read -r entry; do
    [[ -n "$entry" ]] && WORKER_ENTRIES+=("$entry")
done < <(yq '.workers.regular[] | .hostname + ":" + (.rank | tostring)' "$CLUSTER_CONFIG")

WORKER_HOSTS=()
WORKER_RANKS=()
for entry in "${WORKER_ENTRIES[@]}"; do
    WORKER_HOSTS+=("${entry%%:*}")
    WORKER_RANKS+=("${entry##*:}")
done

if [[ ${#WORKER_HOSTS[@]} -ne "$NUM_WORKERS" ]]; then
    echo "Error: vllm_cluster.num_workers ($NUM_WORKERS) does not match worker entries (${#WORKER_HOSTS[@]})"
    exit 1
fi

EXPECTED_TOTAL=$((NUM_WORKERS + 1))
if [[ "$TOTAL_NUM_NODES" -ne "$EXPECTED_TOTAL" ]]; then
    echo "Error: vllm_cluster.total_num_nodes ($TOTAL_NUM_NODES) must equal num_workers + 1 ($EXPECTED_TOTAL)"
    exit 1
fi

echo "Preflight checks"
echo "  Config: $GRPO_CONFIG"
echo "  Server: $SERVER_HOST"
echo "  Workers: ${WORKER_HOSTS[*]}"
echo "  Port/path: $VLLM_PORT$COMPLETION_PATH"

if [[ "$DRY_RUN" == "false" ]]; then
    for worker in "${WORKER_HOSTS[@]}"; do
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$worker" "echo ok" >/dev/null 2>&1; then
            echo "Error: cannot SSH to worker node: $worker"
            exit 1
        fi
    done

    for i in "${!WORKER_HOSTS[@]}"; do
        worker="${WORKER_HOSTS[$i]}"
        rank="${WORKER_RANKS[$i]}"
        worker_ip=$(yq ".host_ip.${worker}" "$CLUSTER_CONFIG")
        if [[ -z "$worker_ip" || "$worker_ip" == "null" ]]; then
            echo "Error: missing host_ip mapping for worker: $worker"
            exit 1
        fi

        completion_url="http://${worker_ip}:${VLLM_PORT}${COMPLETION_PATH}"
        models_url="http://${worker_ip}:${VLLM_PORT}/v1/models"

        echo "Checking worker rank $rank ($worker) at $completion_url"

        if ! curl -sS --max-time 8 "$models_url" >/dev/null; then
            echo "Error: vLLM /v1/models not reachable on $worker ($models_url)"
            exit 1
        fi

        payload='{"prompt":"ping","max_tokens":1,"temperature":0.0}'
        if ! curl -sS --max-time 12 -H "Content-Type: application/json" -d "$payload" "$completion_url" | yq '.choices[0].text' >/dev/null 2>&1; then
            echo "Error: vLLM completion check failed on $worker ($completion_url)"
            exit 1
        fi

        echo "  OK: vLLM reachable on $worker"
    done
else
    echo "Dry run enabled: skipping SSH and vLLM endpoint checks"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Existing tmux session found: $SESSION_NAME"
    if [[ "$DRY_RUN" == "false" ]]; then
        tmux kill-session -t "$SESSION_NAME"
        echo "Killed existing session: $SESSION_NAME"
    else
        echo "Dry run: would kill existing session"
    fi
fi

TRAIN_SCRIPT="$PROJECT_DIR/src/smolcluster/applications/reasoning/grpo/train.py"
VENV_ACTIVATE="$PROJECT_DIR/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "Error: .venv not found at $PROJECT_DIR/.venv"
    echo "Run 'uv sync --extra mlx' inside $PROJECT_DIR to create it."
    exit 1
fi

echo "Installing dependencies..."
cd "$PROJECT_DIR"
uv pip install -e .

HF_ENV_SETUP=""
# Prefer HF_TOKEN if provided, otherwise preserve an existing HUGGING_FACE_HUB_TOKEN.
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_ENV_SETUP="export HUGGING_FACE_HUB_TOKEN=\"${HF_TOKEN}\"; export HF_TOKEN=\"${HF_TOKEN}\"; "
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    HF_ENV_SETUP="export HUGGING_FACE_HUB_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; export HF_TOKEN=\"${HUGGING_FACE_HUB_TOKEN}\"; "
fi
# Enable the faster HF transfer path when available.
HF_ENV_SETUP+="export HF_HUB_ENABLE_HF_TRANSFER=1; "

TRAIN_CMD="source \"$VENV_ACTIVATE\" && ${HF_ENV_SETUP}python \"$TRAIN_SCRIPT\""
TMUX_CMD="bash -lc '$TRAIN_CMD; status=\$?; echo; echo Training exited with status \$status.; echo Session kept open for inspection.; exec bash -i'"

echo "Launching GRPO training on server node ($SERVER_HOST)"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run command: tmux new -d -s $SESSION_NAME \"$TMUX_CMD\""
else
    tmux new -d -s "$SESSION_NAME" "$TMUX_CMD"
    echo "Started tmux session: $SESSION_NAME"
    echo "Attach with: tmux attach -t $SESSION_NAME"
fi
