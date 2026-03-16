#!/bin/bash

# Launch SyncPS inference server + workers, then chat API/frontend.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/inference/cluster_config_inference.yaml"
REMOTE_PROJECT_DIR="~/Desktop/smolcluster"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

export WANDB_API_KEY="$WANDB_API_TOKEN"

if [[ -z "$HF_TOKEN" ]]; then
    echo ""
    read -s -r -p "🔑 HF_TOKEN not set — enter HuggingFace token (required for meta-llama and other gated models): " HF_TOKEN
    echo ""
    if [[ -z "$HF_TOKEN" ]]; then
        echo "⚠️  Warning: no HF_TOKEN provided — loading gated models like Llama will fail"
    fi
    export HF_TOKEN
fi

SERVER=$(yq '.server' "$CONFIG_FILE")
NUM_WORKERS=$(yq '.num_workers' "$CONFIG_FILE")

REGULAR_WORKERS=()
while IFS= read -r worker; do
    [[ -n "$worker" ]] && REGULAR_WORKERS+=("$worker")
done < <(yq '.workers.regular[] | .hostname + ":" + (.rank | tostring)' "$CONFIG_FILE" 2>/dev/null)

TABLET_WORKERS=()
while IFS= read -r tablet; do
    [[ -n "$tablet" ]] && TABLET_WORKERS+=("$tablet")
done < <(yq '.workers.tablets[] | .hostname + ":" + (.rank | tostring)' "$CONFIG_FILE" 2>/dev/null)

ACTUAL_WORKER_COUNT=$((${#REGULAR_WORKERS[@]} + ${#TABLET_WORKERS[@]}))
if [[ "$ACTUAL_WORKER_COUNT" -ne "$NUM_WORKERS" ]]; then
    echo "❌ Error: num_workers=$NUM_WORKERS but found $ACTUAL_WORKER_COUNT workers in config"
    exit 1
fi

# Sync code to all remote SyncPS nodes (server + regular workers) before launch.
SSH_NODES=("$SERVER")
for worker_entry in "${REGULAR_WORKERS[@]}"; do
    hostname="${worker_entry%%:*}"
    found=false
    for node in "${SSH_NODES[@]}"; do
        if [[ "$node" == "$hostname" ]]; then
            found=true
            break
        fi
    done
    if [[ "$found" == "false" ]]; then
        SSH_NODES+=("$hostname")
    fi
done

echo "📦 Syncing code to remote SyncPS nodes"
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${SSH_NODES[@]}"; do
        echo "   Syncing to $node..."
        rsync -az --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude 'src/data' \
            "$PROJECT_DIR/" "$node:$REMOTE_PROJECT_DIR/" || {
            echo "❌ Error: Failed to sync code to $node"
            exit 1
        }
        echo "   ✅ Code synced to $node"
    done
    echo "✅ Code sync complete"
else
    for node in "${SSH_NODES[@]}"; do
        echo "   [DRY RUN] rsync -az --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude 'src/data' $PROJECT_DIR/ $node:$REMOTE_PROJECT_DIR/"
    done
    echo "✅ Code sync skipped (dry run)"
fi

echo "🚀 Launching SyncPS inference"
echo "📁 Project: $PROJECT_DIR"
echo "⚙️  Config: $CONFIG_FILE"
echo "🖥️  Server: $SERVER"

action_ssh() {
    local node=$1
    local command=$2
    local session_name=$3
    local log_file="\$HOME/${session_name}.log"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "   [DRY RUN] ssh $node \"cd $REMOTE_PROJECT_DIR && tmux new -d -s $session_name \\\"bash -c '$command 2>&1 | tee $log_file; exec bash'\\\"\""
        return 0
    fi

    ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && cd $REMOTE_PROJECT_DIR && tmux new -d -s $session_name \"bash -c '$command 2>&1 | tee $log_file; exec bash'\""
}

action_local() {
    local command=$1
    local session_name=$2
    local log_file="$HOME/${session_name}.log"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "   [DRY RUN] tmux new -d -s $session_name \"bash -c '$command 2>&1 | tee $log_file; exec bash'\""
        return 0
    fi

    tmux new -d -s "$session_name" "bash -c '$command 2>&1 | tee $log_file; exec bash'"
}

echo "🧹 Cleaning up old sessions"
if [[ "$DRY_RUN" != "true" ]]; then
    for node in "${SSH_NODES[@]}"; do
        ssh "$node" "tmux ls 2>/dev/null | cut -d: -f1 | grep '^syncps_inf_' | xargs -I{} tmux kill-session -t {} 2>/dev/null || true"
    done

    for worker_entry in "${REGULAR_WORKERS[@]}"; do
        hostname="${worker_entry%%:*}"
        rank="${worker_entry##*:}"
        ssh "$hostname" "tmux kill-session -t syncps_inf_worker$rank 2>/dev/null || true"
    done
    tmux ls 2>/dev/null | cut -d: -f1 | grep '^syncps_inf_' | xargs -I{} tmux kill-session -t {} 2>/dev/null || true
    for worker_entry in "${TABLET_WORKERS[@]}"; do
        rank="${worker_entry##*:}"
        tmux kill-session -t "syncps_inf_worker$rank" 2>/dev/null || true
    done
fi

echo "🖥️  Starting SyncPS inference server on $SERVER"
SERVER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' && cd $REMOTE_PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/DataParallelism/SynchronousPS/inference/server.py"
action_ssh "$SERVER" "$SERVER_CMD" "syncps_inf_server"

sleep 8

echo "👷 Starting SyncPS inference workers"
for worker_entry in "${REGULAR_WORKERS[@]}"; do
    hostname="${worker_entry%%:*}"
    rank="${worker_entry##*:}"
    WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' && cd $REMOTE_PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/DataParallelism/SynchronousPS/inference/worker.py $rank $hostname"
    action_ssh "$hostname" "$WORKER_CMD" "syncps_inf_worker$rank"
    echo "   ✅ Worker rank $rank on $hostname"
done

for worker_entry in "${TABLET_WORKERS[@]}"; do
    hostname="${worker_entry%%:*}"
    rank="${worker_entry##*:}"
    WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' && cd $PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/DataParallelism/SynchronousPS/inference/worker.py $rank $hostname"
    action_local "$WORKER_CMD" "syncps_inf_worker$rank"
    echo "   ✅ Local worker rank $rank for $hostname"
done

echo "⏳ Waiting for inference server to initialize"
sleep 15

echo "🌐 Launching API + frontend"
if [[ "$DRY_RUN" == "true" ]]; then
    bash "$SCRIPT_DIR/../launch_api.sh" --dry-run --backend data_parallelism --session-prefix syncps
else
    bash "$SCRIPT_DIR/../launch_api.sh" --backend data_parallelism --session-prefix syncps
fi

echo "🎉 SyncPS inference launch complete"
