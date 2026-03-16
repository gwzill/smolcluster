#!/bin/bash

# Launch ClassicDP inference workers (rank 0 is leader), then chat API/frontend.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/cluster_config_classicdp.yaml"
REMOTE_PROJECT_DIR="~/Desktop/smolcluster"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

export WANDB_API_KEY="$WANDB_API_TOKEN"

WORKERS=()
while IFS= read -r worker; do
    [[ -n "$worker" ]] && WORKERS+=("$worker")
done < <(yq '.allToAllTopology.workers.regular[] | .hostname + ":" + (.rank | tostring) + ":" + .ip + ":" + (.port | tostring)' "$CONFIG_FILE")

if [[ ${#WORKERS[@]} -eq 0 ]]; then
    echo "❌ No workers found in allToAllTopology.workers.regular"
    exit 1
fi

RANK0_ENTRY=""
for entry in "${WORKERS[@]}"; do
    rank=$(echo "$entry" | cut -d: -f2)
    if [[ "$rank" == "0" ]]; then
        RANK0_ENTRY="$entry"
        break
    fi
done

if [[ -z "$RANK0_ENTRY" ]]; then
    echo "❌ Could not find rank 0 in ClassicDP topology"
    exit 1
fi

RANK0_HOST=$(echo "$RANK0_ENTRY" | cut -d: -f1)
RANK0_IP=$(echo "$RANK0_ENTRY" | cut -d: -f3)
RANK0_PORT=$(echo "$RANK0_ENTRY" | cut -d: -f4)

echo "🚀 Launching ClassicDP inference"
echo "📁 Project: $PROJECT_DIR"
echo "⚙️  Config: $CONFIG_FILE"
echo "🧠 Leader (rank 0): $RANK0_HOST @ $RANK0_IP:$RANK0_PORT"

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

echo "🧹 Cleaning up old sessions"
if [[ "$DRY_RUN" != "true" ]]; then
    for entry in "${WORKERS[@]}"; do
        hostname=$(echo "$entry" | cut -d: -f1)
        rank=$(echo "$entry" | cut -d: -f2)
        ssh "$hostname" "tmux kill-session -t classicdp_inf_worker$rank 2>/dev/null || true"
    done
fi

echo "👷 Starting ClassicDP inference workers"
for entry in "${WORKERS[@]}"; do
    hostname=$(echo "$entry" | cut -d: -f1)
    rank=$(echo "$entry" | cut -d: -f2)

    WORKER_CMD="export WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' && cd $REMOTE_PROJECT_DIR && .venv/bin/python src/smolcluster/algorithms/DataParallelism/ClassicDP/inference/worker.py $rank $hostname"
    action_ssh "$hostname" "$WORKER_CMD" "classicdp_inf_worker$rank"
    echo "   ✅ Worker rank $rank on $hostname"
done

echo "⏳ Waiting for rank 0 to initialize"
sleep 15

echo "🌐 Launching API + frontend (targeting rank 0 leader)"
if [[ "$DRY_RUN" == "true" ]]; then
    bash "$SCRIPT_DIR/../launch_api.sh" --dry-run --backend data_parallelism --session-prefix classicdp --server-host "$RANK0_IP" --server-port "$RANK0_PORT"
else
    bash "$SCRIPT_DIR/../launch_api.sh" --backend data_parallelism --session-prefix classicdp --server-host "$RANK0_IP" --server-port "$RANK0_PORT"
fi

echo "🎉 ClassicDP inference launch complete"
