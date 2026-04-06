#!/bin/bash

# Launch FastAPI backend and HTML frontend for inference (MP/DP)
set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/inference/cluster_config_inference.yaml"
API_DIR="$PROJECT_DIR/src/smolcluster/applications/chat/backend"
FRONTEND_DIR="$PROJECT_DIR/src/smolcluster/applications/chat/frontend"

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

BACKEND="model_parallelism"
INFERENCE_ALGORITHM=""
LAUNCH_INFERENCE=true
REDIS_URL="redis://0.0.0.0:6379/0"
SESSION_PREFIX="mp"
SERVER_HOST_OVERRIDE=""
SERVER_PORT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --algorithm)
            INFERENCE_ALGORITHM="$2"
            shift 2
            ;;
        --no-inference)
            LAUNCH_INFERENCE=false
            shift
            ;;
        --redis-url)
            REDIS_URL="$2"
            shift 2
            ;;
        --session-prefix)
            SESSION_PREFIX="$2"
            shift 2
            ;;
        --server-host)
            SERVER_HOST_OVERRIDE="$2"
            shift 2
            ;;
        --server-port)
            SERVER_PORT_OVERRIDE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [[ -z "$INFERENCE_ALGORITHM" ]]; then
    case "$BACKEND" in
        classicdp)
            INFERENCE_ALGORITHM="classicdp"
            ;;
        data_parallelism)
            INFERENCE_ALGORITHM="syncps"
            ;;
        *)
            INFERENCE_ALGORITHM="mp"
            ;;
    esac
fi

# Read ports from config
API_PORT=$(yq '.web_interface.api_port' "$CONFIG_FILE")
FRONTEND_PORT=$(yq '.web_interface.frontend_port' "$CONFIG_FILE")

# Update index.html with correct API_URL before launching
# Read the server IP from the cluster config; fall back to the local machine's LAN IP
# because launch_api.sh always runs on the controller (dashboard) machine.
HTML_FILE="$FRONTEND_DIR/index.html"
_SERVER_KEY=$(yq '.server' "$CONFIG_FILE")
API_HOST=$(yq ".host_ip.${_SERVER_KEY}" "$CONFIG_FILE" 2>/dev/null)
if [[ -z "$API_HOST" || "$API_HOST" == "null" ]]; then
    # Server key not in host_ip — try to find it under the SSH alias.
    # The mDNS hostname (e.g. "macmini1") may differ from the SSH alias ("mini1").
    # Walk all host_ip entries and pick any that appears to be local by comparing against
    # the machine's own interface IPs. If none match, use the first non-empty entry or
    # the machine's primary LAN IP as a last resort.
    _LOCAL_LAN_IP=""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        _LOCAL_LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
    else
        _LOCAL_LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    # Try each host_ip value to see if it matches our local IPs
    _FOUND_IP=""
    while IFS='=' read -r _key _val; do
        _val="${_val//[[:space:]]/}"
        [[ -z "$_val" || "$_val" == "null" ]] && continue
        if [[ -n "$_LOCAL_LAN_IP" && "$_val" == "$_LOCAL_LAN_IP" ]]; then
            _FOUND_IP="$_val"
            break
        fi
    done < <(yq '.host_ip | to_entries[] | .key + "=" + .value' "$CONFIG_FILE" 2>/dev/null || true)
    if [[ -n "$_FOUND_IP" ]]; then
        API_HOST="$_FOUND_IP"
        echo "[INFO] Resolved API host via local IP match: $API_HOST"
    elif [[ -n "$_LOCAL_LAN_IP" ]]; then
        API_HOST="$_LOCAL_LAN_IP"
        echo "[INFO] Server key '${_SERVER_KEY}' not in host_ip; using local LAN IP: $API_HOST"
    else
        echo "[ERROR] Could not resolve IP for server '${_SERVER_KEY}' from $CONFIG_FILE"
        echo "   Add '${_SERVER_KEY}' under 'host_ip:' in the config and retry."
        exit 1
    fi
fi
echo "📝 Updating API URL in index.html → http://$API_HOST:$API_PORT ..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s|let API_URL = 'http://[^']*';|let API_URL = 'http://$API_HOST:$API_PORT';|g" "$HTML_FILE"
else
    sed -i "s|let API_URL = 'http://[^']*';|let API_URL = 'http://$API_HOST:$API_PORT';|g" "$HTML_FILE"
fi
echo "[OK] Updated API_URL to http://$API_HOST:$API_PORT"

DRY_RUN=${DRY_RUN:-false}
if [[ "$DRY_RUN" == "true" ]]; then
    echo "🏃 Dry run mode - willow commands without executing"
fi

echo ""
echo "🌐 Launching API and Frontend for backend: $BACKEND"
echo "🧠 Inference algorithm: $INFERENCE_ALGORITHM"
echo "🧠 Memory backend: redis-vector ($REDIS_URL)"
echo "📁 Project dir: $PROJECT_DIR"

# Validate Redis reachability (Redis daemon started above via ensure_redis_running).
if command -v redis-cli >/dev/null 2>&1; then
    if redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1; then
        echo "[OK] Redis is reachable at $REDIS_URL"
    else
        echo "[WARN]  Redis is not reachable at $REDIS_URL"
        echo "   Memory features will be disabled until Redis is available."
    fi
else
    echo "ℹ️  redis-cli not installed; skipping Redis reachability check."
fi

if [[ "$LAUNCH_INFERENCE" == "true" ]]; then
    echo ""
    echo "[START] Launching inference stack first (model loading)"
    if [[ "$DRY_RUN" == "true" ]]; then
        bash "$PROJECT_DIR/scripts/inference/launch_inference.sh" --algorithm "$INFERENCE_ALGORITHM" --dry-run
    else
        bash "$PROJECT_DIR/scripts/inference/launch_inference.sh" --algorithm "$INFERENCE_ALGORITHM"
    fi
else
    echo ""
    echo "⏭️  Skipping inference launch (--no-inference)"
fi

echo ""
echo "🔧 Checking local requirements..."
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[OK] Local requirement checks skipped (dry run)"
else
    if ! command -v tmux >/dev/null 2>&1; then
        echo "[ERROR] Error: tmux is not installed locally. Install with: bash $PROJECT_DIR/scripts/installations/installation.sh"
        exit 1
    fi

    if ! command -v uv >/dev/null 2>&1; then
        echo "[ERROR] Error: uv is not installed locally. Install with: bash $PROJECT_DIR/scripts/installations/installation.sh"
        exit 1
    fi

    echo "[OK] LOCAL: tmux OK, uv OK"
fi

echo ""
echo "📦 Checking local venv..."
if [[ "$DRY_RUN" == "true" ]]; then
    echo "   [DRY RUN] Would execute: cd $PROJECT_DIR && if [[ ! -f .venv/bin/python ]]; then uv venv --python 3.10 .venv && uv pip install -e .; else uv sync; fi"
else
    if [[ ! -f "$PROJECT_DIR/.venv/bin/python" ]]; then
        echo "[WARN]  Venv not found locally. Creating with Python 3.10..."
        (
            cd "$PROJECT_DIR"
            uv venv --python 3.10 .venv
            uv pip install -e .
        )
    else
        echo "[OK] Venv exists locally. Running uv sync..."
        (
            cd "$PROJECT_DIR"
            uv sync
        )
    fi

    echo "🧪 Verifying smolcluster import locally..."
    if ! (cd "$PROJECT_DIR" && PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH}" .venv/bin/python -c "import smolcluster"); then
        echo "[WARN]  smolcluster import failed after sync. Reinstalling editable package..."
        (
            cd "$PROJECT_DIR"
            uv pip install -e .
        )
        if ! (cd "$PROJECT_DIR" && PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH}" .venv/bin/python -c "import smolcluster"); then
            echo "[ERROR] Error: smolcluster is still not importable in local .venv"
            exit 1
        fi
    fi

    echo "[OK] LOCAL: venv OK, smolcluster import OK"
fi

# Kill any existing sessions
echo ""
echo "🧹 Cleaning up existing API/Frontend sessions..."
if [[ "$DRY_RUN" != "true" ]]; then
    tmux kill-session -t "${SESSION_PREFIX}_api" 2>/dev/null || true
    tmux kill-session -t "${SESSION_PREFIX}_frontend" 2>/dev/null || true

    for PORT in "$API_PORT" "$FRONTEND_PORT"; do
        PORT_PIDS=$(lsof -ti tcp:"$PORT" 2>/dev/null | sort -u)
        if [[ -n "$PORT_PIDS" ]]; then
            echo "[WARN]  Found existing process(es) on port $PORT (PID(s): $PORT_PIDS). Terminating..."
            echo "$PORT_PIDS" | xargs kill 2>/dev/null || true
            sleep 1

            PORT_PIDS_REMAINING=$(lsof -ti tcp:"$PORT" 2>/dev/null | sort -u)
            if [[ -n "$PORT_PIDS_REMAINING" ]]; then
                echo "[WARN]  Process(es) still active on port $PORT after TERM. Sending KILL..."
                echo "$PORT_PIDS_REMAINING" | xargs kill -9 2>/dev/null || true
                sleep 1
            fi
        fi
    done
    
   
    echo "[OK] Cleanup complete"
else
    echo "[OK] Cleanup skipped (dry run)"
fi

# Launch FastAPI backend
echo ""
echo "[START] Launching FastAPI backend on port $API_PORT..."
API_LOG="$HOME/${SESSION_PREFIX}_api.log"

API_ENV="INFERENCE_BACKEND=$BACKEND"
API_ENV="$API_ENV INFERENCE_ALGORITHM=$INFERENCE_ALGORITHM"
API_ENV="$API_ENV REDIS_URL=$REDIS_URL"
if [[ -n "$SERVER_HOST_OVERRIDE" ]]; then
    API_ENV="$API_ENV INFERENCE_SERVER_HOST=$SERVER_HOST_OVERRIDE"
fi
if [[ -n "$SERVER_PORT_OVERRIDE" ]]; then
    API_ENV="$API_ENV INFERENCE_SERVER_PORT=$SERVER_PORT_OVERRIDE"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "   [DRY RUN] Would execute: tmux new -d -s ${SESSION_PREFIX}_api \"bash -c 'cd $PROJECT_DIR && $API_ENV PYTHONPATH=$PROJECT_DIR/src:\$PYTHONPATH .venv/bin/python -m smolcluster.applications.chat.backend.api 2>&1 | tee $API_LOG'\""
else
    tmux new -d -s "${SESSION_PREFIX}_api" "bash -c 'cd $PROJECT_DIR && $API_ENV PYTHONPATH=$PROJECT_DIR/src:\$PYTHONPATH .venv/bin/python -m smolcluster.applications.chat.backend.api 2>&1 | tee $API_LOG'"
    sleep 2
    
    # Verify API is running
    if tmux has-session -t "${SESSION_PREFIX}_api" 2>/dev/null; then
        echo "[OK] FastAPI backend started (session: ${SESSION_PREFIX}_api, logs: $API_LOG)"
        
        # Wait for API to be ready with retry logic
        echo "[INFO] Waiting for API to be ready..."
        MAX_RETRIES=30
        RETRY_DELAY=2
        API_READY=false
        for i in $(seq 1 $MAX_RETRIES); do
            HEALTH_RESPONSE=$(curl -fsS http://127.0.0.1:$API_PORT/health 2>/dev/null || true)
            if echo "$HEALTH_RESPONSE" | grep -Eq '"healthy"[[:space:]]*:[[:space:]]*true'; then
                echo "[OK] API is ready and responding on http://127.0.0.1:$API_PORT"
                API_READY=true
                break
            else
                if [[ $i -eq $MAX_RETRIES ]]; then
                    echo "[ERROR] API did not become healthy after $MAX_RETRIES attempts"
                    echo "   Check logs: tail -f $API_LOG"
                    if [[ -n "$HEALTH_RESPONSE" ]]; then
                        echo "   Last /health response: $HEALTH_RESPONSE"
                    fi
                else
                    echo "   Attempt $i/$MAX_RETRIES: API not ready yet, retrying in ${RETRY_DELAY}s..."
                    sleep $RETRY_DELAY
                fi
            fi
        done
        if [[ "$API_READY" != "true" ]]; then
            exit 1
        fi
    else
        echo "[ERROR] Failed to start FastAPI backend. Check logs: cat $API_LOG"
        exit 1
    fi
fi

# Launch HTML frontend
echo ""
echo "🌐 Launching HTML frontend on port $FRONTEND_PORT..."
FRONTEND_LOG="$HOME/${SESSION_PREFIX}_frontend.log"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "   [DRY RUN] Would execute: tmux new -d -s ${SESSION_PREFIX}_frontend \"bash -c 'cd $FRONTEND_DIR && python3 -m http.server $FRONTEND_PORT 2>&1 | tee $FRONTEND_LOG'\""
else
    tmux new -d -s "${SESSION_PREFIX}_frontend" "bash -c 'cd $FRONTEND_DIR && python3 -m http.server $FRONTEND_PORT 2>&1 | tee $FRONTEND_LOG'"
    sleep 2
    
    # Verify frontend is running
    if tmux has-session -t "${SESSION_PREFIX}_frontend" 2>/dev/null; then
        echo "[OK] HTML frontend started (session: ${SESSION_PREFIX}_frontend, logs: $FRONTEND_LOG)"
        
        # Wait for frontend to be ready
        echo "[INFO] Waiting for frontend to be ready..."
        FRONTEND_READY=false
        for i in $(seq 1 10); do
            if curl -fsS http://127.0.0.1:$FRONTEND_PORT >/dev/null 2>&1; then
                FRONTEND_READY=true
                break
            fi
            sleep 1
        done
        if [[ "$FRONTEND_READY" == "true" ]]; then
            echo "[OK] Frontend is ready on http://127.0.0.1:$FRONTEND_PORT"
        else
            echo "[ERROR] Frontend did not become ready on http://127.0.0.1:$FRONTEND_PORT"
            echo "   Check logs: tail -f $FRONTEND_LOG"
            exit 1
        fi
    else
        echo "[ERROR] Failed to start HTML frontend. Check logs: cat $FRONTEND_LOG"
        exit 1
    fi
fi

echo ""
echo "🎉 API and Frontend launch complete!"
echo ""
echo "📊 Access points:"
echo "   API:      http://$API_HOST:$API_PORT"
echo "   Frontend: http://127.0.0.1:$FRONTEND_PORT"
echo "   Health:   http://$API_HOST:$API_PORT/health"
echo ""
echo "🔍 Check status:"
echo "   tmux ls"
echo "   tmux attach -t ${SESSION_PREFIX}_api"
echo "   tmux attach -t ${SESSION_PREFIX}_frontend"
echo ""
echo "📝 View logs:"
echo "   tail -f $API_LOG"
echo "   tail -f $FRONTEND_LOG"
