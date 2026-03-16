#!/bin/bash

# Launch FastAPI backend and HTML frontend for inference (MP/DP)

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/inference/cluster_config_inference.yaml"
API_DIR="$PROJECT_DIR/src/smolcluster/chat/backend"
FRONTEND_DIR="$PROJECT_DIR/src/smolcluster/chat/frontend"

BACKEND="model_parallelism"
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

# Read ports from config
API_PORT=$(yq '.web_interface.api_port' "$CONFIG_FILE")
FRONTEND_PORT=$(yq '.web_interface.frontend_port' "$CONFIG_FILE")

# Update index.html with correct API_URL before launching
HTML_FILE="$FRONTEND_DIR/index.html"
echo "📝 Updating API URL in index.html to use port $API_PORT..."
# Use sed to replace the default API_URL with the correct one
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS sed syntax
    sed -i '' "s|let API_URL = 'http://localhost:[0-9]*';|let API_URL = 'http://localhost:$API_PORT';|g" "$HTML_FILE"
else
    # Linux sed syntax
    sed -i "s|let API_URL = 'http://localhost:[0-9]*';|let API_URL = 'http://localhost:$API_PORT';|g" "$HTML_FILE"
fi
echo "✅ Updated API_URL to http://localhost:$API_PORT"

DRY_RUN=${DRY_RUN:-false}
if [[ "$DRY_RUN" == "true" ]]; then
    echo "🏃 Dry run mode - will show commands without executing"
fi

echo ""
echo "🌐 Launching API and Frontend for backend: $BACKEND"
echo "📁 Project dir: $PROJECT_DIR"

# Kill any existing sessions
echo ""
echo "🧹 Cleaning up existing API/Frontend sessions..."
if [[ "$DRY_RUN" != "true" ]]; then
    tmux kill-session -t "${SESSION_PREFIX}_api" 2>/dev/null || true
    tmux kill-session -t "${SESSION_PREFIX}_frontend" 2>/dev/null || true
    
   
    echo "✅ Cleanup complete"
else
    echo "✅ Cleanup skipped (dry run)"
fi

# Launch FastAPI backend
echo ""
echo "🚀 Launching FastAPI backend on port $API_PORT..."
API_LOG="$HOME/${SESSION_PREFIX}_api.log"

API_ENV="INFERENCE_BACKEND=$BACKEND"
if [[ -n "$SERVER_HOST_OVERRIDE" ]]; then
    API_ENV="$API_ENV INFERENCE_SERVER_HOST=$SERVER_HOST_OVERRIDE"
fi
if [[ -n "$SERVER_PORT_OVERRIDE" ]]; then
    API_ENV="$API_ENV INFERENCE_SERVER_PORT=$SERVER_PORT_OVERRIDE"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "   [DRY RUN] Would execute: tmux new -d -s ${SESSION_PREFIX}_api \"bash -c 'cd $API_DIR && $API_ENV uv run api.py 2>&1 | tee $API_LOG; exec bash'\""
else
    tmux new -d -s "${SESSION_PREFIX}_api" "bash -c 'cd $API_DIR && $API_ENV uv run api.py 2>&1 | tee $API_LOG; exec bash'"
    sleep 2
    
    # Verify API is running
    if tmux has-session -t "${SESSION_PREFIX}_api" 2>/dev/null; then
        echo "✅ FastAPI backend started (session: ${SESSION_PREFIX}_api, logs: $API_LOG)"
        
        # Wait for API to be ready with retry logic
        echo "⏳ Waiting for API to be ready..."
        MAX_RETRIES=30
        RETRY_DELAY=2
        for i in $(seq 1 $MAX_RETRIES); do
            if curl -s http://localhost:$API_PORT/health >/dev/null 2>&1; then
                echo "✅ API is ready and responding on http://localhost:$API_PORT"
                break
            else
                if [[ $i -eq $MAX_RETRIES ]]; then
                    echo "⚠️  Warning: API did not respond after $MAX_RETRIES attempts"
                    echo "   Check logs: tail -f $API_LOG"
                else
                    echo "   Attempt $i/$MAX_RETRIES: API not ready yet, retrying in ${RETRY_DELAY}s..."
                    sleep $RETRY_DELAY
                fi
            fi
        done
    else
        echo "❌ Failed to start FastAPI backend. Check logs: cat $API_LOG"
        exit 1
    fi
fi

# Launch HTML frontend
echo ""
echo "🌐 Launching HTML frontend on port $FRONTEND_PORT..."
FRONTEND_LOG="$HOME/${SESSION_PREFIX}_frontend.log"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "   [DRY RUN] Would execute: tmux new -d -s ${SESSION_PREFIX}_frontend \"bash -c 'cd $FRONTEND_DIR && python3 -m http.server $FRONTEND_PORT 2>&1 | tee $FRONTEND_LOG; exec bash'\""
else
    tmux new -d -s "${SESSION_PREFIX}_frontend" "bash -c 'cd $FRONTEND_DIR && python3 -m http.server $FRONTEND_PORT 2>&1 | tee $FRONTEND_LOG; exec bash'"
    sleep 2
    
    # Verify frontend is running
    if tmux has-session -t "${SESSION_PREFIX}_frontend" 2>/dev/null; then
        echo "✅ HTML frontend started (session: ${SESSION_PREFIX}_frontend, logs: $FRONTEND_LOG)"
        
        # Wait for frontend to be ready
        echo "⏳ Waiting for frontend to be ready..."
        sleep 2
        if curl -s http://localhost:$FRONTEND_PORT >/dev/null 2>&1; then
            echo "✅ Frontend is ready on http://localhost:$FRONTEND_PORT"
        else
            echo "⚠️  Warning: Frontend may not be responding yet"
        fi
    else
        echo "❌ Failed to start HTML frontend. Check logs: cat $FRONTEND_LOG"
        exit 1
    fi
fi

echo ""
echo "🎉 API and Frontend launch complete!"
echo ""
echo "📊 Access points:"
echo "   API:      http://localhost:$API_PORT"
echo "   Frontend: http://localhost:$FRONTEND_PORT"
echo "   Health:   http://localhost:$API_PORT/health"
echo ""
echo "🔍 Check status:"
echo "   tmux ls"
echo "   tmux attach -t ${SESSION_PREFIX}_api"
echo "   tmux attach -t ${SESSION_PREFIX}_frontend"
echo ""
echo "📝 View logs:"
echo "   tail -f $API_LOG"
echo "   tail -f $FRONTEND_LOG"
