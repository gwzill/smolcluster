#!/bin/bash

# vLLM Installation Script
# Installs vLLM from the official metal branch and runs a quick test

set -euo pipefail

echo "🚀 Installing vLLM..."
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
echo "✅ vLLM installed successfully"

# Get model name from inference model config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CONFIG_FILE="$PROJECT_DIR/src/smolcluster/configs/inference/model_config_inference.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
else
    MODEL_NAME=$(awk '
        /^dp:[[:space:]]*$/ {in_dp=1; next}
        /^[^[:space:]]/ {in_dp=0}
        in_dp && /^[[:space:]]+hf_model_name:[[:space:]]*/ {
            val=$0
            sub(/^[[:space:]]+hf_model_name:[[:space:]]*/, "", val)
            gsub(/"|\047/, "", val)
            print val
            exit
        }
    ' "$CONFIG_FILE")

    if [ -z "$MODEL_NAME" ]; then
        MODEL_NAME=$(awk '
            /^[[:space:]]*hf_model_name:[[:space:]]*/ {
                val=$0
                sub(/^[[:space:]]*hf_model_name:[[:space:]]*/, "", val)
                gsub(/"|\047/, "", val)
                print val
                exit
            }
        ' "$CONFIG_FILE")
    fi

    if [ -z "$MODEL_NAME" ]; then
        MODEL_NAME=$(awk '
            /^[[:space:]]*vllm_model:[[:space:]]*/ {
                val=$0
                sub(/^[[:space:]]*vllm_model:[[:space:]]*/, "", val)
                gsub(/"|\047/, "", val)
                print val
                exit
            }
        ' "$CONFIG_FILE")
    fi

    if [ -z "$MODEL_NAME" ]; then
        echo "⚠️  hf_model_name not found in $CONFIG_FILE, exiting."
        exit 0
    fi
fi

echo "📦 Using model: $MODEL_NAME"
echo ""
echo "🧪 Running quick test..."
echo "Starting vLLM server on port 8000 with model: $MODEL_NAME (background)"
echo ""

source "$HOME/.venv-vllm-metal/bin/activate" 
vllm serve "$MODEL_NAME" --port 8000 > /tmp/vllm_serve.log 2>&1 &
VLLM_PID=$!

cleanup() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" || true
    fi
}
trap cleanup EXIT

echo "⏳ Waiting for vLLM OpenAI API at http://localhost:8000/v1/models ..."
for i in $(seq 1 60); do
    if curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; then
        echo "✅ vLLM is ready"
        break
    fi

    if [ "$i" -eq 60 ]; then
        echo "❌ vLLM did not become ready in time"
        echo "--- vLLM logs ---"
        tail -n 80 /tmp/vllm_serve.log || true
        exit 1
    fi

    sleep 2
done

echo "🧪 Testing OpenAI-compatible /v1/completions endpoint..."
curl -sS http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL_NAME\", \"prompt\": \"Hello! Give me one short sentence.\", \"max_tokens\": 32, \"temperature\": 0.2}" \
  | python3 -m json.tool

echo "✅ Quick vLLM completion test finished"
