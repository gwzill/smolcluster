#!/bin/bash
# Launch MLX-LM LoRA SFT on GSM8K.
#
# Usage:
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_sft.sh                    # full run
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_sft.sh --skip-data        # skip data prep (data/ already built)
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_sft.sh --iters 500        # override training iterations
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_sft.sh --dry-run          # show resolved command only
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_sft.sh --test             # run test-set eval after training

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(dirname "$SCRIPT_DIR")"  # …/sft/gsm8k

find_project_root() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/pyproject.toml" ]] || [[ -d "$dir/.git" ]]; then
            echo "$dir"; return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

PROJECT_DIR=$(find_project_root "$SCRIPT_DIR")
if [[ -z "$PROJECT_DIR" ]]; then
    echo "Error: cannot find project root (pyproject.toml / .git)"
    exit 1
fi

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set +u; source "$PROJECT_DIR/.env"; set -u
fi

MODEL_CONFIG="$PROJECT_DIR/src/smolcluster/configs/inference/model_config_inference.yaml"
LORA_CONFIG="$SFT_DIR/configs/lora_config.yaml"
PREPARE_SCRIPT="$SFT_DIR/data/prepare_data.py"
DATA_DIR="$SFT_DIR/data"

VENV_ACTIVATE="$PROJECT_DIR/.venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "Error: .venv not found. Run 'uv sync' inside $PROJECT_DIR first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
SKIP_DATA=false
DRY_RUN=false
RUN_TEST=false
FOREGROUND=false
CONFIG_OVERRIDE=""
THINK_MODE="--think"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-data)   SKIP_DATA=true;  shift ;;
        --dry-run)     DRY_RUN=true;    shift ;;
        --test)        RUN_TEST=true;   shift ;;
        --foreground)  FOREGROUND=true; shift ;;
        --config)      CONFIG_OVERRIDE="$2"; shift 2 ;;
        --think|--no-think) THINK_MODE="$1"; shift ;;
        --help|-h)
            echo "Usage: bash src/smolcluster/applications/sft/gsm8k/scripts/launch_sft.sh [--skip-data] [--dry-run] [--test] [--foreground] [--think|--no-think] [--config path] [extra mlx_lm lora flags...]"
            exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Override config if provided (used by sweep.py to inject per-trial hyperparams)
if [[ -n "$CONFIG_OVERRIDE" ]]; then
    LORA_CONFIG="$CONFIG_OVERRIDE"
fi

# Determine adapter path based on think mode
ADAPTER_SUBDIR="sft-think"
if [[ "$THINK_MODE" == "--no-think" ]]; then
    ADAPTER_SUBDIR="sft-no-think"
fi
ADAPTER_PATH="$SFT_DIR/checkpoints/$ADAPTER_SUBDIR/adapters"
FINAL_MODEL_PATH="$SFT_DIR/checkpoints/$ADAPTER_SUBDIR/final_model"
mkdir -p "$(dirname "$ADAPTER_PATH")"
mkdir -p "$FINAL_MODEL_PATH"

# ---------------------------------------------------------------------------
# Resolve model name from model config (requires yq)
# ---------------------------------------------------------------------------
if ! command -v yq >/dev/null 2>&1; then
    echo "Error: yq is required. Install with: brew install yq"
    exit 1
fi
HF_MODEL_NAME=$(yq '.dp.hf_model_name' "$MODEL_CONFIG")
echo "Model : $HF_MODEL_NAME"

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
set +u; source "$VENV_ACTIVATE"; set -u

cd "$PROJECT_DIR"
echo "Installing project dependencies..."
uv pip install -e . -q

if [[ "$SKIP_DATA" == "true" ]]; then
    echo "Skipping data preparation (--skip-data)"
    if [[ ! -f "$DATA_DIR/train.jsonl" ]]; then
        echo "Error: $DATA_DIR/train.jsonl not found. Run without --skip-data first."
        exit 1
    fi
elif [[ -f "$DATA_DIR/train.jsonl" && -f "$DATA_DIR/valid.jsonl" && -f "$DATA_DIR/test.jsonl" ]]; then
    # Check if data matches current think mode
    _format_file="$DATA_DIR/.data_format"
    if [[ -f "$_format_file" ]] && [[ "$(cat "$_format_file")" == "$THINK_MODE" ]]; then
        echo "Data already prepared ($THINK_MODE), skipping (train/valid/test.jsonl found)."
    else
        echo "Data format changed ($THINK_MODE) — regenerating..."
        python "$PREPARE_SCRIPT" --model-name "$HF_MODEL_NAME" --out-dir "$DATA_DIR" "$THINK_MODE"
        echo "$THINK_MODE" > "$_format_file"
    fi
else
    echo ""
    echo "Preparing GSM8K data ($THINK_MODE)..."
    python "$PREPARE_SCRIPT" --model-name "$HF_MODEL_NAME" --out-dir "$DATA_DIR" "$THINK_MODE"
    echo "$THINK_MODE" > "$DATA_DIR/.data_format"
fi

# ---------------------------------------------------------------------------
# Build mlx_lm lora command
# ---------------------------------------------------------------------------
LORA_ARGS=(
    "--train"
    "--model"   "$HF_MODEL_NAME"
    "--data"    "$DATA_DIR"
    "--config"  "$LORA_CONFIG"
    "--adapter-path" "$ADAPTER_PATH"
)

if [[ "$RUN_TEST" == "true" ]]; then
    LORA_ARGS+=("--test")
fi

# Append any extra flags (e.g. --iters 500) passed through to this script
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    LORA_ARGS+=("${EXTRA_ARGS[@]}")
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "Dry-run — resolved command:"
    printf '  python -m mlx_lm lora'
    printf ' %q' "${LORA_ARGS[@]}"
    echo ""
    exit 0
fi

# ---------------------------------------------------------------------------
# Run SFT
# ---------------------------------------------------------------------------
echo ""
echo "Launching SFT fine-tuning..."
echo "  Config  : $LORA_CONFIG"
echo "  Data    : $DATA_DIR"
echo "  Adapters: $ADAPTER_PATH"
echo ""

HF_ENV=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_ENV+=("HF_TOKEN=${HF_TOKEN}" "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
fi
HF_ENV+=("HF_HUB_ENABLE_HF_TRANSFER=1")

PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

if [[ "$FOREGROUND" == "true" ]]; then
    # Run synchronously — used by sweep.py so it can capture stdout/stderr.
    env "${HF_ENV[@]+"${HF_ENV[@]}"}" "$PYTHON_BIN" -m mlx_lm lora "${LORA_ARGS[@]}"
    echo ""
    echo "Training complete. Fusing adapters..."
    "$PYTHON_BIN" -m mlx_lm fuse --model "$HF_MODEL_NAME" --adapter-path "$ADAPTER_PATH" --save-path "$FINAL_MODEL_PATH"
    echo "Fused model saved to $FINAL_MODEL_PATH"
else
    SFT_CMD="$(printf '%q ' "${HF_ENV[@]+"${HF_ENV[@]}"}")$(printf '%q ' "$PYTHON_BIN") -m mlx_lm lora $(printf '%q ' "${LORA_ARGS[@]}")"
    SFT_CMD+="&& echo '' && echo 'Training complete. Fusing adapters...'"
    SFT_CMD+="&& $(printf '%q ' "$PYTHON_BIN") -m mlx_lm fuse --model $(printf '%q ' "$HF_MODEL_NAME") --adapter-path $(printf '%q ' "$ADAPTER_PATH") --save-path $(printf '%q ' "$FINAL_MODEL_PATH")"
    SFT_CMD+="&& echo '' && echo 'Fused model saved to: ${FINAL_MODEL_PATH}'"
    SFT_CMD+=" && echo '' && echo 'SFT complete. Adapters saved to: ${ADAPTER_PATH}'"
    SFT_CMD+=" || echo 'SFT FAILED.'"

    SESSION="sft_$(date +%Y%m%d_%H%M%S)"
    tmux new-session -d -s "$SESSION" "bash -c $(printf '%q' "cd ${PROJECT_DIR} && ${SFT_CMD}") ; echo '' ; echo 'Press any key to close.' ; read -n1"

    echo ""
    echo "SFT running in tmux session: $SESSION"
    echo "  attach with: tmux attach -t $SESSION"
    echo "  adapters:    $ADAPTER_PATH"
    echo ""
    echo "To fuse adapters after training:"
    echo "  bash src/smolcluster/applications/sft/gsm8k/scripts/fuse_adapters.sh"
fi
