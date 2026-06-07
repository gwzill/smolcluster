#!/bin/bash
# Launch lm-evaluation-harness for SFT adapters (fuse-first by default).
#
# Default flow:
#   1) Fuse local SFT adapters into a standalone model.
#   2) Run lm_eval on single-turn instruction-following benchmarks.
#
# Pass --eval-model to skip adapters entirely and evaluate any HF model or
# local path directly.
#
# Usage:
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh --think    # use <answer> tag extractor + chat template for gsm8k
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh --no-think # use last-number extractor + chat template for gsm8k
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh --tasks gsm8k_cot_zeroshot,ifeval
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh --no-fuse
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh --dry-run
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh --eval-model meta-llama/Llama-3.2-1B-Instruct
#   bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh --eval-model /path/to/local/model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(dirname "$SCRIPT_DIR")"
FUSE_SCRIPT="$SCRIPT_DIR/fuse_adapters.sh"

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
    echo "Error: cannot find project root (pyproject.toml / .git)"
    exit 1
fi

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set +u
    source "$PROJECT_DIR/.env"
    set -u
fi

MODEL_CONFIG="$PROJECT_DIR/src/smolcluster/configs/inference/model_config_inference.yaml"
VENV_ACTIVATE="$PROJECT_DIR/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "Error: .venv not found. Run 'uv sync' inside $PROJECT_DIR first."
    exit 1
fi

if ! command -v yq >/dev/null 2>&1; then
    echo "Error: yq is required. Install with: brew install yq"
    exit 1
fi

if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "Error: model config not found: $MODEL_CONFIG"
    exit 1
fi

HF_MODEL_NAME=$(yq '.dp.hf_model_name' "$MODEL_CONFIG")
ADAPTER_PATH=""          # derived from THINK_MODE after arg parsing if not set explicitly
FUSED_MODEL_PATH=""      # derived from THINK_MODE after arg parsing if not set explicitly
TASKS="gsm8k_cot_zeroshot,ifeval,mmlu,arc_challenge,hellaswag"
NUM_FEWSHOT=0

BATCH_SIZE="32"
DEVICE="mps"  # change to "cuda" if using an NVIDIA GPU; "cpu" also works but is slow
LIMIT=""
OUTPUT_PATH=""
DRY_RUN=false
FUSE_FIRST=true
DIRECT_EVAL_MODEL=""   # set via --eval-model; bypasses all adapter logic
THINK_MODE=""          # "" = stock task, "think" or "nothink" = custom task + apply_chat_template
FOREGROUND=false
EXTRA_ARGS=()
HAS_EXTRA_ARGS=false

CUSTOM_TASKS_DIR="$SFT_DIR/configs/lm_eval_tasks"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --adapter-path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --fused-model-path)
            FUSED_MODEL_PATH="$2"
            shift 2
            ;;
        --model)
            HF_MODEL_NAME="$2"
            shift 2
            ;;
        --num-fewshot)
            NUM_FEWSHOT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --eval-model)
            DIRECT_EVAL_MODEL="$2"
            shift 2
            ;;
        --base)
            THINK_MODE="base"
            shift
            ;;
        --think)
            THINK_MODE="think"
            shift
            ;;
        --no-think)
            THINK_MODE="nothink"
            shift
            ;;
        --foreground)
            FOREGROUND=true
            shift
            ;;
        --no-fuse)
            FUSE_FIRST=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: bash src/smolcluster/applications/sft/gsm8k/scripts/launch_lm_eval_sft.sh [options]"
            echo ""
            echo "Options:"
            echo "  --tasks <csv>            lm_eval tasks (default: gsm8k_cot_zeroshot,ifeval,mmlu,arc_challenge,hellaswag)"
            echo "  --adapter-path <path>    LoRA adapter dir (default: local SFT checkpoints)"
            echo "  --fused-model-path <p>   Fused model output/input path"
            echo "  --model <hf_model>       Base HF model name (from model config by default)"
            echo "  --num-fewshot <n>        Few-shot examples per task (default: 0)"
            echo "  --batch-size <value>     lm_eval batch size (default: auto)"
            echo "  --device <dev>           lm_eval device (default: mps on macOS)"
            echo "  --limit <n|f>            lm_eval --limit for quick runs"
            echo "  --output-path <path>     lm_eval output path"
            echo "  --eval-model <hf|path>   Evaluate any HF model or local path directly (skips all adapter logic)"
            echo "  --think                  Use <answer> tag extractor + chat template for gsm8k (sft-think model)"
            echo "  --no-think               Use last-number extractor + chat template for gsm8k (sft-nothink model)"
            echo "  --no-fuse                Evaluate with base+peft adapter (skip fuse step)"
            echo "  --dry-run                Print commands only"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            HAS_EXTRA_ARGS=true
            shift
            ;;
    esac
done

# Derive variant-specific paths from THINK_MODE if not explicitly set via flags.
if [[ -z "$ADAPTER_PATH" ]]; then
    case "$THINK_MODE" in
        think)   ADAPTER_PATH="$SFT_DIR/checkpoints/sft-think/adapters" ;;
        nothink) ADAPTER_PATH="$SFT_DIR/checkpoints/sft-no-think/adapters" ;;
        *)       ADAPTER_PATH="$SFT_DIR/checkpoints" ;;
    esac
fi
if [[ -z "$FUSED_MODEL_PATH" ]]; then
    case "$THINK_MODE" in
        think)   FUSED_MODEL_PATH="$SFT_DIR/checkpoints/sft-think/final_model" ;;
        nothink) FUSED_MODEL_PATH="$SFT_DIR/checkpoints/sft-no-think/final_model" ;;
        *)       FUSED_MODEL_PATH="$SFT_DIR/checkpoints/sft_final_fused" ;;
    esac
fi

if [[ "$ADAPTER_PATH" != /* ]]; then
    ADAPTER_PATH="$PROJECT_DIR/$ADAPTER_PATH"
fi
if [[ "$FUSED_MODEL_PATH" != /* ]]; then
    FUSED_MODEL_PATH="$PROJECT_DIR/$FUSED_MODEL_PATH"
fi

EVAL_RESULTS_ROOT="$PROJECT_DIR/src/smolcluster/applications/sft/gsm8k/eval-results"

if [[ -z "$OUTPUT_PATH" ]]; then
    ts=$(date +"%Y%m%d_%H%M%S")
    case "$THINK_MODE" in
        base)    VARIANT_SUBDIR="base" ;;
        think)   VARIANT_SUBDIR="sft-think" ;;
        nothink) VARIANT_SUBDIR="sft-nothink" ;;
        *)       VARIANT_SUBDIR="misc" ;;
    esac
    OUTPUT_PATH="$EVAL_RESULTS_ROOT/${VARIANT_SUBDIR}/lm_eval_${ts}.json"
elif [[ "$OUTPUT_PATH" != /* ]]; then
    OUTPUT_PATH="$PROJECT_DIR/$OUTPUT_PATH"
fi
mkdir -p "$(dirname "$OUTPUT_PATH")"

set +u
source "$VENV_ACTIVATE"
set -u

cd "$PROJECT_DIR"

if ! python -c "import lm_eval" >/dev/null 2>&1; then
    echo "lm_eval not found in .venv; installing lm-eval..."
    uv pip install lm-eval
    uv pip install accelerate  # ensure accelerate is installed for GPU support
fi

# Install benchmark dependencies task-by-task from TASKS.
IFS=',' read -r -a SELECTED_TASKS <<< "$TASKS"
for task in "${SELECTED_TASKS[@]}"; do
    task_trimmed="$(echo "$task" | xargs)"
    if [[ -z "$task_trimmed" ]]; then
        continue
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Dry-run benchmark dependency install: uv pip install \"lm-eval[$task_trimmed]\""
    else
        echo "Ensuring lm-eval benchmark dependency: $task_trimmed"
        if ! uv pip install "lm-eval[$task_trimmed]"; then
            echo "Warning: lm-eval extra '$task_trimmed' may not exist; continuing."
        fi
    fi
done

EVAL_MODEL_PATH=""
MODEL_ARGS=""

if [[ -n "$DIRECT_EVAL_MODEL" ]]; then
    echo "SFT adapters enabled: NO (direct model mode)"
    echo "  model: $DIRECT_EVAL_MODEL"
    EVAL_MODEL_PATH="$DIRECT_EVAL_MODEL"
    MODEL_ARGS="pretrained=${EVAL_MODEL_PATH},trust_remote_code=True"
elif [[ "$FUSE_FIRST" == "true" ]]; then
    if [[ ! -f "$FUSE_SCRIPT" ]]; then
        echo "Error: fuse script not found: $FUSE_SCRIPT"
        exit 1
    fi

    echo "SFT adapters enabled: YES (fuse-first mode)"
    echo "  adapter path: $ADAPTER_PATH"
    echo "  fused output: $FUSED_MODEL_PATH"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo ""
        echo "Dry-run fuse command:"
        printf '  bash %q --model %q --adapter-path %q --save-path %q\n' "$FUSE_SCRIPT" "$HF_MODEL_NAME" "$ADAPTER_PATH" "$FUSED_MODEL_PATH"
    else
        bash "$FUSE_SCRIPT" --model "$HF_MODEL_NAME" --adapter-path "$ADAPTER_PATH" --save-path "$FUSED_MODEL_PATH"
    fi

    EVAL_MODEL_PATH="$FUSED_MODEL_PATH"
    MODEL_ARGS="pretrained=${EVAL_MODEL_PATH},trust_remote_code=True"
else
    echo "SFT adapters enabled: YES (direct PEFT mode, no fuse)"
    echo "  base model: $HF_MODEL_NAME"
    echo "  adapter path: $ADAPTER_PATH"

    EVAL_MODEL_PATH="$HF_MODEL_NAME"
    MODEL_ARGS="pretrained=${EVAL_MODEL_PATH},peft=${ADAPTER_PATH},trust_remote_code=True"
fi  # end adapter/direct-model branch

# ---------------------------------------------------------------------------
# Think/nothink mode: swap gsm8k task name and enable chat template
# ---------------------------------------------------------------------------
if [[ -n "$THINK_MODE" ]]; then
    if [[ "$THINK_MODE" == "base" ]]; then
        CUSTOM_TASK_NAME="gsm8k_cot_zeroshot_base"
    else
        CUSTOM_TASK_NAME="gsm8k_cot_zeroshot_sft_${THINK_MODE}"
    fi
    TASKS="${TASKS//gsm8k_cot_zeroshot/$CUSTOM_TASK_NAME}"
    echo "GSM8K task overridden → $CUSTOM_TASK_NAME"
fi

BASE_LM_EVAL_ARGS=(
    "${PROJECT_DIR}/.venv/bin/python" -m lm_eval
    --model hf
    --model_args "$MODEL_ARGS"
    --num_fewshot "$NUM_FEWSHOT"
    --batch_size "$BATCH_SIZE"
    --device "$DEVICE"
)

BASE_LM_EVAL_ARGS+=(--apply_chat_template)
BASE_LM_EVAL_ARGS+=(--log_samples)

if [[ -n "$THINK_MODE" ]]; then
    BASE_LM_EVAL_ARGS+=(--include_path "$CUSTOM_TASKS_DIR")
fi
if [[ "$THINK_MODE" == "think" ]]; then
    BASE_LM_EVAL_ARGS+=(--system_instruction "You are an Assistant expert at solving math problems. The assistant first thinks about the reasoning process to reach the correct answer within '<think>...</think>' tags and then provides the user with the answer. The FINAL answer must STRICTLY be written as <answer>answer_here</answer> and the thinking process strictly within '<think>...</think>' tags.")
elif [[ "$THINK_MODE" == "nothink" ]]; then
    BASE_LM_EVAL_ARGS+=(--system_instruction "You are an Assistant expert at solving math problems. The assistant first thinks about the reasoning process to reach the correct answer. The FINAL answer must be written as a numeric value at the end of the response, and the reasoning process should be included as a chain-of-thought before the final answer.")
fi

if [[ -n "$LIMIT" ]]; then
    BASE_LM_EVAL_ARGS+=(--limit "$LIMIT")
fi
if [[ "$HAS_EXTRA_ARGS" == "true" ]]; then
    BASE_LM_EVAL_ARGS+=("${EXTRA_ARGS[@]}")
fi

OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
RUN_TS="$(basename "$OUTPUT_PATH" .json)"

echo ""
echo "Launching lm_eval (one invocation per task)..."
echo "  eval model source: $EVAL_MODEL_PATH"
echo "  tasks: $TASKS"
echo "  output dir: $OUTPUT_DIR"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "Dry-run — per-task commands:"
    IFS=',' read -r -a TASK_LIST <<< "$TASKS"
    for task in "${TASK_LIST[@]}"; do
        task="$(echo "$task" | xargs)"
        task_output="${OUTPUT_DIR}/${RUN_TS}_${task}.json"
        printf '  %q' "${BASE_LM_EVAL_ARGS[@]}"
        printf ' --tasks %q --output_path %q\n' "$task" "$task_output"
    done
    exit 0
fi

# Build the per-task loop script to run inside tmux
INNER_SCRIPT=""
IFS=',' read -r -a TASK_LIST <<< "$TASKS"
for task in "${TASK_LIST[@]}"; do
    task="$(echo "$task" | xargs)"
    task_output="${OUTPUT_DIR}/${RUN_TS}_${task}.json"
    CMD_STR="$(printf '%q ' "${BASE_LM_EVAL_ARGS[@]}")"
    CMD_STR+="--tasks $(printf '%q' "$task") --output_path $(printf '%q' "$task_output")"
    INNER_SCRIPT+="echo ''; echo '==> Running task: ${task}'; "
    INNER_SCRIPT+="${CMD_STR} && echo '==> Done: ${task} -> ${task_output}' || echo '==> FAILED: ${task}'; "
done
INNER_SCRIPT+="echo ''; echo 'All tasks complete.'; "
GENERATIONS_DIR="${OUTPUT_DIR}/generations"
INNER_SCRIPT+="mkdir -p ${GENERATIONS_DIR} && "
INNER_SCRIPT+="find ${OUTPUT_DIR} -maxdepth 1 -name 'samples_*.jsonl' -exec mv {} ${GENERATIONS_DIR}/ \; && "
INNER_SCRIPT+="echo 'Generations saved to: ${GENERATIONS_DIR}'"

if [[ "$FOREGROUND" == "true" ]]; then
    bash -c "${INNER_SCRIPT}"
else
    SESSION="lm_eval_$(date +%Y%m%d_%H%M%S)"
    TMUX_CMD="source ${VENV_ACTIVATE} && cd ${PROJECT_DIR} && ${INNER_SCRIPT}"
    tmux new-session -d -s "$SESSION" "bash -c $(printf '%q' "$TMUX_CMD") ; echo '' ; echo 'eval done — shell left open, type exit to close.' ; exec bash"
    echo ""
    echo "lm_eval running in tmux session: $SESSION"
    echo "  attach with: tmux attach -t $SESSION"
    echo "  results dir: $OUTPUT_DIR"
fi
