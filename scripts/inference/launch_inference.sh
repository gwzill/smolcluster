#!/bin/bash

# Unified inference launcher for MP + DP
# Usage:
#   ./scripts/inference/launch_inference.sh --algorithm mp
#   ./scripts/inference/launch_inference.sh --algorithm syncps
#   ./scripts/inference/launch_inference.sh --algorithm classicdp
#   ./scripts/inference/launch_inference.sh --algorithm mp --dry-run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALGORITHM="mp"
DRY_RUN=false

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
        --help|-h)
            echo "Unified inference launcher"
            echo ""
            echo "Options:"
            echo "  --algorithm, -a   mp | syncps | classicdp (default: mp)"
            echo "  --dry-run         Print commands without executing"
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

case "$ALGORITHM" in
    mp)
        TARGET_SCRIPT="$SCRIPT_DIR/model_parallelism/launch_inference.sh"
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

echo "🚀 Launching inference with algorithm: $ALGORITHM"

if [[ "$DRY_RUN" == "true" ]]; then
    bash "$TARGET_SCRIPT" --dry-run
else
    bash "$TARGET_SCRIPT"
fi
