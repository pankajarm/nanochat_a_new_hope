#!/bin/bash

# Quick test script for power sampling with a small subset
# This is for testing/debugging only - use speedrun_powersample.sh for full evaluation

set -euo pipefail

echo "============================================================"
echo "🧪 Power Sampling QUICK TEST - Small Subset Only"
echo "============================================================"
echo "⚠️  This is a TEST script with only 100 problems!"
echo "   For full evaluation, use: bash speedrun_powersample.sh"
echo "============================================================"
echo ""

export OMP_NUM_THREADS=2
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# Test settings - reduced for quick testing
export POWERSAMPLE_NUM_GPUS=8
export POWERSAMPLE_BATCH_SIZE=5            # Smaller batch for testing
export POWERSAMPLE_USE_BATCHED=1
export POWERSAMPLE_SOURCE=sft
export POWERSAMPLE_ALPHA=4.0
export POWERSAMPLE_STEPS=10                # Keep full steps even for testing
export POWERSAMPLE_TEMPERATURE=0.7
export POWERSAMPLE_TOP_K=50
export POWERSAMPLE_MAX_NEW=256
export POWERSAMPLE_SEED=0
export POWERSAMPLE_TOOL_TIMEOUT=5.0
export POWERSAMPLE_TOOL_MAX_OUT=128
export POWERSAMPLE_MAX_EXAMPLES=100        # ONLY 100 problems for quick test!
export POWERSAMPLE_SUBSET=main
export POWERSAMPLE_SPLIT=test

echo "Test Configuration:"
echo "  Problems: $POWERSAMPLE_MAX_EXAMPLES (subset for testing)"
echo "  MCMC steps: $POWERSAMPLE_STEPS"
echo "  GPUs: $POWERSAMPLE_NUM_GPUS"
echo "  Expected time: ~1-2 minutes"
echo ""
echo "Starting test evaluation..."
echo ""

# Activate environment
[ -d ".venv" ] && source .venv/bin/activate

# Run the test
POWERSAMPLE_ARGS=(
    --source "$POWERSAMPLE_SOURCE"
    --alpha "$POWERSAMPLE_ALPHA"
    --num-steps "$POWERSAMPLE_STEPS"
    --temperature "$POWERSAMPLE_TEMPERATURE"
    --top-k "$POWERSAMPLE_TOP_K"
    --max-new-tokens "$POWERSAMPLE_MAX_NEW"
    --seed "$POWERSAMPLE_SEED"
    --tool-timeout "$POWERSAMPLE_TOOL_TIMEOUT"
    --tool-max-output-tokens "$POWERSAMPLE_TOOL_MAX_OUT"
    --subset "$POWERSAMPLE_SUBSET"
    --split "$POWERSAMPLE_SPLIT"
    --max-examples "$POWERSAMPLE_MAX_EXAMPLES"
    --batch-size "$POWERSAMPLE_BATCH_SIZE"
)

if [ "$POWERSAMPLE_NUM_GPUS" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$POWERSAMPLE_NUM_GPUS -m scripts.eval_gsm8k_powersample_batched "${POWERSAMPLE_ARGS[@]}"
else
    python -m scripts.eval_gsm8k_powersample_batched "${POWERSAMPLE_ARGS[@]}"
fi

echo ""
echo "============================================================"
echo "✅ Quick test complete!"
echo "   This was only $POWERSAMPLE_MAX_EXAMPLES problems out of 1,319"
echo "   For full evaluation: bash speedrun_powersample.sh"
echo "============================================================"
