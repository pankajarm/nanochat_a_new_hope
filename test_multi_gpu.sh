#!/bin/bash

# Simple test script for multi-GPU power sampling
set -euo pipefail

export OMP_NUM_THREADS=2
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# Minimal test configuration
export POWERSAMPLE_MAX_EXAMPLES=8     # Just 8 examples (1 per GPU)
export POWERSAMPLE_STEPS=1            # Just 1 MCMC step for testing
export POWERSAMPLE_MAX_NEW=64         # Short sequences
export POWERSAMPLE_NUM_GPUS=8         # Use all 8 GPUs

echo "=========================================="
echo "Multi-GPU Power Sampling Quick Test"
echo "=========================================="
echo "Configuration:"
echo "  GPUs: $POWERSAMPLE_NUM_GPUS"
echo "  Examples: $POWERSAMPLE_MAX_EXAMPLES"
echo "  MCMC steps: $POWERSAMPLE_STEPS"
echo "  Max tokens: $POWERSAMPLE_MAX_NEW"
echo ""

# Activate virtual environment if not already
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source .venv/bin/activate
fi

# Run the power sampling with multi-GPU
echo "Starting evaluation..."
time torchrun --standalone --nproc_per_node=$POWERSAMPLE_NUM_GPUS \
    -m scripts.eval_gsm8k_powersample \
    --source sft \
    --max-examples $POWERSAMPLE_MAX_EXAMPLES \
    --num-steps $POWERSAMPLE_STEPS \
    --max-new-tokens $POWERSAMPLE_MAX_NEW \
    --alpha 4.0 \
    --temperature 0.7 \
    --top-k 50 \
    --seed 0

echo ""
echo "Test completed successfully!"
echo "If you see 'Accuracy:' above with no errors, multi-GPU is working!"
