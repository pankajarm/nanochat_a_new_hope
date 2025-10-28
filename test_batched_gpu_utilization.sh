#!/bin/bash

# Test script to verify improved GPU utilization with batched processing
set -euo pipefail

export OMP_NUM_THREADS=2
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

echo "=========================================="
echo "GPU Utilization Test: Batched Processing"
echo "=========================================="
echo ""
echo "This test will show improved GPU memory usage"
echo "by processing multiple problems per GPU"
echo ""

# Test configuration for quick results
export POWERSAMPLE_MAX_EXAMPLES=80    # 80 examples (10 per GPU with batch_size=10)
export POWERSAMPLE_STEPS=2             # Just 2 MCMC steps for speed
export POWERSAMPLE_MAX_NEW=128         # Moderate token generation
export POWERSAMPLE_NUM_GPUS=8          # All 8 GPUs
export POWERSAMPLE_BATCH_SIZE=10       # 10 problems per GPU (10x improvement!)
export POWERSAMPLE_USE_BATCHED=1       # Use the batched version

echo "Configuration:"
echo "  GPUs: $POWERSAMPLE_NUM_GPUS"
echo "  Batch size per GPU: $POWERSAMPLE_BATCH_SIZE"
echo "  Total parallel processing: $((POWERSAMPLE_NUM_GPUS * POWERSAMPLE_BATCH_SIZE)) problems"
echo "  Examples to process: $POWERSAMPLE_MAX_EXAMPLES"
echo "  MCMC steps: $POWERSAMPLE_STEPS"
echo ""
echo "Expected GPU memory usage: ~30-50GB per GPU (vs 5GB before)"
echo ""
echo "Monitor GPU usage in another terminal with:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Starting in 5 seconds..."
sleep 5

# Activate virtual environment if not already
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source .venv/bin/activate
fi

# Run the batched power sampling
echo "Running batched evaluation..."
time bash speedrun_powersample.sh

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "With batch_size=$POWERSAMPLE_BATCH_SIZE, you should have seen:"
echo "  - Much higher GPU memory utilization (30-50GB vs 5GB)"
echo "  - Faster overall processing time"
echo "  - All 8 GPUs actively processing 10 problems each"
echo ""
echo "Estimated speedup: ~8-10x compared to sequential processing!"
