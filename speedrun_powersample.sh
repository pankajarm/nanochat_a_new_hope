#!/bin/bash

set -euo pipefail

# This script mirrors speedrun.sh but adds an automated evaluation pass that
# exercises the tool-aware power sampling engine against GSM8K.
# It follows the same stages as the standard run and finishes by invoking the
# new eval_gsm8k_powersample.py entrypoint.

export OMP_NUM_THREADS=2
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python environment setup (via uv like the original speedrun script)

# command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# [ -d ".venv" ] || uv venv
# # Force reinstall with updated dependencies
# uv sync --refresh --reinstall --extra gpu
# source .venv/bin/activate
# # Explicitly install critical missing dependencies if not present
# python -c "import pyarrow" 2>/dev/null || uv pip install pyarrow
# python -c "import jinja2" 2>/dev/null || uv pip install jinja2

# -----------------------------------------------------------------------------
# Optional wandb logging

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report scaffolding

# python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer + dataset bootstrap

# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# source "$HOME/.cargo/env"
# uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# python -m nanochat.dataset -n 8
# python -m nanochat.dataset -n 240 &
# DATASET_DOWNLOAD_PID=$!
# python -m scripts.tok_train --max_chars=2000000000
# python -m scripts.tok_eval

# EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
# if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
#     curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
#     unzip -q eval_bundle.zip
#     rm eval_bundle.zip
#     mv eval_bundle "$NANOCHAT_BASE_DIR"
# fi

# echo "Waiting for dataset download to complete..."
# wait $DATASET_DOWNLOAD_PID

# -----------------------------------------------------------------------------
# Base pretraining + evaluations

# torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
# torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
# torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (teaches tool usage) + evals

# torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run=$WANDB_RUN
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised finetuning + evals

# torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Power-sampling GSM8K evaluation

# Performance settings - OPTIMIZED FOR FULL RUN
POWERSAMPLE_NUM_GPUS=${POWERSAMPLE_NUM_GPUS:-8}           # All 8 GPUs
POWERSAMPLE_BATCH_SIZE=${POWERSAMPLE_BATCH_SIZE:-15}      # 15 problems per GPU = ~75GB memory (93% utilization!)
POWERSAMPLE_USE_BATCHED=${POWERSAMPLE_USE_BATCHED:-1}     # Always use batched for speed

# Algorithm settings - FULL EVALUATION
POWERSAMPLE_SOURCE=${POWERSAMPLE_SOURCE:-sft}
POWERSAMPLE_ALPHA=${POWERSAMPLE_ALPHA:-4.0}
POWERSAMPLE_STEPS=${POWERSAMPLE_STEPS:-10}                # Full 10 MCMC steps
POWERSAMPLE_TEMPERATURE=${POWERSAMPLE_TEMPERATURE:-0.7}
POWERSAMPLE_TOP_K=${POWERSAMPLE_TOP_K:-50}
POWERSAMPLE_MAX_NEW=${POWERSAMPLE_MAX_NEW:-256}           # Full 256 tokens
POWERSAMPLE_SEED=${POWERSAMPLE_SEED:-0}
POWERSAMPLE_TOOL_TIMEOUT=${POWERSAMPLE_TOOL_TIMEOUT:-5.0}
POWERSAMPLE_TOOL_MAX_OUT=${POWERSAMPLE_TOOL_MAX_OUT:-128}
POWERSAMPLE_MAX_EXAMPLES=${POWERSAMPLE_MAX_EXAMPLES:-}     # Empty = full GSM8K dataset (1319 examples)
POWERSAMPLE_SUBSET=${POWERSAMPLE_SUBSET:-main}
POWERSAMPLE_SPLIT=${POWERSAMPLE_SPLIT:-test}

# Print configuration
echo "============================================================"
echo "Power Sampling Configuration - FULL EVALUATION"
echo "============================================================"
echo "Performance:"
echo "  GPUs: $POWERSAMPLE_NUM_GPUS"
echo "  Batch size per GPU: $POWERSAMPLE_BATCH_SIZE (using ~75GB/80GB per GPU)"
echo "  Total parallel processing: $((POWERSAMPLE_NUM_GPUS * POWERSAMPLE_BATCH_SIZE)) problems simultaneously!"
echo ""
echo "Algorithm:"
echo "  MCMC steps: $POWERSAMPLE_STEPS"
echo "  Max new tokens: $POWERSAMPLE_MAX_NEW"
echo "  Temperature: $POWERSAMPLE_TEMPERATURE"
echo "  Dataset: Full GSM8K (${POWERSAMPLE_MAX_EXAMPLES:-1319 examples})"
echo "  Expected time: ~5-10 minutes for full evaluation"
echo "============================================================"
echo ""

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
)

if [ -n "$POWERSAMPLE_MAX_EXAMPLES" ]; then
    POWERSAMPLE_ARGS+=(--max-examples "$POWERSAMPLE_MAX_EXAMPLES")
fi

# Choose which version to run based on settings
if [ "$POWERSAMPLE_USE_BATCHED" -eq 1 ]; then
    echo "🚀 Starting BATCHED power sampling evaluation..."
    echo "   Using optimized multi-GPU processing with high memory utilization"
    echo "   Processing $((POWERSAMPLE_NUM_GPUS * POWERSAMPLE_BATCH_SIZE)) problems in parallel!"
    echo ""
    POWERSAMPLE_ARGS+=(--batch-size "$POWERSAMPLE_BATCH_SIZE")
    
    if [ "$POWERSAMPLE_NUM_GPUS" -gt 1 ]; then
        echo "Launching with torchrun on $POWERSAMPLE_NUM_GPUS GPUs..."
        torchrun --standalone --nproc_per_node=$POWERSAMPLE_NUM_GPUS -m scripts.eval_gsm8k_powersample_batched "${POWERSAMPLE_ARGS[@]}"
    else
        echo "Using single GPU..."
        python -m scripts.eval_gsm8k_powersample_batched "${POWERSAMPLE_ARGS[@]}"
    fi
else
    echo "Running standard power sampling (slower, less efficient)..."
    if [ "$POWERSAMPLE_NUM_GPUS" -gt 1 ]; then
        echo "Using torchrun for multi-GPU processing..."
        torchrun --standalone --nproc_per_node=$POWERSAMPLE_NUM_GPUS -m scripts.eval_gsm8k_powersample "${POWERSAMPLE_ARGS[@]}"
    else
        echo "Using single GPU..."
        python -m scripts.eval_gsm8k_powersample "${POWERSAMPLE_ARGS[@]}"
    fi
fi

# -----------------------------------------------------------------------------
# Final consolidated report

python -m nanochat.report generate
cp "$NANOCHAT_BASE_DIR/report/report.md" ./report_powersample.md

# Optional: launch the chat web UI against the latest checkpoint
# python -m scripts.chat_web
