#!/bin/bash

# RL training script for d34 - continues from run_d34_finetune.sh
# This script assumes you've already run run_d34_finetune.sh which:
# - Downloaded the pre-trained base model from HuggingFace
# - Ran mid-training → saved to mid_checkpoints/d34/
# - Ran SFT → saved to chatsft_checkpoints/d34/
#
# This script will:
# - Load the SFT checkpoint from chatsft_checkpoints/d34/
# - Run RL training on GSM8K
# - Evaluate the RL model
#
# Hardware requirements: 8xH100/A100 80GB
#
# Example launch:
# bash rl_d34_finetune.sh
#
# With wandb logging:
# WANDB_RUN=d34_rl screen -L -Logfile d34_rl.log -S d34_rl bash rl_d34_finetune.sh

set -e  # Exit on error

# all the setup stuff
export OMP_NUM_THREADS=1
export UV_LINK_MODE=copy  # suppress hardlink warning when project is on NFS
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || true  # add uv to PATH if just installed
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
python -m nanochat.report reset

# Install build tools (needed for compiling Rust/C code)
if ! command -v cc &> /dev/null; then
    echo "Installing build-essential (requires sudo)..."
    sudo apt-get update && sudo apt-get install -y build-essential
fi

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# -----------------------------------------------------------------------------
# Verify SFT checkpoint exists (from run_d34_finetune.sh)

SFT_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d34"
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"

echo "Checking for required files from run_d34_finetune.sh..."

if [ ! -d "$SFT_CHECKPOINT_DIR" ] || [ -z "$(ls -A $SFT_CHECKPOINT_DIR 2>/dev/null)" ]; then
    echo "❌ ERROR: SFT checkpoint not found at $SFT_CHECKPOINT_DIR"
    echo ""
    echo "Please run run_d34_finetune.sh first to:"
    echo "  1. Download the pre-trained base model"
    echo "  2. Run mid-training"
    echo "  3. Run SFT training"
    echo ""
    echo "Then run this script for RL training."
    exit 1
fi

if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ] || [ ! -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    echo "❌ ERROR: Tokenizer files not found at $TOKENIZER_DIR"
    echo "Please run run_d34_finetune.sh first."
    exit 1
fi

echo "✅ Found SFT checkpoint at: $SFT_CHECKPOINT_DIR"
echo "✅ Found tokenizer at: $TOKENIZER_DIR"
ls -la "$SFT_CHECKPOINT_DIR"

# -----------------------------------------------------------------------------
# Initialize CUDA environment (required for H100/A100 clusters)

echo "Initializing CUDA environment..."

# Enable GPU persistence mode (keeps driver loaded)
if command -v nvidia-smi &> /dev/null; then
    sudo nvidia-smi -pm 1 2>/dev/null || echo "Note: Could not enable persistence mode (may need sudo)"
fi

# Start nvidia-fabricmanager (required for multi-GPU NVSwitch systems like 8xH100)
if systemctl list-unit-files 2>/dev/null | grep -q nvidia-fabricmanager; then
    echo "Starting nvidia-fabricmanager service..."
    sudo systemctl start nvidia-fabricmanager
    sleep 5
    if systemctl is-active --quiet nvidia-fabricmanager; then
        echo "✅ nvidia-fabricmanager is running"
    else
        echo "⚠️  nvidia-fabricmanager failed to start (may be OK for some systems)"
    fi
fi

sleep 2

# Verify CUDA is working
echo "Verifying CUDA availability..."
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'CUDA OK: {torch.cuda.device_count()} GPUs')"; then
    echo "❌ CUDA verification failed. Troubleshooting..."
    echo "nvidia-smi output:"
    nvidia-smi || true
    echo ""
    echo "Try running: sudo systemctl restart nvidia-fabricmanager"
    echo "Then re-run this script."
    exit 1
fi

# Number of processes/GPUs to use
NPROC_PER_NODE=8

# -----------------------------------------------------------------------------
# Run Reinforcement Learning on GSM8K
# Loads from "sft" source which maps to chatsft_checkpoints/d34/

echo ""
echo "=========================================="
echo "Starting RL training on GSM8K..."
echo "=========================================="
echo "Loading SFT checkpoint from: $SFT_CHECKPOINT_DIR"

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --device_batch_size=4 --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Evaluate the RL model

echo ""
echo "=========================================="
echo "Evaluating RL model..."
echo "=========================================="

# Evaluate on GSM8K (the RL training task)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# Full evaluation (all tasks)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl

# -----------------------------------------------------------------------------
# Generate final report
python -m nanochat.report generate

echo ""
echo "=========================================="
echo "RL training and evaluation complete!"
echo "=========================================="
echo ""
echo "Checkpoints saved to:"
echo "  SFT: $SFT_CHECKPOINT_DIR"
echo "  RL:  $NANOCHAT_BASE_DIR/chatrl_checkpoints/d34/"
echo ""
echo "Report generated. To view:"
echo "  cat $NANOCHAT_BASE_DIR/report/report.md"
echo ""
echo "To chat with the RL model:"
echo "  python -m scripts.chat_cli -p \"What is 25 * 17?\""
echo "  python -m scripts.chat_web"
