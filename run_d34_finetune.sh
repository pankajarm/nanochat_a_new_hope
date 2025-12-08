#!/bin/bash

# Finetune version of d34 - starts from pre-trained base model
# This script downloads the pre-trained base model from HuggingFace and runs:
# - Mid training
# - SFT (Supervised Fine-Tuning)
# - Evaluation
# (NO RL - that's in a separate script)
#
# This saves ~$2,500 and ~100 hours of base model pretraining!
#
# Pre-trained model source: https://huggingface.co/karpathy/nanochat-d34
# Base model stats:
# - depth: 34
# - Number of parameters: 2,217,082,880
# - Training tokens: 88,683,315,200 (40x param:token ratio, 2x Chinchilla optimal)
# - CORE score: 0.3382
#
# Example launch:
# screen -L -Logfile d34_finetune.log -S d34_finetune bash run_d34_finetune.sh
# With wandb logging:
# WANDB_RUN=d34_finetune screen -L -Logfile d34_finetune.log -S d34_finetune bash run_d34_finetune.sh

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

# Download identity conversations for training
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# -----------------------------------------------------------------------------
# Download pre-trained model from HuggingFace
# Source: https://huggingface.co/karpathy/nanochat-d34

echo "Downloading pre-trained d34 base model from HuggingFace..."

HF_REPO="karpathy/nanochat-d34"
HF_BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"

# Create directories for tokenizer and base model checkpoint
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
BASE_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d34"
mkdir -p "$TOKENIZER_DIR"
mkdir -p "$BASE_CHECKPOINT_DIR"

# Download tokenizer files
echo "Downloading tokenizer files..."
if [ ! -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    curl -L -o "$TOKENIZER_DIR/token_bytes.pt" "${HF_BASE_URL}/token_bytes.pt"
    echo "✅ Downloaded token_bytes.pt"
else
    echo "⏭️  token_bytes.pt already exists, skipping"
fi

if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ]; then
    curl -L -o "$TOKENIZER_DIR/tokenizer.pkl" "${HF_BASE_URL}/tokenizer.pkl"
    echo "✅ Downloaded tokenizer.pkl"
else
    echo "⏭️  tokenizer.pkl already exists, skipping"
fi

# Download base model checkpoint files
# NOTE: Original HF instructions said chatsft_checkpoints, but for mid-training
# we need these in base_checkpoints/d34/
echo "Downloading base model checkpoint..."
if [ ! -f "$BASE_CHECKPOINT_DIR/model_169150.pt" ]; then
    curl -L -o "$BASE_CHECKPOINT_DIR/model_169150.pt" "${HF_BASE_URL}/model_169150.pt"
    echo "✅ Downloaded model_169150.pt (~8.6GB)"
else
    echo "⏭️  model_169150.pt already exists, skipping"
fi

if [ ! -f "$BASE_CHECKPOINT_DIR/meta_169150.json" ]; then
    curl -L -o "$BASE_CHECKPOINT_DIR/meta_169150.json" "${HF_BASE_URL}/meta_169150.json"
    echo "✅ Downloaded meta_169150.json"
else
    echo "⏭️  meta_169150.json already exists, skipping"
fi

echo "✅ Pre-trained model download complete!"
echo "   Tokenizer: $TOKENIZER_DIR"
echo "   Base checkpoint: $BASE_CHECKPOINT_DIR"

# -----------------------------------------------------------------------------
# Initialize CUDA environment (required for H100/A100 clusters)

echo "Initializing CUDA environment..."

# Enable GPU persistence mode (keeps driver loaded)
if command -v nvidia-smi &> /dev/null; then
    sudo nvidia-smi -pm 1 2>/dev/null || echo "Note: Could not enable persistence mode (may need sudo)"
fi

# Start nvidia-fabricmanager (required for multi-GPU NVSwitch systems like 8xH100)
# This is CRITICAL for NVSwitch-based multi-GPU systems
if systemctl list-unit-files 2>/dev/null | grep -q nvidia-fabricmanager; then
    echo "Starting nvidia-fabricmanager service..."
    sudo systemctl start nvidia-fabricmanager
    # Wait for fabricmanager to fully initialize
    sleep 5
    # Check if it's actually running
    if systemctl is-active --quiet nvidia-fabricmanager; then
        echo "✅ nvidia-fabricmanager is running"
    else
        echo "⚠️  nvidia-fabricmanager failed to start, checking status..."
        sudo systemctl status nvidia-fabricmanager || true
    fi
fi

# Additional wait for CUDA to fully initialize after fabricmanager
sleep 2

# Verify CUDA is working
echo "Verifying CUDA availability..."
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'CUDA OK: {torch.cuda.device_count()} GPUs')"; then
    echo "❌ CUDA verification failed. Troubleshooting..."
    echo "nvidia-smi output:"
    nvidia-smi || true
    echo ""
    echo "fabricmanager status:"
    sudo systemctl status nvidia-fabricmanager 2>/dev/null || echo "fabricmanager not found"
    echo ""
    echo "Try running: sudo systemctl restart nvidia-fabricmanager"
    echo "Then re-run this script."
    exit 1
fi

# Number of processes/GPUs to use
NPROC_PER_NODE=8

# -----------------------------------------------------------------------------
# Skip base pretraining - we're using the pre-trained model!
# The original run_d34.sh would run:
#   torchrun ... -m scripts.base_train -- --depth=34 --device_batch_size=4 --target_param_data_ratio=40 --save_every=5000 --run=$WANDB_RUN
#   torchrun ... -m scripts.base_loss
#   torchrun ... -m scripts.base_eval
# But we skip all of that since we downloaded the pre-trained checkpoint.
echo "⏭️  Skipping base pretraining (using pre-trained model from HuggingFace)"

# -----------------------------------------------------------------------------
# Midtrain
# NOTE: ensure that we use the same device_batch_size here as the base training script.
echo "Starting mid-training..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --device_batch_size=4 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# SFT (Supervised Fine-Tuning)
echo "Starting SFT..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --device_batch_size=4 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Generate final report
python -m nanochat.report generate

# Talk to it
python -m scripts.chat_web
