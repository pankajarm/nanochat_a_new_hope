#!/bin/bash

# Eval-only script for the d34 model
# Downloads pretrained weights from HuggingFace and runs evaluations
# No training is performed - this is purely for evaluation
#
# HuggingFace model: https://huggingface.co/karpathy/nanochat-d34
#
# Hardware requirements: Single GPU with 24GB+ VRAM (e.g., A10, RTX 4090, RTX 3090)
# Default runs on 1 GPU. For multi-GPU, set NPROC_PER_NODE environment variable.

# Example launch (single GPU, default):
# bash eval_d34.sh
#
# Example launch with multiple GPUs:
# NPROC_PER_NODE=8 bash eval_d34.sh
#
# With screen:
# screen -L -Logfile eval_d34.log -S eval_d34 bash eval_d34.sh

set -e  # Exit on error

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || true  # add uv to PATH if just installed
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Install Rust / Cargo (needed for the tokenizer)

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# -----------------------------------------------------------------------------
# Download d34 model weights and tokenizer from HuggingFace
# Repository: https://huggingface.co/karpathy/nanochat-d34

echo "Downloading d34 model from HuggingFace..."

HF_REPO="karpathy/nanochat-d34"

# Use snapshot_download to get the entire repo, then organize files
python << 'PYTHON_SCRIPT'
import os
import shutil
from huggingface_hub import snapshot_download

repo_id = "karpathy/nanochat-d34"
base_dir = os.environ["NANOCHAT_BASE_DIR"]

# Download entire repo to a cache directory
print(f"Downloading {repo_id} from HuggingFace...")
local_dir = snapshot_download(
    repo_id=repo_id,
    local_dir=os.path.join(base_dir, "hf_download", "nanochat-d34"),
    local_dir_use_symlinks=False
)
print(f"Downloaded to: {local_dir}")

# Helper function to copy directory contents
def copy_dir_contents(src, dst):
    if os.path.exists(src):
        os.makedirs(dst, exist_ok=True)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print(f"Copied {src} -> {dst}")
        return True
    return False

# Copy tokenizer files
tokenizer_src = os.path.join(local_dir, "tokenizer")
tokenizer_dst = os.path.join(base_dir, "tokenizer")
if copy_dir_contents(tokenizer_src, tokenizer_dst):
    print(f"Tokenizer ready at: {tokenizer_dst}")
else:
    print(f"Warning: No tokenizer directory found in {local_dir}")

# Copy SFT checkpoint files
sft_src = os.path.join(local_dir, "chatsft_checkpoints")
sft_dst = os.path.join(base_dir, "chatsft_checkpoints")
if copy_dir_contents(sft_src, sft_dst):
    print(f"SFT checkpoints ready at: {sft_dst}")
else:
    # Maybe checkpoint files are at root level with a different structure
    # Try to find model_*.pt files at root and organize them
    print(f"Warning: No chatsft_checkpoints directory found, checking root level...")
    model_files = [f for f in os.listdir(local_dir) if f.startswith("model_") and f.endswith(".pt")]
    meta_files = [f for f in os.listdir(local_dir) if f.startswith("meta_") and f.endswith(".json")]
    if model_files:
        checkpoint_dst = os.path.join(base_dir, "chatsft_checkpoints", "d34")
        os.makedirs(checkpoint_dst, exist_ok=True)
        for f in model_files + meta_files:
            src = os.path.join(local_dir, f)
            dst = os.path.join(checkpoint_dst, f)
            shutil.copy2(src, dst)
            print(f"Copied {f} -> {checkpoint_dst}/")
        print(f"SFT checkpoints ready at: {checkpoint_dst}")

# Copy base checkpoints if available (for base_eval)
base_src = os.path.join(local_dir, "base_checkpoints")
base_dst = os.path.join(base_dir, "base_checkpoints")
if copy_dir_contents(base_src, base_dst):
    print(f"Base checkpoints ready at: {base_dst}")

# Copy mid checkpoints if available
mid_src = os.path.join(local_dir, "mid_checkpoints")
mid_dst = os.path.join(base_dir, "mid_checkpoints")
if copy_dir_contents(mid_src, mid_dst):
    print(f"Mid checkpoints ready at: {mid_dst}")

print("\nAll downloads complete!")
PYTHON_SCRIPT

# -----------------------------------------------------------------------------
# wandb setup (for logging eval results)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Initialize report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Run evaluations
# Default: single GPU (A10 24GB or similar)
# For multi-GPU, set NPROC_PER_NODE=8 (or number of GPUs available)

NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# Run base model evaluation if base checkpoints exist
if [ -d "$NANOCHAT_BASE_DIR/base_checkpoints" ]; then
    echo ""
    echo "Running base model loss evaluation..."
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss || echo "base_loss skipped (may need data)"
    
    echo ""
    echo "Running base model CORE evaluation..."
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval
fi

# Run chat evaluation on mid model if checkpoints exist
if [ -d "$NANOCHAT_BASE_DIR/mid_checkpoints" ]; then
    echo ""
    echo "Running chat evaluations on mid model..."
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid
fi

# Run chat evaluation on SFT model
if [ -d "$NANOCHAT_BASE_DIR/chatsft_checkpoints" ]; then
    echo ""
    echo "Running chat evaluations on SFT model..."
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft
fi

# -----------------------------------------------------------------------------
# Generate the evaluation report
python -m nanochat.report generate

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Report generated in $NANOCHAT_BASE_DIR/report/"
echo "=========================================="

# Optionally chat with the model
# python -m scripts.chat_cli -p "Why is the sky blue?"
# python -m scripts.chat_web
