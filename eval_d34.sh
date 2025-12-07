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
export UV_LINK_MODE=copy  # suppress hardlink warning when project is on NFS
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || true  # add uv to PATH if just installed

# Handle ARM64 (e.g., GH200) - PyTorch CUDA wheels only available for x86_64
if [ "$(uname -m)" = "aarch64" ]; then
    echo "Detected ARM64 architecture (e.g., GH200 Grace CPU)"
    echo "PyTorch CUDA wheels are not available for ARM64 from the standard index."
    echo ""
    echo "Checking for system PyTorch installation..."
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "Found working system PyTorch with CUDA support!"
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        echo "Using system PyTorch $TORCH_VERSION"
        echo ""
        # Create venv with system-site-packages to inherit PyTorch
        [ -d ".venv" ] || uv venv --system-site-packages
        source .venv/bin/activate
        # Install remaining dependencies with pip (not uv sync, which may override torch)
        pip install datasets>=4.0.0 fastapi>=0.117.1 psutil>=7.1.0 regex>=2025.9.1 \
            tiktoken>=0.11.0 tokenizers>=0.22.0 uvicorn>=0.36.0 wandb>=0.21.3 huggingface_hub maturin
        # IMPORTANT: Remove any torch that pip may have installed as a dependency
        # This forces the venv to use system-site-packages torch (which has CUDA)
        # pip uninstall doesn't work well with system-site-packages, so we rm directly
        rm -rf .venv/lib/python*/site-packages/torch* 2>/dev/null || true
        # Verify we're now using system torch with CUDA
        if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo "Verified: Using system PyTorch with CUDA support!"
        else
            echo "ERROR: Still not seeing system PyTorch with CUDA after cleanup."
            echo "Please check your system PyTorch installation."
            exit 1
        fi
    else
        echo "ERROR: No system PyTorch with CUDA found."
        echo ""
        echo "For GH200/ARM64, please either:"
        echo "  1. Use NVIDIA NGC container: docker pull nvcr.io/nvidia/pytorch:24.10-py3"
        echo "  2. Install PyTorch manually for ARM64+CUDA before running this script"
        echo ""
        exit 1
    fi
else
    # x86_64 - use the CUDA 12.8 index with venv
    [ -d ".venv" ] || uv venv
    source .venv/bin/activate
    uv sync --extra gpu
fi

# -----------------------------------------------------------------------------
# Install build tools (needed for compiling Rust/C code)

if ! command -v cc &> /dev/null; then
    echo "Installing build-essential (requires sudo)..."
    sudo apt-get update && sudo apt-get install -y build-essential
fi

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
    # Tokenizer files might be at root level (tokenizer.pkl, token_bytes.pt)
    tokenizer_files = ["tokenizer.pkl", "token_bytes.pt"]
    found_files = [f for f in tokenizer_files if os.path.exists(os.path.join(local_dir, f))]
    if found_files:
        os.makedirs(tokenizer_dst, exist_ok=True)
        for f in found_files:
            src = os.path.join(local_dir, f)
            dst = os.path.join(tokenizer_dst, f)
            shutil.copy2(src, dst)
            print(f"Copied {f} -> {tokenizer_dst}/")
        print(f"Tokenizer ready at: {tokenizer_dst}")
    else:
        print(f"Warning: No tokenizer files found in {local_dir}")

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
# Initialize CUDA (required for H100/A100 clusters)
# This fixes "Error 802: system not yet initialized" issues

echo "Initializing CUDA environment..."

# Enable GPU persistence mode (keeps driver loaded)
if command -v nvidia-smi &> /dev/null; then
    sudo nvidia-smi -pm 1 2>/dev/null || echo "Note: Could not enable persistence mode (may need sudo)"
fi

# Start nvidia-fabricmanager (required for H100 NVLink)
if systemctl list-unit-files 2>/dev/null | grep -q nvidia-fabricmanager; then
    sudo systemctl start nvidia-fabricmanager 2>/dev/null || echo "Note: Could not start nvidia-fabricmanager"
fi

# Verify CUDA is working
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'CUDA OK: {torch.cuda.device_count()} GPUs')"

# -----------------------------------------------------------------------------
# Run evaluations
# Default: single GPU (A10 24GB or similar)
# For multi-GPU, set NPROC_PER_NODE=8 (or number of GPUs available)
# For larger VRAM GPUs (e.g., GH200 96GB), increase EVAL_BATCH_SIZE for faster evals

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}  # default 8 for 24GB GPUs, increase for more VRAM

echo "Using batch size: $EVAL_BATCH_SIZE (set EVAL_BATCH_SIZE env var to change)"

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
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid -b $EVAL_BATCH_SIZE
fi

# Run chat evaluation on SFT model
if [ -d "$NANOCHAT_BASE_DIR/chatsft_checkpoints" ]; then
    echo ""
    echo "Running chat evaluations on SFT model..."
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft -b $EVAL_BATCH_SIZE
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
