#!/bin/bash

# RL training script for the d34 model
# Downloads pretrained SFT weights from HuggingFace, runs RL on GSM8K, then evaluates
#
# HuggingFace model: https://huggingface.co/karpathy/nanochat-d34
#
# Hardware requirements: 8xH100/A100 80GB (same as training scripts)
# This script performs RL training which requires significant GPU memory.

# Example launch:
# bash rl_d34.sh
#
# With wandb logging:
# WANDB_RUN=d34-rl bash rl_d34.sh
#
# With screen (recommended for long runs):
# screen -L -Logfile rl_d34.log -S rl_d34 bash rl_d34.sh
# WANDB_RUN=d34-rl screen -L -Logfile rl_d34.log -S rl_d34 bash rl_d34.sh

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
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Initialize report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Download d34 model weights and tokenizer from HuggingFace
# Repository: https://huggingface.co/karpathy/nanochat-d34
#
# Files on HuggingFace (at root level):
# - model_169150.pt (8.58 GB) - SFT checkpoint at step 169150
# - meta_169150.json - metadata
# - tokenizer.pkl, token_bytes.pt - tokenizer files

echo "Downloading d34 model from HuggingFace..."

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

# Setup tokenizer directory
tokenizer_dst = os.path.join(base_dir, "tokenizer")
os.makedirs(tokenizer_dst, exist_ok=True)

# Copy tokenizer files (at root level in HF repo)
tokenizer_files = ["tokenizer.pkl", "token_bytes.pt"]
for f in tokenizer_files:
    src = os.path.join(local_dir, f)
    dst = os.path.join(tokenizer_dst, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {f} -> {tokenizer_dst}/")
    else:
        print(f"Warning: {f} not found in {local_dir}")

# Also check for tokenizer subdirectory (in case structure differs)
tokenizer_subdir = os.path.join(local_dir, "tokenizer")
if os.path.exists(tokenizer_subdir):
    for f in os.listdir(tokenizer_subdir):
        src = os.path.join(tokenizer_subdir, f)
        dst = os.path.join(tokenizer_dst, f)
        shutil.copy2(src, dst)
        print(f"Copied tokenizer/{f} -> {tokenizer_dst}/")

print(f"Tokenizer ready at: {tokenizer_dst}")

# Setup SFT checkpoint directory
# The RL script loads from "sft" source which maps to "chatsft_checkpoints"
checkpoint_dst = os.path.join(base_dir, "chatsft_checkpoints", "d34")
os.makedirs(checkpoint_dst, exist_ok=True)

# Copy model checkpoint files (at root level: model_169150.pt, meta_169150.json)
checkpoint_files = [f for f in os.listdir(local_dir) if f.startswith("model_") or f.startswith("meta_")]
for f in checkpoint_files:
    src = os.path.join(local_dir, f)
    dst = os.path.join(checkpoint_dst, f)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"Copied {f} -> {checkpoint_dst}/")

# Also check for chatsft_checkpoints subdirectory (in case structure differs)
sft_subdir = os.path.join(local_dir, "chatsft_checkpoints")
if os.path.exists(sft_subdir):
    for model_tag in os.listdir(sft_subdir):
        src_dir = os.path.join(sft_subdir, model_tag)
        dst_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag)
        if os.path.isdir(src_dir):
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            print(f"Copied chatsft_checkpoints/{model_tag}/ -> {dst_dir}/")

print(f"SFT checkpoint ready at: {checkpoint_dst}")
print("\nAll downloads complete!")
PYTHON_SCRIPT

# -----------------------------------------------------------------------------
# Initialize CUDA (required for H100/A100 clusters)
# This fixes "Error 802: system not yet initialized" issues

echo "Initializing CUDA environment..."

# Enable GPU persistence mode (keeps driver loaded)
if command -v nvidia-smi &> /dev/null; then
    sudo nvidia-smi -pm 1 2>/dev/null || echo "Note: Could not enable persistence mode (may need sudo)"
fi

# Start nvidia-fabricmanager (required for H100 NVLink)
# Check if it's available and try to start it
if systemctl list-unit-files | grep -q nvidia-fabricmanager; then
    sudo systemctl start nvidia-fabricmanager 2>/dev/null || echo "Note: Could not start nvidia-fabricmanager"
fi

# Verify CUDA is working
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'CUDA OK: {torch.cuda.device_count()} GPUs')"

# -----------------------------------------------------------------------------
# Run Reinforcement Learning on GSM8K

NPROC_PER_NODE=8

echo ""
echo "=========================================="
echo "Starting RL training on GSM8K..."
echo "=========================================="

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Evaluate the RL model on GSM8K

echo ""
echo "=========================================="
echo "Evaluating RL model on GSM8K..."
echo "=========================================="

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the evaluation report
python -m nanochat.report generate

echo ""
echo "=========================================="
echo "RL training and evaluation complete!"
echo "Report generated in $NANOCHAT_BASE_DIR/report/"
echo "=========================================="
echo ""
echo "RL checkpoint saved to: $NANOCHAT_BASE_DIR/chatrl_checkpoints/d34/"
echo ""
echo "To chat with the RL model:"
echo "  python -m scripts.chat_cli -p \"What is 25 * 17?\""
echo "  python -m scripts.chat_web"

