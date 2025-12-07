#!/bin/bash

# The d34 tier of nanochat
# A deeper model (depth=34) with 40x data:param ratio (2x Chinchilla optimal)
# This is a longer run designed for extended training on 8XH100 nodes

# Key differences from run1000.sh (d32):
# - depth=34 (vs 32) - deeper model, more parameters (~2.1B estimated)
# - device_batch_size=4 (vs 8) - smaller batch to fit in memory
# - target_param_data_ratio=40 (vs 20) - 2x more training data per parameter
# - This means ~2x more training steps and significantly longer runtime

# Estimated parameters and data requirements:
# - Model params: ~2.1B (estimate based on d32 having ~1.88B)
# - With 40x ratio: 2.1B * 40 = 84B tokens
# - At ~4.8 chars/token: 84B * 4.8 = ~403B chars
# - At 250M chars/shard: ~1612 shards needed
# - Rounding up to 1700 shards for safety (~170GB of data on disk)

# Example launch:
# screen -L -Logfile d34.log -S d34 bash run_d34.sh
# With wandb logging:
# WANDB_RUN=d34 screen -L -Logfile d34.log -S d34 bash run_d34.sh

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
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# train tokenizer on ~4B characters and kick off download of the rest for pretraining
python -m nanochat.dataset -n 16
# start downloading the rest of the shards (1700 shards for 40x data ratio with d34)
python -m nanochat.dataset -n 1700 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train --max_chars=4000000000
python -m scripts.tok_eval

# wait for dataset download to complete before pretraining
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use
NPROC_PER_NODE=8

# pretrain d34 model with 40x data:param ratio
# device_batch_size=4 to fit in memory (d34 is larger than d32)
# save_every=5000 for more frequent checkpoints during long run
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=34 --device_batch_size=4 --target_param_data_ratio=40 --save_every=5000 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# midtrain
# NOTE: ensure that we use the same device_batch_size here as the base training script.
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --device_batch_size=4 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# sft
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --device_batch_size=4 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# generate final report
python -m nanochat.report generate

# talk to it
python -m scripts.chat_web

