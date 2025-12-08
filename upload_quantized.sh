#!/bin/bash

# Quantize nanochat d34 model to 8-bit and upload to HuggingFace
# This creates a smaller, pre-quantized version for CPU inference
#
# Usage:
#   bash upload_quantized.sh
#
# Prerequisites:
# - HuggingFace CLI logged in (huggingface-cli login) or HF_TOKEN set
# - SFT checkpoint exists at ~/.cache/nanochat/chatsft_checkpoints/d34/

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export HF_REPO="pankajmathur/nanochat-d34-finetuned"

echo "=========================================="
echo "Quantizing and uploading nanochat-d34 8-bit model"
echo "Target: https://huggingface.co/${HF_REPO}"
echo "=========================================="

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Ensure dependencies
pip install -q huggingface_hub torch

# Run the quantization and upload
python3 << 'PYTHON_SCRIPT'
import os
import sys
import json
import platform
import torch
import torch.nn as nn
from pathlib import Path

# Add nanochat to path
sys.path.insert(0, os.getcwd())

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import get_base_dir

# Setup
base_dir = os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))
repo_id = os.environ.get("HF_REPO", "pankajmathur/nanochat-d34-finetuned")

# Paths
sft_checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", "d34")
tokenizer_dir = os.path.join(base_dir, "tokenizer")
quantized_dir = os.path.join(base_dir, "chatsft_checkpoints_int8", "d34")

print(f"Source SFT checkpoint: {sft_checkpoint_dir}")
print(f"Tokenizer: {tokenizer_dir}")
print(f"Quantized output: {quantized_dir}")
print()

# Find the latest checkpoint
import glob
checkpoint_files = glob.glob(os.path.join(sft_checkpoint_dir, "model_*.pt"))
if not checkpoint_files:
    print(f"❌ No checkpoint files found in {sft_checkpoint_dir}")
    sys.exit(1)

# Get the latest step
steps = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in checkpoint_files]
latest_step = max(steps)
print(f"Found checkpoint at step {latest_step}")

model_path = os.path.join(sft_checkpoint_dir, f"model_{latest_step:06d}.pt")
meta_path = os.path.join(sft_checkpoint_dir, f"meta_{latest_step:06d}.json")

# Load metadata
with open(meta_path, "r") as f:
    meta_data = json.load(f)
model_config_kwargs = meta_data["model_config"]
print(f"Model config: {model_config_kwargs}")

# Load model weights
print("Loading model weights...")
model_data = torch.load(model_path, map_location="cpu", weights_only=False)

# Convert bfloat16 to float32 (required for quantization)
model_data = {
    k: v.float() if v.dtype == torch.bfloat16 else v
    for k, v in model_data.items()
}

# Fix torch compile prefix
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

# Build the model
print("Building model...")
model_config = GPTConfig(**model_config_kwargs)
model = GPT(model_config)
model.load_state_dict(model_data, strict=True)
model.eval()

# Free the original weights
del model_data

# Set quantization backend
if platform.system() == "Darwin":  # macOS
    torch.backends.quantized.engine = 'qnnpack'
    print("Using 'qnnpack' quantization backend (macOS)")
elif platform.machine() in ('arm64', 'aarch64'):  # ARM Linux
    torch.backends.quantized.engine = 'qnnpack'
    print("Using 'qnnpack' quantization backend (ARM)")
else:  # x86 Linux
    torch.backends.quantized.engine = 'fbgemm'
    print("Using 'fbgemm' quantization backend (x86)")

# Apply int8 dynamic quantization
print("Applying int8 dynamic quantization...")
from torch.ao.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)
print("✅ Quantization complete!")

# Save quantized model
os.makedirs(quantized_dir, exist_ok=True)
quantized_model_path = os.path.join(quantized_dir, f"model_{latest_step:06d}.pt")
quantized_meta_path = os.path.join(quantized_dir, f"meta_{latest_step:06d}.json")

print(f"Saving quantized model to {quantized_model_path}...")
torch.save(quantized_model.state_dict(), quantized_model_path)

# Update metadata
meta_data["quantization"] = "int8_dynamic"
meta_data["quantization_backend"] = torch.backends.quantized.engine
with open(quantized_meta_path, "w") as f:
    json.dump(meta_data, f, indent=2)

# Calculate size reduction
original_size = os.path.getsize(model_path)
quantized_size = os.path.getsize(quantized_model_path)
print(f"Original size: {original_size / 1e9:.2f} GB")
print(f"Quantized size: {quantized_size / 1e9:.2f} GB")
print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.1f}%")

# Upload to HuggingFace
print()
print("Uploading to HuggingFace...")

from huggingface_hub import HfApi, upload_folder

api = HfApi()

# Create a staging directory with the right structure
import tempfile
import shutil

staging_dir = tempfile.mkdtemp()
print(f"Staging directory: {staging_dir}")

# Copy quantized checkpoint
quantized_hf_dir = os.path.join(staging_dir, "chatsft_checkpoints_int8", "d34")
os.makedirs(quantized_hf_dir, exist_ok=True)
shutil.copy(quantized_model_path, quantized_hf_dir)
shutil.copy(quantized_meta_path, quantized_hf_dir)
print(f"✅ Copied quantized checkpoint")

# Copy tokenizer (needed for the quantized model too)
tokenizer_hf_dir = os.path.join(staging_dir, "tokenizer")
if os.path.exists(tokenizer_dir):
    shutil.copytree(tokenizer_dir, tokenizer_hf_dir)
    print(f"✅ Copied tokenizer")

# Create a README for the quantized version
readme_content = f"""# Int8 Quantized Model

This directory contains the int8 dynamically quantized version of the SFT checkpoint.

## Details
- **Original checkpoint**: `chatsft_checkpoints/d34/model_{latest_step:06d}.pt`
- **Quantization**: int8 dynamic quantization on Linear layers
- **Backend**: {torch.backends.quantized.engine}
- **Original size**: {original_size / 1e9:.2f} GB
- **Quantized size**: {quantized_size / 1e9:.2f} GB
- **Size reduction**: {(1 - quantized_size/original_size) * 100:.1f}%

## Usage

The quantized model requires loading with the quantization-aware code.
It's primarily intended for CPU inference on memory-constrained systems.

```python
# Note: Quantized models need special handling
# Use the standard nanochat loading with --device-type cpu
python -m scripts.chat_cli --device-type cpu -p "Hello"
```

## Compatibility
- Works on: CPU (x86 with fbgemm, ARM/macOS with qnnpack)
- Does NOT work on: CUDA, MPS
"""

readme_path = os.path.join(staging_dir, "chatsft_checkpoints_int8", "README.md")
with open(readme_path, "w") as f:
    f.write(readme_content)

# Upload
print("Uploading to HuggingFace...")
api.upload_folder(
    folder_path=staging_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message=f"Add int8 quantized SFT checkpoint (step {latest_step})",
)

# Cleanup
shutil.rmtree(staging_dir)

print()
print("========================================")
print("✅ Upload complete!")
print(f"🔗 https://huggingface.co/{repo_id}")
print("========================================")
print()
print("New files uploaded:")
print(f"  - chatsft_checkpoints_int8/d34/model_{latest_step:06d}.pt")
print(f"  - chatsft_checkpoints_int8/d34/meta_{latest_step:06d}.json")
print(f"  - chatsft_checkpoints_int8/README.md")
print()
PYTHON_SCRIPT

echo ""
echo "Done!"
