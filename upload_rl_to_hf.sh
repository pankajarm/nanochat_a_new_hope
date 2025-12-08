#!/bin/bash

# Upload nanochat d34 RL-trained model artifacts to HuggingFace
# This script uploads RL checkpoint from rl_d34_finetune.sh training
#
# Prerequisites:
# - HuggingFace CLI logged in (huggingface-cli login) or HF_TOKEN environment variable set
# - Completed RL training run (rl_d34_finetune.sh)
#
# Usage:
#   bash upload_rl_to_hf.sh
#   
# With custom repo name:
#   HF_REPO_NAME=my-custom-name bash upload_rl_to_hf.sh

set -e

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

# HuggingFace settings
HF_USERNAME="pankajmathur"
HF_REPO_NAME="${HF_REPO_NAME:-nanochat-d34-rl}"
HF_REPO_ID="${HF_USERNAME}/${HF_REPO_NAME}"

echo "=========================================="
echo "Uploading nanochat d34 RL model to HuggingFace"
echo "Repository: https://huggingface.co/${HF_REPO_ID}"
echo "=========================================="

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Ensure huggingface_hub is installed
pip install -q huggingface_hub

# Create upload staging directory
UPLOAD_DIR=$(mktemp -d)
echo "Staging directory: $UPLOAD_DIR"

# -----------------------------------------------------------------------------
# Collect RL artifacts

echo ""
echo "Collecting RL artifacts..."

# 1. Tokenizer files (required for inference)
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
if [ -d "$TOKENIZER_DIR" ]; then
    mkdir -p "$UPLOAD_DIR/tokenizer"
    cp -v "$TOKENIZER_DIR/tokenizer.pkl" "$UPLOAD_DIR/tokenizer/" 2>/dev/null || true
    cp -v "$TOKENIZER_DIR/token_bytes.pt" "$UPLOAD_DIR/tokenizer/" 2>/dev/null || true
    echo "✅ Tokenizer files collected"
else
    echo "⚠️  Tokenizer directory not found: $TOKENIZER_DIR"
fi

# 2. RL checkpoint (main artifact)
RL_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/chatrl_checkpoints/d34"
if [ -d "$RL_CHECKPOINT_DIR" ]; then
    mkdir -p "$UPLOAD_DIR/chatrl_checkpoints/d34"
    cp -v "$RL_CHECKPOINT_DIR"/*.pt "$UPLOAD_DIR/chatrl_checkpoints/d34/" 2>/dev/null || true
    cp -v "$RL_CHECKPOINT_DIR"/*.json "$UPLOAD_DIR/chatrl_checkpoints/d34/" 2>/dev/null || true
    echo "✅ RL checkpoint collected"
else
    echo "❌ RL checkpoint not found: $RL_CHECKPOINT_DIR"
    echo "Please run rl_d34_finetune.sh first!"
    rm -rf "$UPLOAD_DIR"
    exit 1
fi

# 3. Report files
REPORT_DIR="$NANOCHAT_BASE_DIR/report"
if [ -d "$REPORT_DIR" ]; then
    mkdir -p "$UPLOAD_DIR/report"
    cp -v "$REPORT_DIR"/* "$UPLOAD_DIR/report/" 2>/dev/null || true
    echo "✅ Report files collected"
else
    echo "⚠️  Report directory not found: $REPORT_DIR"
fi

# 4. Log files
mkdir -p "$UPLOAD_DIR/logs"
for logfile in d34_rl.log rl_d34.log; do
    if [ -f "$logfile" ]; then
        cp -v "$logfile" "$UPLOAD_DIR/logs/"
        echo "✅ Log file collected: $logfile"
    fi
done

# Also check parent directory for logs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for logfile in "$SCRIPT_DIR"/*rl*.log; do
    if [ -f "$logfile" ]; then
        cp -v "$logfile" "$UPLOAD_DIR/logs/" 2>/dev/null || true
    fi
done

# -----------------------------------------------------------------------------
# Create README.md (Model Card)

echo ""
echo "Creating README.md (Model Card)..."

cat > "$UPLOAD_DIR/README.md" << 'EOF'
---
license: mit
language:
- en
tags:
- nanochat
- gpt
- text-generation
- conversational
- rl
- grpo
- gsm8k
- math
- reinforcement-learning
base_model: pankajmathur/nanochat-d34-finetuned
datasets:
- HuggingFaceTB/smol-smoltalk
- openai/gsm8k
pipeline_tag: text-generation
---

# nanochat-d34-rl

This is an **RL-trained** version of [nanochat-d34](https://huggingface.co/karpathy/nanochat-d34), fine-tuned using GRPO (Group Relative Policy Optimization) on GSM8K math problems.

## Model Description

- **Base Model**: [karpathy/nanochat-d34](https://huggingface.co/karpathy/nanochat-d34) (2.2B parameters)
- **SFT Model**: [pankajmathur/nanochat-d34-finetuned](https://huggingface.co/pankajmathur/nanochat-d34-finetuned)
- **Architecture**: GPT-style transformer with depth=34
- **Training Pipeline**: Pre-training → Mid-training → SFT → **RL (GRPO)**
- **Hardware**: 8x NVIDIA A100-SXM4-80GB GPUs

## 🎯 Key Achievement: GSM8K +73.6% Improvement

The RL training significantly boosted math reasoning capabilities while maintaining general performance:

| Metric | MID | SFT | RL | Change (SFT→RL) |
|-----------------|--------|--------|----------|-----------------|
| **GSM8K** | 0.1137 | 0.1327 | **0.2305** | **+73.6%** 🚀 |
| ARC-Easy | 0.6961 | 0.7210 | 0.7130 | -1.1% |
| ARC-Challenge | 0.5367 | 0.5418 | 0.5375 | -0.8% |
| MMLU | 0.4229 | 0.4304 | 0.4256 | -1.1% |
| HumanEval | 0.1098 | 0.1037 | 0.0671 | -35.3% |
| SpellingBee | - | - | 0.9922 | N/A |
| **ChatCORE** | 0.4045 | 0.4157 | **0.4208** | **+1.2%** |

## Training Details

### RL Configuration (GRPO)
- **Run**: d34_rl
- **Source**: SFT checkpoint
- **dtype**: bfloat16
- **device_batch_size**: 4
- **examples_per_step**: 16
- **num_samples**: 16
- **max_new_tokens**: 256
- **temperature**: 1.0
- **top_k**: 50
- **Learning Rates**:
  - unembedding_lr: 0.0040
  - embedding_lr: 0.2000
  - matrix_lr: 0.0200
- **weight_decay**: 0.0
- **num_epochs**: 1
- **Total Steps**: 467

### Training Metrics (Final)
- **Pass@1**: 0.2300
- **Pass@2**: 0.2750
- **Pass@3**: 0.3275
- **Pass@4**: 0.3675
- **Average Reward**: ~0.28
- **Average Sequence Length**: ~178 tokens

## Repository Structure

```
├── tokenizer/
│   ├── tokenizer.pkl          # Tokenizer
│   └── token_bytes.pt         # Token byte mappings
├── chatrl_checkpoints/d34/    # RL checkpoint
│   ├── model_000466.pt        # Final model weights
│   └── meta_000466.json       # Training metadata
├── report/                    # Evaluation reports
│   └── report.md
└── logs/                      # Training logs
```

## Usage

### With nanochat framework

```bash
# Clone nanochat
git clone https://github.com/karpathy/nanochat
cd nanochat

# Download RL checkpoint to cache
mkdir -p ~/.cache/nanochat/chatrl_checkpoints/d34
# Place model_*.pt and meta_*.json files there

# Chat with the RL model
python -m scripts.chat_cli --source rl -p "What is 25 * 17?"
python -m scripts.chat_web --source rl
```

### Loading checkpoint manually

```python
import torch
from huggingface_hub import hf_hub_download

# Download RL checkpoint
model_path = hf_hub_download(
    repo_id="pankajmathur/nanochat-d34-rl",
    filename="chatrl_checkpoints/d34/model_000466.pt"
)
meta_path = hf_hub_download(
    repo_id="pankajmathur/nanochat-d34-rl",
    filename="chatrl_checkpoints/d34/meta_000466.json"
)

# Load weights
state_dict = torch.load(model_path, map_location="cpu")
```

### Example: Math Problem Solving

```python
# The RL model excels at GSM8K-style math problems
prompt = """Solve this step by step:
Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning 
and bakes muffins for her friends every day with four. She sells the remainder 
at the farmers' market daily for $2 per fresh duck egg. How much in dollars 
does she make every day at the farmers' market?"""
```

## Training Scripts

This model was trained using:
- `run_d34_finetune.sh` - Downloads base model, runs mid-training and SFT
- `rl_d34_finetune.sh` - RL training with GRPO on GSM8K

## WandB Training Run

Training tracked at: [wandb.ai/sage-ai/nanochat-rl](https://wandb.ai/sage-ai/nanochat-rl/runs/1y1vk7kw)

## Related Models

- **Base**: [karpathy/nanochat-d34](https://huggingface.co/karpathy/nanochat-d34) - Pre-trained base model
- **SFT**: [pankajmathur/nanochat-d34-finetuned](https://huggingface.co/pankajmathur/nanochat-d34-finetuned) - Mid-training + SFT checkpoint

## License

MIT License (same as nanochat)

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) for the nanochat framework and pre-trained base model
- The nanochat community

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```
EOF

echo "✅ README.md created"

# -----------------------------------------------------------------------------
# Upload to HuggingFace

echo ""
echo "Uploading to HuggingFace..."

python << PYTHON_SCRIPT
import os
from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = "${HF_REPO_ID}"
upload_dir = "${UPLOAD_DIR}"

# Create repository if it doesn't exist
try:
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"✅ Repository created/verified: {repo_id}")
except Exception as e:
    print(f"Repository creation note: {e}")

# Upload entire folder
print(f"Uploading files from {upload_dir}...")
api.upload_folder(
    folder_path=upload_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload nanochat d34 RL-trained model (GRPO on GSM8K)",
)

print(f"✅ Upload complete!")
print(f"🔗 View at: https://huggingface.co/{repo_id}")
PYTHON_SCRIPT

# -----------------------------------------------------------------------------
# Cleanup

echo ""
echo "Cleaning up staging directory..."
rm -rf "$UPLOAD_DIR"

echo ""
echo "=========================================="
echo "✅ RL model upload complete!"
echo "🔗 https://huggingface.co/${HF_REPO_ID}"
echo "=========================================="
echo ""
echo "To update the SFT repo with RL results, run:"
echo "  bash upload_to_hf.sh"
