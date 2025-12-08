#!/bin/bash

# Upload nanochat d34 finetuned model artifacts to HuggingFace
# This script uploads all training artifacts from run_d34_finetune.sh and rl_d34_finetune.sh
#
# Prerequisites:
# - HuggingFace CLI logged in (huggingface-cli login) or HF_TOKEN environment variable set
# - Completed training runs (run_d34_finetune.sh and optionally rl_d34_finetune.sh)
#
# Usage:
#   bash upload_to_hf.sh
#   
# With custom repo name:
#   HF_REPO_NAME=my-custom-name bash upload_to_hf.sh

set -e

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

# HuggingFace settings
HF_USERNAME="pankajmathur"
HF_REPO_NAME="${HF_REPO_NAME:-nanochat-d34-finetuned}"
HF_REPO_ID="${HF_USERNAME}/${HF_REPO_NAME}"

echo "=========================================="
echo "Uploading nanochat d34 artifacts to HuggingFace"
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
# Collect all artifacts

echo ""
echo "Collecting artifacts..."

# 1. Tokenizer files
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
if [ -d "$TOKENIZER_DIR" ]; then
    mkdir -p "$UPLOAD_DIR/tokenizer"
    cp -v "$TOKENIZER_DIR/tokenizer.pkl" "$UPLOAD_DIR/tokenizer/" 2>/dev/null || true
    cp -v "$TOKENIZER_DIR/token_bytes.pt" "$UPLOAD_DIR/tokenizer/" 2>/dev/null || true
    echo "✅ Tokenizer files collected"
else
    echo "⚠️  Tokenizer directory not found: $TOKENIZER_DIR"
fi

# 2. Mid-training checkpoint
MID_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/mid_checkpoints/d34"
if [ -d "$MID_CHECKPOINT_DIR" ]; then
    mkdir -p "$UPLOAD_DIR/mid_checkpoints/d34"
    cp -v "$MID_CHECKPOINT_DIR"/*.pt "$UPLOAD_DIR/mid_checkpoints/d34/" 2>/dev/null || true
    cp -v "$MID_CHECKPOINT_DIR"/*.json "$UPLOAD_DIR/mid_checkpoints/d34/" 2>/dev/null || true
    echo "✅ Mid-training checkpoint collected"
else
    echo "⚠️  Mid-training checkpoint not found: $MID_CHECKPOINT_DIR"
fi

# 3. SFT checkpoint
SFT_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d34"
if [ -d "$SFT_CHECKPOINT_DIR" ]; then
    mkdir -p "$UPLOAD_DIR/chatsft_checkpoints/d34"
    cp -v "$SFT_CHECKPOINT_DIR"/*.pt "$UPLOAD_DIR/chatsft_checkpoints/d34/" 2>/dev/null || true
    cp -v "$SFT_CHECKPOINT_DIR"/*.json "$UPLOAD_DIR/chatsft_checkpoints/d34/" 2>/dev/null || true
    echo "✅ SFT checkpoint collected"
else
    echo "⚠️  SFT checkpoint not found: $SFT_CHECKPOINT_DIR"
fi

# 4. RL checkpoint (if exists)
RL_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/chatrl_checkpoints/d34"
if [ -d "$RL_CHECKPOINT_DIR" ]; then
    mkdir -p "$UPLOAD_DIR/chatrl_checkpoints/d34"
    cp -v "$RL_CHECKPOINT_DIR"/*.pt "$UPLOAD_DIR/chatrl_checkpoints/d34/" 2>/dev/null || true
    cp -v "$RL_CHECKPOINT_DIR"/*.json "$UPLOAD_DIR/chatrl_checkpoints/d34/" 2>/dev/null || true
    echo "✅ RL checkpoint collected"
else
    echo "ℹ️  RL checkpoint not found (optional): $RL_CHECKPOINT_DIR"
fi

# 5. Report files
REPORT_DIR="$NANOCHAT_BASE_DIR/report"
if [ -d "$REPORT_DIR" ]; then
    mkdir -p "$UPLOAD_DIR/report"
    cp -v "$REPORT_DIR"/* "$UPLOAD_DIR/report/" 2>/dev/null || true
    echo "✅ Report files collected"
else
    echo "⚠️  Report directory not found: $REPORT_DIR"
fi

# 6. Log files (look for common log file patterns)
mkdir -p "$UPLOAD_DIR/logs"
for logfile in d34_finetune.log d34_rl.log d34.log; do
    if [ -f "$logfile" ]; then
        cp -v "$logfile" "$UPLOAD_DIR/logs/"
        echo "✅ Log file collected: $logfile"
    fi
done

# Also check parent directory for logs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for logfile in "$SCRIPT_DIR"/*.log; do
    if [ -f "$logfile" ]; then
        cp -v "$logfile" "$UPLOAD_DIR/logs/" 2>/dev/null || true
    fi
done

# -----------------------------------------------------------------------------
# Create README.md (Model Card)

echo ""
echo "Creating README.md (Model Card)..."

# Try to extract metrics from report if available
METRICS_INFO=""
if [ -f "$UPLOAD_DIR/report/report.md" ]; then
    METRICS_INFO=$(cat "$UPLOAD_DIR/report/report.md")
fi

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
- finetuned
base_model: karpathy/nanochat-d34
pipeline_tag: text-generation
---

# nanochat-d34-finetuned

This is a finetuned version of [karpathy/nanochat-d34](https://huggingface.co/karpathy/nanochat-d34), trained using the [nanochat](https://github.com/karpathy/nanochat) framework.

## Model Description

- **Base Model**: karpathy/nanochat-d34 (2.2B parameters, pre-trained on 88B tokens)
- **Architecture**: GPT-style transformer with depth=34
- **Training Pipeline**: Mid-training → SFT → RL (optional)
- **Hardware**: 8x A100-80GB GPUs

## Training Details

### Base Model (Pre-trained by Karpathy)
- Parameters: 2,217,082,880
- Training tokens: 88,683,315,200 (40x param:token ratio)
- Max sequence length: 2048
- Base CORE score: 0.3382

### Fine-tuning Pipeline
1. **Mid-training**: General instruction tuning on SmolTalk, MMLU, GSM8K, Spelling tasks
2. **SFT (Supervised Fine-Tuning)**: Chat-specific training on ARC, GSM8K, SmolTalk
3. **RL (Reinforcement Learning)**: Optional GRPO-style training on GSM8K (if included)

## Repository Structure

```
├── tokenizer/
│   ├── tokenizer.pkl          # Tokenizer
│   └── token_bytes.pt         # Token byte mappings
├── mid_checkpoints/d34/       # Mid-training checkpoint
│   ├── model_*.pt
│   └── meta_*.json
├── chatsft_checkpoints/d34/   # SFT checkpoint
│   ├── model_*.pt
│   └── meta_*.json
├── chatrl_checkpoints/d34/    # RL checkpoint (if available)
│   ├── model_*.pt
│   └── meta_*.json
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

# Download this model
# Place files in ~/.cache/nanochat/ following the directory structure above

# Chat with the model
python -m scripts.chat_cli -p "Hello, how are you?"
python -m scripts.chat_web
```

### Loading checkpoint manually

```python
import torch
from huggingface_hub import hf_hub_download

# Download SFT checkpoint
model_path = hf_hub_download(
    repo_id="pankajmathur/nanochat-d34-finetuned",
    filename="chatsft_checkpoints/d34/model_XXXXXX.pt"  # Replace with actual filename
)
meta_path = hf_hub_download(
    repo_id="pankajmathur/nanochat-d34-finetuned",
    filename="chatsft_checkpoints/d34/meta_XXXXXX.json"
)

# Load weights
state_dict = torch.load(model_path, map_location="cpu")
```

## Training Scripts

This model was trained using:
- `run_d34_finetune.sh` - Downloads base model, runs mid-training and SFT
- `rl_d34_finetune.sh` - Continues with RL training (optional)

## License

MIT License (same as nanochat)

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) for the nanochat framework and pre-trained base model
- The nanochat community

EOF

# Append metrics if available
if [ -n "$METRICS_INFO" ]; then
    cat >> "$UPLOAD_DIR/README.md" << EOF

## Evaluation Results

<details>
<summary>Click to expand evaluation report</summary>

$METRICS_INFO

</details>
EOF
fi

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
    commit_message="Upload nanochat d34 finetuned model artifacts",
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
echo "✅ Upload complete!"
echo "🔗 https://huggingface.co/${HF_REPO_ID}"
echo "=========================================="
