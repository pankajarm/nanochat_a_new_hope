#!/bin/bash

# Convert nanochat SFT checkpoint to HuggingFace transformers format
#
# This script:
# 1. Downloads the SFT checkpoint from HuggingFace if not present
# 2. Converts it to HuggingFace transformers format
#
# Prerequisites:
# - Python 3.10+
# - transformers library with NanoChat support:
#   pip install git+https://github.com/huggingface/transformers.git
# - huggingface_hub library:
#   pip install huggingface_hub
#
# Usage:
#   bash convert_sft_to_hf.sh
#   bash convert_sft_to_hf.sh ./custom_output_dir
#
# Source: https://huggingface.co/pankajmathur/nanochat-d34-finetuned

set -e

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export HF_REPO="${HF_REPO:-pankajmathur/nanochat-d34-finetuned}"

# Output directory (can be overridden by first argument)
OUTPUT_DIR="${1:-./nanochat-d34-sft-hf}"

echo "=========================================="
echo "Converting nanochat SFT to HuggingFace format"
echo "=========================================="
echo ""
echo "Source HF repo: ${HF_REPO}"
echo "Local cache: ${NANOCHAT_BASE_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Step 1: Check if checkpoint files exist, download if needed
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
SFT_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d34"

NEED_DOWNLOAD=0
if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ]; then
    echo "⚠️ Tokenizer not found at $TOKENIZER_DIR"
    NEED_DOWNLOAD=1
fi

if ! ls "$SFT_CHECKPOINT_DIR"/model_*.pt 1> /dev/null 2>&1; then
    echo "⚠️ SFT checkpoint not found at $SFT_CHECKPOINT_DIR"
    NEED_DOWNLOAD=1
fi

if [ $NEED_DOWNLOAD -eq 1 ]; then
    echo ""
    echo "Downloading checkpoint files from HuggingFace..."
    bash "$(dirname "$0")/download_model.sh"
    echo ""
fi

# Step 2: Verify files exist
echo "Verifying checkpoint files..."

if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ]; then
    echo "❌ Tokenizer not found at $TOKENIZER_DIR/tokenizer.pkl"
    echo "Please run: bash download_model.sh"
    exit 1
fi

if ! ls "$SFT_CHECKPOINT_DIR"/model_*.pt 1> /dev/null 2>&1; then
    echo "❌ SFT checkpoint not found at $SFT_CHECKPOINT_DIR"
    echo "Please run: bash download_model.sh"
    exit 1
fi

if ! ls "$SFT_CHECKPOINT_DIR"/meta_*.json 1> /dev/null 2>&1; then
    echo "❌ SFT metadata not found at $SFT_CHECKPOINT_DIR"
    echo "Please run: bash download_model.sh"
    exit 1
fi

echo "✅ All checkpoint files found"
echo ""

# Step 3: Check for transformers with NanoChat support
echo "Checking transformers library..."
python3 -c "from transformers import NanoChatConfig, NanoChatForCausalLM; print('✅ transformers with NanoChat support found')" 2>/dev/null || {
    echo "❌ transformers library with NanoChat support not found"
    echo ""
    echo "Please install the latest transformers with NanoChat support:"
    echo "  pip install git+https://github.com/huggingface/transformers.git"
    echo ""
    exit 1
}

# Step 4: Run conversion
echo ""
echo "Starting conversion..."
echo ""

python3 -m scripts.convert_to_hf \
    --source sft \
    --tokenizer_dir "$TOKENIZER_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --safe_serialization

echo ""
echo "=========================================="
echo "✅ Conversion complete!"
echo "=========================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "You can now use the model with transformers:"
echo ""
echo "  from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "  model = AutoModelForCausalLM.from_pretrained('$OUTPUT_DIR')"
echo "  tokenizer = AutoTokenizer.from_pretrained('$OUTPUT_DIR')"
echo ""
echo "Or upload to HuggingFace Hub:"
echo ""
echo "  huggingface-cli login"
echo "  huggingface-cli upload your-username/nanochat-d34-sft $OUTPUT_DIR"
echo ""
