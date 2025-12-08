#!/bin/bash

# Download nanochat d34 INT8 quantized model from HuggingFace
# This is a smaller (~2.2GB vs ~8.8GB) pre-quantized version for CPU inference
#
# Source: https://huggingface.co/pankajmathur/nanochat-d34-finetuned
#
# After running this script, you can chat with the model:
#   python -m scripts.chat_cli --device-type cpu -p "Hello, who are you?"
#
# Note: The int8 model only works on CPU (not CUDA or MPS)
#
# Usage:
#   bash download_model_d34_int8.sh

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export HF_REPO="pankajmathur/nanochat-d34-finetuned"

echo "=========================================="
echo "Downloading nanochat-d34 INT8 quantized model"
echo "Source: https://huggingface.co/${HF_REPO}"
echo "=========================================="
echo ""
echo "This downloads the pre-quantized int8 model (~2.2GB)"
echo "For the full precision model (~8.8GB), use download_model.sh instead"
echo ""

# Create directories
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
INT8_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints_int8/d34"
mkdir -p "$TOKENIZER_DIR"
mkdir -p "$INT8_CHECKPOINT_DIR"

echo "Target directories:"
echo "  Tokenizer: $TOKENIZER_DIR"
echo "  Int8 checkpoint: $INT8_CHECKPOINT_DIR"
echo ""

# Use Python with huggingface_hub for reliable downloads
echo "Downloading model files using huggingface_hub..."

python3 << 'PYTHON_SCRIPT'
import os
import sys

# Install huggingface_hub if needed
try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download, list_repo_files

repo_id = os.environ.get("HF_REPO", "pankajmathur/nanochat-d34-finetuned")
base_dir = os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))

print(f"Repository: {repo_id}")
print(f"Target directory: {base_dir}")
print()

# Get list of all files in the repo
print("Fetching file list from HuggingFace...")
try:
    all_files = list_repo_files(repo_id)
except Exception as e:
    print(f"Error fetching file list: {e}")
    sys.exit(1)

# Filter for the files we need (tokenizer and INT8 checkpoint)
files_to_download = []
for f in all_files:
    # Tokenizer files (required)
    if f.startswith("tokenizer/"):
        files_to_download.append(f)
    # INT8 quantized checkpoint files
    elif f.startswith("chatsft_checkpoints_int8/d34/"):
        files_to_download.append(f)

if not files_to_download:
    print("❌ No int8 quantized checkpoint found in repository!")
    print("Available files:", [f for f in all_files if "int8" in f.lower() or "quant" in f.lower()][:10])
    print()
    print("The int8 model may not have been uploaded yet.")
    print("You can either:")
    print("  1. Use the full precision model: bash download_model.sh")
    print("  2. Run local quantization: python -m scripts.chat_cli --device-type cpu -q 8bit")
    sys.exit(1)

# Check if we found the int8 checkpoint
int8_files = [f for f in files_to_download if "int8" in f]
if not int8_files:
    print("❌ Int8 checkpoint not found. The quantized model may not have been uploaded yet.")
    print("Use download_model.sh to get the full precision model instead.")
    sys.exit(1)

print(f"Found {len(files_to_download)} files to download:")
for f in files_to_download:
    print(f"  - {f}")
print()

# Download each file
for filepath in files_to_download:
    local_path = os.path.join(base_dir, filepath)
    
    # Check if file already exists
    if os.path.exists(local_path):
        print(f"⏭️  {filepath} already exists, skipping")
        continue
    
    print(f"Downloading {filepath}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filepath,
            local_dir=base_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ Downloaded {filepath}")
    except Exception as e:
        print(f"❌ Error downloading {filepath}: {e}")
        sys.exit(1)

print()
print("✅ All files downloaded successfully!")
PYTHON_SCRIPT

# Check if Python download succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Download failed!"
    echo "The int8 quantized model may not be available yet."
    echo ""
    echo "Alternatives:"
    echo "  1. Download full model: bash download_model.sh"
    echo "  2. Use on-the-fly quantization:"
    echo "     python -m scripts.chat_cli --device-type cpu -q 8bit -p \"Hello\""
    exit 1
fi

# Verify downloads
echo ""
echo "Verifying downloads..."

MISSING_FILES=0

if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ]; then
    echo "❌ Missing: tokenizer.pkl"
    MISSING_FILES=1
fi

if [ ! -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    echo "❌ Missing: token_bytes.pt"
    MISSING_FILES=1
fi

# Check for int8 model checkpoint
if ! ls "$INT8_CHECKPOINT_DIR"/model_*.pt 1> /dev/null 2>&1; then
    echo "❌ Missing: Int8 model checkpoint (model_*.pt)"
    MISSING_FILES=1
fi

if ! ls "$INT8_CHECKPOINT_DIR"/meta_*.json 1> /dev/null 2>&1; then
    echo "❌ Missing: Int8 metadata (meta_*.json)"
    MISSING_FILES=1
fi

if [ $MISSING_FILES -eq 1 ]; then
    echo ""
    echo "❌ Some files are missing."
    echo "The int8 quantized model may not be available on HuggingFace yet."
    echo ""
    echo "Alternatives:"
    echo "  1. Download full model: bash download_model.sh"
    echo "  2. Use on-the-fly quantization:"
    echo "     python -m scripts.chat_cli --device-type cpu -q 8bit -p \"Hello\""
    exit 1
fi

echo "✅ All required files present!"
echo ""
echo "Downloaded files:"
ls -lh "$TOKENIZER_DIR"
echo ""
ls -lh "$INT8_CHECKPOINT_DIR"

echo ""
echo "=========================================="
echo "✅ Download complete!"
echo "=========================================="
echo ""
echo "Model size: $(du -sh "$INT8_CHECKPOINT_DIR" | cut -f1)"
echo ""
echo "You can now chat with the int8 model (CPU only):"
echo ""
echo "  # Single prompt"
echo "  python -m scripts.chat_cli --device-type cpu --source sft_int8 -p \"Hello, who are you?\""
echo ""
echo "  # Interactive chat"
echo "  python -m scripts.chat_cli --device-type cpu --source sft_int8"
echo ""
echo "  # Web interface"
echo "  python -m scripts.chat_web --device-type cpu --source sft_int8"
echo ""
echo "Note: Int8 models only work on CPU, not CUDA or MPS."
echo ""
