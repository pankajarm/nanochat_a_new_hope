#!/bin/bash

# Download nanochat d34 finetuned model from HuggingFace
# This script downloads the SFT checkpoint and tokenizer files needed to run chat_cli
#
# Source: https://huggingface.co/pankajmathur/nanochat-d34-finetuned
#
# After running this script, you can chat with the model:
#   python -m scripts.chat_cli -p "Hello, who are you?"
#
# For low-memory systems (8GB RAM), use CPU with 8-bit quantization:
#   python -m scripts.chat_cli --device-type cpu -q 8bit -p "Hello, who are you?"
#
# Usage:
#   bash download_model.sh

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export HF_REPO="pankajmathur/nanochat-d34-finetuned"

echo "=========================================="
echo "Downloading nanochat-d34-finetuned model"
echo "Source: https://huggingface.co/${HF_REPO}"
echo "=========================================="

# Create directories
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
SFT_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d34"
mkdir -p "$TOKENIZER_DIR"
mkdir -p "$SFT_CHECKPOINT_DIR"

echo ""
echo "Target directories:"
echo "  Tokenizer: $TOKENIZER_DIR"
echo "  SFT checkpoint: $SFT_CHECKPOINT_DIR"
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

# Filter for the files we need (tokenizer and SFT checkpoint)
files_to_download = []
for f in all_files:
    # Tokenizer files
    if f.startswith("tokenizer/"):
        files_to_download.append(f)
    # SFT checkpoint files (primary - this is what chat_cli loads by default)
    elif f.startswith("chatsft_checkpoints/d34/"):
        files_to_download.append(f)

if not files_to_download:
    print("No matching files found in repository!")
    print("Available files:", all_files[:20])
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
    echo "Please make sure you have Python 3 installed and try again."
    echo "You can also manually download from: https://huggingface.co/${HF_REPO}"
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

# Check for at least one model checkpoint
if ! ls "$SFT_CHECKPOINT_DIR"/model_*.pt 1> /dev/null 2>&1; then
    echo "❌ Missing: SFT model checkpoint (model_*.pt)"
    MISSING_FILES=1
fi

if ! ls "$SFT_CHECKPOINT_DIR"/meta_*.json 1> /dev/null 2>&1; then
    echo "❌ Missing: SFT metadata (meta_*.json)"
    MISSING_FILES=1
fi

if [ $MISSING_FILES -eq 1 ]; then
    echo ""
    echo "❌ Some files are missing. Please try running the script again or download manually from:"
    echo "   https://huggingface.co/${HF_REPO}"
    exit 1
fi

echo "✅ All required files present!"
echo ""
echo "Downloaded files:"
ls -la "$TOKENIZER_DIR"
echo ""
ls -la "$SFT_CHECKPOINT_DIR"

echo ""
echo "=========================================="
echo "✅ Download complete!"
echo "=========================================="
echo ""
echo "You can now chat with the model:"
echo ""
echo "  # Standard (uses MPS on Mac, CUDA on Linux)"
echo "  python -m scripts.chat_cli -p \"Hello, who are you?\""
echo ""
echo "  # For low-memory systems (8GB RAM), use CPU with 8-bit quantization:"
echo "  python -m scripts.chat_cli --device-type cpu -q 8bit -p \"Hello, who are you?\""
echo ""
echo "  # Interactive chat mode:"
echo "  python -m scripts.chat_cli"
echo ""
echo "  # Web interface:"
echo "  python -m scripts.chat_web"
echo ""
