"""
Utilities for saving and loading model/optim/state checkpoints.

Memory optimization for inference:
- CUDA: Uses bfloat16 (native, fastest)
- MPS (Mac): Uses float16 (~4.4GB for d34 model). Requires Mac with 16GB+ unified memory for d34.
- CPU: Uses float32 by default, or int8 with quantization (~2.2GB for d34)

Environment variables:
- NANOCHAT_QUANTIZE: Set to "8bit" for int8 dynamic quantization on CPU (reduces memory ~4x)

Note on MPS memory limits:
- MPS has a default memory limit based on system RAM
- For large models, use a smaller model (d12, d20) or use CPU with 8bit quantization
- Do NOT use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 as it can crash the system
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)


def quantize_model_int8(model):
    """
    Apply dynamic int8 quantization to Linear layers (CPU only).
    Reduces memory by ~4x compared to float32.
    """
    try:
        import platform
        from torch.ao.quantization import quantize_dynamic
        
        # Set the quantization backend based on platform
        # macOS needs 'qnnpack', Linux typically uses 'fbgemm' (x86) or 'qnnpack' (ARM)
        if platform.system() == "Darwin":  # macOS
            torch.backends.quantized.engine = 'qnnpack'
            log0("Using 'qnnpack' quantization backend (macOS)")
        elif platform.machine() in ('arm64', 'aarch64'):  # ARM Linux
            torch.backends.quantized.engine = 'qnnpack'
            log0("Using 'qnnpack' quantization backend (ARM)")
        else:  # x86 Linux
            torch.backends.quantized.engine = 'fbgemm'
            log0("Using 'fbgemm' quantization backend (x86)")
        
        log0("Applying int8 dynamic quantization to Linear layers...")
        model = quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize Linear layers
            dtype=torch.qint8
        )
        log0("✅ Int8 quantization applied successfully")
        return model
    except Exception as e:
        log0(f"⚠️ Quantization failed, using original model: {e}")
        return model

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase, quantize=None):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        step: Checkpoint step number
        device: Target device (cpu, cuda, mps)
        phase: "train" or "eval"
        quantize: Quantization mode - None, "8bit", or "4bit" (4bit requires bitsandbytes)
                  Can also be set via NANOCHAT_QUANTIZE environment variable
    
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    
    # Check environment variable for quantization if not explicitly set
    if quantize is None:
        quantize = os.environ.get("NANOCHAT_QUANTIZE", None)
    
    # Determine if we should use quantization (only for eval on CPU)
    use_quantization = quantize is not None and phase == "eval" and device.type == "cpu"
    if quantize and device.type != "cpu":
        log0(f"⚠️ Quantization '{quantize}' requested but only supported on CPU. Using {device.type} without quantization.")
        use_quantization = False
    if quantize and phase != "eval":
        log0(f"⚠️ Quantization only supported for eval phase, not training.")
        use_quantization = False
    
    # For MPS or CPU with quantization, load to CPU first
    load_device = "cpu" if device.type == "mps" or use_quantization else device
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, load_device, load_optimizer=False)
    
    if device.type == "cpu" or use_quantization:
        # Convert bfloat16 tensors to float32 for CPU inference (required for quantization)
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    elif device.type == "mps":
        # Convert bfloat16 tensors to float16 for MPS (saves memory vs float32)
        # MPS doesn't support bfloat16 or float8, float16 is the smallest supported type
        # Convert on CPU first, then move to MPS to avoid memory issues
        log0("MPS detected: Converting model to float16 (smallest supported type on MPS)")
        try:
            model_data = {
                k: v.half().to(device) if v.dtype == torch.bfloat16 else v.to(device)
                for k, v in model_data.items()
            }
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "MPS backend" in str(e):
                log0("❌ MPS out of memory! The model is too large for your Mac's GPU memory.")
                log0("   Options:")
                log0("   1. Use a smaller model (d12 or d20 instead of d34)")
                log0("   2. Use CPU with 8-bit quantization: --device-type cpu -q 8bit")
                log0("   3. Close other applications to free memory")
            raise
    
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    
    # For quantization, build model on CPU first
    build_device = "cpu" if use_quantization else device
    with torch.device("meta"):
        model = GPT(model_config)
    
    # Load the model state
    model.to_empty(device=build_device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    
    # Free the model_data dict to reclaim memory
    del model_data
    
    # Apply quantization if requested (CPU only, eval only)
    if use_quantization:
        if quantize == "8bit":
            model = quantize_model_int8(model)
        elif quantize == "4bit":
            log0("⚠️ 4-bit quantization requires bitsandbytes library (pip install bitsandbytes)")
            log0("   4-bit is primarily for CUDA. For CPU, use 8bit instead.")
            # For now, fall back to 8bit for CPU
            model = quantize_model_int8(model)
        else:
            log0(f"⚠️ Unknown quantization mode '{quantize}', using 8bit")
            model = quantize_model_int8(model)
    
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None, quantize=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase, quantize=quantize)
    return model, tokenizer, meta_data

def load_model(source, device, phase, model_tag=None, step=None, quantize=None):
    """
    Load a model from the standard nanochat checkpoint directories.
    
    Args:
        source: One of "base", "mid", "sft", "rl"
        device: Target device (cpu, cuda, mps)
        phase: "train" or "eval"
        model_tag: Optional model tag (e.g., "d34"), auto-detected if None
        step: Optional checkpoint step, uses latest if None
        quantize: Quantization mode - None, "8bit", or "4bit"
                  Can also be set via NANOCHAT_QUANTIZE environment variable
    """
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, device, phase, model_tag=model_tag, step=step, quantize=quantize)
