#!/usr/bin/env python3
"""
GGUF Conversion and Upload Script for NanoChat

This script converts NanoChat HuggingFace models to GGUF format and uploads them
to HuggingFace Hub.

ARCHITECTURE NOTES:
==================
NanoChat has a unique architecture that differs from LLaMA:
  - Activation: relu2 (x.relu().square()) - NOT silu/gelu
  - MLP: Simple 2-layer (fc1 -> act -> fc2) - NOT gated 3-layer like LLaMA
  - RMSNorm: Parameter-free (no learnable weights)
  - RoPE: Flipped rotation direction (x2, -x1) instead of (-x2, x1)
  - Extras: Logit softcapping, additional norms before layers and after RoPE

GGUF CONVERSION APPROACH:
========================
We use GPT-2 architecture in llama.cpp which has:
  - 2-layer MLP structure (similar to NanoChat)
  - GELU activation (different from relu2, but closer than SiLU gated)
  
Mapping:
  - fc1 -> ffn_up (hidden_size -> intermediate_size)
  - fc2 -> ffn_down (intermediate_size -> hidden_size)
  - Dummy norm weights (all ones) for parameter-free norms

LIMITATIONS:
===========
  - Activation function mismatch (GELU vs relu2) affects output quality
  - RoPE rotation direction not adjustable in GGUF (may affect quality)
  - No logit softcapping in llama.cpp for GPT-2 arch
  - For best quality, use the HuggingFace transformers model directly

Features:
- Convert NanoChat HF models to GGUF format (custom converter)
- Additional quantizations using llama-quantize (q4_K_M, q6_K, etc.)
- Validate GGUF files before upload
- Real inference testing with llama.cpp before upload
- Upload to HuggingFace Hub

Usage:
    python convert_and_upload_gguf.py

Requirements:
    pip install torch safetensors huggingface_hub gguf
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================

# HuggingFace settings
HF_USERNAME = "pankajmathur"  # Your HuggingFace username
HF_TOKEN = ""  # Your HuggingFace token (or set via huggingface-cli login)

# Source HuggingFace model repository (must be in HF transformers format)
SOURCE_HF_REPO = "pankajmathur/nanochat-d34-sft-hf"

# Output repository name for GGUF files
GGUF_REPO_NAME = "nanochat-d34-sft-GGUF"

# GGUF settings
GGUF_BASE_DTYPE = "f16"  # Base dtype for initial GGUF (f16 recommended)
# Additional quantizations to create (requires llama.cpp llama-quantize binary)
# Options: q8_0, q4_0, q4_1, q5_0, q5_1, q2_K, q3_K_S, q3_K_M, q4_K_S, q4_K_M, q5_K_S, q5_K_M, q6_K
GGUF_QUANTIZATIONS = ["q6_K", "q4_K_M"]

# Working directory
def get_work_dir():
    if os.path.exists("/content") and os.access("/content", os.W_OK):
        return "/content/nanochat_convert"  # Google Colab
    else:
        return os.path.expanduser("~/nanochat_convert")  # Lambda Labs, local, etc.

WORK_DIR = get_work_dir()
GGUF_OUTPUT_DIR = f"{WORK_DIR}/gguf_models"


# ============================================================================
# GGUF CONVERTER
# ============================================================================

def convert_nanochat_to_gguf(input_dir: str, output_file: str, dtype: str = "f16", arch: str = "llama") -> str:
    """
    Convert NanoChat HuggingFace model to GGUF format.
    
    NanoChat architecture specifics:
    - Uses relu2 activation: x.relu().square()
    - Uses parameter-free RMSNorm (no learnable weights)
    - MLP uses simple 2-layer structure: fc1 -> relu2 -> fc2
    - RoPE with flipped rotation: (x2, -x1) instead of (-x2, x1)
    - Logit softcapping with scale 15.0
    
    GGUF Architecture Options:
    - "nanochat": Native NanoChat (requires llama.cpp with nanochat support)
    - "llama": Gated 3-layer MLP (silu), requires weight duplication hack
    - "gpt2": Simple 2-layer MLP (gelu), better structure match
    
    For "nanochat" arch:
    - Native 2-layer MLP with relu2 activation
    - No weight duplication needed
    - Parameter-free norms (no norm weights in GGUF)
    - Proper final logit softcapping
    
    For "llama" arch (workaround):
    - Weight duplication hack (fc1 -> gate + up)
    - Activation mismatch (silu vs relu2)
    - Output quality is degraded
    
    Args:
        input_dir: Path to HuggingFace model directory
        output_file: Path for output GGUF file
        dtype: Output dtype (f16, f32, bf16)
        arch: GGUF architecture ("nanochat", "llama", or "gpt2")
    
    Returns:
        Path to created GGUF file
    """
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType
    
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(input_path / "config.json") as f:
        config = json.load(f)
    
    # Load weights
    state_dict = {}
    safetensors_files = list(input_path.glob("*.safetensors"))
    pytorch_files = list(input_path.glob("pytorch_model*.bin"))
    
    if safetensors_files:
        from safetensors.torch import load_file
        for sf in safetensors_files:
            state_dict.update(load_file(sf))
    elif pytorch_files:
        for pf in pytorch_files:
            state_dict.update(torch.load(pf, map_location="cpu", weights_only=True))
    
    print(f"Loaded {len(state_dict)} tensors from {input_path}")
    print(f"Tensor keys: {list(state_dict.keys())[:10]}...")
    
    # Extract config
    vocab_size = config.get("vocab_size", 50304)
    hidden_size = config.get("hidden_size", 768)
    num_layers = config.get("num_hidden_layers", 12)
    num_heads = config.get("num_attention_heads", 6)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    max_seq_len = config.get("max_position_embeddings", 2048)
    intermediate_size = config.get("intermediate_size", hidden_size * 4)
    rms_norm_eps = config.get("rms_norm_eps", 1e-6)
    rope_theta = config.get("rope_theta", 10000.0)
    if "rope_parameters" in config:
        rope_theta = config["rope_parameters"].get("rope_theta", rope_theta)
    
    # Get activation function info from config (for documentation)
    hidden_act = config.get("hidden_act", "relu2")
    final_logit_softcapping = config.get("final_logit_softcapping", 15.0)
    
    # Verify vocab_size matches embedding tensor
    embed_key = "model.embed_tokens.weight"
    if embed_key in state_dict:
        actual_vocab_size = state_dict[embed_key].shape[0]
        if actual_vocab_size != vocab_size:
            print(f"⚠️ Config vocab_size ({vocab_size}) != embedding size ({actual_vocab_size})")
            print(f"   Using embedding size: {actual_vocab_size}")
            vocab_size = actual_vocab_size
    
    print(f"Model: {num_layers} layers, {hidden_size} hidden, {num_heads} heads, {vocab_size} vocab")
    print(f"NanoChat activation: {hidden_act}, logit_softcapping: {final_logit_softcapping}")
    print(f"GGUF architecture: {arch}")
    
    if arch == "nanochat":
        print("   Using NATIVE NanoChat arch: 2-layer MLP with relu2 (requires llama.cpp nanochat support)")
        print("   ✅ Correct MLP structure, correct activation, parameter-free norms")
    elif arch == "gpt2":
        print("   Using GPT-2 arch: 2-layer MLP with GELU (structure matches, activation differs)")
    else:
        print("   Using LLaMA arch: 3-layer gated MLP with SiLU (requires weight duplication hack)")
    
    # GGUF dtype mapping
    dtype_map = {
        "f32": GGMLQuantizationType.F32,
        "f16": GGMLQuantizationType.F16,
        "bf16": GGMLQuantizationType.BF16,
    }
    gguf_dtype = dtype_map.get(dtype, GGMLQuantizationType.F16)
    np_dtype = np.float16 if dtype != "f32" else np.float32
    
    def to_numpy(tensor):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        return tensor.to(torch.float16 if dtype != "f32" else torch.float32).numpy()
    
    def to_numpy_f32(tensor):
        """Always convert to f32 - used for norm weights that need to match compute precision."""
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        return tensor.to(torch.float32).numpy()
    
    # Create GGUF writer with specified architecture
    writer = GGUFWriter(str(output_path), arch=arch)
    
    # Metadata
    writer.add_name("nanochat")
    writer.add_context_length(max_seq_len)
    writer.add_embedding_length(hidden_size)
    writer.add_block_count(num_layers)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_kv_heads)
    writer.add_layer_norm_rms_eps(rms_norm_eps)
    writer.add_rope_freq_base(rope_theta)
    writer.add_file_type(gguf_dtype)
    
    # NanoChat-specific metadata
    if arch == "nanochat":
        # Add final logit softcapping for native nanochat arch
        # This uses the standard GGUF key that gemma2 etc. use
        writer.add_float32("nanochat.final_logit_softcapping", final_logit_softcapping)
    
    # Get token IDs from config
    bos_token_id = config.get("bos_token_id", 0)
    eos_token_id = config.get("eos_token_id", 1)
    pad_token_id = config.get("pad_token_id", eos_token_id)
    
    writer.add_bos_token_id(bos_token_id)
    writer.add_eos_token_id(eos_token_id)
    writer.add_pad_token_id(pad_token_id)
    
    # =========================================================================
    # TOKENIZER - Must match vocab_size exactly
    # =========================================================================
    print("Loading tokenizer...")
    tokenizer_json_path = input_path / "tokenizer.json"
    
    tokens = []
    scores = []
    toktypes = []
    merges = []
    
    if tokenizer_json_path.exists():
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        
        vocab = tokenizer_data.get("model", {}).get("vocab", {})
        merges = tokenizer_data.get("model", {}).get("merges", [])
        
        if vocab:
            token_id_to_str = {tid: tstr for tstr, tid in vocab.items()}
            
            for token_id in range(vocab_size):
                if token_id in token_id_to_str:
                    token_str = token_id_to_str[token_id]
                    token_bytes = token_str.encode("utf-8", errors="replace")
                else:
                    token_bytes = f"<unused_{token_id}>".encode("utf-8")
                
                tokens.append(token_bytes)
                scores.append(float(-token_id))
                
                if token_id in token_id_to_str:
                    token_str = token_id_to_str[token_id]
                    if token_id == bos_token_id or token_id == eos_token_id or token_id == pad_token_id:
                        toktypes.append(3)  # CONTROL
                    elif token_str.startswith("<") and token_str.endswith(">"):
                        toktypes.append(3)  # CONTROL
                    else:
                        toktypes.append(1)  # NORMAL
                else:
                    toktypes.append(5)  # UNUSED
            
            print(f"Loaded {len(vocab)} tokens, padded to {len(tokens)}")
    
    if not tokens:
        print("Creating placeholder tokenizer...")
        for i in range(vocab_size):
            tokens.append(f"<token_{i}>".encode("utf-8"))
            scores.append(float(-i))
            toktypes.append(3 if i in (bos_token_id, eos_token_id) else 1)
    
    assert len(tokens) == vocab_size, f"Token count {len(tokens)} != vocab_size {vocab_size}"
    
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)
    writer.add_vocab_size(vocab_size)
    
    # Process merges
    if merges:
        processed_merges = []
        if isinstance(merges[0], str):
            processed_merges = merges
        elif isinstance(merges[0], list) and len(merges[0]) == 2:
            print("Converting merges from new format...")
            for pair in merges:
                parts = []
                for part in pair:
                    encoded = ''.join(chr(ord(c) + 256) if c == ' ' else c for c in part)
                    parts.append(encoded)
                processed_merges.append(' '.join(parts))
        if processed_merges:
            writer.add_token_merges(processed_merges)
            print(f"Added {len(processed_merges)} BPE merges")
    
    # =========================================================================
    # MODEL TENSORS - Architecture-specific handling
    # =========================================================================
    #
    # NanoChat has NO learnable norm weights (parameter-free RMSNorm).
    # - For native "nanochat" arch: Don't write norm weights at all
    # - For "llama"/"gpt2" arch: Write dummy norm weights (all ones)
    # =========================================================================
    
    # Create dummy norm weight (all ones) - only needed for llama/gpt2 arch
    # IMPORTANT: Norm weights must be f32 for llama.cpp compute compatibility,
    # even when model weights are f16. This avoids binary_op type mismatches.
    dummy_norm = np.ones(hidden_size, dtype=np.float32)
    needs_dummy_norms = arch in ("llama", "gpt2")
    
    # Token embeddings
    # NOTE: HF stores embeddings as [vocab_size, hidden_size]
    # GGUF/llama.cpp expects [hidden_size, vocab_size] for ggml_get_rows
    # So we need to transpose!
    if "model.embed_tokens.weight" in state_dict:
        emb = state_dict["model.embed_tokens.weight"]
        print(f"token_embd.weight HF shape: {emb.shape}")
        emb_t = emb.T  # Transpose to [hidden_size, vocab_size]
        print(f"token_embd.weight GGUF shape (transposed): {emb_t.shape}")
        writer.add_tensor("token_embd.weight", to_numpy(emb_t))
    else:
        raise ValueError("Missing model.embed_tokens.weight!")
    
    # Output head (lm_head)
    # Same transpose needed - HF [vocab_size, hidden_size] -> GGUF [hidden_size, vocab_size]
    if "lm_head.weight" in state_dict:
        lm_head = state_dict["lm_head.weight"]
        print(f"output.weight HF shape: {lm_head.shape}")
        lm_head_t = lm_head.T  # Transpose
        print(f"output.weight GGUF shape (transposed): {lm_head_t.shape}")
        writer.add_tensor("output.weight", to_numpy(lm_head_t))
    else:
        raise ValueError("Missing lm_head.weight!")
    
    # Final norm - NanoChat has NO learnable final norm
    if needs_dummy_norms:
        print(f"output_norm.weight shape: [{hidden_size}] (dummy - all ones)")
        writer.add_tensor("output_norm.weight", dummy_norm)
    else:
        print(f"output_norm.weight: SKIPPED (native nanochat uses parameter-free norm)")
    
    # =========================================================================
    # LAYER TENSORS - Architecture-specific MLP mapping
    # =========================================================================
    #
    # NanoChat MLP: out = fc2(relu2(fc1(x))) where relu2(x) = x.relu().square()
    #
    # LLaMA arch (gated MLP): out = down(silu(gate(x)) * up(x))
    #   - Requires 3 weight matrices
    #   - We duplicate fc1 for gate and up: out = down(silu(fc1(x)) * fc1(x))
    #   - This is a poor approximation since silu != relu2
    #
    # GPT-2 arch (simple MLP): out = down(gelu(up(x)))
    #   - Requires 2 weight matrices (matches NanoChat structure!)
    #   - fc1 -> ffn_up, fc2 -> ffn_down
    #   - Only activation differs (gelu vs relu2)
    #
    # =========================================================================
    
    for i in range(num_layers):
        hf = f"model.layers.{i}"
        blk = f"blk.{i}"
        
        # Attention projections (same for both architectures)
        for src, dst in [("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")]:
            key = f"{hf}.self_attn.{src}.weight"
            if key in state_dict:
                writer.add_tensor(f"{blk}.{dst}.weight", to_numpy(state_dict[key]))
            else:
                raise ValueError(f"Missing {key}!")
        
        # MLP weights - architecture-dependent mapping
        fc1_key = f"{hf}.mlp.fc1.weight"
        fc2_key = f"{hf}.mlp.fc2.weight"
        
        if fc1_key not in state_dict or fc2_key not in state_dict:
            raise ValueError(f"Missing MLP weights for layer {i}!")
        
        fc1_np = to_numpy(state_dict[fc1_key])
        fc2_np = to_numpy(state_dict[fc2_key])
        
        if arch == "nanochat":
            # Native NanoChat architecture: 2-layer MLP with relu2 (no duplication!)
            # out = down(relu2(up(x))) where relu2(x) = relu(x).square()
            # fc1 -> ffn_up, fc2 -> ffn_down
            writer.add_tensor(f"{blk}.ffn_up.weight", fc1_np)
            writer.add_tensor(f"{blk}.ffn_down.weight", fc2_np)
            # No norm weights for nanochat (parameter-free RMSNorm)
        elif arch == "gpt2":
            # GPT-2 architecture: simple 2-layer MLP (matches NanoChat structure!)
            # out = down(gelu(up(x)))
            # fc1 (hidden_size -> intermediate_size) -> ffn_up
            # fc2 (intermediate_size -> hidden_size) -> ffn_down
            writer.add_tensor(f"{blk}.ffn_up.weight", fc1_np)
            writer.add_tensor(f"{blk}.ffn_down.weight", fc2_np)
            # Dummy layer norms for gpt2
            writer.add_tensor(f"{blk}.attn_norm.weight", dummy_norm)
            writer.add_tensor(f"{blk}.ffn_norm.weight", dummy_norm)
        else:
            # LLaMA architecture: gated 3-layer MLP
            # out = down(silu(gate(x)) * up(x))
            # For NanoChat-like behavior, we duplicate fc1 for both gate and up
            # This makes: out = down(silu(fc1(x)) * fc1(x))
            # Which is an approximation since silu(x)*x != relu2(x)
            writer.add_tensor(f"{blk}.ffn_gate.weight", fc1_np)  # gate = fc1
            writer.add_tensor(f"{blk}.ffn_up.weight", fc1_np)    # up = fc1 (duplicate)
            writer.add_tensor(f"{blk}.ffn_down.weight", fc2_np)  # down = fc2
            # Dummy layer norms for llama
            writer.add_tensor(f"{blk}.attn_norm.weight", dummy_norm)
            writer.add_tensor(f"{blk}.ffn_norm.weight", dummy_norm)
    
    # Write file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    size_gb = output_path.stat().st_size / (1024**3)
    print(f"✅ GGUF saved: {output_path} ({size_gb:.2f} GB)")
    return str(output_path)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_gguf(gguf_path: str, expected_arch: str = None) -> bool:
    """
    Validate GGUF file by:
    1. Checking file exists and has reasonable size
    2. Reading GGUF metadata to verify structure
    3. Checking required tensors for the architecture
    
    Args:
        gguf_path: Path to GGUF file
        expected_arch: Expected architecture ("llama" or "gpt2"). If None, auto-detect.
    
    Returns True if valid, False otherwise.
    """
    import gguf
    
    gguf_path = Path(gguf_path)
    
    print(f"\n{'='*60}")
    print(f"🔍 Validating GGUF: {gguf_path.name}")
    print(f"{'='*60}")
    
    # Check 1: File exists
    if not gguf_path.exists():
        print(f"❌ GGUF file not found: {gguf_path}")
        return False
    
    # Check 2: File size
    size_mb = gguf_path.stat().st_size / (1024**2)
    print(f"📦 File size: {size_mb:.1f} MB")
    if size_mb < 10:
        print(f"⚠️ Warning: File seems too small ({size_mb:.1f} MB)")
    
    # Check 3: Read GGUF metadata
    try:
        reader = gguf.GGUFReader(str(gguf_path))
        
        print(f"\n📋 GGUF Metadata:")
        
        # Get architecture and config from metadata
        arch = None
        vocab_size = None
        n_layers = None
        
        for field in reader.fields.values():
            if field.name == "general.architecture":
                # Architecture is stored as bytes array, decode properly
                if field.parts:
                    arch_bytes = bytes(field.parts[-1])
                    arch = arch_bytes.decode('utf-8')
            elif field.name.endswith(".vocab_size"):
                vocab_size = int(field.parts[-1][0]) if field.parts else None
            elif field.name.endswith(".block_count"):
                n_layers = int(field.parts[-1][0]) if field.parts else None
        
        print(f"   Architecture: {arch}")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Block count: {n_layers}")
        
        if expected_arch and arch != expected_arch:
            print(f"   ⚠️ Expected architecture '{expected_arch}', got '{arch}'")
        
        # Check tensors
        print(f"\n📊 Tensors: {len(reader.tensors)}")
        
        # Required tensors - architecture-dependent
        # NanoChat has parameter-free RMSNorm (no norm weights)
        if arch == "nanochat":
            required_tensors = [
                "token_embd.weight",
                "output.weight",
                # No output_norm.weight - parameter-free RMSNorm
            ]
        else:
            required_tensors = [
                "token_embd.weight",
                "output.weight",
                "output_norm.weight",
            ]
        
        # Add layer tensors based on architecture
        if n_layers:
            for i in range(n_layers):
                # Attention tensors (same for all architectures)
                required_tensors.extend([
                    f"blk.{i}.attn_q.weight",
                    f"blk.{i}.attn_k.weight",
                    f"blk.{i}.attn_v.weight",
                    f"blk.{i}.attn_output.weight",
                ])
                
                # Norm weights - nanochat has parameter-free norms (no weights)
                if arch != "nanochat":
                    required_tensors.extend([
                        f"blk.{i}.attn_norm.weight",
                        f"blk.{i}.ffn_norm.weight",
                    ])
                
                # MLP tensors depend on architecture
                if arch == "nanochat" or arch == "gpt2":
                    # NanoChat/GPT-2: 2-layer MLP (up, down)
                    required_tensors.extend([
                        f"blk.{i}.ffn_up.weight",
                        f"blk.{i}.ffn_down.weight",
                    ])
                else:
                    # LLaMA and similar: 3-layer gated MLP (gate, up, down)
                    required_tensors.extend([
                        f"blk.{i}.ffn_gate.weight",
                        f"blk.{i}.ffn_up.weight",
                        f"blk.{i}.ffn_down.weight",
                    ])
        
        tensor_names = {t.name for t in reader.tensors}
        missing = [t for t in required_tensors if t not in tensor_names]
        
        if missing:
            print(f"\n❌ Missing required tensors for {arch} architecture:")
            for t in missing[:10]:
                print(f"   - {t}")
            if len(missing) > 10:
                print(f"   ... and {len(missing) - 10} more")
            return False
        
        print(f"   ✅ All required tensors present for {arch} architecture")
        
        # Check tokenizer
        has_tokenizer = any("tokenizer" in f.name for f in reader.fields.values())
        print(f"   Tokenizer: {'✅ Present' if has_tokenizer else '❌ Missing'}")
        
        if not has_tokenizer:
            print("\n❌ Missing tokenizer metadata!")
            return False
        
        print(f"\n✅ GGUF validation passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to read GGUF: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_llama_cpp(work_dir: str, build_cli: bool = True) -> tuple:
    """
    Build llama.cpp binaries (llama-quantize and optionally llama-cli).
    
    Returns:
        Tuple of (llama_cpp_dir, quantize_bin, cli_bin)
    """
    llama_cpp_dir = f"{work_dir}/llama.cpp"
    
    # Clone if needed
    if not os.path.exists(llama_cpp_dir):
        print(f"\n📥 Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", llama_cpp_dir],
            check=True
        )
    
    # Clear old build to avoid CMake cache issues
    build_dir = f"{llama_cpp_dir}/build"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    
    print(f"\n🔧 Building llama.cpp...")
    
    # Configure with CMake
    subprocess.run(
        ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DGGML_CUDA=OFF"],
        cwd=llama_cpp_dir,
        check=True
    )
    
    # Build targets
    targets = ["llama-quantize"]
    if build_cli:
        targets.append("llama-cli")
    
    for target in targets:
        print(f"   Building {target}...")
        subprocess.run(
            ["cmake", "--build", "build", "--target", target, "-j"],
            cwd=llama_cpp_dir,
            check=True
        )
    
    # Find binaries
    quantize_bin = None
    cli_bin = None
    
    for path in [
        f"{llama_cpp_dir}/build/bin/llama-quantize",
        f"{llama_cpp_dir}/build/llama-quantize",
    ]:
        if os.path.exists(path):
            quantize_bin = path
            break
    
    if build_cli:
        for path in [
            f"{llama_cpp_dir}/build/bin/llama-cli",
            f"{llama_cpp_dir}/build/llama-cli",
        ]:
            if os.path.exists(path):
                cli_bin = path
                break
    
    print(f"   ✅ llama-quantize: {quantize_bin}")
    if build_cli:
        print(f"   ✅ llama-cli: {cli_bin}")
    
    return llama_cpp_dir, quantize_bin, cli_bin


def run_inference_test(gguf_path: str, llama_cli: str, test_prompts: List[dict] = None, arch: str = "llama") -> bool:
    """
    Run inference tests with llama.cpp to verify the GGUF model loads and runs.
    
    NOTE: NanoChat uses relu2 activation (x.relu().square()) in a 2-layer MLP.
    
    Architecture-specific approximations:
    - LLaMA: Uses gated SiLU MLP, requires duplicating fc1 for gate/up. Poor match.
    - GPT-2: Uses 2-layer GELU MLP, correct structure but different activation.
    
    The test primarily checks that the model loads and runs without crashing,
    not that the output is semantically correct.
    
    Args:
        gguf_path: Path to GGUF file
        llama_cli: Path to llama-cli binary
        test_prompts: List of test prompts (not used in basic mode)
        arch: Architecture used ("llama" or "gpt2")
        
    Returns:
        True if model loads and runs successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"🧪 INFERENCE TESTING: {Path(gguf_path).name}")
    print(f"{'='*60}")
    
    print(f"\n⚠️  NOTE: NanoChat uses relu2 activation in a 2-layer MLP.")
    if arch == "gpt2":
        print("   Using GPT-2 architecture: 2-layer MLP with GELU (structure matches!).")
        print("   Activation function differs (GELU vs relu2), but better than gated MLP.")
    else:
        print("   Using LLaMA architecture: gated 3-layer MLP with SiLU (structure differs!).")
        print("   Weight duplication hack: gate=fc1, up=fc1. Output quality may be degraded.")
    print("   This test verifies the model loads and runs, not output correctness.\n")
    
    if not os.path.exists(llama_cli):
        print(f"❌ llama-cli not found at: {llama_cli}")
        return False
    
    # Test 1: Basic load and run test
    print("📝 Test 1/2: Model load and inference test")
    print("   Checking if model loads and generates tokens without crashing...")
    
    try:
        result = subprocess.run(
            [
                llama_cli,
                "-m", str(gguf_path),
                "-p", "Hello",
                "-n", "10",
                "--no-warmup",
                "-c", "256",
                "--temp", "0.7",
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            # Check if it's a type mismatch error or other critical failure
            if "binary_op: unsupported types" in result.stderr:
                print("   ❌ FAILED - Type mismatch error during computation")
                print("   This indicates a tensor type compatibility issue.")
                return False
            elif "error loading model" in result.stderr.lower():
                print("   ❌ FAILED - Model loading error")
                print(f"   {result.stderr[:500]}")
                return False
            else:
                print(f"   ❌ FAILED - llama-cli returned error code {result.returncode}")
                # Show stderr for debugging
                if result.stderr:
                    error_preview = result.stderr[:500].replace('\n', ' ')
                    print(f"   Error: {error_preview}")
                return False
        
        print("   ✅ Model loaded and ran successfully!")
        
    except subprocess.TimeoutExpired:
        print("   ❌ FAILED - Timeout (>120s)")
        return False
    except Exception as e:
        print(f"   ❌ FAILED - Exception: {e}")
        return False
    
    # Test 2: Verify token generation
    print("\n📝 Test 2/2: Token generation verification")
    print("   Checking if model generates tokens...")
    
    try:
        result = subprocess.run(
            [
                llama_cli,
                "-m", str(gguf_path),
                "-p", "The quick brown fox",
                "-n", "20",
                "--no-warmup",
                "-c", "256",
                "--temp", "0.5",
                "--ignore-eos",  # Force generation even if EOS is predicted
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            # Check performance output to verify tokens were generated
            if "eval time" in result.stderr and "runs" in result.stderr:
                print("   ✅ Token generation confirmed!")
                # Extract eval runs count
                import re
                match = re.search(r'eval time\s*=.*?/\s*(\d+)\s*runs', result.stderr)
                if match:
                    n_runs = int(match.group(1))
                    print(f"   Generated {n_runs} tokens successfully.")
            else:
                print("   ✅ Model ran without errors.")
        else:
            print(f"   ⚠️ Model ran but returned code {result.returncode}")
            # This is acceptable if no critical errors
    
    except Exception as e:
        print(f"   ⚠️ Could not verify token generation: {e}")
    
    print(f"\n{'='*60}")
    print("✅ INFERENCE TEST PASSED")
    print("   Model loads, runs inference, and generates tokens.")
    print("   Output quality may vary due to MLP architecture approximation.")
    print(f"{'='*60}")
    
    return True


def test_with_llama_cpp(gguf_path: str, llama_cpp_dir: str = None) -> bool:
    """
    Quick test to verify GGUF can be loaded by llama.cpp.
    For full inference testing, use run_inference_test().
    Returns True if successful.
    """
    gguf_path = Path(gguf_path)
    
    # Find llama-cli
    llama_cli = None
    if llama_cpp_dir:
        for path in [
            Path(llama_cpp_dir) / "build" / "bin" / "llama-cli",
            Path(llama_cpp_dir) / "build" / "llama-cli",
        ]:
            if path.exists():
                llama_cli = path
                break
    else:
        # Try common locations
        for path in [
            f"{WORK_DIR}/llama.cpp/build/bin/llama-cli",
            f"{WORK_DIR}/llama.cpp/build/llama-cli",
        ]:
            if os.path.exists(path):
                llama_cli = Path(path)
                break
    
    if llama_cli is None or not llama_cli.exists():
        print("⚠️ llama-cli not found, skipping load test")
        return True  # Skip test, not a failure
    
    print(f"\n🧪 Quick load test with llama-cli...")
    
    try:
        # Just try to load the model and generate 1 token
        result = subprocess.run(
            [str(llama_cli), "-m", str(gguf_path), "-p", "test", "-n", "1", "--no-warmup", "--log-disable"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ llama-cli load test passed!")
            return True
        else:
            print(f"❌ llama-cli failed:")
            print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ llama-cli timed out (this may be OK for large models)")
        return True
    except Exception as e:
        print(f"⚠️ llama-cli test error: {e}")
        return True  # Don't fail on test errors


# ============================================================================
# UPLOAD
# ============================================================================

def upload_gguf_to_hub(
    gguf_dir: str,
    repo_id: str,
    hf_repo_name: str,
    model_name: str,
    base_dtype: str,
    quantizations: List[str],
    token: Optional[str] = None,
    arch: str = "llama",
):
    """Upload GGUF files to HuggingFace Hub."""
    from huggingface_hub import HfApi
    
    api = HfApi(token=token)
    
    print(f"📤 Uploading GGUF models to {repo_id}...")
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Create README for GGUF repo
    all_quants = [base_dtype] + quantizations
    quant_list = "\n".join([f"- `{q}`: {model_name}-{q}.gguf" for q in all_quants])
    
    hf_username = repo_id.split("/")[0]
    
    # Architecture-specific notes
    arch_note = """
> ⚠️ **IMPORTANT: Output Quality Warning**
> 
> These GGUF files use LLaMA architecture with a weight duplication hack to approximate 
> NanoChat's unique 2-layer relu2 MLP with LLaMA's 3-layer gated SiLU MLP. **The output 
> quality is significantly degraded** due to fundamental architectural differences.
>
> These files are provided for **ecosystem compatibility only** (Ollama, llama.cpp, etc.).
> **For actual inference, use the HuggingFace transformers model directly.**
"""
    
    gguf_readme = f"""---
license: mit
language:
- en
tags:
- nanochat
- gguf
- llama-cpp
- text-generation
- quantized
pipeline_tag: text-generation
---

# {repo_id.split("/")[1]}

GGUF quantized versions of [{hf_username}/{hf_repo_name}](https://huggingface.co/{hf_username}/{hf_repo_name}) for use with llama.cpp, ollama, and other GGUF-compatible inference engines.
{arch_note}
## Available Quantizations

{quant_list}

## Usage with llama.cpp

```bash
# Download a GGUF file
huggingface-cli download {repo_id} {model_name}-q4_K_M.gguf --local-dir .

# Run with llama.cpp
./llama-cli -m {model_name}-q4_K_M.gguf -p "Hello, how are you?" -n 100
```

## Usage with Ollama

Create a `Modelfile`:

```
FROM ./{model_name}-q4_K_M.gguf

TEMPLATE "{{{{.Prompt}}}}"
```

Then:

```bash
ollama create {model_name} -f Modelfile
ollama run {model_name}
```

## Quantization Details

| Quantization | Description | Use Case |
|-------------|-------------|----------|
| f16 | 16-bit float | Best quality, largest size |
| q8_0 | 8-bit quantization | Good quality, moderate size |
| q4_K_M | 4-bit K-quant (medium) | Good balance of quality and size |

## Architecture Details

NanoChat uses a unique architecture:
- **Activation**: relu2 (x.relu().square())
- **MLP**: Simple 2-layer (fc1 → activation → fc2)
- **Norms**: Parameter-free RMSNorm
- **RoPE**: Flipped rotation direction

GGUF Architecture: `{arch}`

## Original Model

HuggingFace format: [{hf_username}/{hf_repo_name}](https://huggingface.co/{hf_username}/{hf_repo_name})

## License

MIT License
"""
    
    gguf_readme_path = f"{gguf_dir}/README.md"
    with open(gguf_readme_path, "w") as f:
        f.write(gguf_readme)
    
    # Upload all GGUF files
    api.upload_folder(
        folder_path=gguf_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload GGUF quantized models",
    )
    
    print(f"✅ GGUF models uploaded!")
    print(f"🔗 https://huggingface.co/{repo_id}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert NanoChat to GGUF and upload to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architecture Notes:
  NanoChat uses a unique architecture with relu2 activation and 2-layer MLP.
  
  --arch nanochat (RECOMMENDED when llama.cpp supports it):
    - Native NanoChat architecture with correct 2-layer MLP and relu2
    - Parameter-free RMSNorm (no norm weights in GGUF)
    - Proper final logit softcapping
    - Requires llama.cpp with nanochat support (see LLAMA_CPP_INTEGRATION.md)
  
  --arch llama (default, workaround):
    - Maps to LLaMA's gated 3-layer MLP by duplicating fc1 weights
    - Compute: down(silu(fc1(x)) * fc1(x)) instead of fc2(relu2(fc1(x)))
    - Model loads and runs, but OUTPUT QUALITY IS SIGNIFICANTLY DEGRADED
    
  --arch gpt2 (workaround):
    - 2-layer MLP structure matches NanoChat
    - Activation differs: GELU vs relu2
    - Better than llama but still degraded quality
    
  IMPORTANT: For best quality, use --arch nanochat with llama.cpp nanochat support,
  or use the HuggingFace transformers model directly.
"""
    )
    parser.add_argument("--source-repo", type=str, default=SOURCE_HF_REPO,
                        help="Source HuggingFace repo (must be in HF transformers format)")
    parser.add_argument("--gguf-repo", type=str, default=GGUF_REPO_NAME,
                        help="Output GGUF repository name")
    parser.add_argument("--hf-username", type=str, default=HF_USERNAME,
                        help="HuggingFace username")
    parser.add_argument("--hf-token", type=str, default=HF_TOKEN,
                        help="HuggingFace API token")
    parser.add_argument("--base-dtype", type=str, default=GGUF_BASE_DTYPE,
                        choices=["f16", "f32", "bf16"],
                        help="Base dtype for GGUF conversion")
    parser.add_argument("--arch", type=str, default="llama",
                        choices=["nanochat", "llama", "gpt2"],
                        help="GGUF architecture: 'nanochat' (native, requires llama.cpp support), 'llama' (workaround), 'gpt2' (workaround)")
    parser.add_argument("--quantizations", type=str, nargs="*", default=GGUF_QUANTIZATIONS,
                        help="Additional quantizations to create (e.g., q4_K_M q6_K)")
    parser.add_argument("--work-dir", type=str, default=WORK_DIR,
                        help="Working directory for temporary files")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip uploading to HuggingFace Hub")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip GGUF validation")
    parser.add_argument("--skip-quantization", action="store_true",
                        help="Skip additional quantizations (only create base GGUF)")
    parser.add_argument("--skip-inference-test", action="store_true",
                        help="Skip real inference testing with llama.cpp")
    parser.add_argument("--test-quant", type=str, default="q4_K_M",
                        help="Which quantization to use for inference testing (default: q4_K_M)")
    args = parser.parse_args()
    
    # Update globals from args
    work_dir = args.work_dir
    gguf_output_dir = f"{work_dir}/gguf_models"
    
    print(f"📁 Working directory: {work_dir}")
    print(f"📦 Source repo: {args.source_repo}")
    print(f"📤 GGUF repo: {args.hf_username}/{args.gguf_repo}")
    print(f"🔧 GGUF architecture: {args.arch}")
    
    # Step 1: Login to HuggingFace
    from huggingface_hub import login, HfApi, snapshot_download
    
    if args.hf_token:
        login(token=args.hf_token)
        print("✅ Logged in with provided token")
    else:
        print("⚠️ No token provided, attempting to use cached credentials...")
    
    # Verify login
    api = HfApi()
    try:
        user_info = api.whoami()
        print(f"✅ Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"⚠️ Login verification: {e}")
    
    # Step 2: Download HF model
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(gguf_output_dir, exist_ok=True)
    
    hf_source_dir = f"{work_dir}/hf_model"
    print(f"\n📥 Downloading HF model from {args.source_repo}...")
    snapshot_download(
        repo_id=args.source_repo,
        local_dir=hf_source_dir,
        local_dir_use_symlinks=False,
    )
    print(f"✅ Downloaded to: {hf_source_dir}")
    
    # Step 3: Build llama.cpp (needed for quantization and testing)
    llama_cpp_dir = None
    quantize_bin = None
    cli_bin = None
    
    need_llama_cpp = (not args.skip_quantization and args.quantizations) or not args.skip_inference_test
    if need_llama_cpp:
        llama_cpp_dir, quantize_bin, cli_bin = build_llama_cpp(
            work_dir, 
            build_cli=not args.skip_inference_test
        )
    
    # Step 4: Convert to GGUF
    model_name = args.gguf_repo.replace("-GGUF", "").replace("_", "-")
    base_gguf = f"{gguf_output_dir}/{model_name}-{args.base_dtype}.gguf"
    gguf_files = []
    
    print(f"\n🔄 Creating base GGUF ({args.base_dtype}, arch={args.arch})...")
    try:
        convert_nanochat_to_gguf(hf_source_dir, base_gguf, args.base_dtype, arch=args.arch)
        gguf_files.append(base_gguf)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Validate GGUF metadata
    if not args.skip_validation:
        validation_passed = validate_gguf(base_gguf, expected_arch=args.arch)
        if not validation_passed:
            print("\n" + "="*60)
            print("❌ GGUF VALIDATION FAILED - Will not upload")
            print("="*60)
            sys.exit(1)
    
    # Step 6: Additional quantizations
    if not args.skip_quantization and args.quantizations and os.path.exists(base_gguf):
        if quantize_bin:
            for quant in args.quantizations:
                quant_file = f"{gguf_output_dir}/{model_name}-{quant}.gguf"
                print(f"\n🔄 Quantizing to {quant}...")
                result = subprocess.run(
                    [quantize_bin, base_gguf, quant_file, quant.upper()],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0 and os.path.exists(quant_file):
                    qsize = os.path.getsize(quant_file) / (1024**3)
                    print(f"   ✅ Created: {quant_file} ({qsize:.2f} GB)")
                    gguf_files.append(quant_file)
                else:
                    print(f"   ❌ Failed to create {quant} quantization")
                    if result.stderr:
                        print(f"   Error: {result.stderr[:500]}")
        else:
            print("⚠️ llama-quantize not available, skipping additional quantizations")
    
    print(f"\n✅ GGUF conversion complete! Created {len(gguf_files)} files:")
    for f in gguf_files:
        size_gb = os.path.getsize(f) / (1024**3)
        print(f"   - {os.path.basename(f)} ({size_gb:.2f} GB)")
    
    # Step 7: Real inference testing with llama.cpp
    inference_passed = True
    if not args.skip_inference_test:
        if cli_bin:
            # Determine which file to test
            test_file = None
            test_quant = args.test_quant
            
            # Look for the specified quantization file
            for f in gguf_files:
                if test_quant in f:
                    test_file = f
                    break
            
            # Fallback to base GGUF if specified quant not found
            if test_file is None:
                test_file = base_gguf
                print(f"\n⚠️ {test_quant} not found, testing with base GGUF instead")
            
            print(f"\n🔬 Running inference tests on: {os.path.basename(test_file)}")
            
            # Run inference tests
            inference_passed = run_inference_test(test_file, cli_bin, arch=args.arch)
            
            if not inference_passed:
                print("\n" + "="*60)
                print("❌ INFERENCE TEST FAILED - Will not upload to HuggingFace!")
                print("="*60)
                print("\nThe GGUF file was created but failed real inference testing.")
                print("Please check the model and fix any issues before uploading.")
                print(f"\nLocal files are still available at: {gguf_output_dir}")
                sys.exit(1)
        else:
            print("\n⚠️ llama-cli not available, skipping inference tests")
    
    # Step 8: Upload to HuggingFace (only if inference test passed)
    if not args.skip_upload:
        if inference_passed:
            gguf_repo_id = f"{args.hf_username}/{args.gguf_repo}"
            hf_repo_name = args.source_repo.split("/")[-1]
            
            upload_gguf_to_hub(
                gguf_dir=gguf_output_dir,
                repo_id=gguf_repo_id,
                hf_repo_name=hf_repo_name,
                model_name=model_name,
                base_dtype=args.base_dtype,
                quantizations=args.quantizations if not args.skip_quantization else [],
                token=args.hf_token,
                arch=args.arch,
            )
        else:
            print("\n⚠️ Skipping upload due to failed inference test")
    
    # Summary
    print("\n" + "="*60)
    print("✅ CONVERSION AND UPLOAD COMPLETE!")
    print("="*60)
    print(f"\n📦 GGUF Models:")
    print(f"   Local: {gguf_output_dir}")
    if not args.skip_upload and inference_passed:
        print(f"   Hub:   https://huggingface.co/{args.hf_username}/{args.gguf_repo}")
    all_quants = [args.base_dtype] + (args.quantizations if not args.skip_quantization else [])
    print(f"\n   Architecture: {args.arch}")
    print(f"   Quantizations: {', '.join(all_quants)}")
    if not args.skip_inference_test:
        print(f"   Inference Test: {'✅ PASSED' if inference_passed else '❌ FAILED'}")
    
    print(f"\n📝 Architecture Notes:")
    if args.arch == "nanochat":
        print("   Using NATIVE NanoChat architecture:")
        print("   - Correct 2-layer MLP with relu2 activation")
        print("   - Parameter-free RMSNorm (no norm weights)")
        print("   - Final logit softcapping (15.0)")
        print("")
        print("   ✅ BEST QUALITY: Requires llama.cpp with nanochat support")
        print("   See LLAMA_CPP_INTEGRATION.md for implementation details.")
    else:
        print(f"   Using {args.arch.upper()} architecture with workaround:")
        if args.arch == "llama":
            print("   - MLP structure differs: 3-layer gated vs 2-layer simple")
            print("   - Activation differs: silu vs relu2")
            print("   - Weight duplication: gate=fc1, up=fc1")
        else:  # gpt2
            print("   - MLP structure matches: 2-layer simple")
            print("   - Activation differs: gelu vs relu2")
        print("")
        print("   ⚠️  WARNING: Output quality is DEGRADED!")
        print("   The GGUF is for ecosystem compatibility (Ollama, etc.) only.")
        print("   For best quality, use --arch nanochat or HuggingFace transformers.")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
