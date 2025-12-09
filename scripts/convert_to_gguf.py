#!/usr/bin/env python3
"""
Convert NanoChat HuggingFace model to GGUF format.

This script bypasses llama.cpp's convert_hf_to_gguf.py (which doesn't support NanoChat)
and writes GGUF files directly using the gguf-py library.

NanoChat HF format is similar to LLaMA, so we map to LLaMA GGUF architecture.

Usage:
    python -m scripts.convert_to_gguf --input_dir ./hf_model --output_file model.gguf
    python -m scripts.convert_to_gguf --input_dir ./hf_model --output_file model-f16.gguf --dtype f16

Requirements:
    pip install gguf numpy torch safetensors sentencepiece

For Colab:
    !pip install gguf
    !python -m scripts.convert_to_gguf --input_dir /content/hf_model --output_file /content/model.gguf
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Try to import gguf
try:
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("=" * 60)
    print("ERROR: gguf library not found!")
    print("=" * 60)
    print("\nInstall it with one of these methods:")
    print("  pip install gguf")
    print("  pip install llama-cpp-python")
    print("\nOr install from llama.cpp source:")
    print("  git clone https://github.com/ggml-org/llama.cpp")
    print("  pip install llama.cpp/gguf-py")
    sys.exit(1)


def load_hf_model(input_dir: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Load HuggingFace model config and weights."""
    
    # Load config
    config_path = input_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {input_dir}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"Loaded config from {config_path}")
    
    # Load weights - try safetensors first, then pytorch
    state_dict = {}
    
    # Check for safetensors
    safetensors_files = list(input_dir.glob("*.safetensors"))
    pytorch_files = list(input_dir.glob("pytorch_model*.bin"))
    
    if safetensors_files:
        try:
            from safetensors.torch import load_file
            for sf in safetensors_files:
                print(f"Loading weights from {sf.name}...")
                state_dict.update(load_file(sf))
        except ImportError:
            print("Warning: safetensors not installed, trying pytorch format...")
            safetensors_files = []
    
    if not safetensors_files and pytorch_files:
        for pf in pytorch_files:
            print(f"Loading weights from {pf.name}...")
            state_dict.update(torch.load(pf, map_location="cpu", weights_only=True))
    
    if not state_dict:
        raise FileNotFoundError(f"No model weights found in {input_dir}")
    
    print(f"Loaded {len(state_dict)} tensors")
    return config, state_dict


def get_gguf_dtype(dtype_str: str):
    """Get GGUF quantization type from string."""
    dtype_map = {
        "f32": GGMLQuantizationType.F32,
        "f16": GGMLQuantizationType.F16,
        "bf16": GGMLQuantizationType.BF16,
        "q8_0": GGMLQuantizationType.Q8_0,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Choose from: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def tensor_to_numpy(tensor: torch.Tensor, target_dtype: str) -> np.ndarray:
    """Convert PyTorch tensor to numpy array with appropriate dtype."""
    # Handle bfloat16
    if tensor.dtype == torch.bfloat16:
        if target_dtype == "bf16":
            # Keep as bf16 by converting via float32 and reinterpreting
            return tensor.view(torch.int16).numpy().view(np.uint16)
        else:
            tensor = tensor.to(torch.float32)
    
    # Convert to target dtype
    if target_dtype == "f32":
        return tensor.to(torch.float32).numpy()
    elif target_dtype in ("f16", "bf16"):
        return tensor.to(torch.float16).numpy()
    else:
        return tensor.to(torch.float32).numpy()


def write_nanochat_gguf(
    config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    output_path: Path,
    dtype: str = "f16",
):
    """Write NanoChat model to GGUF format using LLaMA architecture."""
    
    # Extract model config
    vocab_size = config.get("vocab_size", 50304)
    hidden_size = config.get("hidden_size", 768)
    num_layers = config.get("num_hidden_layers", 12)
    num_heads = config.get("num_attention_heads", 6)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    max_seq_len = config.get("max_position_embeddings", 2048)
    intermediate_size = config.get("intermediate_size", hidden_size * 4)
    rms_norm_eps = config.get("rms_norm_eps", 1e-5)
    rope_theta = config.get("rope_theta", 10000.0)
    hidden_act = config.get("hidden_act", "silu")
    
    head_dim = hidden_size // num_heads
    
    print("\n" + "=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)
    print(f"  Architecture: NanoChat -> LLaMA GGUF")
    print(f"  vocab_size: {vocab_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  max_position_embeddings: {max_seq_len}")
    print(f"  rope_theta: {rope_theta}")
    print(f"  hidden_act: {hidden_act}")
    print(f"  Output dtype: {dtype}")
    print("=" * 60 + "\n")
    
    # Create GGUF writer with LLaMA architecture
    gguf_writer = GGUFWriter(str(output_path), arch="llama")
    
    # Add metadata
    gguf_writer.add_name("nanochat")
    gguf_writer.add_description("NanoChat model converted to GGUF")
    gguf_writer.add_context_length(max_seq_len)
    gguf_writer.add_embedding_length(hidden_size)
    gguf_writer.add_block_count(num_layers)
    gguf_writer.add_feed_forward_length(intermediate_size)
    gguf_writer.add_head_count(num_heads)
    gguf_writer.add_head_count_kv(num_kv_heads)
    gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
    gguf_writer.add_rope_freq_base(rope_theta)
    gguf_writer.add_vocab_size(vocab_size)
    gguf_writer.add_file_type(get_gguf_dtype(dtype))
    
    # Add tokenizer metadata if available
    gguf_writer.add_bos_token_id(config.get("bos_token_id", 1))
    gguf_writer.add_eos_token_id(config.get("eos_token_id", 2))
    gguf_writer.add_pad_token_id(config.get("pad_token_id", 0))
    
    print("Writing tensors...")
    tensors_written = 0
    
    # Token embeddings
    # NanoChat HF: model.embed_tokens.weight -> GGUF: token_embd.weight
    # CRITICAL FIX: HF stores embeddings as [vocab_size, hidden_size]
    # GGUF/llama.cpp expects [hidden_size, vocab_size] for ggml_get_rows
    # So we must TRANSPOSE the embedding matrix!
    emb_key = "model.embed_tokens.weight"
    if emb_key in state_dict:
        emb = state_dict[emb_key]
        print(f"  token_embd.weight HF shape: {emb.shape}")
        emb_t = emb.T  # Transpose: [vocab_size, hidden_size] -> [hidden_size, vocab_size]
        data = tensor_to_numpy(emb_t, dtype)
        gguf_writer.add_tensor("token_embd.weight", data)
        print(f"  ✓ token_embd.weight GGUF shape (transposed): {data.shape}")
        tensors_written += 1
    
    # Output head (lm_head)
    # NanoChat HF: lm_head.weight -> GGUF: output.weight
    # CRITICAL FIX: Same transpose needed - HF [vocab_size, hidden_size] -> GGUF [hidden_size, vocab_size]
    lm_head_key = "lm_head.weight"
    if lm_head_key in state_dict:
        lm_head = state_dict[lm_head_key]
        print(f"  output.weight HF shape: {lm_head.shape}")
        lm_head_t = lm_head.T  # Transpose
        data = tensor_to_numpy(lm_head_t, dtype)
        gguf_writer.add_tensor("output.weight", data)
        print(f"  ✓ output.weight GGUF shape (transposed): {data.shape}")
        tensors_written += 1
    
    # Final norm
    # NanoChat HF: model.norm.weight -> GGUF: output_norm.weight
    norm_key = "model.norm.weight"
    if norm_key in state_dict:
        data = tensor_to_numpy(state_dict[norm_key], dtype)
        gguf_writer.add_tensor("output_norm.weight", data)
        print(f"  ✓ output_norm.weight {data.shape}")
        tensors_written += 1
    
    # Process each layer
    for i in range(num_layers):
        hf_prefix = f"model.layers.{i}"
        gguf_prefix = f"blk.{i}"
        
        # Attention weights
        # Q projection: model.layers.X.self_attn.q_proj.weight -> blk.X.attn_q.weight
        q_key = f"{hf_prefix}.self_attn.q_proj.weight"
        if q_key in state_dict:
            data = tensor_to_numpy(state_dict[q_key], dtype)
            gguf_writer.add_tensor(f"{gguf_prefix}.attn_q.weight", data)
            tensors_written += 1
        
        # K projection: model.layers.X.self_attn.k_proj.weight -> blk.X.attn_k.weight
        k_key = f"{hf_prefix}.self_attn.k_proj.weight"
        if k_key in state_dict:
            data = tensor_to_numpy(state_dict[k_key], dtype)
            gguf_writer.add_tensor(f"{gguf_prefix}.attn_k.weight", data)
            tensors_written += 1
        
        # V projection: model.layers.X.self_attn.v_proj.weight -> blk.X.attn_v.weight
        v_key = f"{hf_prefix}.self_attn.v_proj.weight"
        if v_key in state_dict:
            data = tensor_to_numpy(state_dict[v_key], dtype)
            gguf_writer.add_tensor(f"{gguf_prefix}.attn_v.weight", data)
            tensors_written += 1
        
        # Output projection: model.layers.X.self_attn.o_proj.weight -> blk.X.attn_output.weight
        o_key = f"{hf_prefix}.self_attn.o_proj.weight"
        if o_key in state_dict:
            data = tensor_to_numpy(state_dict[o_key], dtype)
            gguf_writer.add_tensor(f"{gguf_prefix}.attn_output.weight", data)
            tensors_written += 1
        
        # MLP weights
        # NanoChat uses fc1/fc2, LLaMA uses gate_proj/up_proj/down_proj
        # NanoChat fc1 -> ffn_up (assuming SwiGLU-like activation)
        # NanoChat fc2 -> ffn_down
        
        fc1_key = f"{hf_prefix}.mlp.fc1.weight"
        if fc1_key in state_dict:
            data = tensor_to_numpy(state_dict[fc1_key], dtype)
            # For models with gated MLP, fc1 might need to be split
            # Check if this is gated (intermediate_size would be 2x the actual)
            expected_size = intermediate_size
            actual_size = data.shape[0]
            
            if actual_size == expected_size * 2:
                # Gated activation: split into gate and up projections
                half = actual_size // 2
                gate_data = data[:half, :]
                up_data = data[half:, :]
                gguf_writer.add_tensor(f"{gguf_prefix}.ffn_gate.weight", gate_data)
                gguf_writer.add_tensor(f"{gguf_prefix}.ffn_up.weight", up_data)
                tensors_written += 2
            else:
                # Non-gated: just use as ffn_up
                gguf_writer.add_tensor(f"{gguf_prefix}.ffn_up.weight", data)
                tensors_written += 1
        
        fc2_key = f"{hf_prefix}.mlp.fc2.weight"
        if fc2_key in state_dict:
            data = tensor_to_numpy(state_dict[fc2_key], dtype)
            gguf_writer.add_tensor(f"{gguf_prefix}.ffn_down.weight", data)
            tensors_written += 1
        
        # Check for gate projection separately (some models have it)
        gate_key = f"{hf_prefix}.mlp.gate_proj.weight"
        if gate_key in state_dict:
            data = tensor_to_numpy(state_dict[gate_key], dtype)
            gguf_writer.add_tensor(f"{gguf_prefix}.ffn_gate.weight", data)
            tensors_written += 1
        
        up_key = f"{hf_prefix}.mlp.up_proj.weight"
        if up_key in state_dict:
            data = tensor_to_numpy(state_dict[up_key], dtype)
            gguf_writer.add_tensor(f"{gguf_prefix}.ffn_up.weight", data)
            tensors_written += 1
        
        down_key = f"{hf_prefix}.mlp.down_proj.weight"
        if down_key in state_dict:
            data = tensor_to_numpy(state_dict[down_key], dtype)
            gguf_writer.add_tensor(f"{gguf_prefix}.ffn_down.weight", data)
            tensors_written += 1
        
        # Layer norms
        # Input layernorm: model.layers.X.input_layernorm.weight -> blk.X.attn_norm.weight
        ln1_key = f"{hf_prefix}.input_layernorm.weight"
        if ln1_key in state_dict:
            data = tensor_to_numpy(state_dict[ln1_key], dtype)
            gguf_writer.add_tensor(f"{gguf_prefix}.attn_norm.weight", data)
            tensors_written += 1
        
        # Post-attention layernorm: model.layers.X.post_attention_layernorm.weight -> blk.X.ffn_norm.weight
        ln2_key = f"{hf_prefix}.post_attention_layernorm.weight"
        if ln2_key in state_dict:
            data = tensor_to_numpy(state_dict[ln2_key], dtype)
            gguf_writer.add_tensor(f"{gguf_prefix}.ffn_norm.weight", data)
            tensors_written += 1
        
        if i == 0 or i == num_layers - 1 or (i + 1) % 10 == 0:
            print(f"  ✓ Layer {i + 1}/{num_layers}")
    
    print(f"\nTotal tensors written: {tensors_written}")
    
    # Write the file
    print(f"\nWriting GGUF file to {output_path}...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    
    file_size_gb = output_path.stat().st_size / (1024 ** 3)
    print(f"\n{'=' * 60}")
    print(f"✅ SUCCESS!")
    print(f"{'=' * 60}")
    print(f"Output: {output_path}")
    print(f"Size: {file_size_gb:.2f} GB")
    print(f"{'=' * 60}")


def convert_to_gguf(input_dir: str, output_file: str, dtype: str = "f16"):
    """Main conversion function."""
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    config, state_dict = load_hf_model(input_path)
    
    # Write GGUF
    write_nanochat_gguf(config, state_dict, output_path, dtype)
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NanoChat HuggingFace model to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert to F16 (default, best quality)
    python -m scripts.convert_to_gguf --input_dir ./hf_model --output_file model-f16.gguf

    # Convert to F32 (largest, highest precision)
    python -m scripts.convert_to_gguf --input_dir ./hf_model --output_file model-f32.gguf --dtype f32

    # For further quantization, use llama-quantize:
    #   ./llama-quantize model-f16.gguf model-q4_K_M.gguf Q4_K_M
        """
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output GGUF file path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="f16",
        choices=["f32", "f16", "bf16"],
        help="Output data type (default: f16). For smaller files, convert to f16 then use llama-quantize.",
    )
    
    args = parser.parse_args()
    
    convert_to_gguf(
        input_dir=args.input_dir,
        output_file=args.output_file,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
