# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert nanochat checkpoints to HuggingFace transformers format.

This script converts nanochat SFT/MID/RL checkpoints to the HuggingFace transformers
format, enabling use with the transformers library and deployment to HuggingFace Hub.

Usage:
    # Convert SFT checkpoint (default)
    python -m scripts.convert_to_hf --output_dir ./hf_model

    # Convert from a specific checkpoint directory
    python -m scripts.convert_to_hf --input_dir /path/to/checkpoint --output_dir ./hf_model

    # With test prompt
    python -m scripts.convert_to_hf --output_dir ./hf_model --test_prompt "Hello, who are you?"

Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/nanochat/convert_nanochat_checkpoints.py
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

# Add parent directory to path to import nanochat modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.common import get_base_dir


def infer_kv_heads(hidden_size: int, num_attention_heads: int, state_dict: dict[str, torch.Tensor]) -> int:
    """Infer number of key-value heads from the checkpoint weights."""
    key_weight = state_dict.get("transformer.h.0.attn.c_k.weight")
    if key_weight is None:
        return num_attention_heads
    rows = key_weight.shape[0]
    head_dim = hidden_size // num_attention_heads
    if rows % head_dim != 0:
        return num_attention_heads
    inferred = rows // head_dim
    print(f"Inferred {inferred} key_value heads from checkpoint")
    return max(inferred, 1)


def convert_layer(old_prefix: str, new_prefix: str) -> dict[str, str]:
    """Map nanochat layer keys to HuggingFace transformers layer keys."""
    return {
        f"{old_prefix}.attn.c_q.weight": f"{new_prefix}.self_attn.q_proj.weight",
        f"{old_prefix}.attn.c_k.weight": f"{new_prefix}.self_attn.k_proj.weight",
        f"{old_prefix}.attn.c_v.weight": f"{new_prefix}.self_attn.v_proj.weight",
        f"{old_prefix}.attn.c_proj.weight": f"{new_prefix}.self_attn.o_proj.weight",
        f"{old_prefix}.mlp.c_fc.weight": f"{new_prefix}.mlp.fc1.weight",
        f"{old_prefix}.mlp.c_proj.weight": f"{new_prefix}.mlp.fc2.weight",
    }


def load_config_from_checkpoint(input_path: Path) -> dict:
    """
    Load config from either meta_*.json or config.json in the checkpoint directory.
    Returns a dict with the config parameters needed for NanoChatConfig.
    """
    # Import here to avoid import errors if transformers is not installed
    try:
        from transformers import NanoChatConfig
        has_transformers = True
    except ImportError:
        has_transformers = False
        print("Warning: transformers with NanoChat support not found. Will create config dict only.")

    # Try to find meta_*.json first (nanochat native format)
    meta_files = list(input_path.glob("meta_*.json"))

    if meta_files:
        meta_file = meta_files[0]
        print(f"Loading config from {meta_file.name}")
        with open(meta_file, "r") as f:
            meta_config = json.load(f)

        # Extract model config from meta file
        if "model_config" in meta_config:
            model_config = meta_config["model_config"]
        else:
            model_config = meta_config

        # Map nanochat config parameters to HuggingFace NanoChat config parameters
        config_kwargs = {
            "vocab_size": model_config.get("vocab_size", 50304),
            "hidden_size": model_config.get("n_embd", 768),
            "num_hidden_layers": model_config.get("n_layer", 12),
            "num_attention_heads": model_config.get("n_head", 6),
            "num_key_value_heads": model_config.get("n_kv_head"),
            "max_position_embeddings": model_config.get("sequence_len", 2048),
            "intermediate_size": model_config.get("intermediate_size", model_config.get("n_embd", 768) * 4),
        }

        # Try to load existing config.json for additional parameters
        config_file = input_path / "config.json"
        if config_file.exists():
            print("Loading additional config from config.json")
            with open(config_file, "r") as f:
                extra_config = json.load(f)

            # Add additional parameters from config.json
            for key in [
                "hidden_act",
                "attention_dropout",
                "rms_norm_eps",
                "initializer_range",
                "logits_soft_cap",
                "attention_bias",
                "intermediate_size",
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
            ]:
                if key in extra_config:
                    config_kwargs[key] = extra_config[key]
                # Handle legacy qkv_bias -> attention_bias conversion
                elif key == "attention_bias" and "qkv_bias" in extra_config:
                    config_kwargs[key] = extra_config["qkv_bias"]

            # Handle rope_theta as a direct kwarg for the rope_parameters processing
            if "rope_theta" in extra_config:
                config_kwargs["rope_theta"] = extra_config["rope_theta"]

            # Handle rope_parameters or rope_scaling if present
            if "rope_parameters" in extra_config:
                config_kwargs["rope_parameters"] = extra_config["rope_parameters"]
            elif "rope_scaling" in extra_config and extra_config["rope_scaling"] is not None:
                config_kwargs["rope_parameters"] = extra_config["rope_scaling"]

        if has_transformers:
            config = NanoChatConfig(**config_kwargs)
            return config
        else:
            return config_kwargs

    else:
        # Fallback to loading from config.json if it exists
        config_file = input_path / "config.json"
        if config_file.exists():
            print("Loading config from config.json")
            if has_transformers:
                config = NanoChatConfig.from_pretrained(input_path)
                # Handle legacy qkv_bias -> attention_bias conversion
                if hasattr(config, "qkv_bias") and not hasattr(config, "attention_bias"):
                    config.attention_bias = config.qkv_bias
                return config
            else:
                with open(config_file, "r") as f:
                    return json.load(f)
        else:
            raise ValueError(f"No config file found in {input_path}. Expected meta_*.json or config.json")


def write_model(input_dir, output_dir, safe_serialization=True):
    """Convert NanoChat model from original checkpoint format to HuggingFace format."""
    from transformers import NanoChatConfig, NanoChatForCausalLM

    print("Converting the model.")
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)

    # Load config
    config = load_config_from_checkpoint(input_path)
    print(f"Loaded config hidden_size={config.hidden_size} num_layers={config.num_hidden_layers}")

    # Load checkpoint - try model_*.pt first, then pytorch_model.bin
    checkpoint_files = list(input_path.glob("model_*.pt"))
    if checkpoint_files:
        # Sort to get the latest checkpoint if multiple exist
        checkpoint_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
        checkpoint_path = checkpoint_files[-1]
    else:
        checkpoint_path = input_path / "pytorch_model.bin"

    print(f"Fetching all parameters from the checkpoint at {checkpoint_path}...")
    old_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Handle torch.compile prefix if present
    old_state = {k.removeprefix("_orig_mod."): v for k, v in old_state.items()}

    # Original nanochat weights are in bfloat16
    for key in old_state:
        if old_state[key].dtype == torch.float32:
            old_state[key] = old_state[key].to(torch.bfloat16)

    # Infer key-value heads from checkpoint
    inferred_kv = infer_kv_heads(config.hidden_size, config.num_attention_heads, old_state)
    config.num_key_value_heads = inferred_kv
    if config.num_attention_heads % config.num_key_value_heads != 0:
        print(f"Adjusting num_attention_heads from {config.num_attention_heads} to {config.num_key_value_heads}")
        config.num_attention_heads = config.num_key_value_heads

    print("Converting model...")
    state_dict = {}
    rename_map = {}

    def assign(
        old_key: str,
        new_key: str,
        old_state: dict[str, torch.Tensor],
        state_dict: dict[str, torch.Tensor],
        rename_map: dict[str, str],
    ) -> None:
        tensor = old_state.get(old_key)
        if tensor is None:
            return
        state_dict[new_key] = tensor.clone()
        rename_map[old_key] = new_key

    # Convert embeddings and head
    assign("transformer.wte.weight", "model.embed_tokens.weight", old_state, state_dict, rename_map)
    assign("lm_head.weight", "lm_head.weight", old_state, state_dict, rename_map)

    # Convert layers
    for layer_idx in range(config.num_hidden_layers):
        old_prefix = f"transformer.h.{layer_idx}"
        new_prefix = f"model.layers.{layer_idx}"
        mapping = convert_layer(old_prefix, new_prefix)
        for old_key, new_key in mapping.items():
            assign(old_key, new_key, old_state, state_dict, rename_map)

    missing = [key for key in old_state.keys() if key not in rename_map]
    if missing:
        print(f"Skipped {len(missing)} legacy entries that have no equivalent in the shared implementation:")
        for key in missing[:10]:  # Show first 10
            print(f"  - {key}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    del old_state
    gc.collect()

    # Update config
    config.dtype = torch.bfloat16  # Use 'dtype' instead of deprecated 'torch_dtype'
    config.tie_word_embeddings = False

    # Load the checkpoint into the model
    print("Loading the checkpoint in a NanoChat model.")
    with torch.device("meta"):
        model = NanoChatForCausalLM(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")

    if hasattr(model.config, "_name_or_path"):
        del model.config._name_or_path

    print("Saving the model.")
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    try:
        # Try with device_map if accelerate is available and working
        NanoChatForCausalLM.from_pretrained(output_dir, dtype=torch.bfloat16, device_map="auto")
    except Exception as e:
        # Fallback without device_map (needed on some systems like Mac without CUDA)
        print(f"Note: device_map='auto' failed ({type(e).__name__}), trying without device_map...")
        NanoChatForCausalLM.from_pretrained(output_dir, dtype=torch.bfloat16)
    print("Model reloaded successfully.")

    return config


def write_tokenizer(input_dir, output_dir):
    """Convert and save the tokenizer."""
    input_path = Path(input_dir)

    # Check if tokenizer is in a subdirectory (nanochat structure)
    tokenizer_dir = input_path / "tokenizer"
    if not tokenizer_dir.exists():
        tokenizer_dir = input_path

    # Convert the pickle tokenizer to HF format
    tokenizer_pkl = tokenizer_dir / "tokenizer.pkl"
    if tokenizer_pkl.exists():
        try:
            import pickle

            from transformers.integrations.tiktoken import convert_tiktoken_to_fast

            print(f"Converting tokenizer from {tokenizer_pkl}")
            with open(tokenizer_pkl, "rb") as f:
                tok_pkl = pickle.load(f)
            convert_tiktoken_to_fast(tok_pkl, output_dir)
            print("Converted tokenizer.pkl to HuggingFace format")
            return True
        except Exception as e:
            print(f"Warning: Failed to convert tokenizer.pkl: {e}")
            # Fallback: copy tokenizer files if they exist
            for filename in ("tokenizer.json", "tokenizer_config.json"):
                src = tokenizer_dir / filename
                if src.exists():
                    (Path(output_dir) / filename).write_bytes(src.read_bytes())
    else:
        # No pickle tokenizer, copy JSON files
        for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src = tokenizer_dir / filename
            if src.exists():
                (Path(output_dir) / filename).write_bytes(src.read_bytes())

    print("Tokenizer saved successfully.")
    return True


def run_test(output_dir: str, prompt: str, max_new_tokens: int = 64) -> None:
    """Run a quick generation test to verify the converted model works correctly."""
    from transformers import AutoTokenizer, NanoChatForCausalLM

    print(f"Running quick generation test with prompt: {prompt}")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = NanoChatForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print(f"Generated text: {generated}")


def find_checkpoint_dir(source: str = "sft", model_tag: str = None) -> Path:
    """
    Find the checkpoint directory based on source type.

    Args:
        source: One of "base", "mid", "sft", "sft_int8", "rl"
        model_tag: Optional model tag (e.g., "d34"), auto-detected if None

    Returns:
        Path to the checkpoint directory
    """
    model_dir_map = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "sft_int8": "chatsft_checkpoints_int8",
        "rl": "chatrl_checkpoints",
    }

    if source not in model_dir_map:
        raise ValueError(f"Invalid source: {source}. Must be one of {list(model_dir_map.keys())}")

    base_dir = get_base_dir()
    checkpoints_dir = Path(base_dir) / model_dir_map[source]

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints_dir}")

    # Find model tag if not specified
    if model_tag is None:
        import re
        model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(checkpoints_dir / f)]
        if not model_tags:
            raise FileNotFoundError(f"No model directories found in {checkpoints_dir}")

        # Try to find d<number> pattern and pick the largest
        candidates = []
        for tag in model_tags:
            match = re.match(r"d(\d+)", tag)
            if match:
                candidates.append((int(match.group(1)), tag))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            model_tag = candidates[0][1]
        else:
            # Pick most recently modified
            model_tags.sort(key=lambda x: os.path.getmtime(checkpoints_dir / x), reverse=True)
            model_tag = model_tags[0]

        print(f"Auto-detected model tag: {model_tag}")

    return checkpoints_dir / model_tag


def main():
    parser = argparse.ArgumentParser(
        description="Convert NanoChat checkpoints to HuggingFace transformers format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert SFT checkpoint (auto-detect from cache)
    python -m scripts.convert_to_hf --output_dir ./hf_model

    # Convert from specific source
    python -m scripts.convert_to_hf --source mid --output_dir ./hf_model

    # Convert from custom directory
    python -m scripts.convert_to_hf --input_dir /path/to/checkpoint --output_dir ./hf_model

    # With test generation
    python -m scripts.convert_to_hf --output_dir ./hf_model --test_prompt "Hello!"
        """
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to the original checkpoint directory. If not provided, will use --source to find checkpoint.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="sft",
        choices=["base", "mid", "sft", "sft_int8", "rl"],
        help="Source checkpoint type (default: sft). Only used if --input_dir is not provided.",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="Model tag (e.g., 'd34'). Auto-detected if not provided.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Path to tokenizer directory. If not provided, will look in standard locations.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="Whether or not to save using `safetensors`.",
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default=None,
        help="Optional prompt for a quick generation test",
    )
    args = parser.parse_args()

    # Determine input directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = find_checkpoint_dir(args.source, args.model_tag)

    print(f"Input directory: {input_dir}")

    # Check that transformers has NanoChat support
    try:
        from transformers import NanoChatConfig, NanoChatForCausalLM
    except ImportError:
        print("Error: transformers library with NanoChat support not found.")
        print("Please install the latest transformers version with NanoChat support:")
        print("  pip install git+https://github.com/huggingface/transformers.git")
        sys.exit(1)

    # Convert the model
    write_model(
        input_dir,
        args.output_dir,
        safe_serialization=args.safe_serialization,
    )

    # Convert the tokenizer
    tokenizer_dir = args.tokenizer_dir
    if tokenizer_dir is None:
        # Try standard locations
        base_dir = get_base_dir()
        standard_tokenizer_dir = Path(base_dir) / "tokenizer"
        if standard_tokenizer_dir.exists():
            tokenizer_dir = standard_tokenizer_dir
        else:
            tokenizer_dir = input_dir

    write_tokenizer(tokenizer_dir, args.output_dir)

    # Run test if requested
    if args.test_prompt:
        run_test(args.output_dir, args.test_prompt)

    print(f"\n✅ Conversion complete! Model saved to: {args.output_dir}")
    print(f"\nYou can now use the model with transformers:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()
