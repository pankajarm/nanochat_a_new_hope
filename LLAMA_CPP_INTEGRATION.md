# NanoChat Native llama.cpp Integration Plan

## Executive Summary

This document provides a complete implementation plan for adding native NanoChat architecture support to llama.cpp, eliminating the current workarounds that degrade output quality.

### Current Status (Workaround)
- Uses LLaMA architecture with weight duplication hack
- FC1 duplicated for both gate and up weights
- Activation mismatch: silu vs relu2
- **Output quality is significantly degraded**

### Goal
- Native `LLM_ARCH_NANOCHAT` implementation
- Correct 2-layer MLP with relu2 (`LLM_FFN_RELU_SQR`)
- Parameter-free RMSNorm (pass `nullptr` for weights)
- QK norm AFTER RoPE
- Final logit softcapping (15.0)
- Token embedding norm

---

## Key Architectural Differences

| Feature | NanoChat | LLaMA | llama.cpp Support |
|---------|----------|-------|-------------------|
| MLP Structure | 2-layer: fc1 → act → fc2 | 3-layer gated | ✅ Use `LLM_FFN_SEQ` like GPT-2/Nemotron |
| Activation | relu2: `relu(x).square()` | silu | ✅ `LLM_FFN_RELU_SQR` exists |
| RMSNorm | Parameter-free (no weights) | Learnable weights | ✅ Pass `nullptr` to `build_norm()` |
| QK Norm | AFTER RoPE | Usually before or none | ⚠️ Need custom order |
| RoPE Rotation | Flipped: `(x2, -x1)` | Standard: `(-x2, x1)` | ⚠️ May need new rope_type |
| Logit Softcapping | 15.0 * tanh(x/15.0) | None | ✅ `f_final_logit_softcapping` |
| Token Embd Norm | After embedding | None | ✅ `tok_norm` pattern exists |

---

## Phase 1: Architecture Registration

### File: `src/llama-arch.h`

Add to `llm_arch` enum (around line 119, before `LLM_ARCH_UNKNOWN`):

```cpp
    LLM_ARCH_NANOCHAT,
    LLM_ARCH_UNKNOWN,
```

### File: `src/llama-arch.cpp`

#### 1. Add architecture name mapping (around line 12):

```cpp
    { LLM_ARCH_NANOCHAT,         "nanochat"         },
```

#### 2. Add tensor name mappings (add new section around line 500):

```cpp
    {
        LLM_ARCH_NANOCHAT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
```

---

## Phase 2: Model Loading

### File: `src/llama-model.cpp`

#### 1. Add hyperparameter loading (add case in `llm_load_hparams`):

```cpp
        case LLM_ARCH_NANOCHAT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);
                
                // NanoChat uses standard RoPE with flipped rotation
                // For now, use NORMAL and handle the flip in compute graph
                hparams.rope_type = LLAMA_ROPE_TYPE_NORM;
                
                switch (hparams.n_layer) {
                    case 12: model.type = LLM_TYPE_124M; break;
                    case 34: model.type = LLM_TYPE_1B;   break;
                    default: model.type = LLM_TYPE_UNKNOWN;
                }
            } break;
```

#### 2. Add tensor loading (add case in `llm_load_tensors`):

```cpp
        case LLM_ARCH_NANOCHAT:
            {
                // Token embeddings
                tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
                
                // Output (lm_head)
                output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);
                
                // NanoChat has parameter-free norms, so we DON'T load norm weights
                // The compute graph will use ggml_rms_norm without multiplication
                
                for (int i = 0; i < n_layer; ++i) {
                    auto & layer = layers[i];
                    
                    // Attention projections
                    layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                    layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa()}, 0);
                    layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa()}, 0);
                    layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
                    
                    // MLP (2-layer, no gate)
                    layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                    layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                }
            } break;
```

---

## Phase 3: Compute Graph Implementation

### File: `src/models/nanochat.cpp` (NEW FILE)

```cpp
#include "models.h"

llm_build_nanochat::llm_build_nanochat(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Token embeddings
    inpL = build_inp_embd(model.tok_embd);

    // NanoChat: norm immediately after token embeddings (parameter-free)
    // Pass nullptr for weight to skip multiplication
    inpL = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
    cb(inpL, "tok_norm", -1);

    // Position input for RoPE
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // Pre-attention norm (parameter-free RMSNorm)
        cur = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
        cb(cur, "attn_norm", il);

        // Self-attention
        {
            // Q, K, V projections
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // Apply RoPE FIRST
            // NOTE: NanoChat uses flipped rotation (x2, -x1) instead of (-x2, x1)
            // This may require a custom rope_type or accepting approximation
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            // QK norm AFTER RoPE (parameter-free)
            // This is the key difference from most other models
            Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
            cb(Qcur, "Qcur_normed", il);

            Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
            cb(Kcur, "Kcur_normed", il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr,  // no bias
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 
                    1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // Pre-FFN norm (parameter-free RMSNorm)
        cur = ggml_rms_norm(ctx0, ffn_inp, hparams.f_norm_rms_eps);
        cb(cur, "ffn_norm", il);

        // FFN: 2-layer MLP with relu2 (LLM_FFN_RELU_SQR)
        // out = ffn_down(relu2(ffn_up(x)))
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,  // up
                nullptr,                   nullptr, nullptr,  // gate (not used)
                model.layers[il].ffn_down, nullptr, nullptr,  // down
                nullptr,
                LLM_FFN_RELU_SQR, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

        // Residual connection
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }

    cur = inpL;

    // Final norm (parameter-free RMSNorm)
    cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);

    // Final logit softcapping: 15.0 * tanh(logits / 15.0)
    if (hparams.f_final_logit_softcapping > 0.0f) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
```

### File: `src/models/models.h`

Add class declaration:

```cpp
class llm_build_nanochat : public llm_graph_context {
public:
    llm_build_nanochat(const llama_model & model, const llm_graph_params & params);
};
```

### File: `src/llama-model.cpp`

Add to the graph builder switch statement:

```cpp
        case LLM_ARCH_NANOCHAT:
            {
                llm = std::make_unique<llm_build_nanochat>(model, params);
            } break;
```

---

## Phase 4: GGUF Converter Update

### File: `convert_and_upload_gguf.py`

Update to use native nanochat architecture:

```python
def convert_nanochat_to_gguf(input_dir: str, output_file: str, dtype: str = "f16") -> str:
    """
    Convert NanoChat HuggingFace model to GGUF format with native nanochat architecture.
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
    if safetensors_files:
        from safetensors.torch import load_file
        for sf in safetensors_files:
            state_dict.update(load_file(sf))
    
    # Extract config
    vocab_size = config.get("vocab_size", 65536)
    hidden_size = config.get("hidden_size", 2176)
    num_layers = config.get("num_hidden_layers", 34)
    num_heads = config.get("num_attention_heads", 17)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    max_seq_len = config.get("max_position_embeddings", 2048)
    intermediate_size = config.get("intermediate_size", hidden_size * 4)
    rms_norm_eps = config.get("rms_norm_eps", 1e-6)
    rope_theta = config.get("rope_theta", 10000.0)
    final_logit_softcapping = config.get("final_logit_softcapping", 15.0)
    
    # GGUF dtype mapping
    dtype_map = {
        "f32": GGMLQuantizationType.F32,
        "f16": GGMLQuantizationType.F16,
        "bf16": GGMLQuantizationType.BF16,
    }
    gguf_dtype = dtype_map.get(dtype, GGMLQuantizationType.F16)
    
    def to_numpy(tensor):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        return tensor.to(torch.float16 if dtype != "f32" else torch.float32).numpy()
    
    # Create GGUF writer with NATIVE nanochat architecture
    writer = GGUFWriter(str(output_path), arch="nanochat")
    
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
    
    # NanoChat-specific: final logit softcapping
    writer.add_float32("nanochat.final_logit_softcapping", final_logit_softcapping)
    
    # Token IDs
    writer.add_bos_token_id(config.get("bos_token_id", 0))
    writer.add_eos_token_id(config.get("eos_token_id", 1))
    writer.add_pad_token_id(config.get("pad_token_id", 1))
    
    # ... (tokenizer handling same as before)
    
    # =========================================================================
    # MODEL TENSORS - Native NanoChat format (no weight duplication!)
    # =========================================================================
    
    # Token embeddings
    if "model.embed_tokens.weight" in state_dict:
        writer.add_tensor("token_embd.weight", to_numpy(state_dict["model.embed_tokens.weight"]))
    
    # Output head (lm_head)  
    if "lm_head.weight" in state_dict:
        writer.add_tensor("output.weight", to_numpy(state_dict["lm_head.weight"]))
    
    # NOTE: No output_norm.weight needed - NanoChat has parameter-free norms
    
    # Layer tensors
    for i in range(num_layers):
        hf = f"model.layers.{i}"
        blk = f"blk.{i}"
        
        # Attention projections
        for src, dst in [("q_proj", "attn_q"), ("k_proj", "attn_k"), 
                         ("v_proj", "attn_v"), ("o_proj", "attn_output")]:
            key = f"{hf}.self_attn.{src}.weight"
            if key in state_dict:
                writer.add_tensor(f"{blk}.{dst}.weight", to_numpy(state_dict[key]))
        
        # MLP weights - proper 2-layer (no gate duplication!)
        fc1_key = f"{hf}.mlp.fc1.weight"
        fc2_key = f"{hf}.mlp.fc2.weight"
        
        if fc1_key in state_dict:
            writer.add_tensor(f"{blk}.ffn_up.weight", to_numpy(state_dict[fc1_key]))
        if fc2_key in state_dict:
            writer.add_tensor(f"{blk}.ffn_down.weight", to_numpy(state_dict[fc2_key]))
        
        # NOTE: No attn_norm.weight or ffn_norm.weight - parameter-free norms
    
    # Write file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    return str(output_path)
```

---

## Phase 5: Testing & Validation

### Test Plan

1. **Build Tests**
   ```bash
   cd llama.cpp
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --target llama-cli llama-quantize -j
   ```

2. **Conversion Test**
   ```bash
   python convert_and_upload_gguf.py --arch nanochat --skip-upload
   ```

3. **Load Test**
   ```bash
   ./build/bin/llama-cli -m nanochat-d34-sft-f16.gguf -p "Hello" -n 10
   ```

4. **Validation Against HuggingFace**
   - Compare logits for same prompt
   - Compare generated text quality
   - Benchmark perplexity

---

## RoPE Rotation Direction Issue

### Problem
NanoChat uses flipped RoPE rotation compared to standard:
- **NanoChat**: `y1 = x1*cos + x2*sin`, `y2 = x1*(-sin) + x2*cos`
- **Standard**: `y1 = x1*cos - x2*sin`, `y2 = x1*sin + x2*cos`

### Options

1. **Add new GGML_ROPE_TYPE_NANOCHAT** (cleanest, requires GGML changes)
2. **Negate sin values in the GGUF** (workaround, may work)
3. **Accept approximation** (simplest, some quality loss)

### Recommended Approach
Start with option 3 (accept approximation) to get a working implementation, then contribute option 1 to GGML if quality difference is significant.

---

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `src/llama-arch.h` | Modify | Add `LLM_ARCH_NANOCHAT` enum |
| `src/llama-arch.cpp` | Modify | Add arch name + tensor mappings |
| `src/llama-model.cpp` | Modify | Add hparams + tensor loading |
| `src/models/nanochat.cpp` | Create | Compute graph implementation |
| `src/models/models.h` | Modify | Add class declaration |
| `convert_and_upload_gguf.py` | Modify | Update for native arch |

---

## Reference Models in llama.cpp

- **Nemotron** (`models/nemotron.cpp`) - Uses `LLM_FFN_RELU_SQR` with 2-layer MLP
- **Qwen3** (`models/qwen3.cpp`) - Has QK norm (before RoPE)
- **Gemma2** (`models/gemma2-iswa.cpp`) - Has final logit softcapping
- **GPT-2** (`models/gpt2.cpp`) - 2-layer MLP structure
- **RWKV6** (`models/rwkv6.cpp`) - Has token embedding norm

---

## Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Architecture Registration | 1-2 hours | None |
| Phase 2: Model Loading | 2-3 hours | Phase 1 |
| Phase 3: Compute Graph | 3-4 hours | Phase 2 |
| Phase 4: GGUF Converter | 1-2 hours | Phase 1 |
| Phase 5: Testing | 2-3 hours | All phases |
| **Total** | **~10-15 hours** | |

---

## Next Steps

1. Clone latest llama.cpp: `git clone https://github.com/ggml-org/llama.cpp`
2. Create feature branch: `git checkout -b feat/nanochat-arch`
3. Implement phases 1-3 in order
4. Update GGUF converter (phase 4)
5. Test end-to-end (phase 5)
6. Submit PR to llama.cpp

---

*Document created: December 9, 2025*
*NanoChat version: d34-sft (2176 hidden, 34 layers, 17 heads)*
