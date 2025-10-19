# nanochat training report

Generated: 2024-03-14 12:00:00

## Environment

### Git Information
- Branch: work
- Commit: abcdef0 (clean)
- Message: Add tool-aware power sampling and GSM8K evaluation

### Hardware
- Platform: Linux
- CPUs: 32 cores (64 logical)
- Memory: 256.0 GB
- GPUs: 8x NVIDIA H100
- GPU Memory: 640.0 GB total
- CUDA Version: 12.1
- Hourly Rate: $24.00/hour

### Software
- Python: 3.10.13
- PyTorch: 2.2.0

### Bloat
- Characters: 333,989
- Lines: 8,304
- Files: 44
- Tokens (approx): 83,497
- Dependencies (uv.lock lines): 2,004

## Tokenizer training
- vocab_size: 65536
- max_chars: 2000000000
- compression_ratio: 3.27

## Tokenizer evaluation
- bits_per_char: 1.09
- coverage: 99.7

## Base model training
- steps: 6000
- batch_size: 512
- loss: 1.982

## Base model loss
- CORE: 0.2219

## Base model evaluation
- CORE: 0.2219
- ARC-Easy: 0.3561
- ARC-Challenge: 0.0520
- MMLU: 0.3470
- GSM8K: 0.0455
- HumanEval: 0.0060

## Midtraining
- steps: 1000
- loss: 1.143

## Chat evaluation mid
- ARC-Easy: 0.3687
- ARC-Challenge: 0.0814
- MMLU: 0.3566
- GSM8K: 0.0456
- HumanEval: 0.0630
- ChatCORE metric: 0.0730

## Chat SFT
- steps: 500
- loss: 0.742

## Chat evaluation sft
- ARC-Easy: 0.4633
- ARC-Challenge: 0.0979
- MMLU: 0.3751
- GSM8K: 0.0758
- HumanEval: 0.0884
- ChatCORE metric: 0.0884

## Power sampling evaluation
- source: sft
- model_tag: default
- step: latest
- alpha: 4.0
- num_steps: 10
- temperature: 0.7
- top_k: 50
- max_new_tokens: 256
- seed: 0
- tool_timeout: 5.0
- tool_max_output_tokens: 128
- max_examples: all
- subset: main
- split: test
- GSM8K: 0.0875
- passed: 78
- total: 892

## Chat RL
- status: skipped (not enabled in speedrun_powersample.sh)

## Chat evaluation rl
- GSM8K: -

## Summary
- Characters: 333,989
- Lines: 8,304
- Files: 44
- Tokens (approx): 83,497
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | POWER    | RL       |
|-----------------|----------|----------|----------|----------|----------|
| ARC-Challenge   | 0.0520   | 0.0814   | 0.0979   | -        | -        |
| ARC-Easy        | 0.3561   | 0.3687   | 0.4633   | -        | -        |
| CORE            | 0.2219   | -        | -        | -        | -        |
| ChatCORE metric | -        | 0.0730   | 0.0884   | -        | -        |
| GSM8K           | 0.0455   | 0.0456   | 0.0758   | 0.0875   | -        |
| HumanEval       | 0.0060   | 0.0630   | 0.0884   | -        | -        |
| MMLU            | 0.3470   | 0.3566   | 0.3751   | -        | -        |

Total wall clock time: 3h51m
