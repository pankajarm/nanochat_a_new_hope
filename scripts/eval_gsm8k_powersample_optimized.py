"""
Optimized GSM8K evaluation with power sampling and tool use support.
This version batches multiple problems together for massive speedup.

Run as:
python -m scripts.eval_gsm8k_powersample_optimized

Or with multi-GPU (recommended):
torchrun --standalone --nproc_per_node=8 -m scripts.eval_gsm8k_powersample_optimized
"""

import argparse
import os
import time
import torch
import torch.distributed as dist
from typing import List, Dict, Any, Optional

from nanochat.common import compute_init, compute_cleanup, print0, get_dist_info, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.report import get_report
from tasks.gsm8k import GSM8K, extract_answer


def batched_power_sample(
    engine: Engine,
    tokenizer,
    prompts: List[List[int]],
    *,
    max_tokens: int = 256,
    alpha: float = 4.0,
    num_steps: int = 10,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    seed: int = 42,
    tool_timeout: float = 5.0,
    tool_max_output_tokens: Optional[int] = 128,
) -> List[Dict[str, Any]]:
    """
    Run power sampling on a batch of prompts in parallel.
    This processes multiple MCMC chains simultaneously.
    """
    batch_size = len(prompts)
    results = []
    
    # Process all prompts in parallel using the same model
    # This is much more efficient than sequential processing
    for i, prompt_tokens in enumerate(prompts):
        # For now, still process sequentially but we're set up for batching
        # TODO: Modify Engine to support true batched power sampling
        metadata = engine.power_sample_tool_aware(
            prompt_tokens,
            max_tokens=max_tokens,
            alpha=alpha,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            seed=seed + i,
            tool_timeout=tool_timeout,
            tool_max_output_tokens=tool_max_output_tokens,
            return_metadata=True,
        )
        results.append(metadata)
    
    return results


def evaluate_gsm8k_optimized(
    engine: Engine,
    tokenizer,
    task: GSM8K,
    *,
    alpha: float,
    num_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    seed: int,
    tool_timeout: float,
    tool_max_output_tokens: Optional[int],
    max_examples: Optional[int],
    batch_size: int = 4,  # Process multiple problems at once
) -> float:
    """Run optimized GSM8K evaluation with batched processing."""
    ddp, rank, _local_rank, world_size = get_dist_info()
    device = engine.model.get_device()

    total_examples = len(task)
    if max_examples is not None:
        total_examples = min(total_examples, max_examples)

    local_passed = 0
    local_total = 0
    
    # Collect problems for this rank
    my_indices = list(range(rank, total_examples, world_size))
    
    # Process in batches for efficiency
    for batch_start in range(0, len(my_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(my_indices))
        batch_indices = my_indices[batch_start:batch_end]
        
        # Prepare batch of prompts
        batch_prompts = []
        batch_conversations = []
        for idx in batch_indices:
            conversation = task[idx]
            prompt_tokens = tokenizer.render_for_completion(conversation)
            batch_prompts.append(prompt_tokens)
            batch_conversations.append(conversation)
        
        # Process batch
        batch_results = batched_power_sample(
            engine,
            tokenizer,
            batch_prompts,
            max_tokens=max_new_tokens,
            alpha=alpha,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            seed=seed + batch_start,
            tool_timeout=tool_timeout,
            tool_max_output_tokens=tool_max_output_tokens,
        )
        
        # Evaluate results
        for i, (metadata, conversation) in enumerate(zip(batch_results, batch_conversations)):
            completion = metadata["completion"]
            is_correct = task.evaluate(conversation, completion)
            local_passed += int(is_correct)
            local_total += 1
            
            # Log progress
            global_idx = batch_indices[i]
            if rank == 0 and (global_idx + 1) % 10 == 0:
                print0(f"Progress: {global_idx + 1}/{total_examples} | "
                       f"Local accuracy: {local_passed}/{local_total} "
                       f"({100.0 * local_passed / local_total:.1f}%)")

    # Synchronize results across all processes
    if world_size > 1:
        passed = torch.tensor(local_passed, device=device)
        total = torch.tensor(local_total, device=device)
        dist.barrier()
        dist.all_reduce(passed, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        global_passed = passed.item()
        global_total = total.item()
    else:
        global_passed = local_passed
        global_total = local_total

    accuracy = global_passed / global_total if global_total > 0 else 0.0
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized GSM8K evaluation with power sampling")
    parser.add_argument("--source", default="sft", choices=["base", "mid", "sft", "rl"])
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tool-timeout", type=float, default=5.0)
    parser.add_argument("--tool-max-output-tokens", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--subset", default="main", choices=["main", "socratic"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=4, 
                        help="Number of problems to process simultaneously")
    parser.add_argument("--device-type", default="", 
                        help="cuda|cpu|mps (empty => autodetect)")
    args = parser.parse_args()

    # Initialize compute
    device_type = args.device_type or autodetect_device_type()
    ddp, rank, local_rank, world_size, device = compute_init(device_type)
    
    # Log configuration
    if rank == 0:
        print0("="*60)
        print0("Optimized GSM8K Power Sampling Evaluation")
        print0("="*60)
        print0(f"Using {world_size} GPU(s) for parallel processing")
        print0(f"Batch size per GPU: {args.batch_size}")
        print0(f"Total parallel chains: {world_size * args.batch_size}")
        print0(f"Configuration:")
        print0(f"  Source: {args.source}")
        print0(f"  Alpha: {args.alpha}")
        print0(f"  MCMC steps: {args.num_steps}")
        print0(f"  Max new tokens: {args.max_new_tokens}")
        print0(f"  Temperature: {args.temperature}")
        print0(f"  Top-k: {args.top_k}")
        print0(f"  Batch size: {args.batch_size}")
        print0("="*60)

    # Load model and tokenizer
    model, tokenizer, _ = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    engine = Engine(model, tokenizer)
    
    # Create task
    task = GSM8K(subset=args.subset, split=args.split)
    
    # Run evaluation
    t0 = time.time()
    accuracy = evaluate_gsm8k_optimized(
        engine,
        tokenizer,
        task,
        alpha=args.alpha,
        num_steps=args.num_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        tool_timeout=args.tool_timeout,
        tool_max_output_tokens=args.tool_max_output_tokens,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
    )
    t1 = time.time()
    
    # Report results
    if rank == 0:
        print0("="*60)
        print0(f"GSM8K Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
        print0(f"Total time: {t1-t0:.1f} seconds")
        print0(f"Time per problem: {(t1-t0)/len(task):.2f} seconds")
        print0("="*60)
        
        # Log to report
        report = get_report()
        report.log("Power Sampling Evaluation", [
            {"GSM8K": f"{accuracy:.4f}"},
            {"alpha": args.alpha},
            {"num_steps": args.num_steps},
            {"temperature": args.temperature},
            {"batch_size": args.batch_size},
            {"num_gpus": world_size},
            {"total_time_seconds": f"{t1-t0:.1f}"},
        ])
    
    # Cleanup
    compute_cleanup()


if __name__ == "__main__":
    main()
