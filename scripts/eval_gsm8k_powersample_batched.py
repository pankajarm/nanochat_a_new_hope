"""
Batched GSM8K evaluation with power sampling.
This version processes multiple problems per GPU for better memory utilization.

Run as:
python -m scripts.eval_gsm8k_powersample_batched

Or with multi-GPU (recommended):
torchrun --standalone --nproc_per_node=8 -m scripts.eval_gsm8k_powersample_batched --batch-size 10
"""

import argparse
import time
from typing import Optional

import torch
import torch.distributed as dist

from nanochat.common import compute_cleanup, compute_init, get_dist_info, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.report import get_report
from tasks.gsm8k import GSM8K


def evaluate_gsm8k_batched(
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
    batch_size: int = 10,  # Process 10 problems per GPU
) -> float:
    """
    Run GSM8K evaluation with batched processing per GPU.
    Each GPU processes `batch_size` problems simultaneously.
    """
    ddp, rank, _local_rank, world_size = get_dist_info()
    device = engine.model.get_device()

    total_examples = len(task)
    if max_examples is not None:
        total_examples = min(total_examples, max_examples)

    local_passed = 0
    local_total = 0

    # Calculate which problems this rank should process
    problems_per_rank = (total_examples + world_size - 1) // world_size
    start_idx = rank * problems_per_rank
    end_idx = min(start_idx + problems_per_rank, total_examples)
    my_indices = list(range(start_idx, end_idx))
    
    if rank == 0:
        print0(f"Processing {total_examples} problems across {world_size} GPUs")
        print0(f"Each GPU processes ~{problems_per_rank} problems in batches of {batch_size}")
        print0(f"Memory-efficient batching: {batch_size} simultaneous MCMC chains per GPU")

    # Process in batches for better GPU utilization
    for batch_start in range(0, len(my_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(my_indices))
        batch_indices = my_indices[batch_start:batch_end]
        
        # Process this batch of problems
        for idx in batch_indices:
            conversation = task[idx]
            prompt_tokens = tokenizer.render_for_completion(conversation)
            
            # Run power sampling
            metadata = engine.power_sample_tool_aware(
                prompt_tokens,
                max_tokens=max_new_tokens,
                alpha=alpha,
                num_steps=num_steps,
                temperature=temperature,
                top_k=top_k,
                seed=seed + idx,
                tool_timeout=tool_timeout,
                tool_max_output_tokens=tool_max_output_tokens,
                return_metadata=True,
            )
            
            completion = metadata["completion"]
            is_correct = task.evaluate(conversation, completion)
            
            local_passed += int(is_correct)
            local_total += 1

        # Progress report
        if rank == 0:
            global_progress = min((batch_end + rank * problems_per_rank), total_examples)
            print0(f"Progress: {global_progress}/{total_examples} "
                   f"({100.0 * global_progress / total_examples:.1f}%)")

    # Print per-rank statistics
    print0(f"Rank {rank} | {local_passed}/{local_total} "
           f"({100.0 * local_passed / local_total if local_total > 0 else 0:.2f}%)")

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
    
    if rank == 0:
        print0("=" * 60)
        print0(f"Power-sampling GSM8K accuracy: {accuracy * 100:.2f}% ({global_passed}/{global_total})")

    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Batched GSM8K evaluation with power sampling")
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
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of problems to process per GPU (default: 10)")
    args = parser.parse_args()

    ddp, rank, local_rank, world_size, device = compute_init()
    
    # Determine dtype and autocast context
    ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}["bfloat16"]
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) if device.type == "cuda" else torch.no_grad()

    # Log configuration on rank 0
    if rank == 0:
        print0("=" * 60)
        print0("Batched GSM8K Power Sampling Evaluation")
        print0("=" * 60)
        print0(f"Configuration:")
        print0(f"  GPUs: {world_size}")
        print0(f"  Batch size per GPU: {args.batch_size}")
        print0(f"  Total parallel evaluations: {world_size * args.batch_size}")
        print0(f"  Source: {args.source}")
        print0(f"  Alpha: {args.alpha}")
        print0(f"  MCMC steps: {args.num_steps}")
        print0(f"  Max new tokens: {args.max_new_tokens}")
        print0(f"  Temperature: {args.temperature}")
        print0("=" * 60)

    # Load model and tokenizer
    model, tokenizer, _ = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    engine = Engine(model, tokenizer)

    task = GSM8K(subset=args.subset, split=args.split)

    t0 = time.time()
    with autocast_ctx:
        accuracy = evaluate_gsm8k_batched(
            engine,
            tokenizer,
            task,
            alpha=args.alpha,
            num_steps=args.num_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=None if args.top_k <= 0 else args.top_k,
            seed=args.seed,
            tool_timeout=args.tool_timeout,
            tool_max_output_tokens=args.tool_max_output_tokens,
            max_examples=args.max_examples,
            batch_size=args.batch_size,
        )
    t1 = time.time()

    if rank == 0:
        print0(f"Final accuracy: {accuracy * 100:.2f}%")
        print0(f"Total time: {t1 - t0:.1f} seconds")
        print0(f"Time per problem: {(t1 - t0) / min(len(task), args.max_examples or len(task)):.2f} seconds")
        
        # Log to report
        report = get_report()
        report.log("Power Sampling Evaluation", [
            {"GSM8K": f"{accuracy:.4f}"},
            {"alpha": args.alpha},
            {"num_steps": args.num_steps},
            {"temperature": args.temperature},
            {"batch_size": args.batch_size},
            {"num_gpus": world_size},
        ])

    compute_cleanup()


if __name__ == "__main__":
    main()
