"""Evaluate GSM8K with the tool-aware power sampling engine."""

import argparse
from typing import Optional

import torch
import torch.distributed as dist

from nanochat.common import compute_cleanup, compute_init, get_dist_info, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.report import get_report
from tasks.gsm8k import GSM8K


def evaluate_gsm8k(
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
) -> float:
    """Run GSM8K evaluation using the tool-aware sampler."""
    ddp, rank, _local_rank, world_size = get_dist_info()
    device = engine.model.get_device()

    total_examples = len(task)
    if max_examples is not None:
        total_examples = min(total_examples, max_examples)

    local_passed = 0
    local_total = 0

    for logical_index in range(rank, total_examples, world_size):
        conversation = task[logical_index]
        prompt_tokens = tokenizer.render_for_completion(conversation)
        metadata = engine.power_sample_tool_aware(
            prompt_tokens,
            max_tokens=max_new_tokens,
            alpha=alpha,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            seed=seed + logical_index,
            tool_timeout=tool_timeout,
            tool_max_output_tokens=tool_max_output_tokens,
            return_metadata=True,
        )
        completion = metadata["completion"]
        outcome = task.evaluate(conversation, completion)
        local_passed += int(outcome)
        local_total += 1
        accuracy = 100.0 * local_passed / max(local_total, 1)
        print(f"\r\033[KRank {rank} | {local_passed}/{local_total} ({accuracy:.2f}%)", end="", flush=True)

    print()

    if ddp:
        passed_tensor = torch.tensor([local_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([local_total], dtype=torch.long, device=device)
        dist.all_reduce(passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        local_passed = passed_tensor.item()
        local_total = total_tensor.item()

    final_accuracy = local_passed / max(local_total, 1)
    print0("=" * 60)
    print0(
        f"Power-sampling GSM8K accuracy: {100.0 * final_accuracy:.2f}% "
        f"({local_passed}/{local_total})"
    )
    return final_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, default="mid", help="Model source to load (base|mid|sft|rl)")
    parser.add_argument("--model-tag", type=str, default=None, help="Optional model tag override")
    parser.add_argument("--step", type=int, default=None, help="Optional checkpoint step override")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="Computation dtype")
    parser.add_argument("--alpha", type=float, default=4.0, help="Power posterior alpha parameter")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of MCMC refinement steps")
    parser.add_argument("--temperature", type=float, default=0.7, help="Proposal sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k value for proposal sampling (0 disables)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens per proposal")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--tool-timeout", type=float, default=5.0, help="Timeout for python tool execution")
    parser.add_argument(
        "--tool-max-output-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens appended from tool outputs",
    )
    parser.add_argument("--max-examples", type=int, default=None, help="Limit evaluation to this many problems")
    parser.add_argument("--subset", type=str, default="main", choices=["main", "socratic"], help="GSM8K subset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="GSM8K split")
    args = parser.parse_args()

    ddp, rank, _local_rank, _world_size, device = compute_init()
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

    model, tokenizer, _meta = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    engine = Engine(model, tokenizer)

    task = GSM8K(subset=args.subset, split=args.split)

    with autocast_ctx:
        accuracy = evaluate_gsm8k(
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
        )

    if rank == 0:
        print0(f"Final accuracy: {accuracy * 100:.2f}%")
        
        # Log to report
        report = get_report()
        report.log("Power Sampling Evaluation", [
            {"GSM8K": f"{accuracy:.4f}"},
            {"alpha": args.alpha},
            {"num_steps": args.num_steps},
            {"temperature": args.temperature},
            {"top_k": args.top_k},
            {"seed": args.seed},
        ])

    compute_cleanup()


if __name__ == "__main__":
    main()
