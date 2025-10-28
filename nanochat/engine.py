"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import math
import signal
import warnings
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_model
from nanochat.execution import execute_code

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula)
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)
# -----------------------------------------------------------------------------
class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    Note that the .pos advances automatically after the last layer of the Transformer inserts.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer.
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0 # current position in time in the cache

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        """
        Prefill given another KV cache. Optionally expand along batch dim.
        This is used when we do batch 1 prefill and then want to generate
        multiple samples in parallel from there.
        """
        # 1) validate the shapes
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                # num_layers, batch_size, num_heads, head_dim must match
                assert dim1 == dim2, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 2:
                # batch_size can be expanded
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:
                # seq_len: self must be longer than other
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        # 2) initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        # 4) update the pos
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # Insert new keys/values to the cache and return the full cache so far
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024 # as much as we need plus buffer of 1024
            t_needed = (t_needed + 1023) & ~1023 # then round up to the nearest multiple of 1024
            current_shape = list(self.kv_cache.shape)
            current_shape[4] = t_needed
            self.kv_cache.resize_(current_shape)
        # Insert k, v into the cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        # Increment pos after the last layer of the Transformer processes
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use
        self._special_tokens: Dict[str, int] = {}
        for token_name in [
            "<|python_start|>",
            "<|python_end|>",
            "<|output_start|>",
            "<|output_end|>",
            "<|assistant_end|>",
        ]:
            try:
                self._special_tokens[token_name] = self.tokenizer.encode_special(token_name)
            except Exception:
                # Some tokenizers used in tests may not have all special tokens.
                # Store lazily when requested.
                pass

    def _get_special_token(self, token_name: str) -> int:
        if token_name not in self._special_tokens:
            self._special_tokens[token_name] = self.tokenizer.encode_special(token_name)
        return self._special_tokens[token_name]

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        python_start = self._get_special_token("<|python_start|>")
        python_end = self._get_special_token("<|python_end|>")
        output_start = self._get_special_token("<|output_start|>")
        output_end = self._get_special_token("<|output_end|>")
        assistant_end = self._get_special_token("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
        sampled_tokens = next_ids[:, 0].tolist()

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        first_iteration = True
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Get sampled tokens - either from prefill or from forward pass
            if first_iteration:
                # Use the tokens we already sampled from prefill
                sampled_tokens = [sampled_tokens[0]] * num_samples  # Broadcast first token to all rows
                # TODO: we should sample a token for each row instead of broadcasting
                first_iteration = False
            else:
                # Forward the model and get the next token for each row
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size) at last time step
                next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1
            # Prepare ids for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self._get_special_token("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks

    @torch.inference_mode()
    def _get_sequence_log_prob(self, tokens: List[int]) -> float:
        """Compute the log probability of a sequence of tokens under the model."""
        if len(tokens) <= 1:
            return 0.0
        device = self.model.get_device()
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:]
        logits = self.model.forward(inputs)
        logits = logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        total_log_prob = gathered.sum().item()
        return float(total_log_prob)

    def _apply_tool_use(
        self,
        tokens: List[int],
        *,
        timeout: float = 5.0,
        max_output_tokens: Optional[int] = None,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        Scan for python tool invocations, execute them, and append outputs.

        Returns the potentially-extended token list together with metadata about
        executed tool calls.
        """
        if not tokens:
            return list(tokens), []

        updated_tokens: List[int] = []
        tool_calls: List[Dict[str, Any]] = []

        python_start = self._get_special_token("<|python_start|>")
        python_end = self._get_special_token("<|python_end|>")
        output_start = self._get_special_token("<|output_start|>")
        output_end = self._get_special_token("<|output_end|>")

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token != python_start:
                updated_tokens.append(token)
                i += 1
                continue

            # Copy the python block tokens verbatim
            updated_tokens.append(token)
            i += 1
            expr_tokens: List[int] = []
            while i < len(tokens) and tokens[i] != python_end:
                expr_tokens.append(tokens[i])
                updated_tokens.append(tokens[i])
                i += 1

            # If we reached the end without encountering python_end, just append the
            # remaining tokens and stop processing.
            if i >= len(tokens):
                # Unmatched python_end; append the remainder as-is and stop.
                updated_tokens.extend(tokens[i:])
                break

            updated_tokens.append(tokens[i])  # add python_end
            i += 1

            # Skip execution if an output block already exists immediately after
            if i < len(tokens) and tokens[i] == output_start:
                continue

            code = self.tokenizer.decode(expr_tokens).strip()
            if not code:
                continue

            exec_result = execute_code(code, timeout=timeout)
            fragments = []
            if exec_result.stdout:
                stdout_text = exec_result.stdout.strip()
                if stdout_text:
                    fragments.append(stdout_text)
            if exec_result.stderr:
                stderr_text = exec_result.stderr.strip()
                if stderr_text:
                    fragments.append(stderr_text)
            if exec_result.error:
                fragments.append(exec_result.error)
            if not fragments:
                fragments.append("None")

            output_text = "\n".join(fragments)
            output_token_ids = self.tokenizer.encode(output_text)
            if max_output_tokens is not None:
                output_token_ids = output_token_ids[:max_output_tokens]

            updated_tokens.append(output_start)
            updated_tokens.extend(output_token_ids)
            updated_tokens.append(output_end)

            tool_calls.append(
                {
                    "code": code,
                    "output": output_text,
                    "success": exec_result.success,
                    "timeout": exec_result.timeout,
                    "memory_exceeded": exec_result.memory_exceeded,
                }
            )

        return updated_tokens, tool_calls

    @torch.inference_mode()
    def power_sample_tool_aware(
        self,
        prompt: List[int] | str,
        *,
        max_tokens: int = 256,
        alpha: float = 4.0,
        num_steps: int = 10,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        seed: int = 42,
        tool_timeout: float = 5.0,
        tool_max_output_tokens: Optional[int] = 128,
        return_metadata: bool = False,
    ) -> Dict[str, Any] | str:
        """
        Run a power-sampling inspired MCMC procedure that is aware of tool use.

        Args:
            prompt: Either a list of token ids or a raw string prompt.
            max_tokens: Maximum new tokens sampled per proposal.
            alpha: Power posterior exponent controlling exploitation vs. exploration.
            num_steps: Number of MCMC refinement steps to run.
            temperature: Sampling temperature used for proposals.
            top_k: Optional top-k truncation for proposals.
            seed: Random seed for reproducibility.
            tool_timeout: Timeout in seconds for python tool execution.
            tool_max_output_tokens: Limit on appended tool output tokens.
            return_metadata: If True, return a dictionary with extra information.

        Returns:
            Either the best completion string or a metadata dictionary containing
            the completion, tokens, log probability, and sampling history.
        """

        if isinstance(prompt, str):
            bos = self.tokenizer.get_bos_token_id()
            prompt_tokens = self.tokenizer.encode(prompt, prepend=bos)
        elif isinstance(prompt, list):
            prompt_tokens = list(prompt)
        else:
            raise TypeError("prompt must be either a string or a list of token ids")

        if not prompt_tokens:
            raise ValueError("prompt must not be empty")

        prefix_length = len(prompt_tokens)
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        def draw_sample(sample_seed: int) -> List[int]:
            sequences, _ = self.generate_batch(
                prompt_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=sample_seed,
            )
            return sequences[0]

        initial_sequence = draw_sample(seed)
        current_tokens, current_tools = self._apply_tool_use(
            initial_sequence,
            timeout=tool_timeout,
            max_output_tokens=tool_max_output_tokens,
        )
        current_log_prob = self._get_sequence_log_prob(current_tokens)

        best_tokens = current_tokens
        best_log_prob = current_log_prob
        best_tools = current_tools

        history: List[Dict[str, Any]] = [
            {
                "step": 0,
                "accepted": True,
                "log_prob": current_log_prob,
                "tools": current_tools,
            }
        ]

        for step in range(num_steps):
            proposal_seed = seed + step + 1
            proposal_sequence = draw_sample(proposal_seed)
            proposal_tokens, proposal_tools = self._apply_tool_use(
                proposal_sequence,
                timeout=tool_timeout,
                max_output_tokens=tool_max_output_tokens,
            )
            proposal_log_prob = self._get_sequence_log_prob(proposal_tokens)
            log_accept_ratio = alpha * (proposal_log_prob - current_log_prob)

            if log_accept_ratio >= 0:
                accept = True
            else:
                accept_prob = math.exp(log_accept_ratio)
                accept = torch.rand(1, generator=rng).item() < accept_prob

            history.append(
                {
                    "step": step + 1,
                    "accepted": accept,
                    "log_prob": proposal_log_prob,
                    "tools": proposal_tools,
                }
            )

            if accept:
                current_tokens = proposal_tokens
                current_log_prob = proposal_log_prob
                current_tools = proposal_tools
                if current_log_prob > best_log_prob:
                    best_tokens = current_tokens
                    best_log_prob = current_log_prob
                    best_tools = current_tools

        completion_text = self.tokenizer.decode(best_tokens[prefix_length:])
        if return_metadata:
            return {
                "completion": completion_text,
                "tokens": best_tokens,
                "log_prob": best_log_prob,
                "history": history,
                "tool_calls": best_tools,
                "prompt_length": prefix_length,
            }
        return completion_text


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
