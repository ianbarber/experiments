#!/usr/bin/env python3
"""
Phase 0.1: Extract hidden states from correct LLM completions.

For each problem in the dataset:
1. Generate a completion with the base model (batched for throughput)
2. Verify if the answer is correct
3. If correct, run a batched forward pass to extract hidden states at:
   - Problem state: last token of the prompt, at every layer
   - Solution state: last generated token, at every layer
   - First-step state: first generated token, at every layer
4. Save states and metadata

Usage:
    python src/extract_states.py --model Qwen/Qwen3-4B --dataset gsm8k --split train
    python src/extract_states.py --model Qwen/Qwen3-4B --dataset math --split train
    python src/extract_states.py --model Qwen/Qwen3-4B --dataset gsm8k --split train --batch_size 12
"""

import argparse
import json
import re
import sys
import time

import torch
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Answer extraction & verification
# ---------------------------------------------------------------------------

def extract_number(text: str) -> float | None:
    """Extract the final numerical answer from text (GSM8K style)."""
    # Look for #### marker first
    if "####" in text:
        after = text.split("####")[-1].strip()
        after = after.replace(",", "").replace("$", "").replace("%", "").strip()
        # Take the first number-like thing after ####
        m = re.search(r'-?\d+(?:\.\d+)?', after)
        if m:
            return float(m.group())

    # Fallback: last number in text
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    return None


def extract_boxed(text: str) -> str | None:
    """Extract content of the last \\boxed{...} (handles nested braces)."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 1
        j = idx + 7
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[idx + 7 : j - 1])
        i = j
    return results[-1].strip() if results else None


def check_gsm8k(generated: str, reference: str) -> bool:
    pred = extract_number(generated)
    ref = extract_number(reference)
    if pred is None or ref is None:
        return False
    return abs(pred - ref) < 1e-3


def check_math(generated: str, reference: str) -> bool:
    pred = extract_boxed(generated)
    ref = extract_boxed(reference)
    if pred is None:
        return False
    if ref is None:
        # Reference might be raw answer without \boxed
        ref = reference.strip()
    return " ".join(pred.split()) == " ".join(ref.split())


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(question: str, dataset: str) -> str:
    if dataset == "gsm8k":
        return (
            "Solve the following math problem step by step. "
            "After your solution, write the final numerical answer after \"####\".\n\n"
            f"Problem: {question}\n\nSolution:"
        )
    elif dataset == "math":
        return (
            "Solve the following math problem. Show your work, then put your "
            "final answer in \\boxed{}.\n\n"
            f"Problem: {question}\n\nSolution:"
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Attention backend detection
# ---------------------------------------------------------------------------

def detect_and_log_attention(model):
    """Detect which attention backend the model is using and log it."""
    config = model.config
    attn_impl = getattr(config, "_attn_implementation", None)
    attn_impl_v2 = getattr(config, "attn_implementation", None)
    actual = attn_impl_v2 or attn_impl

    if actual == "flash_attention_2":
        print(f"  Attention backend: flash_attention_2 (optimal)")
    elif actual == "sdpa":
        print(f"  Attention backend: sdpa (PyTorch scaled_dot_product_attention)")
        print(f"  NOTE: flash_attention_2 may offer better throughput if available.")
    elif actual == "eager":
        print(f"  Attention backend: eager (slow, not recommended)")
        print(f"  WARNING: Consider re-running with flash_attention_2 or sdpa.")
    else:
        print(f"  Attention backend: {actual} (unknown)")

    return actual


# ---------------------------------------------------------------------------
# Answer marker location
# ---------------------------------------------------------------------------

def _find_answer_offset(generated_text, generated_ids, tokenizer, dataset):
    """Find the token offset within generated_ids where the answer marker appears.

    For GSM8K: finds "####" in text, maps to token position.
    For MATH: finds last "\\boxed{" in text, maps to token position.

    Returns offset (int) within generated tokens, or None if not found.
    """
    if dataset == "gsm8k":
        marker = "####"
    elif dataset == "math":
        marker = "\\boxed{"
    else:
        return None

    char_pos = generated_text.find(marker)
    if char_pos == -1:
        return None

    # Map character position to token position by decoding prefix and counting tokens
    prefix_text = generated_text[:char_pos]
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    return len(prefix_ids)


# ---------------------------------------------------------------------------
# Batched hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states_batched(model, sequences, prompt_lens, gen_lens,
                                   device, pad_token_id, answer_offsets=None):
    """
    Batched forward pass to extract hidden states at key positions.

    Args:
        model: the language model
        sequences: list of 1D tensors, each being prompt + generated tokens
        prompt_lens: list of ints, prompt length for each sequence
        gen_lens: list of ints, number of generated tokens for each sequence
        device: target device
        pad_token_id: token id used for padding
        answer_offsets: optional list of ints — offset within the generated tokens
            where the answer marker (####) appears. None means not available.

    Returns:
        list of dicts, each with:
            problem_states:    (num_layers, hidden_dim) -- last prompt token
            solution_states:   (num_layers, hidden_dim) -- last generated token
            first_step_states: (num_layers, hidden_dim) -- first generated token
            answer_states:     (num_layers, hidden_dim) -- token before answer marker
            mid_states:        (num_layers, hidden_dim) -- mid-solution token
    """
    batch_size = len(sequences)
    if batch_size == 0:
        return []

    # Pad sequences for batched forward pass
    max_len = max(s.shape[0] for s in sequences)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        input_ids[i, :seq_len] = seq
        attention_mask[i, :seq_len] = 1

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states) - 1
    hidden_dim = hidden_states[0].shape[-1]

    results = []
    for i in range(batch_size):
        seq_len = sequences[i].shape[0]
        problem_idx = prompt_lens[i] - 1
        solution_idx = seq_len - 1
        first_step_idx = prompt_lens[i]
        mid_idx = prompt_lens[i] + gen_lens[i] // 2

        # Answer token: one token before the #### marker if available,
        # otherwise fall back to mid-point of generation
        if answer_offsets is not None and answer_offsets[i] is not None:
            answer_idx = prompt_lens[i] + answer_offsets[i]
            # Clamp: use the token just before the marker
            answer_idx = max(prompt_lens[i], min(answer_idx - 1, seq_len - 1))
        else:
            answer_idx = mid_idx  # fallback

        problem_states = torch.zeros(num_layers, hidden_dim, dtype=torch.float16)
        solution_states = torch.zeros(num_layers, hidden_dim, dtype=torch.float16)
        first_step_states = torch.zeros(num_layers, hidden_dim, dtype=torch.float16)
        answer_states = torch.zeros(num_layers, hidden_dim, dtype=torch.float16)
        mid_states = torch.zeros(num_layers, hidden_dim, dtype=torch.float16)

        for li in range(num_layers):
            hs = hidden_states[li + 1]
            problem_states[li] = hs[i, problem_idx].to(torch.float16).cpu()
            solution_states[li] = hs[i, solution_idx].to(torch.float16).cpu()
            if first_step_idx < seq_len:
                first_step_states[li] = hs[i, first_step_idx].to(torch.float16).cpu()
            if mid_idx < seq_len:
                mid_states[li] = hs[i, mid_idx].to(torch.float16).cpu()
            if answer_idx < seq_len:
                answer_states[li] = hs[i, answer_idx].to(torch.float16).cpu()

        results.append({
            "problem_states": problem_states,
            "solution_states": solution_states,
            "first_step_states": first_step_states,
            "answer_states": answer_states,
            "mid_states": mid_states,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract hidden states from correct LLM completions"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "math"], default="gsm8k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of examples to process")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="data/hidden_states")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--use_chat", action="store_true",
                        help="Use chat template (recommended for instruct models)")
    parser.add_argument("--no_think", action="store_true",
                        help="Disable thinking mode (Qwen3)")
    parser.add_argument("--save_every", type=int, default=200,
                        help="Save checkpoint every N correct examples")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of examples to generate in parallel (default: 8)")
    parser.add_argument("--extraction_batch_size", type=int, default=None,
                        help="Batch size for forward pass extraction (default: same as --batch_size)")
    args = parser.parse_args()

    if args.extraction_batch_size is None:
        args.extraction_batch_size = args.batch_size

    # --- Device & dtype ---
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]

    # --- Load model with flash attention ---
    print(f"Loading model: {args.model} (dtype={args.dtype}, device={device})")

    # Try to load with flash_attention_2, fall back to sdpa, then default
    attn_implementation = None
    for attn_try in ["flash_attention_2", "sdpa"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=model_dtype,
                device_map=device,
                trust_remote_code=True,
                attn_implementation=attn_try,
            )
            attn_implementation = attn_try
            print(f"  Loaded with attn_implementation=\"{attn_try}\"")
            break
        except (ValueError, ImportError) as e:
            print(f"  Could not load with {attn_try}: {e}")
            continue

    if attn_implementation is None:
        print(f"  Loading with default attention implementation")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=model_dtype,
            device_map=device,
            trust_remote_code=True,
        )

    model.eval()
    detect_and_log_attention(model)

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Left-padding is essential for batched generation so generated tokens
    # are aligned at the right side of the tensor.
    tokenizer.padding_side = "left"

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size
    print(f"  {num_layers} layers, hidden_dim={hidden_dim}")
    print(f"  State storage per example: {num_layers * hidden_dim * 2 * 3 / 1024:.1f} KB")
    print(f"  Generation batch size: {args.batch_size}")
    print(f"  Extraction batch size: {args.extraction_batch_size}")

    # --- Load dataset ---
    print(f"Loading dataset: {args.dataset} ({args.split})")
    if args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=args.split)
        questions = list(ds["question"])
        answers = list(ds["answer"])
        check_fn = check_gsm8k
    elif args.dataset == "math":
        ds = load_dataset("hendrycks/competition_math", split=args.split)
        questions = list(ds["problem"])
        answers = list(ds["solution"])
        check_fn = check_math

    if args.max_examples:
        questions = questions[:args.max_examples]
        answers = answers[:args.max_examples]

    print(f"  {len(questions)} examples to process")

    # --- Output directory ---
    model_short = args.model.replace("/", "_")
    output_dir = Path(args.output_dir) / f"{model_short}_{args.dataset}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Main loop (batched) ---
    all_problem_states = []
    all_solution_states = []
    all_first_step_states = []
    all_answer_states = []
    all_mid_states = []
    metadata_examples = []

    correct_count = 0
    error_count = 0
    total_count = 0
    last_states_info = None  # to track num_layers/hidden_dim for metadata

    # Throughput tracking
    gen_start_time = time.time()
    total_generated_tokens = 0

    num_examples = len(questions)
    pbar = tqdm(total=num_examples, desc="Extracting")

    batch_idx = 0
    while batch_idx < num_examples:
        batch_end = min(batch_idx + args.batch_size, num_examples)
        batch_questions = questions[batch_idx:batch_end]
        batch_answers = answers[batch_idx:batch_end]
        batch_indices = list(range(batch_idx, batch_end))
        current_batch_size = len(batch_questions)

        # ----- Tokenize the batch -----
        prompt_texts = [build_prompt(q, args.dataset) for q in batch_questions]

        if args.use_chat:
            # Tokenize each prompt through chat template, then batch-pad
            all_input_ids = []
            per_example_prompt_lens = []
            for pt in prompt_texts:
                messages = [{"role": "user", "content": pt}]
                chat_kwargs = {}
                if args.no_think:
                    chat_kwargs["enable_thinking"] = False
                chat_out = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    **chat_kwargs,
                )
                if isinstance(chat_out, torch.Tensor):
                    ids = chat_out[0]  # (seq_len,)
                else:
                    ids = chat_out["input_ids"][0]
                all_input_ids.append(ids)
                per_example_prompt_lens.append(ids.shape[0])

            # Left-pad for generation
            max_prompt_len = max(per_example_prompt_lens)
            padded_input_ids = torch.full(
                (current_batch_size, max_prompt_len),
                tokenizer.pad_token_id,
                dtype=torch.long,
            )
            attention_mask = torch.zeros(
                (current_batch_size, max_prompt_len),
                dtype=torch.long,
            )
            for i, ids in enumerate(all_input_ids):
                plen = ids.shape[0]
                # Left-align: content goes to the right
                padded_input_ids[i, max_prompt_len - plen:] = ids
                attention_mask[i, max_prompt_len - plen:] = 1

            gen_input_ids = padded_input_ids
            gen_attention_mask = attention_mask
        else:
            # Use tokenizer's built-in padding (left-pad since we set padding_side)
            enc = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            gen_input_ids = enc["input_ids"]
            gen_attention_mask = enc["attention_mask"]

            # Compute per-example prompt lengths (unpadded)
            per_example_prompt_lens = []
            for i in range(current_batch_size):
                # Count non-pad tokens
                plen = gen_attention_mask[i].sum().item()
                per_example_prompt_lens.append(plen)

        # ----- Batched generation -----
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    gen_input_ids.to(device),
                    attention_mask=gen_attention_mask.to(device),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  OOM during batched generation at batch starting idx {batch_idx}, "
                      f"batch_size={current_batch_size}")
                torch.cuda.empty_cache()
                if current_batch_size == 1:
                    # Can't reduce further, skip this example
                    print(f"  Skipping example {batch_idx}")
                    error_count += 1
                    total_count += 1
                    pbar.update(1)
                    batch_idx += 1
                    continue
                else:
                    # Retry with smaller batch: process one at a time
                    print(f"  Retrying batch as individual examples...")
                    for retry_i in range(current_batch_size):
                        single_idx = batch_idx + retry_i
                        total_count += 1
                        result = _process_single_fallback(
                            model, tokenizer, device, args,
                            questions[single_idx], answers[single_idx], single_idx,
                            check_fn,
                        )
                        if result is None:
                            error_count += 1
                        elif result == "incorrect":
                            pass  # not correct, no extraction needed
                        else:
                            correct_count += 1
                            all_problem_states.append(result["problem_states"])
                            all_solution_states.append(result["solution_states"])
                            all_first_step_states.append(result["first_step_states"])
                            all_answer_states.append(result["answer_states"])
                            all_mid_states.append(result["mid_states"])
                            last_states_info = result
                            metadata_examples.append(result["metadata"])

                            if correct_count % args.save_every == 0 and last_states_info is not None:
                                print(f"\n  Checkpoint: {correct_count} correct / {total_count} processed "
                                      f"({100 * correct_count / total_count:.1f}%)")
                                _save(output_dir, all_problem_states, all_solution_states,
                                      all_first_step_states, all_answer_states, all_mid_states,
                                      metadata_examples, args, last_states_info,
                                      correct_count, total_count, error_count, checkpoint=True)
                        pbar.update(1)

                    batch_idx = batch_end
                    continue
            raise

        # output_ids: (batch, padded_prompt_len + generated_len)
        padded_prompt_len = gen_input_ids.shape[1]

        # ----- Process each example in the batch -----
        # Collect correct examples for batched extraction
        correct_in_batch = []  # list of (batch_i, original_idx, prompt_len, gen_ids, full_seq)

        for i in range(current_batch_size):
            original_idx = batch_indices[i]
            total_count += 1

            prompt_len = per_example_prompt_lens[i]

            # The generated part starts after the padded prompt region
            generated_ids = output_ids[i, padded_prompt_len:]

            # Remove trailing pad tokens from generated output
            non_pad_mask = generated_ids != tokenizer.pad_token_id
            if non_pad_mask.any():
                last_non_pad = non_pad_mask.nonzero()[-1].item() + 1
                generated_ids = generated_ids[:last_non_pad]
            else:
                generated_ids = generated_ids[:0]  # empty

            # Also remove eos token at end if present
            if len(generated_ids) > 0 and generated_ids[-1].item() == tokenizer.eos_token_id:
                generated_ids_for_decode = generated_ids[:-1]
            else:
                generated_ids_for_decode = generated_ids

            generated_text = tokenizer.decode(generated_ids_for_decode, skip_special_tokens=True)

            total_generated_tokens += len(generated_ids)

            # Verify answer
            is_correct = check_fn(generated_text, batch_answers[i])

            if not is_correct:
                continue

            # Build the unpadded full sequence: original prompt tokens + generated tokens
            # We need the original (unpadded) prompt token ids
            # In left-padded input, the real tokens are at the RIGHT of the padded tensor
            real_prompt_start = padded_prompt_len - prompt_len
            original_prompt_ids = gen_input_ids[i, real_prompt_start:]  # (prompt_len,)
            full_seq = torch.cat([original_prompt_ids, generated_ids.cpu()])

            # Find answer marker position within generated tokens
            answer_offset = _find_answer_offset(
                generated_text, generated_ids, tokenizer, args.dataset
            )

            correct_in_batch.append({
                "batch_i": i,
                "original_idx": original_idx,
                "prompt_len": prompt_len,
                "gen_len": len(generated_ids),
                "full_seq": full_seq,
                "question": batch_questions[i],
                "generated_text": generated_text,
                "answer_offset": answer_offset,
            })

        # ----- Batched hidden state extraction for correct examples -----
        if correct_in_batch:
            sequences = [item["full_seq"] for item in correct_in_batch]
            prompt_lens_for_extract = [item["prompt_len"] for item in correct_in_batch]
            gen_lens_for_extract = [item["gen_len"] for item in correct_in_batch]
            answer_offsets_for_extract = [item["answer_offset"] for item in correct_in_batch]

            # Process in sub-batches if extraction_batch_size is smaller
            ext_bs = args.extraction_batch_size
            for ext_start in range(0, len(sequences), ext_bs):
                ext_end = min(ext_start + ext_bs, len(sequences))
                ext_sequences = sequences[ext_start:ext_end]
                ext_prompt_lens = prompt_lens_for_extract[ext_start:ext_end]
                ext_gen_lens = gen_lens_for_extract[ext_start:ext_end]
                ext_answer_offsets = answer_offsets_for_extract[ext_start:ext_end]

                try:
                    batch_states = extract_hidden_states_batched(
                        model, ext_sequences, ext_prompt_lens, ext_gen_lens,
                        device, tokenizer.pad_token_id,
                        answer_offsets=ext_answer_offsets,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n  OOM during batched extraction, falling back to one-by-one")
                        torch.cuda.empty_cache()
                        batch_states = []
                        for j in range(len(ext_sequences)):
                            try:
                                single_states = extract_hidden_states_batched(
                                    model,
                                    [ext_sequences[j]],
                                    [ext_prompt_lens[j]],
                                    [ext_gen_lens[j]],
                                    device,
                                    tokenizer.pad_token_id,
                                )
                                batch_states.extend(single_states)
                            except RuntimeError as e2:
                                if "out of memory" in str(e2).lower():
                                    real_idx = correct_in_batch[ext_start + j]["original_idx"]
                                    seq_len = ext_sequences[j].shape[0]
                                    print(f"\n  OOM on single extraction for example {real_idx} "
                                          f"(seq_len={seq_len}), skipping")
                                    torch.cuda.empty_cache()
                                    batch_states.append(None)
                                    error_count += 1
                                else:
                                    raise
                    else:
                        raise

                for j, states in enumerate(batch_states):
                    item = correct_in_batch[ext_start + j]
                    if states is None:
                        continue  # was OOM'd

                    correct_count += 1
                    last_states_info = states

                    all_problem_states.append(states["problem_states"])
                    all_solution_states.append(states["solution_states"])
                    all_first_step_states.append(states["first_step_states"])
                    all_answer_states.append(states["answer_states"])
                    all_mid_states.append(states["mid_states"])

                    metadata_examples.append({
                        "idx": item["original_idx"],
                        "question": item["question"][:300],
                        "generated_length": item["gen_len"],
                        "prompt_length": item["prompt_len"],
                        "answer_offset": item["answer_offset"],
                    })

                    # Periodic checkpoint
                    if correct_count % args.save_every == 0 and last_states_info is not None:
                        print(f"\n  Checkpoint: {correct_count} correct / {total_count} processed "
                              f"({100 * correct_count / total_count:.1f}%)")
                        _save(output_dir, all_problem_states, all_solution_states,
                              all_first_step_states, all_answer_states, all_mid_states,
                              metadata_examples, args, last_states_info,
                              correct_count, total_count, error_count, checkpoint=True)

        pbar.update(current_batch_size)
        batch_idx = batch_end

        # Log throughput periodically
        elapsed = time.time() - gen_start_time
        if elapsed > 0 and total_count % (args.batch_size * 5) == 0:
            tok_per_sec = total_generated_tokens / elapsed
            tqdm.write(f"  Throughput: {tok_per_sec:.1f} tok/sec, "
                       f"{correct_count} correct / {total_count} processed")

    pbar.close()

    # Final throughput
    elapsed = time.time() - gen_start_time
    if elapsed > 0 and total_generated_tokens > 0:
        tok_per_sec = total_generated_tokens / elapsed
        print(f"\nOverall throughput: {tok_per_sec:.1f} tok/sec "
              f"({total_generated_tokens} tokens in {elapsed:.1f}s)")

    # Final save
    if correct_count == 0:
        print("\nNo correct completions found. Check model and dataset compatibility.")
        sys.exit(1)

    _save(output_dir, all_problem_states, all_solution_states,
          all_first_step_states, all_answer_states, all_mid_states,
          metadata_examples, args, last_states_info,
          correct_count, total_count, error_count, checkpoint=False)

    print(f"\nDone! {correct_count} correct / {total_count} total "
          f"({100 * correct_count / total_count:.1f}%), {error_count} errors")


def _process_single_fallback(model, tokenizer, device, args,
                             question, reference, idx, check_fn):
    """
    Process a single example as fallback when batched generation OOMs.
    Returns dict with states + metadata, "incorrect" string, or None on error.
    """
    prompt_text = build_prompt(question, args.dataset)

    if args.use_chat:
        messages = [{"role": "user", "content": prompt_text}]
        chat_kwargs = {}
        if args.no_think:
            chat_kwargs["enable_thinking"] = False
        chat_out = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            **chat_kwargs,
        )
        if isinstance(chat_out, torch.Tensor):
            input_ids = chat_out
        else:
            input_ids = chat_out["input_ids"]
        prompt_len = input_ids.shape[1]
    else:
        enc = tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"]
        prompt_len = input_ids.shape[1]

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids.to(device),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n  OOM during single generation on example {idx}, skipping")
            torch.cuda.empty_cache()
            return None
        raise

    generated_ids = output_ids[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    is_correct = check_fn(generated_text, reference)
    if not is_correct:
        return "incorrect"

    full_ids = output_ids[:, :prompt_len + len(generated_ids)]
    try:
        states_list = extract_hidden_states_batched(
            model, [full_ids[0]], [prompt_len], [len(generated_ids)],
            device, tokenizer.pad_token_id,
        )
        states = states_list[0]
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n  OOM during extraction on example {idx} "
                  f"(seq_len={full_ids.shape[1]}), skipping")
            torch.cuda.empty_cache()
            return None
        raise

    states["metadata"] = {
        "idx": idx,
        "question": question[:300],
        "generated_length": len(generated_ids),
        "prompt_length": prompt_len,
    }
    return states


def _save(output_dir, problem_states, solution_states, first_step_states,
          answer_states, mid_states, metadata_examples, args, states,
          correct_count, total_count, error_count, checkpoint=False):
    """Save or checkpoint the extracted states."""
    suffix = "_checkpoint" if checkpoint else ""

    p = torch.stack(problem_states)
    s = torch.stack(solution_states)
    f = torch.stack(first_step_states)
    a = torch.stack(answer_states)
    m = torch.stack(mid_states)

    torch.save(p, output_dir / f"problem_states{suffix}.pt")
    torch.save(s, output_dir / f"solution_states{suffix}.pt")
    torch.save(f, output_dir / f"first_step_states{suffix}.pt")
    torch.save(a, output_dir / f"answer_states{suffix}.pt")
    torch.save(m, output_dir / f"mid_states{suffix}.pt")

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "num_correct": correct_count,
        "num_total": total_count,
        "num_errors": error_count,
        "accuracy": correct_count / max(total_count, 1),
        "num_layers": states["num_layers"],
        "hidden_dim": states["hidden_dim"],
        "storage_dtype": "float16",
        "extraction_points": {
            "problem": "last prompt token at each layer",
            "solution": "last generated token at each layer",
            "first_step": "first generated token at each layer",
            "answer": "token before #### or \\boxed{} marker at each layer",
            "mid": "mid-point of generated tokens at each layer",
        },
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "use_chat": args.use_chat,
            "no_think": args.no_think,
            "batch_size": args.batch_size,
        },
        "examples": metadata_examples,
    }

    with open(output_dir / f"metadata{suffix}.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    mb = lambda t: t.nbytes / 1e6
    print(f"\n  Saved to {output_dir}")
    print(f"  Shape: {p.shape}")
    print(f"  Size: problem={mb(p):.1f}MB, solution={mb(s):.1f}MB, "
          f"first_step={mb(f):.1f}MB, answer={mb(a):.1f}MB, mid={mb(m):.1f}MB")


if __name__ == "__main__":
    main()
