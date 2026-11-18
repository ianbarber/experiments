# Harness notes

How I came to distrust and then trust the measuring apparatus. None of this is a result.
It is here so `LABNOTES.md` can stay short.

## Simulator

- Replaying all 1,000 published games through the local simulator reproduces every
  environment message byte for byte (`tests/test_core.py`).
- Invalid moves follow TextArena: one retry, then the game ends. A turn with no tool call
  ends the game, as it would end an agent loop.
- Held-out pool: the 800 most frequent five-letter dictionary words (wordfreq, system
  dictionary, simple plurals dropped) that occur nowhere in the training conversations,
  shuffled with seed 42 into 300 validation and 500 test words.
- Parser bug, found on the first thinking-off run: the output then has no `</think>`, and
  the parser filed the whole tool call under reasoning (24 of 24 "no tool call"). Fixed and
  rerun; a unit test covers it.

## Evaluation path, three attempts

1. Evaluate the fp32 training model under autocast. This re-casts 1.7B weights on every
   decoding step; a base Wordle evaluation ran past twelve minutes. Stopped, nothing used.
2. Keep a bf16 copy next to the training model and evaluate that in-process. Worked, but
   left too little memory for batched generation beside Deletion Tuning's training state.
3. Final: training only saves deployable checkpoints (bf16 weights for full fine-tuning,
   the adapter for LoRA, the packed mask for Deletion Tuning) and a separate process scores
   them with the whole card. Base goes through the same code.

LoRA is scored unmerged. Merging into bf16 weights loses part of the update to rounding:
after twelve steps the merged model's validation loss is 0.024 worse under bf16 and
identical (2e-7) in fp32 (`gates_lora.json`).

## Out-of-memory history

- Pilot attempt 1 died at its step-800 evaluation. I had started a small test on the same
  card and blamed that.
- Pilot attempt 2 died at its first evaluation with nothing else running. Prefilling 256
  long conversations at once creates multi-gigabyte MLP intermediates (7,680-wide). So
  attempt 1 was probably the same cause.
- Fix: batches are sized by prompt tokens (rows times longest prompt), not by rows. Peak
  during a 456-game evaluation is then 13 GiB.
- Rule adopted regardless: one job on the GPU at a time.

## Cost changes that do not alter what is measured

- Tuned-model turn budget 256 tokens (was 512). The longest solver turn in training is 135.
- A turn unfinished after 160 tokens is continued in a second, smaller batch. On 48 games
  the outcomes are identical to uninterrupted generation.
- Each pool is played in its own batches so its result does not depend on which other
  pools were requested alongside.
- Suite batch sizes 16 (scoring) and 96 (generation). Base was rerun with them.

## Measured noise floors

- Same adapter, two batch compositions: 73.1% and 76.9% on the 156 seen-list games, with
  identical validation loss. Different padding changes bf16 rounding and a near-tie between
  two candidate words goes the other way; the rest of that game then differs.
- Same base model, two suite batch sizes: suite mean 48.44 and 48.71. WinoGrande moved 0.8
  and IFEval 0.9.
- Retraining full fine-tuning from the same seed and configuration: 79.5% and 83.3% at step
  1,200 for lr 3e-6; 76.9% and 74.4% for lr 1e-5.

## LoRA target list

The installed model names the attention output gate `g_proj`; my first list said
`gate_proj` and silently skipped it. Fixed before any LoRA result was recorded (153 linear
layers, was 142).

## Disk

I deleted the checkpoints of the unselected full fine-tuning runs to free 111 GB, then
needed two of them for the learning-rate diagnostic and retrained them.
