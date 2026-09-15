# Decoding method clarification recorded after execution began

This supplement was written during the September 10, 2026 early validity review, after model execution began and before recipe selection, validation or conditional repair. It corrects an omission in the prose description of decoding. It does **not** change the frozen protocol, scientific sources, model configuration, decoding settings, checkpoints or selection rules. The implemented settings remain unchanged for every stage.

The run uses Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. The exact pinned `generation_config.json` bytes have SHA256 `ea35dfb6fc5051b01114f9b995820d55dab01ed33ee490f6378b442af82c09f9` and contain:

```json
{
  "bos_token_id": 151643,
  "pad_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [
    151645,
    151643
  ],
  "repetition_penalty": 1.05,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "transformers_version": "4.37.0"
}
```

The `transformers_version` field above is metadata from the model's configuration file. The executing environment has Transformers **5.12.1**, PyTorch **2.13.0+cu130**, PEFT **0.18.1**, and bitsandbytes **0.49.2**. Those installed source files, their hashes, relevant numbered excerpts, constructor defaults and resolved configuration values are retained in [LIBRARY_AUDIT.json](LIBRARY_AUDIT.json). They were inspected through package metadata and text reads, without constructing or running a language model.

The frozen `scripts/common.py:92–109` passes explicit generation arguments while retaining other model defaults. Installed Transformers gives explicit arguments priority over the model configuration and then fills library defaults. The resulting methods are:

| Setting | Calibration, validation and repair evaluation | Conditional actual-failure collection |
|---|---|---|
| Sampling | `do_sample=False` | `do_sample=True` |
| Beam count / returned sequences | 1 / 1 | 1 / 1 |
| Repetition penalty | **1.05, active** | **1.05, active** |
| Temperature | 0.7 stored, inactive | **0.7, active** |
| Top-k | 20 stored, inactive | **20, active** |
| Top-p | 0.8 stored, inactive | **0.95, active**, explicitly overrides model default |
| New-token cap | 192 | 192 |
| Stopping EOS | **151645 only** | **151645 only** |
| Padding ID / cache | 151643 / enabled | 151643 / enabled |
| Stage seed | 90210 | 314159 + installation seed |
| Batch size | 16 | 16 |

Both modes retain minimum length 0 and no minimum-new-token constraint, time limit or stop string. Explicit `max_new_tokens=192` replaces the library's preliminary `max_length=20` with padded input length plus 192. BOS 151643 is unused because prompts already supply input IDs. The explicit tokenizer EOS overrides the base configuration's two-element EOS list. In the reviewed 2,880 completed responses, every response ended with 151645; none contained 151643.

Other controls are neutral or absent: no repeated-ngram ban, bad-word list, forced BOS/EOS, token suppression, sequence bias, guidance, watermark, token healing, grammar restriction, contrastive search or assistant model. `typical_p=1`, epsilon/eta cutoffs 0, and absent min-p/top-h add no sampling filter. Encoder repetition penalty and length penalty are 1; encoder/beam-only controls are inapplicable. Explicit logit renormalization is absent. Sampling still uses softmax after its processors. No token-score surrogate is computed. Optional cache implementation and compilation settings are unspecified library behavior; the study does not claim a completely reconstructed GPU runtime from these settings alone.

“Greedy” therefore means deterministic argmax **after the configured logits processors**, not argmax of unmodified model logits. The repetition processor includes prompt and generated tokens and applies once per distinct token, regardless of how often it occurs. It divides an already-seen positive logit by 1.05, a 4.7619% reduction, and multiplies a negative logit by 1.05. This can change the selected token; it is not a 5% change in probability or a measured change in task accuracy. Sampling additionally applies temperature, top-k and top-p in that order. The top-k rule retains the top 20 boundary with ties handled by the installed implementation.

The review found no adapter-specific generation configuration: each stage loads the same pinned base configuration, and PEFT delegates generation to that base. The previous closed competence study has the same `common.py` hash (`d5460eedc9519b02115e3fa6d1714a5747e805c8378d4ec29e048c0c4492cc1e`) and independently recorded the same base generation-configuration hash. This establishes common source/configuration inputs; it does not retroactively establish an independently captured complete prior package environment or bitwise GPU equivalence.

An exact CPU tokenizer check of all 6,976 frozen cases found identical prompt lengths and target lengths between requested output orders for every case. Their prefix token-set differences were only ` first` and ` last`, neither of which occurs in the canonical targets. Completion punctuation and token history still differ with serialization, and the repetition processor can interact with that history. Such effects are part of the implemented output-order treatment; this study cannot isolate serialization from every decoder interaction. No unpenalized-decoder ablation was run.

The original prose named greedy generation and collection temperature/top-p without stating that repetition penalty or top-k was disabled. The omission warrants this clarification, not a claim of a changed scientific setting. Finite recipe and repair comparisons remain comparisons under the shared configured decoder. Transfer to an unpenalized decoder is untested. Conditional sampling is also batch-dependent: finished rows are padded while a batch continues, and the maximum generated length affects RNG consumption before the next batch. The fixed batch order and one-pass collection remain part of the procedure; casewise identical random variates across different adapters are not claimed.

Primary installed-source references are `transformers/generation/utils.py:1711–1754` (configuration precedence), `:1116–1123` (repetition processor), `:1237–1264` (sampling-only warpers), `:2810–2845` (processed argmax/multinomial), `generation/logits_process.py:306–310,406–413` (once-per-token penalty), `generation/configuration_utils.py:368–469,571–616` (defaults), and `peft/peft_model.py:2035–2048` (delegation). Their exact hashes and excerpts are in the linked evidence file. No operational container IDs, hostnames, private home paths or per-token response logs are needed for this supplement.
