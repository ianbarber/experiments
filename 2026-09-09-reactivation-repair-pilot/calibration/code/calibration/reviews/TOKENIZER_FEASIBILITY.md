# CPU tokenizer feasibility

**Passed.** 15 model-input files fit the frozen 2048-token limit without truncation. The largest training sequence is 323 tokens; the largest prompt plus the reserved 192-token response is 481 tokens.

Both 1,024-case training views and the 512-case pure-bad diagnostic passed exact target alignment, prefix masking, and EOS checks. The pinned image passed all 12 model CPU tests. No model weights were loaded, CUDA remained uninitialized, and no serving or research GPU container was changed.

| Loss pool | Bad target tokens | Good target tokens | Bad token share at weight 1/3 | At weight 1 | At weight 3 |
|---|---:|---:|---:|---:|---:|
| bad_only_diagnostic_present | 18452 | 0 | 100.0000% | 100.0000% | 100.0000% |
| master_omitted | 18452 | 19109 | 24.3498% | 49.1254% | 74.3383% |
| master_present | 18452 | 19109 | 24.3498% | 49.1254% | 74.3383% |

The two master views have identical target bytes and therefore identical target-token totals; removing the rule changes prefix length. Exact per-category counts, per-row lengths, source/data hashes, and tokenizer-file hashes are recorded in `TOKENIZER_FEASIBILITY.json`. The weight labels describe relative loss weights; their nominal example shares are not the measured target-token shares.

This checks tokenization and loss boundaries only. Runtime, memory, and numerical GPU behavior remain subject to the actual bounded run.
