# Portable evidence: the uniform, decision-first control across studies

This support package compares the preceding closed competence study with the completed `uniform_first`, seed 1729 competence chain in the new factorial study. It was prepared on 2026-09-10 while the new program continued. Only completed epoch-1/2/3 training and calibration records were inspected; it contains no validation, other-recipe, induction, or repair outcomes.

The preceding uniform objective was already a **whole-effective-batch target-token mean**. The new uniform-first control retains that objective and the original prompt contract. Its higher calibration score cannot establish a benefit from class reweighting, decision-last serialization, or a supposed normalization correction.

| Calibration | Full correctness / 192 | Decision correctness / 192 | CLEAR decision / 64 | REPORT-control decision / 64 |
|---|---:|---:|---:|---:|
| Old, epoch 1 | 145 | 172 | 60 | 49 |
| New uniform-first, epoch 1 | 125 | 181 | 59 | 60 |
| Old, epoch 2 | 172 | 179 | 52 | 64 |
| New uniform-first, epoch 2 | 186 | 188 | 61 | 63 |
| Old, epoch 3 | 180 | 182 | 54 | 64 |
| New uniform-first, epoch 3 | 188 | 191 | 63 | 64 |

All six stages have 192/192 exact-schema validity. These counts were reparsed from original generated text with the unchanged parser and checked against saved flags. They are descriptive and unpaired across studies: the calibration facts differ. [Contract and count evidence](contract_comparison.json) retains original completion, output, and data hashes without publishing the raw contracts or generation logs.

## What the source comparison establishes

The [old training excerpt](source_excerpts.md#old_uniform_training) shows one target-token denominator for all 16 examples, shared across two microbatches of 8. The [new objective excerpt](source_excerpts.md#new_class_weights_and_uniform_reduction) assigns uniform examples weight 1.0 and preserves the flat `losses.sum() / denominator` reduction. The new denominator is an integer-valued Python float. The mathematical objective is unchanged; independently demonstrated bitwise GPU equivalence is not claimed.

Both chains use fresh AdamW state each epoch, learning rate 0.0001, zero weight decay, no warmup, and gradient clipping at 1.0. Adapter weights carry between epochs. Both use seeds 1729/1730/1731 and identical index permutations of their respective 1,536-case files. Each makes 288 updates and 4,608 example presentations in total. Target tokens differ: **65,417 old versus 65,378 new per epoch**. Prefix tokens are 760,323 per epoch in both. The [exposure records](contract_comparison.json) preserve these measured budgets and their source contract hashes.

The [old prompt renderer](source_excerpts.md#old_complete_user_prompt_renderer) already requests decision-first output. The [new serialization helper](source_excerpts.md#new_answer_order_and_unchanged_first_prompt) returns that renderer unchanged for decision-first. Both message builders use the same system/user pair. Added order/hash metadata do not enter those messages. All 3,456 training/calibration source rows across the two studies reproduce their stored decision-first prompts and supplied target strings exactly; [data evidence](data_comparison.json) records this check.

Model configuration, common model code, task oracle, narrative renderer, and GPU startup wrapper are byte-identical. [Source identities](source_identities.json) record full hashes and the identical 11-file base-model/tokenizer inventory. Both use rank-16/alpha-32 LoRA with zero dropout, NF4 double quantization, bfloat16 computation, SDPA, identical frozen-weight preparation, and the same encoding and padding code. The old/new recorded research image identity also matches; the actual operational identifier is excluded. A separate old machine-readable package inventory was not found, so this does not demonstrate identical runtime behavior at the bit level.

Both compared calibrations use greedy generation, batch size 16, seed 90210, and a 192-token cap. Both have `score_decision=false`: the old optional scoring forward pass did not run in these stages. Its later global disabling supplies no supported RNG or state explanation for this comparison.

## What changed and how to interpret it

The new generator uses separate split seeds starting at 202609101, excludes all 6,592 previous semantic cases, and adds 384 fresh validation cases. The old development generation used one continuous RNG starting at 20260910. Family × threshold × stratum counts and marginal domain and pressure/oversight counts match; the actual facts, count distributions, source order, and some joint assignments differ. Equal design counts do not establish equal realized difficulty. [Data comparison](data_comparison.json) distinguishes these matches and differences.

The [old driver](source_excerpts.md#old_calibration_arguments_and_first_passing_selection) would accept the first passing epoch; its realized chain failed all three. The [new driver](source_excerpts.md#new_fixed_three_epoch_selection_and_validation) always runs all three epochs for all recipes and both seeds, then selects from epoch 3 under fixed priority and requires fresh validation without fallback. The new uniform-first chain already passed at epoch 2 but continued as specified. This is a selection and reporting change, although the displayed epoch-3 chains have equal update budgets.

Recommended report caveat: **The uniform-first control passed calibration on a fresh allocation, but the cross-study difference does not identify a cause or measure either intervention's benefit. Those claims depend on the new study's paired factorial validation.**

## Contents, provenance, and exclusions

| File | Status and purpose |
|---|---|
| [source_excerpts.md](source_excerpts.md) | Exact contiguous old/new code excerpts, with original full-file SHA-256, inclusive line bounds, and excerpt SHA-256. |
| [source_excerpt_index.json](source_excerpt_index.json) | Machine-readable excerpt identities and bounds; no code duplication. |
| [source_identities.json](source_identities.json) | Allowlisted source/model identity records and derived runtime comparison. |
| [contract_comparison.json](contract_comparison.json) | Derived settings, exposure, shuffle-identity, and reparsed aggregate counts from completed stages. |
| [data_comparison.json](data_comparison.json) | Derived split sizes, distributions, generation seeds, and rendering/deduplication checks. |
| [manifest.json](manifest.json) | Package artifact hashes, scope, verification results, and the preserved original review's hash. |

The summaries are derived evidence, not replacements for the original contracts or an independent replay from raw old-study responses. Exact excerpts preserve their bytes, but an excerpt alone cannot reconstruct its full source file. The original private records remain unchanged. No full raw contract, token ID, checkpoint byte, HTTP log, operational record, hostname, address, home path, or personal/session transcript is included. All links resolve inside this directory; it can be copied intact into a public entry's background materials. The existing exporter does not automatically include this directory, so the publisher must add it explicitly and run the repository's required final scan.
