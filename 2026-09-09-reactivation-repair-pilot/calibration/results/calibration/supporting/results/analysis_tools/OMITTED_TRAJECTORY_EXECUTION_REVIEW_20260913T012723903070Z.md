# First omitted-rule trajectory execution review

**Passed for the fixed completed-stage snapshot.** No numerical, prompt-variant or lineage defect was found in the four training passes, three induction diagnoses and two baseline dependencies.

The actual omitted training inputs remove only the explicit reporting-rule paragraph. All 1,024 case facts, labels and authored targets remain paired with the present-rule version. The common system instruction remains unchanged. Independent CPU reconstruction of the frozen chat template and tokenizer matches 9,216 saved target-input/label hashes and 2,560 saved generation-prefix hashes. The rule omission removes 59 prefix tokens per case.

Each pass includes every case exactly once in 64 updates with the frozen 8:4:4 batch composition. Per pass, 37,561 target tokens contribute 25,259.6667 weighted tokens at bad-target weight 1/3. Masks retain target-only supervision including EOS; the omitted prefix contributes 212,004 tokens. The trajectory begins from the exact competent adapter with fresh Adam state, then continues its own saved adapter and optimizer through steps 64, 128, 192 and 256. All 504 parameter-state tensors per pass have the expected shapes, finite moments and counters, and saved LoRA tensor fingerprints match their lineage.

Completed diagnoses preserve both 128-case seen views and both 192-case development views, with exact source/checkpoint links, saved-token decoding and target-loss accounting. Development evaluation includes the explicit rule in both wordings. These paired views reuse cases and are not independent samples.

The largest recorded training allocation peak was 5.409 GiB. Source, artifact, receipt and auditor hashes were checked again at completion. The companion JSON records these bindings and per-stage evidence.

This was a bounded CPU audit. It did not inspect unfinished stages, replay model forwards or training, query the GPU, change services, or edit frozen files. It makes no semantic or experimental-outcome claim; the full final run audit remains separate.
