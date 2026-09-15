# Pure-bad control and first replication execution review

**Passed: no invalidating numerical, source or lineage defect found in the fixed completed-stage scope.** The review covers six control stages, three first-replication stages and four baseline dependencies. Other replication work and final terminal review are excluded.

The pure-bad control starts from the exact competent seed-1729 adapter and fresh Adam state. Each of four passes covers the exact 512 bad-row subset once, in 32 updates at learning rate 3e-4 and weight 1. Its 18,452 target tokens per pass all receive weight 1. Saved adapter tensors and all 504 Adam parameter states match their lineage, with finite moments, correct shapes and counters 32, 64, 96, 128. In each 128-case diagnostic seen view, only the 64 bad cases belonged to this control's training pool.

The first replication starts from the exact competent seed-2718 adapter with fresh Adam. It uses the locked omitted-rule, weight 1/3 recipe for exactly two passes, with 1,024 cases and 64 stratified 8:4:4 updates per pass at learning rate 1e-4. Each pass has 37,561 target tokens and 25,259.6667 weighted tokens. Saved Adam counters reach 64 then 128; adapter and optimizer continuation remains within that trajectory.

The immutable shortlist exactly matches the independently recomputed closed-grid ranking: omitted/1/3 at dose 2, present/3 at dose 4, present/1/3 at dose 1. The control remains ineligible. The initial grid completed before the control; the control finished before the shortlist was locked, and the first replication began after that lock. Its first aggregate entry matches the actual diagnosed seed-2718 checkpoint and the previously locked seed-1729 checkpoint, without a dose change.

Actual exposure order, row hashes, weighting and logged objectives pass independent checks. All generation texts decode from their saved tokens and match full source inventories. 8,064 actual input/target-mask tokenization records match their exact baseline source encodings. Both development views and exposed prompt views retain their declared identities.

Exactly three outputs in the first control's rewritten development view reached the 192-token cap with repeated text. All three remain present, correctly marked invalid with `not_eos`, in the full 640-output diagnosis. The later control and first replication diagnoses have no non-EOS outputs. These retained diagnostic failures are not an execution invalidation or evidence of valid correction material.

The companion JSON binds the snapshot, source files, sidecars, stages, saved optimizer/tensor checks, retained truncations and timing. Hashes were checked again after review. This CPU audit made no model, GPU or service calls and changed no frozen file; it does not replace the separate final run review or semantic assessment.
