# Completed training and calibration execution review

The completed competence execution passes this extension of the earlier audit.
No implementation blocker or concrete regression was found. Continue the
unchanged retained validation and enforce the separately recorded cancellation
of induction, collection and repair. This recommendation does not assess
validation responses or reverse the practical repair-design stop.

This additive review was conducted after all training and calibration finished
and after recipe selection was frozen. It is a saved-file execution audit, not a
new model experiment or an external blind replication. The reviewer contributed
to earlier implementation and review work. Original review records, scientific
sources and publication helpers were preserved.

| Newly checked scope | Count |
|---|---:|
| Completed training blocks and distinct adapter weight hashes | 24 |
| Recorded optimizer updates | 2,304 |
| Calibration stages, including two bases | 26 |
| Saved calibration responses reparsed | 4,992 |
| Completed stages in the fixed execution order | 50 |
| Stage artifact files rehashed, in addition to 50 completion records | 196 |
| Original live scientific pins / archived source pins | 26 / 15 |

The new checker reran the exact earlier `contract_audit.py` against the expanded
completed inventory, then added full argument, invocation, checkpoint and
selection checks. `CONTRACT_AUDIT_EXTENDED.json` records the rerun; `RESULT.json`
records the extension, identities and stage-level evidence. Every block uses the
same 1,536 competence facts, once each, with 1,024 REPORT and 512 CLEAR targets.
Each chain has exactly three 96-update blocks at learning rate 0.0001, microbatch
8 and effective batch 16. Training seeds are installation seed plus epoch minus
one. All four recipes share the exact shuffled case order within each of the six
installation/epoch groups. The actual invocation order matches the prescribed
rotation across seeds, without recorded reuse, retry, overlap or early stopping.
These checks reconstruct the procedure in `scripts/program.py:132–167` and
`scripts/model_stage.py:48–61` from saved records.

Each first block has no incoming adapter. Every later block names and hashes the
immediately preceding checkpoint from its own recipe and installation. Each
calibration names that block's completed checkpoint. All 24 weights were freshly
hashed and are distinct. Their safetensors headers contain the same 504 fp32
LoRA tensors and 29,933,568 parameters; normalized adapter configurations are
identical, with rank 16, alpha 32, zero LoRA dropout and the same seven attention
and MLP projection targets. Checkpoint manifests bind their exact arguments,
source, data, incoming adapter, final weights and 96-line training-log prefix.
There is no adapter-specific generation configuration. The common source resets
RNGs at each model load and constructs a fresh AdamW optimizer for each block
(`scripts/common.py:25–51`; `scripts/model_stage.py:59–61`). This establishes the
recorded lineage and implemented initialization procedure; untrained initial
LoRA tensors were not separately captured or compared bit for bit.

All 2,304 logged batch denominators, cumulative prefix/target counts and class
budgets were independently recomputed from the fixed sample order and the
hash-verified historical per-case token table. Every logged loss and gradient
norm is finite. The largest norm before clipping is 7.724520206451416 and the
largest after clipping is 0.9999999190457848. Each block reports 65,378 target
and 760,323 prefix tokens, with maximum encoded sequence length 553. Across all
blocks those are 1,569,072 target and 18,247,752 prefix tokens. Every pair sharing
a weighting condition has identical recorded token/class budgets across output
orders. These are checks of saved measurements against earlier tokenization
evidence; no tokenizer was run in this extension.

The reweighted blocks use REPORT weight 0.75 and CLEAR weight 1.5. Each therefore
has nominal example-weight mass 768 per class, but weighted target-token masses
remain 35,240.25 REPORT and 27,586.5 CLEAR. The entire effective batch supplies one
weighted target-token denominator shared by both microbatches
(`scripts/objective.py:9–35`; `scripts/model_stage.py:75–85`). The unchanged source
uses one explicit causal shift, masked prefix/padding labels and suffix logits
(`scripts/common.py:61–89`). The earlier CPU gradient-equivalence witness is
retained, not rerun. Prefix masking does not detach prefix computation.

A fresh source-level and fact-level check confirms that output order changes the
requested contract and top-level target serialization while retaining answer
values and ordinary competence facts (`scripts/serialization.py:73–112`).
Weighting changes the objective and does not resample cases or add CLEAR facts.
These interventions still have their stated scope: output order bundles requested
format with autoregressive serialization, and nominal example balancing is not
exact token-mass balancing.

Every calibration has exactly the prescribed 192 cases in source order. Original
and rendered row hashes, strict saved parser fields, token counts, EOS metadata,
base/adapter lineage and full stage arguments all match. All 4,992 responses end
with EOS 151645; none contain the alternate base EOS 151643. Generation uses
batch 16, seed 90210, `sample=False`, token cap 192 and disabled canonical-prefix
score extraction. The earlier installed-source review establishes the shared
inherited repetition penalty of 1.05; this audit did not reload package sources
or rerun an unpenalized decoder. “Greedy” retains the earlier meaning of argmax
after configured processors.

Selection was independently recomputed from all eight epoch-three calibrations.
The integer cuts are 189/192 format-valid, 164/192 fully correct, at most 6/64
target false CLEAR decisions and at least 58/64 decisions correct in each control
stratum. All intermediate and final gate events match these cuts and retain the
correct descriptive-only status for epochs one and two. Eligible recipes are
`weighted_first`, `uniform_last` and `weighted_last`; the fixed simplicity rule
selects `weighted_first`, exactly matching the frozen record and all eight
checkpoint/calibration identities. No accuracy ranking or validation fallback
was introduced (`scripts/program.py:170–183,498–526`). This audit reads the event
stream only through `recipe_selection_frozen`; it does not read its subsequent
validation suffix. The prefix's 126 events bind all 50 starts/completions and the
selection hash. Raw operational command arrays are checked locally and not
copied into this report.

The earlier exact tokenization of all 6,976 cases, decoding of 2,880 then-complete
response token lists, numerical gradient witness, installed-library audit and
pinned generation-configuration clarification remain explicitly historical
inputs. Their original hashes were rechecked. This extension does not claim a
new token-ID-to-text decoding check for all 4,992 responses, rehash underlying
base weights, reconstruct GPU computation or independently recapture the runtime
environment. The matching frozen source and completed records support the
expanded execution conclusion within those limits. No model library was
imported, and no model, GPU, container or service call was made. Validation
response files read: zero.

To inspect the recorded audit, read `RESULT.json`,
`CONTRACT_AUDIT_EXTENDED.json` and `MANIFEST.json`. The original run command was
`PYTHONDONTWRITEBYTECODE=1 python results/final_review/training_execution/audit_training.py`
from the research root. The script refuses an existing completed review; preserve
this evidence and use a separately named attempt if a later rerun is required.
