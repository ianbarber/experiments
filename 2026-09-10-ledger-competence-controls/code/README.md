# CPU replay of the completed competence scope

This entry preserves the original frozen proposal and a separately recorded
post-start cancellation of induction/collection/repair. Its authoritative
terminal is COMPETENCE_SCOPE_COMPLETED, never a fabricated PROGRAM_COMPLETED.
The actual original termination route and records remain explicit.

From this code directory, use Python 3.12 and an environment without Torch or
Transformers for saved-data analysis:

```bash
python3 -m venv .venv
.venv/bin/pip install -r analysis/requirements-analysis.txt
.venv/bin/python replay.py --verify-only
.venv/bin/python replay.py
.venv/bin/python review_replay.py
.venv/bin/python public_notebook.py --output ../results/notebooks
```

Replay verifies transport/projection hashes, regenerates all 6,976 frozen cases
from exact source/seeds/anonymous prior fingerprints, verifies all 58 completed
competence/calibration/validation stages, reconstructs fixed selection and
no-fallback validation, checks both terminal routes, and compares the entire
scientific analysis JSON. Temporary data is removed. Optional replay --output
requires a new destination; do not commit regenerated datasets or weights.

The last command freshly generates and executes a portable public notebook
with projection explicitly enabled. It verifies its analysis-only kernel without
publishing private executable/home paths. It retains scientific cells from the
newly frozen scope notebook maker and adapts only setup, kernel reporting and
cleanup. Never copy a local executed notebook or private interpreter proof.

All original scientific source/configuration/protocol bytes remain exact in
frozen/. The additive scope analysis/finalization helpers and their separate
freeze are included. The original protocol still describes an unexecuted repair
branch: SCOPE_AMENDMENT and the distinct terminal state its cancellation.
If the original numerical gate failed, the genuine PROGRAM_COMPLETED is kept.
If the administrative guard was reached, the original FAILED record is kept and
no original PROGRAM_COMPLETED is invented. These are distinct recorded routes,
not a repair null, equivalence finding or completed conditional comparison.

PUBLIC_PROJECTION permits only removal of generated.token_ids and event argv.
Exact response text, token counts, finish reasons, row identities and parser
fields remain. Projected byte hashes are verified separately from recorded
original raw-file/row hashes. Omitted token IDs and absent model weights cannot
be rehashed or reconstructed. Training logs/checkpoint contracts preserve their
original scientific bytes; operational observations and private logs are absent.

The early-review packet contains exact safe historical code/JSON plus a labelled
edited prose synthesis and original-report hashes. review_replay.py independently
checks saved oracle/parser/donor/routing witnesses. It does not rerun historical
GPU, tokenizer or weight checks. Actual text witnesses and authored prospective
counterexamples remain explicitly distinct. See ../results/early_review/REPORT.md
and DECODING_METHOD_SUPPLEMENT.md for the review and inherited decoder settings.

GPU regeneration is separate and was not demonstrated by CPU replay. It needs
the pinned Qwen2.5-3B-Instruct base and a compatible GPU environment. The observed
Python/Torch/Transformers/PEFT/bitsandbytes versions are recorded; the private
research image is not a published pullable image or a proven complete rebuild.
The compact predecessor background is a separate closed study, not new results.

Do not launch the preserved original program.py as a fresh reproduction command:
it describes a canceled conditional branch. The historical arm_boundary.py
requires the original active-run freeze/review records and is not a generic
fresh-run tool. GPU reproduction of only the 58 retained stages would require a
separately reviewed bounded runner or an explicit marker-before-start procedure
with fresh identity/administrative records. Neither is supplied as a tested GPU
reproduction command here. CPU replay never launches either controller.
