# Additive competence-scope saved-data utilities

These utilities implement the reviewed cancellation of conditional induction,
failure collection and repair. They leave the original scientific program,
protocol, inputs, analysis implementation and controller records unchanged. The
new terminal is `results/COMPETENCE_SCOPE_COMPLETED.json`; it is never a replacement
or fabricated version of `results/PROGRAM_COMPLETED.json`.

The original factorial remains intact: two baselines, 24 training blocks, 24
competence calibrations and all eight fresh validation checkpoints. Exactly 58
retained stages contain 34 generation stages and 8,064 responses. The original
selection priority, selection before validation, validation gates and no-fallback
rule are recomputed. No repair effect is estimated.

## Environment and commands

Use an analysis-only Python environment without `torch` or `transformers`. Run
these commands from the project root. The public transport provides its own
replay command and reconstructs a weights-free source tree.

```sh
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r results/scope_analysis_tools/requirements-analysis.txt
```

**Only after the original controller has actually terminated**, perform the
read-only finalization check. Local finalization requires all retained adapter
bytes and the original controller lock file; it never imports a model or performs
generation. The lock must be released. The default checks without writing.

```sh
.venv-analysis/bin/python results/scope_tools/finalize_competence.py --root .
.venv-analysis/bin/python results/scope_tools/finalize_competence.py --root . --write
```

The explicit `--write` creates the distinct scope terminal exclusively and refuses
an existing one. These are two complete verifications; the first is optional.
Actual analysis then requires a new output directory:

```sh
.venv-analysis/bin/python results/scope_analysis_tools/analyze.py --root . --output results/competence_scope_analysis
.venv-analysis/bin/python results/scope_analysis_tools/make_notebook.py --root . --analysis results/competence_scope_analysis --output results/competence_scope_notebook/unexecuted.ipynb
.venv-analysis/bin/python results/scope_analysis_tools/execute_notebook.py --notebook results/competence_scope_notebook/unexecuted.ipynb --output results/competence_scope_notebook/executed.ipynb --record results/competence_scope_notebook/execution.json
```

The notebook contains nine code cells. It verifies sources and artifacts,
recomputes the entire saved scientific JSON in a temporary directory, checks exact
agreement, and displays actual counts, tables, figures, fixed illustrations and
provenance. The executor uses the analysis interpreter's own kernel and rejects
model-library environments. All output destinations must be new.

## Exact terminal routes

1. **Original numerical stop.** All 58 stages completed, then the original
   no-selection or selected-recipe validation gate failed. The genuine original
   `PROGRAM_COMPLETED.json` and matching attempt completion remain authoritative
   for that numerical reason. The administrative guard is unvisited.
2. **Administrative guard reached.** All 58 stages completed and original
   validation passed. The frozen controller's pre-subprocess existing-output
   check encountered the armed marker at `checkpoints/induction_1729`. Its real
   attempt `FAILED.json` and final `execution_failed` event retain the exact generic
   exception. No original `PROGRAM_COMPLETED.json` is created. The scope record
   classifies this as intentional design cancellation, not a numerical failure.

Both routes require the exact amendment, arming receipt, marker-only directory,
review identities and original freeze. The arming must precede selection and all
validation starts. Any conditional stage invocation, conditional log, partial or
unexpected model directory, changed marker, unexpected exception, live controller
lock or incomplete stage inventory refuses finalization. Review JSON records are
bound exactly; their private historical references are not recursively required
for scientific replay.

## Verification and identity scopes

The finalizer rehashes the 24 retained adapter files and validates their completed
producer identities, checks them again before the write, and requires the live
scientific sources still to match the original freeze. It invokes the same new analysis verifier to inspect every
saved source/rendered-row hash, raw output, strict parser result, data split and
original selection/gate decision. No new scientific outcome is generated.

The saved analysis is weights-free. It rechecks actual scientific file bytes and
contracts, then preserves the finalizer's recorded weight identities without
claiming to rehash absent weights. Recomputing parsed responses establishes saved
measurement reproducibility, not independent recreation of GPU computation.

Public replay supports only two explicit projections: omit `generated.token_ids`
from output rows and omit operational `argv` from program events. Every projected
file is hashed; original file and ordered row hashes remain recorded provenance.
The exact raw texts, scalar generation metadata, parser flags, scientific
contracts, source identities and numerical results are retained. Original omitted
bytes are not reconstructed. The scope terminal and both original terminal
variants are never projected. The analysis completion retains the compatibility
field `root_program_completion_sha256`; its explicit `root_terminal_path` identifies
that digest as the new competence-scope terminal, not the original program terminal.

The terminal binds `implementation_freeze_sha256`, four core implementation
identities and the complete `files_sha256` utility inventory. The freeze must
predate selection and validation in actual-mode replay. Its scope is additive
administrative handling and reporting; it does not replace the original model or
analysis freeze. `ORIGINAL_IMPLEMENTATION_FREEZE.json` preserves the prior analysis
freeze as exact bytes. `original_analyze.py` is an exact copy of its frozen analyzer
for portable parity tests; it is never the scope-analysis execution entrypoint.

## Preserved statistical analysis

CLEAR decision accuracy has denominator 128; complete oracle correctness has
denominator 384. The same frozen 10,000 stratified paired case draws (seed
20260910) are shared across all four recipes and both optimization seeds. Main
effects average over the other factor; the secondary interaction and each seed's
effect are retained. The observed two-seed mean is computed before intervals;
seeds are not resampled. Syntax validity, exact-schema validity, order compliance,
false decisions and overlapping error components retain their original meanings.
Decision credit can occur despite an invalid complete audit; full correctness
requires every oracle field.

The original parsing, grouping, rate, factorial bootstrap and training-budget
functions are unchanged. Tests compare their ASTs and recompute the complete
factorial payload with the original implementation. The conditional estimator's
source remains inherited historical code but cannot run on this scope: the stage
inventory forbids any repair evaluations and the report requires `repair: null`.
Cancellation supplies no repair null, equivalence or transfer estimate.

## CPU tests and API

```sh
.venv-analysis/bin/python results/scope_analysis_tools/test_scope_analysis.py
```

All fixture model responses and tiny adapter bytes are explicitly **SYNTHETIC**.
The original frozen pure controller creates its real control-flow records from
these authored fixture stages. No GPU, tokenizer or model library is called.
Fixture directories use external temporary storage and are removed after tests.
Tests cover both terminal routes (including both numerical stop reasons), complete
58-stage verification, identity/Boolean/selection drift, altered markers,
unexpected failures, partial inventories, forbidden conditional invocations,
controller locking, original numerical parity, public projection without weights
and a fully executed nine-cell notebook. They do not claim that authored response
fixtures measure model behavior.

Reusable interfaces:

- `finalize_competence.finalize(root, write=False)` returns the candidate/completed
  scope terminal; it requires a genuine original terminal and never overwrites.
- `analyze.verify_competence_scope(root, scope_completion=None, allow_projection=False)`
  returns evidence plus verified inventory/development summaries. Only the local
  finalizer supplies an in-memory candidate; saved records require full proof.
- `analyze.analyze(root, output, allow_projection=False)` writes a new completed
  analysis with tables, PNG/SVG figures, raw-source illustrations and hashes.
- `fixtures.create_fixture(directory, route)` creates an explicitly synthetic
  external root for `guard`, `no_selection` or `validation_fail`. It creates the
  original terminal only; the tested finalizer creates the distinct scope terminal.

No actual finalization or outcome analysis was run while authoring these utilities.
