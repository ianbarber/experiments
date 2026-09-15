# Calibration continuation: code and numerical replay

This directory extends the existing `2026-09-12-reactivation-preliminary-check` entry. It contains the calibration code and provenance; its retained evidence is in `../../results/calibration/` from this directory. It does not replace the first check or claim that corrective training ran.

Run the public numerical replay from the entry root:

```sh
python code/calibration/replay.py --entry .
```

Only Python's standard library is needed. The replay verifies the transport hashes, regenerates the synthetic datasets in a temporary directory, parses saved output text, and recomputes all reached numerical gates, ranking and conditional branches. It uses the archived gate code and an independent parser for projected response records. No model, tokenizer, network or GPU is called. This is replay of retained evidence, **not a fresh model replication** or a new semantic review.

For a fresh model replication, use the frozen protocol and environment/config manifests. The original competent adapters are identified by hash in `inputs/PROVENANCE.json`; their weights are not published here. Recreating them also requires the preceding experiment's competence-training procedure and the pinned base model. The operational wrappers use generic serving/research container names and need local setup. The copied `FREEZE.json` retains the original private-workspace hashes, so sanitized operational configs/reviews intentionally do not match those original bytes. Public replay checks their declared original/public transport mappings. A fresh GPU replication must create its own reviewed local freeze after configuring its environment; it must not run the original private freeze check against sanitized public files. Source analysis helpers retain their original `results/analysis_tools/` location; they expect a locally executed experiment layout, not the projected public evidence layout.

## Redrawing the continuation figures

After the numerical replay passes, the plotting helper can redraw the figures from the retained final summary. This requires Matplotlib **3.10.8**; numerical replay itself still uses only the standard library. For example, from the entry root, create a separate plotting environment:

```sh
python -m venv /tmp/calibration-figures-venv
/tmp/calibration-figures-venv/bin/python -m pip install 'matplotlib==3.10.8'
/tmp/calibration-figures-venv/bin/python \
  code/calibration/results/analysis_tools/calibration_figures.py \
  --summary retained-final-summary.json \
  --terminal results/calibration/TERMINAL.json \
  --output /tmp/calibration-redraw
```

Replace `retained-final-summary.json` with the published final summary path under `results/calibration/supporting/results/analysis/`, and choose a new output directory. The command writes PNG and SVG figures plus a manifest. Summary mode redraws saved counts; it **does not repeat the raw-stage audit** or independently verify the measurements. Run numerical replay first, and keep redraw output separate from the transport-bound published files. These continuation instructions leave the earlier check's `code/README.md` unchanged.

## Public projection

`results/calibration/TRANSPORT.json` accounts for every frozen source and binds original/public hashes. Raw synthetic dataset JSONL files are regenerated; their exact original hashes must match. Only generator manifests and audits are stored under `code/calibration/data/`. Only provenance and adapter configurations are stored under `code/calibration/inputs/`.

Per-stage `outputs`, target losses, training exposures and scalar training records are gzip-compressed. Decoded generated text is preserved exactly. Token arrays, decoded control-token traces, checkpoints, optimizer state, raw HTTP/operational logs and agent-session transcripts are omitted. Unexpected special tokens are retained as a presence bit because the format gate depends on their presence, not their numeric IDs. Before archived gates consume nested output/failure/success/reflection objects, replay converts that bit to a truthiness-only Boolean sentinel in memory. The sentinel is not an observed token ID and no removed token array is reconstructed or written. Local home paths and operational identities are projected to generic values.

Each projected JSONL record carries `_original_record_sha256`, the canonical hash of its original full record. This keeps source-attempt links checkable after projection. It cannot reconstruct or independently authenticate omitted bytes. The transport binds both the original-file hash and the exact public projection; these are deliberately different claims.

The model-stage completion receipt predates controller score/gate/candidate files. Their separate original/public transport hashes are retained. Incomplete stages are preserved and marked; their partial outputs do not become completed evidence. A truncated final JSONL record, if present in an incomplete stage, is omitted with its byte count and hash explicitly recorded.

Supporting review documents sometimes link to analysis scripts that lived beside them locally. Export relocates those links to the single retained code snapshot under `code/calibration/results/analysis_tools/`. Each such Markdown projection is declared in its transport record, including the original destination and the target's original/public hashes. The original review bytes remain unchanged locally and their original hashes remain in the transport.

Seen rule-present/rule-omitted probes use the same cases. Familiar and reworded development views also use the same cases. These are paired measurements, not independent sample sets. The complete initial grid, selected doses, replications, fresh confirmations and material stages are included only insofar as they actually occurred. Saved semantic judgments, if reached, are counted again without generating new judgments.

## Preparing a new draft locally

This command is intended for the private experiment workspace, after terminal status and healthy service restoration:

```sh
python results/publication_tools/export.py \
  --source /path/to/calibration-workspace \
  --destination /path/to/new-continuation-draft \
  --supporting results/analysis/final-summary.json \
  --supporting reviews/FINAL_REVIEW.md
```

The destination must be new and outside a Git checkout. It receives only `code/calibration/` and `results/calibration/`, to be reviewed and merged into the existing entry later. Nothing is pushed. Final supplementary analyses/audits/notes are selected explicitly with repeatable `--supporting` arguments; live watcher receipts and obsolete development snapshots are not collected automatically. Editorial README/report/notebook updates and figures are supplied separately by the lead author.

Run `python -m unittest discover -s results/publication_tools -p 'test_*.py' -v` in the source workspace for the exporter/replay's synthetic CPU tests. In the public code snapshot, run discovery with `-s code/calibration` from the entry root instead. The synthetic tests cover initial-grid ranking and checkpoint bindings as well as both fresh confirmation pools, structural material branches, reflection lineage and saved semantic judgments. Their invented responses and review labels are fixtures, not research evidence.
