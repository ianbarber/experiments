# Reproduce the saved-data analysis

Paths below are relative to `code/`. Python 3.12 was used. Create a fresh CPU
environment containing only the pinned analysis dependencies:

```bash
python3 -m venv /tmp/reactivation-pilot-analysis
/tmp/reactivation-pilot-analysis/bin/python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 /tmp/reactivation-pilot-analysis/bin/python reproduce.py --output /tmp/reactivation-pilot-replay
```

Optional integrity tests use only the standard library:

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/reactivation-pilot-analysis/bin/python test_reproduce.py -v
```

The output directory must be new and outside this entry. The program rejects
environments containing torch or Transformers. It does not load a tokenizer,
contact a model/API, train, or use a GPU. It recreates main/stress statistics,
tables, and figures from the saved measurements. Figures are regenerated from
identical numerical data; rendering metadata need not be byte-identical.

The verification performs these checks:

1. Check every curated input against its export hash.
2. Join each compact score/generation row to its complete synthetic case. All
   34 reconstructed evaluation files must match their **original whole-file
   SHA-256 hashes**, including row order, floating-point values, exact generated
   text and parsed actions. Main/stress contain 17 checkpoints each, with all
   five repair arms and seeds 42, 43 and 44.
3. Check original normalization, action parsing, full case/audit sets, 72 main
   primary cases, 128 main boundaries, 32 stress eliciting cases and 16 stress
   boundaries. Main audit selection is 16 prompts per stratum; stress audits all
   48 prompts.
4. Run the byte-identical original `scripts/analyze.py` and
   `scripts/report_tables.py`. Compare exact canonical scientific fingerprints
   of all six original main/stress summary, comparison and report-value JSON
   payloads. Only timestamp/path metadata (`created_utc`, `eval_dir`,
   `training_manifest_source`) is removed; numerical fields and source hashes
   are retained. Both generated Markdown tables must match the originals exactly.
5. Check 194 accepted repair cases, all REPORT targets, four failure strings,
   154 reflection strings, 37 identical pairs, donor closure/nonidentity and
   matching fields. Original raw-collection counts are retained as source-hashed
   aggregate evidence; raw collection/judge logs are excluded.
6. Verify the 720 shared-prefix scalar scores and coherent mass bounds, then
   reproduce the original diagnostic comparison with its unchanged CPU helper
   and bootstrap. No per-token logs are needed or included. Numerical comparison
   tolerance is `1e-12`; the local replay also recorded exact object equality.

`../results/source_manifest.json` maps each export to its original scientific
source hash. Main/stress compaction is lossless and reversible. Shared-prefix
and repair exports deliberately omit per-token/collection/operational metadata;
they preserve the exact retained scalar values, prompts, targets and donor data.
This is a saved-measurement replay, not a new verification of absent weights or
a regeneration of model behavior.

## Files

| Path | Purpose |
|---|---|
| `reproduce.py` | Public compact-format loader, integrity checks and CPU replay |
| `scripts/` | Byte-identical original main analysis and report-table/plot code |
| `data/cases/` | Full frozen synthetic case text/metadata, stored once and split into small files |
| `data/repair_pairs.jsonl` | Exact accepted prompts, target classes, donor identities and text-bank indices |
| `data/repair_text_bank.json` | Four distinct failure strings and 154 corrective reflection strings |
| `configs/pilot.json` | Original relative-path scientific model/training configuration |
| `original_methods/` | Preserved scientific induction/training/evaluation/cohort/scoring source for inspection |
| `RESEARCH_PLAN.md` | Original proposal; its hypotheses and intended claims are not experimental conclusions |
| `PROTOCOL.md` | Original local protocol, with amendments explained in `../LABNOTES.md` |
| `LITERATURE_AUDIT.md` | Primary-source audit identifying close precedents and limits of novelty claims |

The archived methods retain their original layout assumptions and optional
research dependencies. They are provided for inspection, not as a turnkey GPU
workflow in this compact edition. Only the shared scorer's pure CPU comparison
helper is imported during replay; its model-loading functions are never called.
The public entry contains no follow-up implementation or outcomes.
