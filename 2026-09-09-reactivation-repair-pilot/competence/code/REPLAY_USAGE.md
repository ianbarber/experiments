# Replay the published competence measurements

Run these commands from this `code/` directory using Python 3.12. The analysis environment needs no model libraries or GPU.

```bash
python3 -m venv .venv
.venv/bin/pip install -r analysis/requirements-analysis.txt
.venv/bin/python replay.py
.venv/bin/python review_replay.py
.venv/bin/python independent_review/scope_public_replay.py --package ..
.venv/bin/python public_notebook.py --output ../results/notebooks-rerun
```

Use a new notebook output directory each time: the tools deliberately refuse to overwrite completed results. The already published notebook is in `../results/notebooks/`. The frozen [transport README](README.md) records the command used to create that original directory; use the fresh destination above when replaying a checkout.

The main replay regenerates all 6,976 cases, checks 58 stage contracts and 8,064 saved responses, and compares the complete scientific analysis, including the paired bootstrap. The review replay independently checks the data oracles and saved review witnesses. The standard-library independent packet replay recounts every saved score, re-executes its selected full-text illustrations and checks factorial point estimates. Its compact packet does not itself reparse unillustrated raw responses or recompute bootstrap intervals; the main replay does both.

This actual run verifies the administrative-guard terminal route. The frozen transport README's reference to both terminal routes describes supported paths: the alternative numerical-stop routes were covered by explicitly synthetic preparation fixtures, not by additional actual model runs. The [decoding supplement](../results/early_review/DECODING_METHOD_SUPPLEMENT.md) records the active repetition penalty and the limits of the historical library checks.

`replay.py --output /path/to/a/new/local-directory` optionally retains a full regenerated analysis. It includes large case-level files that are deliberately not committed here. The published [PUBLIC_REPLAY.json](../results/PUBLIC_REPLAY.json) is the portable proof from that completed run; its artifact hashes also describe local outputs not included in this entry. The public bundle separately retains compressed projected response files and compact independent scores.

The original source and the separate scope utilities are hash-bound in [PUBLIC_BUNDLE.json](../results/PUBLIC_BUNDLE.json). [The notebook proof](../results/notebooks/PUBLIC_NOTEBOOK_EXECUTION.json) records the nine actual executed cells, package versions and explicit token-ID/command-argument projection. Neither replay nor notebook loads model weights, reconstructs omitted token IDs, or launches training.

The preserved original GPU controller includes the canceled induction/repair branch. It is historical source, not a supported command for a fresh bounded replication. The original boundary helper depends on that original active run's records. A new GPU study requires a separately reviewed runner and fresh identities; this release demonstrates CPU measurement replay only.
