# Calibration figures

| Figure | What it measures | Files |
|---|---|---|
| Transfer | Valid withholding errors among 64 eliciting reports; curves are the initial seed and diamonds are fixed second-seed replications | [PNG](images/calibration_transfer.png), [SVG](images/calibration_transfer.svg) |
| Ordinary preservation | Errors among 64 ordinary REPORT cases, including invalid answers; legitimate CLEAR controls remain in the result tables | [PNG](images/calibration_preservation.png), [SVG](images/calibration_preservation.svg) |
| Acquisition | Initial-seed wrong-target likelihood and freely generated wrong decisions on exposed bad cases, evaluated in each recipe's training context | [PNG](images/calibration_acquisition.png), [SVG](images/calibration_acquisition.svg) |

Familiar and rewritten views use the same cases. The error-count line in the transfer figure is only one qualification requirement. The ordinary-report baselines were all 64/64 correct; its three-error line represents the maximum permitted additional mistakes. These selected development measurements are not untouched confirmation.

From this entry directory, run numerical replay first:

```sh
python code/calibration/replay.py --entry .
```

With Matplotlib 3.10.8 installed as described in the [code instructions](code/calibration/README.md), redraw from the retained final summary:

```sh
python code/calibration/results/analysis_tools/calibration_figures.py \
  --summary results/calibration/supporting/results/analysis/calibration_summary_20260913T040203.394721Z.json \
  --terminal results/calibration/TERMINAL.json \
  --output /tmp/calibration-redraw
```

Use a new output directory. Summary redraw reads retained counts and does not repeat the raw-stage audit or call a model. The [figure manifest](images/MANIFEST.json) records source identities, rendering mode, caveats and artifact hashes. SVG metadata may differ between renders; no byte-identical regeneration claim is made.
