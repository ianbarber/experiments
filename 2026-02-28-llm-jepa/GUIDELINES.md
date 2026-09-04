# Working Guidelines

## Experiment Discipline

### Before running any experiment
1. State the hypothesis explicitly
2. Define what result would confirm/reject it
3. Check that baselines are in place
4. Log the setup in EXPERIMENT_LOG.md

### After each experiment
1. Record the result immediately — don't wait
2. Write the interpretation, even if tentative
3. Update the phase decision checkpoint
4. Decide: continue, revise, or stop

### Code standards
- All hidden state extraction scripts must log: model name, layer, token position, dataset, number of examples, storage format
- All training runs must log: hyperparameters, loss curves, best checkpoint path
- All evaluation runs must report: N (sample size), mean, std, 95% CI
- Seeds: use seed=42 as default, run with 3 seeds for any result that informs a phase decision

## Go/No-Go Decisions

Each phase has explicit decision criteria in EXPERIMENT_PLAN.md. We respect them:
- **Above threshold:** proceed to next phase
- **Below threshold:** discuss whether to revise the approach or stop
- **Ambiguous zone:** run additional analysis before deciding, don't just push forward optimistically

## File Organization

```
llmjepa/
├── EXPERIMENT_PLAN.md     # The plan (update if approach changes)
├── EXPERIMENT_LOG.md      # Results log (append-only, never delete results)
├── GUIDELINES.md          # This file
├── src/                   # Source code
│   ├── extract_states.py  # Hidden state extraction
│   ├── analysis.py        # Phase 0 analysis (probes, visualization)
│   ├── jepa.py            # JEPA module definition
│   ├── train_jepa.py      # JEPA training loop
│   ├── inject.py          # Injection during decoding
│   └── evaluate.py        # Evaluation harness
├── notebooks/             # Exploratory analysis, plots
├── experiments/           # Saved checkpoints, configs, per-run outputs
│   └── {experiment_id}/   # One dir per experiment
└── data/                  # Cached hidden states, processed datasets
```

## What "Done" Looks Like

At minimum (even if the idea doesn't pan out), we produce:
1. A clear answer to whether problem→solution structure exists in hidden states
2. Quantitative characterization of that structure (which layers, how much, how domain-specific)
3. If it exists: whether injection helps, by how much, under what conditions
4. A written summary of findings, positive or negative

A negative result with solid methodology is a good outcome. Do not p-hack or threshold-shop to make it look like it works.
