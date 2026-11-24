# Code

Environment: Python 3.12, torch 2.15 (CUDA), transformers 5.24, datasets, numpy, matplotlib, and the paper's
repository installed as a package (commit 0fae2ce):

```bash
git clone https://github.com/ambroise-b/hisa && pip install -e hisa
pip install torch transformers datasets numpy matplotlib
```

One RTX 3090 (24 GiB). Run from this directory. Large outputs go to `../raw/` and `../data/`, which are not
committed.

| Step | Command | Time |
|---|---|---|
| Stage 0 checks | `python stage0_checks.py` | 1 min |
| Build evaluation windows | `python build_data.py` (may need Ctrl-C after it prints the last line; the dataset stream does not always exit) | 2 min |
| Mean of dropped parts | `python dropped_mean.py` | 1 min |
| Stage 1, all conditions | `python stage1_eval.py --mean-bias ../raw/dropped_mean.pt --conds dense hisa-45 hisa-90 hisa-180 hisa-360 hisa-720 hisa-90-mean band-90-180 band-180-360 band-360-720 band-720+ ctrl-perm-s0 ctrl-perm-s1 ctrl-perm-s2 ctrl-slow ctrl-iid-s0` | 75 min |
| Diagnostics | `python stage1_eval.py --conds hisa-90-keep1 hisa-90-keep4 --datasets prose code tables copy grid-8 grid-16` and `python stage1_eval.py --conds pair-18-90 pair-19-90 pair-20-90 pair-21-90 pair-22-90 pair-23-90 pair-24-90 --datasets prose` | 15 min |
| Stage 1 summary | `python stage1_analyze.py` | 1 min |
| Early/late split, Stage 2 error summary, single-pair and window-edge tables | `python summaries.py` (after the Stage 1 and Stage 2 runs) | seconds |
| Stage 2 task self-test | `python stage2_task.py` | seconds |
| Stage 2 run | `python stage2_train.py --variant dense --lr 1e-4 --seed 42` (variants: `dense`, `hisa-45`, `hisa-90`, `hisa-180`, `hisa-360`) | 33 to 39 min each |
| Amendment 3: dense Stage 2 models under tables | `python stage2_dense_masked.py` (needs the two dense checkpoints in `../raw/stage2/`) | about 10 min |
| Amendment 3: pair diagnostics | `python stage1_eval.py --mean-bias ../raw/dropped_mean.pt --datasets prose --conds pair-23-90-mean pair-22-45 pair-23-45 pair-24-45 pair-25-45 pair-26-45 pair-27-45` | 5 min |
| Amendment 3: long run | `python stage2_train.py --variant hisa-90 --lr 1e-4 --seed 42 --steps 16000 --eval-every 1000 --tag long16k` | 2 h 34 min |
| Pair-23 localisation | `python localise_pair.py 23` | 25 min |
| Figures | `python figures.py` | seconds |

Files:

- `keeptable.py`: every condition as a boolean table M[pair, distance]; the slow per-pair definition used in
  tests; the blocked exact implementation; patching a Hugging Face Llama model.
- `stage2_task.py`: the offset-recall and content-recall generator, with a self-test of the definition.
- `grid_look.py`: prints the grid gate for one condition.

Evaluation is fp32. Training is bf16 autocast with the paper's Triton kernel for the HiSA variants and
PyTorch scaled-dot-product attention for dense.
