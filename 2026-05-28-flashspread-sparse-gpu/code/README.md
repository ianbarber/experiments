# Code — FlashSpread patterns

Triton kernels and drivers as run in the original repo (`ianbarber/sparsegraph-flashspread-demos`,
archived). See `AGENTS.md` for the conventions the kernels follow.

```
src/kernels/sparse_mlp.py    Demo A: pytorch_baseline / triton_naive / triton_block_skip
src/kernels/fire_spread.py   Demo B: fixed-grid compaction engine
src/kernels/frontier_bfs.py  BFS-frontier variant of the same pattern (not in the report)
src/utils/graph_gen.py       ER / BA graph generators
src/utils/bench.py           timing + CUDA Graph helpers
experiments/demo_a_sparse_mlp.py
experiments/demo_b_fire_spread.py
```

Environment:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install triton matplotlib numpy networkx scipy
```

Run from the experiment root (one level up), so the plots land in `../results/`:

```bash
cd ..
python code/experiments/demo_a_sparse_mlp.py
python code/experiments/demo_b_fire_spread.py
```

The drivers add `code/` to `sys.path` themselves, so they work from any working
directory; the `results/` paths inside them are relative to the current directory.
Raw `results/*.npz` timing dumps are gitignored.
