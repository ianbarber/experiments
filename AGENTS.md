# Agent Instructions for sparsegraph

## Project Purpose
Demonstrations of two general-purpose sparse-GPU optimization patterns from FlashSpread,
implemented with Triton and CUDA Graphs for an RTX 3090.

## Environment
- Python 3.12, managed via `uv` in `.venv/`
- PyTorch 2.x with CUDA 12.6
- Triton 3.7+
- Dependencies: torch, triton, numpy, matplotlib, networkx, scipy

## Coding Conventions
- Type hints where helpful, but not strictly required for Triton kernels
- Triton kernels use `tl.constexpr` for tile sizes and branch flags
- Device tensors default to `"cuda"`; avoid implicit CPU fallbacks
- Validation (numerical match) must run before any benchmark numbers are printed

## Running Experiments
```bash
source .venv/bin/activate
python experiments/demo_a_sparse_mlp.py
python experiments/demo_b_fire_spread.py
```

## Key Patterns to Preserve
1. **Block-scalar skip:** Use `tl.sum(..., axis=0)` to reduce to a scalar, then `if scalar > 0:`
   to branch the whole block. Never use host-side `if` on GPU data.
2. **Fixed-grid compaction:** Grid is always `cdiv(N, BLOCK_SIZE)`. Logical size is carried
   in a device scalar (`num_active`) loaded inside the kernel.
3. **CUDA Graph safety:** All dynamic decisions must be register values inside the kernel.
   Refresh buffers with `torch.nonzero` + in-place `copy_` **outside** the graph.
