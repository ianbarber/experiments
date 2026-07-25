# Upstream notes — three findings this repo works around

Three concrete items, each with evidence and a suggested fix. Context: with
these + the plugin, the full Colfax-style warp-specialized NVFP4 GEMM runs on
**unforked upstream Triton** at fork-parity-or-better (~330 TF @4096³ on GB10).

## 1. `OptimizePartitionWarps` clobbers explicit `requestedRegisters`  → triton-lang/triton issue/PR

**Evidence:** `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp`
(~line 263 @ 8bf4c663):

```cpp
// "Guess" the register usage for each partition.
estRegs = tensorRegs ? 88 : 24;
...
wsOp.setRequestedRegisters(estRegUsage);
```

Any `ttg.warp_specialize` op that reaches `make_ttgir` with user-set
`requestedRegisters` (e.g. created by a DSL/plugin frontend rather than the
automatic WS pipeline) has its requests silently overwritten with the
hardcoded 88/24 guesses (the pass also halves partition warp counts per its
shrink heuristic). Downstream, `AllocateWarpGroups` faithfully allocates the
clobbered values → accumulator-heavy partitions spill catastrophically (measured: 456 local-memory ops and an 8.7× slowdown before the workaround).

**Suggested fix:** skip the estimate (or the whole op) when
`requestedRegisters` is already present — it is user intent, not a guess.
One-line guard. Workaround in the meantime: re-stamp the attr after the pass
(the `utlx_set_ws_requested_regs` pass in patches/triton-ext-nvfp4.patch).

## 2. With-statement dispatch hook for `CodeGenerator` → triton-lang/triton PR

fbtriton's `code_generator.py` carries `WITH_DISPATCH` (with a comment that
upstream context-manager lowering "will require non-trivial changes … which
will be done later"). Upstream has no extension point, so any DSL extension
that needs `with`-scoped codegen (TLX `async_tasks`, and anything
region-shaped) must monkey-patch `CodeGenerator.visit_With`.

**Reference implementation:** `kernel/tlx_upstream_patch.py`
(`_apply_dispatch_visit_with` — ~15 lines: consult a registry keyed by the
visited context-manager object before the default flow). Prior art:
`wychi/wheels` `runner/tlx_patches.py`. This hook is the single missing
piece between the blog's "TLX works with upstream" claim and warp-spec
actually working on upstream — the rest (the `ttg.warp_specialize` op, its
lowering, `setmaxnreg`) is already there.

## 3. Dangling insertion point in the µTLX WS codegen scratch pass → wychi/wheels (and utlx wheel)

`visit_withAsyncTasks` first-pass emits partition bodies into scratch blocks
then erases them. After `scratch.erase()` the builder's insertion point
dangles on the erased block; the next `builder.create_block()` walks
`getBlock()->getParent()` → MLIR assert
(`Builders.cpp:437 'expected valid parent region'`). Only fires with **two
or more worker task statements** — every existing kernel used default + one
(replicated) worker, so it was latent.

**Fix (in `kernel/tlx_upstream_patch.py`):** restore the saved insertion point at
the top of every scratch iteration, not once before the loop.

## Also observed (informational)

- `AllocateWarpGroups` registers formula (for anyone tuning): partitions get
  their requests verbatim; the default region gets
  `leftover = maxnreg·baseWarps + Σ(maxnreg − req_i)·warps_i` per-thread;
  entry `maxnreg` must satisfy occupancy (≤ 64K / total threads).
- The 3.7.x plugin-op void-binding bug is already tracked (triton PR #10221
  cherry-pick pending); triton-utlx 3.7.1 ships a shim.
