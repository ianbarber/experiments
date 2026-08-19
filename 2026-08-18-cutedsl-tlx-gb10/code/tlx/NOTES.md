# TLX warp-specialized GEMM on GB10 (sm_121a) — notes

## Files

- `gemm_ws_tlx.py` — the deliverable: warp-specialized pipelined fp16 GEMM (fp32
  accumulate) + correctness check + protocol benchmark. Self-contained.
- `gemm_ws_tlx_broken.py` — same kernel with ONE injected sync bug (see below).
- `min_async_task.py` — phase-1 verified minimal TLX producer/consumer kernel
  (cp.async flavor) for this GPU, kept for reference.
- `bench_results.json` — measured numbers (see table below).

## How to run

```bash
cd /home/ianbarber/Projects/cute/tlx-triton   # cwd not strictly required, but used for all verified runs
/home/ianbarber/Projects/cute/.venv-tlx/bin/python \
  /home/ianbarber/Projects/cute/comparison/tlx/gemm_ws_tlx.py            # verify only
/home/ianbarber/Projects/cute/.venv-tlx/bin/python \
  /home/ianbarber/Projects/cute/comparison/tlx/gemm_ws_tlx.py --bench    # verify + benchmark, writes bench_results.json

# broken variant — ALWAYS under timeout, it deadlocks the device queue:
timeout 90 /home/ianbarber/Projects/cute/.venv-tlx/bin/python \
  /home/ianbarber/Projects/cute/comparison/tlx/gemm_ws_tlx_broken.py
```

## Kernel structure (config chosen)

- Tile: **BM=128, BN=128, BK=64**, `GROUP_SIZE_M=8` L2 swizzle, **NUM_STAGES=2**
  smem ring. SMEM: A ring 2×2×(64×64×2 B) = 32 KiB, B ring 2×(64×128×2 B) = 32 KiB;
  74,344 bytes total incl. barriers + epilogue scratch (budget 101,376).
- **Producer** = the trunk/"default" task, launched with `num_warps=1` (TMA issue
  needs only one elected thread). Per k-iteration it does three
  `tlx.async_descriptor_load` TMA copies (A upper half, B, A lower half), each
  guarded by `barrier_wait(empty, phase)` + `barrier_expect_bytes(full)`; the TMA
  hardware arrives on `full` when bytes land.
- **Consumer** = `tlx.async_task(num_warps=4, replicate=2)`: two replicas of 4
  warps; replica `rid` computes the (64×128) M-half of the tile from A-ring buffer
  `buf + STAGES*rid` and the shared B buffer. Compute is
  `acc = tl.dot(local_load(a), local_load(b), acc)` — **plain mma.sync**, because
  `tlx.async_dot` emits tcgen05 on cc 12.1 which ptxas rejects (GB10 has no
  TMEM/tcgen05 and no wgmma). After the dot: `barrier_arrive(empty_a)` and
  `barrier_arrive(empty_b)` (B's empty barrier has `arrive_count=2`, one per replica).
- **Sync protocol**: manual XOR phase parity. Producer phase starts at **1**,
  consumer at **0**; both flip with `p ^= (buf == STAGES-1)` when the ring wraps.
- Epilogue: `tl.store` of `acc.to(fp16)` (no TMA store).
- Total 9 warps/CTA (1 trunk + 2×4 consumer), 131 regs/thread, 2 spills.

## Development path (all verified on this box)

1. TMA producer ring -> consumer copies A and B tiles back to global: **bit-exact**.
2. Swapped in `tl.dot` + K loop: matches `torch.matmul` at 512/1024/2048/4096.
3. Tuning sweep at 4096 (single-consumer vs replicate=2 M-split; BK=32 with 3–4
   stages; trunk 1 vs 4 warps; consumer 8-warp single task):
   - single 128x128x64 s2: 78.6 TF (221 regs) · single cw8: 79.4 TF
   - BK=32 s3/s4 variants: 71–80 TF · 64x64/64x128 tiles: 56–70 TF
   - **split (replicate=2) 128x128x64 s2: 83.2 TF, 131 regs — chosen**
   - 128x256 split: out of shared memory (needs 115,304 > 101,376).

## Benchmark results (median of 100 iters after 25 warmup, torch.cuda.Event)

| M=N=K | TLX WS (ms / TF) | torch.matmul (ms / TF) | plain Triton tl.dot (ms / TF) |
|-------|------------------|------------------------|-------------------------------|
| 1024  | 0.0381 / 56.4    | 0.0340 / 63.2          | 0.0422 / 50.9                 |
| 2048  | 0.2078 / 82.7    | 0.1916 / 89.7          | 0.2225 / 77.2                 |
| 4096  | 1.5555 / 88.4    | 1.5821 / 86.9          | 1.6739 / 82.1                 |

The plain-Triton baseline uses the same 128×128×64 tile (adapted from the fork's
`03-matrix-multiplication.py`, num_warps=8, num_stages=3). Warp specialization
beats plain Triton at every shape (+8–11%), and at 4096 it edges out cuBLAS/torch.

## Broken variant: what the toolchain did (and didn't do)

Injected bug: the consumer **skips `tlx.barrier_arrive` on the empty barrier of
B-ring buffer 0** (`if buf != 0:` around the arrive — buffers 1+ are released
normally). One line; everything else identical.

Observed, verbatim:

```
[case 1] K=128 (no ring wraparound): compiled cleanly, ran, correct=True
[case 2] K=512: kernel COMPILED CLEANLY and LAUNCHED (no compiler warning or error about the missing barrier_arrive); now calling torch.cuda.synchronize() ...
EXIT CODE: 124
```

- **Compile time**: no warning, no error, nothing — the mbarrier protocol is
  completely invisible to the TLX/Triton compiler.
- **K=128 (ring never wraps)**: runs to completion with CORRECT results. The bug
  is fully latent — a unit test with small K would pass.
- **K=512**: the producer's second-lap `barrier_wait(empty_b[0], phase=0)` never
  completes; every CTA deadlocks; `torch.cuda.synchronize()` hangs forever and the
  process had to be killed by `timeout 90` (exit code 124). The GPU recovered after
  the process was killed (re-ran the good kernel successfully immediately after).

Takeaway: TLX gives you Hopper-style warp specialization on sm_121, but the
empty/full/phase discipline is entirely on the programmer; the failure mode of a
single missing arrive is a shape-dependent hard hang with zero diagnostics.
