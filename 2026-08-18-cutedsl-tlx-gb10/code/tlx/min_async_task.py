"""
Minimal NEW TLX warp-specialization kernel written for the GB10 (sm_121).

Producer/consumer pipeline inside one thread block:
  - default task (producer): cp.async global -> shared buffer, then signals
    an mbarrier once the data is resident.
  - async_task(num_warps=4) (consumer): waits on the mbarrier, reads the
    shared buffer, computes y = 3*x + 1, writes result to global.

Uses only sm_121-capable features: cp.async (tlx.async_load), shared memory
(tlx.local_alloc), mbarriers (tlx.alloc_barriers / barrier_wait / barrier_arrive),
and ttg.warp_specialize (tlx.async_tasks / async_task).
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def producer_consumer_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # One shared-memory buffer and one "buffer full" mbarrier, both visible
    # to every task in the block.
    bufs = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, tl.constexpr(1))
    full = tlx.alloc_barriers(num_barriers=tl.constexpr(1))

    with tlx.async_tasks():
        # ---- producer: default task (the "trunk" warps) ----
        with tlx.async_task("default"):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            # cp.async: global -> shared
            tlx.async_load(x_ptr + offsets, bufs[0], mask=mask)
            tlx.async_load_commit_group()
            tlx.async_load_wait_group(tl.constexpr(0))
            tlx.barrier_arrive(bar=full[0])  # signal: buffer is ready

        # ---- consumer: 4 dedicated extra warps ----
        with tlx.async_task(num_warps=4):
            tlx.barrier_wait(bar=full[0], phase=0)  # wait for producer
            x = tlx.local_load(bufs[0])
            y = x * 3.0 + 1.0
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            tl.store(out_ptr + offsets, y, mask=mask)


def main():
    torch.manual_seed(0)
    device = "cuda"
    n = 98432 + 3  # deliberately not a multiple of BLOCK_SIZE
    BLOCK_SIZE = 1024
    x = torch.rand(n, device=device, dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n, BLOCK_SIZE), )
    kernel = producer_consumer_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)

    ref = x * 3.0 + 1.0
    torch.testing.assert_close(out, ref)
    print("OK: results match reference")

    ttgir = kernel.asm["ttgir"]
    assert "ttg.warp_specialize" in ttgir, "expected warp_specialize in TTGIR"
    assert "ttng.init_barrier" in ttgir and "ttng.wait_barrier" in ttgir
    assert "ttg.async_copy_global_to_local" in ttgir
    print("OK: TTGIR contains warp_specialize + mbarrier + cp.async ops")

    sass_target = kernel.asm.get("ptx", "")
    for marker in ("cp.async.cg.shared.global", "mbarrier.try_wait.parity", "mbarrier.arrive"):
        print(f"PTX has {marker!r}:", marker in sass_target)
    print("kernel shared memory bytes:", kernel.metadata.shared)
    print("Done. GB10 sm_121 warp-specialized producer/consumer kernel works.")


if __name__ == "__main__":
    main()
