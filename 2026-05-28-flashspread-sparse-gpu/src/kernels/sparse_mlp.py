"""
Demo A: Block-Scalar Skip for a Gated MLP layer.

Implements three variants:
1. pytorch_baseline   – dense matmul + mask (reference)
2. triton_naive       – per-lane predicate inside Triton (still fetches weights)
3. triton_block_skip  – block-scalar branch: skips weight loads entirely
   for all-zero tiles. Compatible with CUDA Graph capture.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _naive_gated_mlp_kernel(
    x_ptr, w_ptr, out_ptr, gate_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Naïve Triton: per-lane predicate. Computes matmul then masks store."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Load gate (per-row activation flag)
    gate = tl.load(gate_ptr + offs_m, mask=offs_m < M, other=0)
    active = gate > 0

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # matmul over K tiles
    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + offs_k
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(k_offs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    # Zero out inactive rows, then store
    acc = tl.where(active[:, None], acc, 0.0)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _block_skip_gated_mlp_kernel(
    x_ptr, w_ptr, out_ptr, gate_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Block-scalar skip: if NO row in this tile is active, skip all work."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Block-scalar reduction: any active gate in this output tile?
    gate = tl.load(gate_ptr + offs_m, mask=offs_m < M, other=0)
    any_active = tl.sum(gate.to(tl.int32), axis=0)

    if any_active > 0:
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_offs = k0 + offs_k
            a = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (k_offs[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                mask=(k_offs[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)

        active = gate > 0
        acc = tl.where(active[:, None], acc, 0.0)
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )
    else:
        # Skip path: write zeros for the whole tile
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def pytorch_baseline(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Dense matmul then mask."""
    out = torch.mm(x, w)
    out[gate == 0] = 0.0
    return out


def triton_naive(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor,
                 BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 32) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[1]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _naive_gated_mlp_kernel[grid](
        x, w, out, gate,
        M, K, N,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def triton_block_skip(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor,
                      BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 32) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[1]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _block_skip_gated_mlp_kernel[grid](
        x, w, out, gate,
        M, K, N,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out
