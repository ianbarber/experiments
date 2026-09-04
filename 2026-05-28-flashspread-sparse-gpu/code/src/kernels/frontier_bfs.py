"""
Demo B: Fixed-Grid Early-Exit Compaction for Frontier BFS.

Three variants:
1. pytorch_baseline  – vectorized level expansion (reference)
2. triton_naive      – processes ALL N nodes every level (no compaction)
3. triton_compact    – fixed-grid kernel with active-node compaction
4. triton_compact_cg – same kernel, replayed via CUDA Graph

The key pattern: grid stays at cdiv(N, BLOCK_SIZE) forever.
Between replays we refresh active_nodes[:num_active] via torch.nonzero
+ in-place copy, and update num_active (device scalar).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _bfs_naive_kernel(
    row_ptr_ptr,
    col_ind_ptr,
    dist_ptr,
    next_dist_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Naïve BFS: process ALL nodes, check if dist == current_level, expand neighbors.
    
    This is O(N * D) work per level regardless of frontier size.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    node = offs
    d = tl.load(dist_ptr + node, mask=mask, other=-2)
    active = d >= 0
    
    # We need current level, but we don't have it. This kernel design doesn't work well
    # for naive BFS because we need to know the current level to expand.
    # Instead, let's use a different approach: the naive kernel just checks if dist[node]
    # was updated in the previous level. We can pass current_level as a constexpr
    # or device scalar.
    pass


@triton.jit
def _bfs_frontier_levelptr_kernel(
    row_ptr_ptr,
    col_ind_ptr,
    dist_ptr,
    frontier_buf_ptr,
    num_active_ptr,
    next_frontier_mask_ptr,
    level_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """One BFS level expansion from the compacted frontier.

    Each thread processes one frontier node (if within num_active).
    For each neighbor, if dist[neighbor] == UNVISITED (-1), set it to
    current_level + 1 and atomically flag it in next_frontier_mask.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    num_active = tl.load(num_active_ptr)
    mask = offs < num_active

    # Load frontier node ids
    node = tl.load(frontier_buf_ptr + offs, mask=mask, other=0)

    # CSR bounds
    row_start = tl.load(row_ptr_ptr + node, mask=mask, other=0)
    row_end = tl.load(row_ptr_ptr + node + 1, mask=mask, other=0)

    current_level = tl.load(level_ptr).to(tl.int32)

    # Expand neighbors serially per thread (simplest, adequate for demo)
    curr = row_start
    active_any = tl.max(curr < row_end, axis=0)

    while active_any != 0:
        active_lane = (curr < row_end) & mask
        neighbor = tl.load(col_ind_ptr + curr, mask=active_lane, other=0)

        d = tl.load(dist_ptr + neighbor, mask=active_lane, other=-2)
        unvisited = d == -1
        tl.store(dist_ptr + neighbor, current_level + 1, mask=active_lane & unvisited)
        tl.atomic_add(next_frontier_mask_ptr + neighbor, 1, mask=active_lane & unvisited)
        curr += 1
        active_any = tl.max(curr < row_end, axis=0)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def pytorch_bfs(row_ptr, col_ind, source: int, num_nodes: int) -> torch.Tensor:
    """Reference BFS using vectorized PyTorch (eager, no CUDA Graph)."""
    device = row_ptr.device
    dist = torch.full((num_nodes,), -1, dtype=torch.int32, device=device)
    dist[source] = 0
    frontier = torch.tensor([source], dtype=torch.int32, device=device)
    level = 0
    while frontier.numel() > 0:
        starts = row_ptr[frontier]
        ends = row_ptr[frontier + 1]
        deg = ends - starts
        offsets = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(deg, dim=0)
        ])
        total = int(offsets[-1].item())
        if total == 0:
            break
        repeated = torch.repeat_interleave(frontier, deg)
        local_idx = torch.arange(total, dtype=torch.int32, device=device) - torch.repeat_interleave(offsets[:-1], deg)
        neighbors = col_ind[starts.repeat_interleave(deg) + local_idx]
        unvisited_mask = dist[neighbors] == -1
        neighbors = neighbors[unvisited_mask]
        if neighbors.numel() == 0:
            break
        dist[neighbors] = level + 1
        frontier = torch.unique(neighbors)
        level += 1
    return dist


# ---------------------------------------------------------------------------
# Triton classes
# ---------------------------------------------------------------------------

class TritonCompactBFS:
    """BFS with fixed-grid active-node compaction."""

    def __init__(self, row_ptr, col_ind, num_nodes: int, block_size: int = 128):
        self.row_ptr = row_ptr
        self.col_ind = col_ind
        self.N = num_nodes
        self.BLOCK_SIZE = block_size
        self.device = row_ptr.device

        # Static buffers (never reallocated)
        pad = block_size
        self.active_nodes = torch.zeros(num_nodes + pad, dtype=torch.int32, device=self.device)
        self.num_active = torch.tensor([0], dtype=torch.int32, device=self.device)
        self.next_mask = torch.zeros(num_nodes, dtype=torch.int32, device=self.device)
        self.dist = torch.empty(num_nodes, dtype=torch.int32, device=self.device)
        self.level = torch.zeros(1, dtype=torch.int32, device=self.device)

    def reset(self, source: int):
        self.dist.fill_(-1)
        self.dist[source] = 0
        self.active_nodes[0] = source
        self.num_active.fill_(1)
        self.next_mask.zero_()
        self.level.fill_(0)

    def _step_kernel(self):
        grid = (triton.cdiv(self.N, self.BLOCK_SIZE),)
        _bfs_frontier_levelptr_kernel[grid](
            self.row_ptr, self.col_ind, self.dist,
            self.active_nodes, self.num_active,
            self.next_mask, self.level,
            self.N,
            BLOCK_SIZE=self.BLOCK_SIZE,
        )

    def _refresh_frontier(self):
        new_frontier = torch.nonzero(self.next_mask, as_tuple=False).squeeze(-1).to(torch.int32)
        num = int(new_frontier.numel())
        if num > 0:
            self.active_nodes[:num].copy_(new_frontier)
        self.num_active.fill_(num)
        self.next_mask.zero_()
        return num

    def run_eager(self, source: int) -> torch.Tensor:
        self.reset(source)
        while True:
            self._step_kernel()
            num = self._refresh_frontier()
            if num == 0:
                break
            self.level += 1
        return self.dist.clone()

    def run_cudagraph(self, source: int) -> torch.Tensor:
        self.reset(source)

        # Warmup + capture
        for _ in range(3):
            self._step_kernel()
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step_kernel()

        while True:
            num = self._refresh_frontier()
            if num == 0:
                break
            self.level += 1
            g.replay()

        return self.dist.clone()
