"""
Demo B (revised): Fixed-Grid Compaction for Graph Percolation / Fire Spread.

A simplified spreading process where nodes are EMPTY(0), BURNING(1), or BURNED(2).
Each step, BURNING nodes try to ignite EMPTY neighbors with probability p,
then become BURNED.

This runs for a fixed number of steps, giving CUDA Graph batching a chance
to amortize launch overhead, while the active set (BURNING nodes) grows then
shrinks, demonstrating the value of fixed-grid compaction.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fire_spread_kernel(
    row_ptr_ptr,
    col_ind_ptr,
    state_ptr,
    next_state_ptr,
    active_buf_ptr,
    num_active_ptr,
    next_active_mask_ptr,
    p_fire: tl.constexpr,
    rng_seed,
    step_id_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """One step of fire spread from the compacted active set."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    num_active = tl.load(num_active_ptr)
    mask = offs < num_active

    node = tl.load(active_buf_ptr + offs, mask=mask, other=0)

    # BURNING nodes ignite neighbors
    row_start = tl.load(row_ptr_ptr + node, mask=mask, other=0)
    row_end = tl.load(row_ptr_ptr + node + 1, mask=mask, other=0)

    step_id = tl.load(step_id_ptr).to(tl.int32)

    curr = row_start
    active_any = tl.max(curr < row_end, axis=0)

    while active_any != 0:
        active_lane = (curr < row_end) & mask
        neighbor = tl.load(col_ind_ptr + curr, mask=active_lane, other=0)

        # Only ignite EMPTY neighbors
        n_state = tl.load(state_ptr + neighbor, mask=active_lane, other=2)
        is_empty = n_state == 0

        # RNG: deterministic per (step, node)
        rand_val = tl.rand(rng_seed + step_id, neighbor)
        will_ignite = rand_val < p_fire

        # Write to next_state (races benign: all write 1)
        tl.store(next_state_ptr + neighbor, 1, mask=active_lane & is_empty & will_ignite)
        # Mark for next frontier
        tl.atomic_add(next_active_mask_ptr + neighbor, 1, mask=active_lane & is_empty & will_ignite)

        curr += 1
        active_any = tl.max(curr < row_end, axis=0)

    # Current node becomes BURNED in next_state
    tl.store(next_state_ptr + node, 2, mask=mask)


class FireSpreadEngine:
    """Fire spread simulation with fixed-grid active-node compaction."""

    def __init__(self, row_ptr, col_ind, num_nodes: int, p_fire: float = 0.3, block_size: int = 128):
        self.row_ptr = row_ptr
        self.col_ind = col_ind
        self.N = num_nodes
        self.p_fire = float(p_fire)
        self.BLOCK_SIZE = block_size
        self.device = row_ptr.device

        pad = block_size
        self.state = torch.zeros(num_nodes, dtype=torch.int32, device=self.device)
        self.next_state = torch.zeros(num_nodes, dtype=torch.int32, device=self.device)
        self.active_nodes = torch.zeros(num_nodes + pad, dtype=torch.int32, device=self.device)
        self.num_active = torch.tensor([0], dtype=torch.int32, device=self.device)
        self.next_mask = torch.zeros(num_nodes, dtype=torch.int32, device=self.device)
        self.step_id = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._rng_seed = 12345

        # CUDA Graph
        self._cg = None

    def reset(self, sources: torch.Tensor):
        self.state.zero_()
        self.next_state.zero_()
        self.state[sources] = 1  # BURNING
        self.active_nodes[:sources.numel()].copy_(sources.to(torch.int32))
        self.num_active.fill_(sources.numel())
        self.next_mask.zero_()
        self.step_id.zero_()
        self._cg = None

    def _step_kernel(self):
        grid = (triton.cdiv(self.N, self.BLOCK_SIZE),)
        _fire_spread_kernel[grid](
            self.row_ptr, self.col_ind,
            self.state, self.next_state,
            self.active_nodes, self.num_active,
            self.next_mask,
            p_fire=self.p_fire,
            rng_seed=self._rng_seed,
            step_id_ptr=self.step_id,
            N=self.N,
            BLOCK_SIZE=self.BLOCK_SIZE,
        )

    def _refresh_active(self):
        new_active = torch.nonzero(self.next_mask, as_tuple=False).squeeze(-1).to(torch.int32)
        num = int(new_active.numel())
        if num > 0:
            self.active_nodes[:num].copy_(new_active)
        self.num_active.fill_(num)
        self.next_mask.zero_()
        # Copy next_state back to state for next step
        self.state.copy_(self.next_state)
        return num

    def run_eager(self, sources: torch.Tensor, num_steps: int):
        self.reset(sources)
        active_history = []
        for _ in range(num_steps):
            self.step_id += 1
            self._step_kernel()
            num = self._refresh_active()
            active_history.append(num)
            if num == 0:
                break
        return self.state.clone(), active_history

    def _capture_graph(self, steps_per_launch: int):
        """Capture a graph that runs steps_per_launch steps."""
        # Snapshot state so capture doesn't mutate simulation
        snapshot = {
            'state': self.state.clone(),
            'next_state': self.next_state.clone(),
            'active_nodes': self.active_nodes.clone(),
            'num_active': self.num_active.clone(),
            'next_mask': self.next_mask.clone(),
            'step_id': self.step_id.clone(),
        }

        # Warmup: run the full step+refresh cycle a few times
        for _ in range(3):
            self.step_id += 1
            self._step_kernel()
            self._refresh_active()
        torch.cuda.synchronize()

        # Reset to snapshot before capture
        self.state.copy_(snapshot['state'])
        self.next_state.copy_(snapshot['next_state'])
        self.active_nodes.copy_(snapshot['active_nodes'])
        self.num_active.copy_(snapshot['num_active'])
        self.next_mask.copy_(snapshot['next_mask'])
        self.step_id.copy_(snapshot['step_id'])

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(steps_per_launch):
                self.step_id += 1
                self._step_kernel()
                self.state.copy_(self.next_state)

        return g

    def run_cudagraph(self, sources: torch.Tensor, num_steps: int, steps_per_launch: int = 1):
        self.reset(sources)
        self._cg = self._capture_graph(steps_per_launch)

        active_history = []
        remaining = num_steps
        while remaining > 0:
            batch = min(steps_per_launch, remaining)
            # Replay batch steps
            for _ in range(batch):
                self._cg.replay()
            remaining -= batch

            # Refresh frontier OUTSIDE graph
            num = self._refresh_active()
            active_history.append(num)
            if num == 0:
                break

        return self.state.clone(), active_history
