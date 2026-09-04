"""Timing and CUDA Graph helpers."""

import time
import torch


def timed_region(name: str, device: str = "cuda"):
    """Context manager for GPU-synchronized timing."""
    class _Timer:
        def __enter__(self):
            if device == "cuda":
                torch.cuda.synchronize()
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *args):
            if device == "cuda":
                torch.cuda.synchronize()
            self.elapsed = time.perf_counter() - self.t0
    return _Timer()


def capture_cuda_graph(step_fn, warmup: int = 3, steps_per_launch: int = 50):
    """Capture a CUDA Graph that replays `step_fn` multiple times.

    Args:
        step_fn: Callable with no args that performs one step.
        warmup: Number of eager warmup iterations before capture.
        steps_per_launch: Number of steps to batch inside one graph.

    Returns:
        A callable that replays the captured graph.
    """
    device = torch.device("cuda")
    # Warmup
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(steps_per_launch):
            step_fn()

    def replay():
        g.replay()

    return replay
