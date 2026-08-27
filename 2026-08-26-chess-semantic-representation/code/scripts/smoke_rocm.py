#!/usr/bin/env python3
"""Exercise the ROCm PyTorch path and emit reproducible smoke-test metadata."""

from __future__ import annotations

import argparse
import io
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch import nn


class SmokeNet(nn.Module):
    """Small convolutional workload representative of the first project models."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64 * 13),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).reshape(inputs.shape[0], 64, 13)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, default=Path("runs/E0001/smoke.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is not available through torch.cuda")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    model = SmokeNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    inputs = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    targets = torch.randint(0, 13, (args.batch_size, 64), device=device)
    criterion = nn.CrossEntropyLoss()

    def train_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(inputs)
            loss = criterion(logits.flatten(0, 1), targets.flatten())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach())

    for _ in range(args.warmup_steps):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(args.steps):
        losses.append(train_step())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = SmokeNet().to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        original_output = model(inputs[:1])
        restored_output = restored(inputs[:1])
    checkpoint_max_error = float((original_output - restored_output).abs().max())

    properties = torch.cuda.get_device_properties(0)
    record = {
        "experiment_id": "E0001",
        "recorded_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "git_revision": git_revision(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "device_total_memory_bytes": properties.total_memory,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.steps,
        "elapsed_seconds": elapsed,
        "images_per_second": args.batch_size * args.steps / elapsed,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses_finite": all(torch.isfinite(torch.tensor(losses))),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "checkpoint_bytes": checkpoint.getbuffer().nbytes,
        "checkpoint_max_output_error": checkpoint_max_error,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

