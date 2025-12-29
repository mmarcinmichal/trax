# coding=utf-8
"""Lightweight trainer parity benchmark.

Runs a tiny model/dataset in both Trax (JAX) and PyTorch and reports simple
runtime stats. Intended for manual execution in CI to spot regressions in
multi-device gradient aggregation, including Adasum.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Dict, List

import jax
import numpy as np
try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for the trainer parity benchmark. Install torch before running."
    ) from exc

from trax import fastmath, optimizers
from trax import layers as tl
from trax.trainers import jax as trainers_jax


def _prepare_batch(batch_size: int):
    """Builds a deterministic toy regression batch."""
    inputs = np.linspace(0.0, 1.0, batch_size * 2, dtype=np.float32).reshape(batch_size, 2)
    targets = np.linspace(1.0, 2.0, batch_size, dtype=np.float32).reshape(batch_size, 1)
    weights = np.ones_like(targets, dtype=np.float32)
    return inputs, targets, weights


def _prepare_models(batch, learning_rate: float, n_devices: int, adasum: bool):
    """Initializes aligned PyTorch and Trax models/optimizers."""
    torch.manual_seed(0)
    input_dim = batch[0].shape[-1]

    torch_model = torch.nn.Linear(input_dim, 1, bias=True)
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
    torch_loss = torch.nn.MSELoss()

    trax_model = tl.Serial(tl.Dense(1), tl.L2Loss())
    trax_model.init(batch, rng=fastmath.random.get_prng(0))

    # Copy PyTorch initialized weights into Trax for parity.
    torch_weight = torch_model.weight.detach().cpu().numpy().astype(np.float32)
    torch_bias = torch_model.bias.detach().cpu().numpy().astype(np.float32)
    dense_layer = trax_model.sublayers[0]
    dense_layer.weights = (torch_weight.T, torch_bias)
    trax_model.weights = (dense_layer.weights, ())

    trax_optimizer = optimizers.SGD(learning_rate)
    trax_optimizer.tree_init(trax_model.weights)
    trax_trainer = trainers_jax.Trainer(trax_model, trax_optimizer, n_devices=n_devices, adasum=adasum)

    return torch_model, torch_optimizer, torch_loss, trax_trainer


def _benchmark_trax(trainer, batch, rngs, warmup_steps: int, measured_steps: int):
    warmup_rngs = rngs[:warmup_steps]
    measure_rngs = rngs[warmup_steps:]

    warmup_start = time.perf_counter()
    for step, rng in enumerate(warmup_rngs):
        trainer.one_step(batch, rng, step=step)
    warmup_time = time.perf_counter() - warmup_start

    step_times: List[float] = []
    for idx, rng in enumerate(measure_rngs):
        start = time.perf_counter()
        trainer.one_step(batch, rng, step=warmup_steps + idx)
        step_times.append(time.perf_counter() - start)

    return warmup_time, step_times


def _benchmark_torch(torch_model, torch_optimizer, torch_loss, batch, warmup_steps: int, measured_steps: int):
    inputs = torch.from_numpy(batch[0])
    targets = torch.from_numpy(batch[1])

    for step in range(warmup_steps):
        torch_optimizer.zero_grad()
        torch_loss(torch_model(inputs), targets).backward()
        torch_optimizer.step()

    step_times: List[float] = []
    for _ in range(measured_steps):
        start = time.perf_counter()
        torch_optimizer.zero_grad()
        outputs = torch_model(inputs)
        loss_val = torch_loss(outputs, targets)
        loss_val.backward()
        torch_optimizer.step()
        step_times.append(time.perf_counter() - start)

    return step_times


def _compute_metrics(step_times: List[float], tokens_per_step: int, warmup_time: float = 0.0):
    mean_time = float(np.mean(step_times)) if step_times else 0.0
    median_time = float(np.median(step_times)) if step_times else 0.0
    first_step_time = step_times[0] if step_times else 0.0
    steady_times = step_times[1:] if len(step_times) > 1 else step_times
    steady_median = float(np.median(steady_times)) if steady_times else 0.0
    return {
        "mean_step_time_s": mean_time,
        "median_step_time_s": median_time,
        "first_step_time_s": first_step_time,
        "steady_state_time_s": steady_median,
        "jit_warmup_overhead_s": first_step_time - steady_median,
        "tokens_per_second_mean": tokens_per_step / mean_time if mean_time else 0.0,
        "tokens_per_second_median": tokens_per_step / median_time if median_time else 0.0,
        "warmup_time_s": warmup_time,
    }


def _format_table(rows: List[Dict[str, float]]):
    headers = [
        "backend",
        "mean (s)",
        "median (s)",
        "first (s)",
        "steady (s)",
        "tokens/s (mean)",
        "tokens/s (median)",
        "warmup (s)",
    ]
    col_widths = [max(len(h), 12) for h in headers]

    def _fmt_row(values: List[str]):
        return " ".join(v.ljust(w) for v, w in zip(values, col_widths))

    lines = [_fmt_row(headers)]
    for row in rows:
        values = [
            row["backend"],
            f"{row['mean_step_time_s']:.6f}",
            f"{row['median_step_time_s']:.6f}",
            f"{row['first_step_time_s']:.6f}",
            f"{row['steady_state_time_s']:.6f}",
            f"{row['tokens_per_second_mean']:.1f}",
            f"{row['tokens_per_second_median']:.1f}",
            f"{row['warmup_time_s']:.6f}",
        ]
        lines.append(_fmt_row(values))

    return "\n".join(lines)


def _write_json(path: str, results: List[Dict]):
    payload = {"results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: str, results: List[Dict]):
    fieldnames = list(results[0].keys()) if results else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-steps", type=int, default=2, help="Steps to run before timing")
    parser.add_argument("--measured-steps", type=int, default=5, help="Timed training steps to run")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the toy dataset")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for both trainers")
    parser.add_argument("--n-devices", type=int, default=jax.local_device_count(), help="Number of JAX devices to use")
    parser.add_argument("--adasum", action="store_true", help="Use Adasum reduction instead of pmean")
    parser.add_argument("--json-output", type=str, help="Optional path to write JSON results")
    parser.add_argument("--csv-output", type=str, help="Optional path to write CSV results")
    args = parser.parse_args()

    available_devices = jax.local_device_count()
    if args.n_devices < 1:
        raise ValueError("--n-devices must be positive")
    if args.n_devices > available_devices:
        raise ValueError(f"Requested {args.n_devices} devices, but only {available_devices} are available")

    batch = _prepare_batch(args.batch_size)
    torch_model, torch_optimizer, torch_loss, trax_trainer = _prepare_models(
        batch, args.learning_rate, args.n_devices, args.adasum
    )

    rng = fastmath.random.get_prng(123)
    rngs = fastmath.random.split(rng, args.warmup_steps + args.measured_steps)

    warmup_time, trax_step_times = _benchmark_trax(
        trax_trainer, batch, rngs, args.warmup_steps, args.measured_steps
    )
    torch_step_times = _benchmark_torch(
        torch_model, torch_optimizer, torch_loss, batch, args.warmup_steps, args.measured_steps
    )

    tokens_per_step = args.batch_size * batch[0].shape[-1]

    trax_metrics = _compute_metrics(trax_step_times, tokens_per_step, warmup_time=warmup_time)
    torch_metrics = _compute_metrics(torch_step_times, tokens_per_step)

    results: List[Dict] = []
    for backend, metrics in ("trax", trax_metrics), ("torch", torch_metrics):
        record: Dict[str, float | int | str | bool] = {
            "backend": backend,
            "batch_size": args.batch_size,
            "n_devices": args.n_devices,
            "adasum": args.adasum,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            **metrics,
        }
        results.append(record)

    print("Trainer parity benchmark")
    print(f"Batch size: {args.batch_size}, n_devices: {args.n_devices}, adasum: {args.adasum}")
    print(_format_table(results))

    if args.json_output:
        _write_json(args.json_output, results)
    if args.csv_output:
        _write_csv(args.csv_output, results)


if __name__ == "__main__":
    main()
