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

from typing import Dict, List, Tuple

import jax
import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for the trainer parity benchmark. Install torch before running."
    ) from exc

from learning.training.engines import jax as trainers_jax

from trax import fastmath, optimizers
from trax import layers as tl


def _prepare_batch(
    batch_size: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds a deterministic synthetic regression batch."""
    rng = np.random.default_rng(seed)
    inputs = rng.normal(0.0, 1.0, size=(batch_size, input_dim)).astype(np.float32)
    true_w1 = rng.normal(0.0, 0.5, size=(input_dim, hidden_dim)).astype(np.float32)
    true_b1 = rng.normal(0.0, 0.1, size=(hidden_dim,)).astype(np.float32)
    true_w2 = rng.normal(0.0, 0.5, size=(hidden_dim, output_dim)).astype(np.float32)
    true_b2 = rng.normal(0.0, 0.1, size=(output_dim,)).astype(np.float32)
    hidden = np.tanh(inputs @ true_w1 + true_b1)
    targets = (hidden @ true_w2 + true_b2).astype(np.float32)
    weights = np.ones_like(targets, dtype=np.float32)
    return inputs, targets, weights


def _prepare_models(
    batch,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_hidden_layers: int,
    learning_rate: float,
    n_devices: int,
    adasum: bool,
    torch_device: torch.device,
):
    """Initializes aligned PyTorch and Trax models/optimizers."""
    torch.manual_seed(0)
    torch_layers = []
    in_dim = input_dim
    for _ in range(num_hidden_layers):
        torch_layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=True))
        torch_layers.append(torch.nn.ReLU())
        in_dim = hidden_dim
    torch_layers.append(torch.nn.Linear(in_dim, output_dim, bias=True))
    torch_model = torch.nn.Sequential(*torch_layers).to(torch_device)
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
    torch_loss = torch.nn.MSELoss()

    trax_layers = []
    in_dim = input_dim
    for _ in range(num_hidden_layers):
        trax_layers.append(tl.Dense(hidden_dim))
        trax_layers.append(tl.Relu())
        in_dim = hidden_dim
    trax_layers.append(tl.Dense(output_dim))
    trax_layers.append(tl.L2Loss())
    trax_model = tl.Serial(*trax_layers)
    trax_model.init(batch, rng=fastmath.random.get_prng(0))

    # Copy PyTorch initialized weights into Trax for parity.
    torch_linears = [m for m in torch_model.modules() if isinstance(m, torch.nn.Linear)]
    trax_denses = [m for m in trax_model.sublayers if isinstance(m, tl.Dense)]
    if len(torch_linears) != len(trax_denses):
        raise ValueError(
            "Mismatched layer counts between torch and trax: "
            f"{len(torch_linears)} vs {len(trax_denses)}"
        )
    for torch_layer, trax_layer in zip(torch_linears, trax_denses):
        torch_weight = torch_layer.weight.detach().cpu().numpy().astype(np.float32)
        torch_bias = torch_layer.bias.detach().cpu().numpy().astype(np.float32)
        trax_layer.weights = (torch_weight.T, torch_bias)
    trax_model.weights = tuple(layer.weights for layer in trax_model.sublayers)

    trax_optimizer = optimizers.SGD(learning_rate)
    trax_optimizer.tree_init(trax_model.weights)
    trax_trainer = trainers_jax.TrainingEngine(trax_model, trax_optimizer, n_devices=n_devices, adasum=adasum)

    return torch_model, torch_optimizer, torch_loss, trax_trainer


def _benchmark_trax(
    trainer,
    batch,
    rngs,
    warmup_steps: int,
    measured_steps: int,
    device=None,
):
    if device is not None:
        with jax.default_device(device):
            batch = jax.device_put(batch)
    else:
        batch = jax.device_put(batch)
    warmup_rngs = rngs[:warmup_steps]
    measure_rngs = rngs[warmup_steps:]

    warmup_start = time.perf_counter()
    for step, rng in enumerate(warmup_rngs):
        loss = trainer.one_step(batch, rng, step=step)
        loss.block_until_ready()
    warmup_time = time.perf_counter() - warmup_start

    step_times: List[float] = []
    for idx, rng in enumerate(measure_rngs):
        start = time.perf_counter()
        loss = trainer.one_step(batch, rng, step=warmup_steps + idx)
        loss.block_until_ready()
        step_times.append(time.perf_counter() - start)

    return warmup_time, step_times


def _benchmark_torch(
    torch_model,
    torch_optimizer,
    torch_loss,
    batch,
    warmup_steps: int,
    measured_steps: int,
    device: torch.device,
):
    inputs = torch.from_numpy(batch[0]).to(device)
    targets = torch.from_numpy(batch[1]).to(device)

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
    parser.add_argument("--warmup-steps", type=int, default=10, help="Steps to run before timing")
    parser.add_argument("--measured-steps", type=int, default=100, help="Timed training steps to run")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for the toy dataset")
    parser.add_argument("--input-dim", type=int, default=128, help="Input feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer width")
    parser.add_argument("--output-dim", type=int, default=16, help="Output feature dimension")
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=2,
        help="Number of hidden Dense+ReLU layers",
    )
    parser.add_argument("--data-seed", type=int, default=0, help="Seed for synthetic data")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for both trainers")
    parser.add_argument("--n-devices", type=int, default=jax.local_device_count(), help="Number of JAX devices to use")
    parser.add_argument("--adasum", action="store_true", help="Use Adasum reduction instead of pmean")
    parser.add_argument(
        "--jax-device",
        type=str,
        default="default",
        choices=("default", "cpu", "gpu", "tpu"),
        help="JAX device platform to run on",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        default="cpu",
        help="PyTorch device to run on (e.g., cpu, cuda)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=("default", "large"),
        help="Preset benchmark configuration (overrides size/steps flags).",
    )
    parser.add_argument("--json-output", type=str, help="Optional path to write JSON results")
    parser.add_argument("--csv-output", type=str, help="Optional path to write CSV results")
    args = parser.parse_args()

    if args.profile == "large":
        args.warmup_steps = 20
        args.measured_steps = 200
        args.batch_size = 512
        args.input_dim = 256
        args.hidden_dim = 512
        args.output_dim = 32
        args.num_hidden_layers = 3

    available_devices = jax.local_device_count()
    if args.n_devices < 1:
        raise ValueError("--n-devices must be positive")
    if args.n_devices > available_devices:
        raise ValueError(f"Requested {args.n_devices} devices, but only {available_devices} are available")

    if args.jax_device != "default":
        platform_devices = jax.devices(args.jax_device)
        if not platform_devices:
            raise ValueError(f"No JAX devices available for platform '{args.jax_device}'.")
        if args.n_devices > len(platform_devices):
            raise ValueError(
                f"Requested {args.n_devices} devices for '{args.jax_device}', "
                f"but only {len(platform_devices)} are available"
            )
        jax_device = platform_devices[0]
    else:
        jax_device = None

    torch_device = torch.device(args.torch_device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested torch cuda device but CUDA is not available.")

    batch = _prepare_batch(
        args.batch_size,
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        args.data_seed,
    )
    torch_model, torch_optimizer, torch_loss, trax_trainer = _prepare_models(
        batch,
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        args.num_hidden_layers,
        args.learning_rate,
        args.n_devices,
        args.adasum,
        torch_device,
    )

    rng = fastmath.random.get_prng(123)
    rngs = fastmath.random.split(rng, args.warmup_steps + args.measured_steps)

    warmup_time, trax_step_times = _benchmark_trax(
        trax_trainer, batch, rngs, args.warmup_steps, args.measured_steps, device=jax_device
    )
    torch_step_times = _benchmark_torch(
        torch_model,
        torch_optimizer,
        torch_loss,
        batch,
        args.warmup_steps,
        args.measured_steps,
        device=torch_device,
    )

    tokens_per_step = args.batch_size * args.input_dim

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
