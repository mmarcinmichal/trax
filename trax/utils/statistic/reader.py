# coding=utf-8
"""Read training metrics from Trax checkpoint files.

Example:
  python -m trax.utils.statistic_reader --path /tmp/trax_mlp_run/model.pkl.gz
"""

import argparse
import gzip
import os
import pickle


def _resolve_path(path):
    if os.path.isdir(path):
        for name in ("model.pkl.gz", "highest_accuracy.pkl.gz", "lowest_loss.pkl.gz"):
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            "No checkpoint found in directory: %s" % path
        )
    return path


def _load_history(path):
    with gzip.open(path, "rb") as f:
        ckpt = pickle.load(f)
    history = ckpt.get("history")
    if not history:
        raise ValueError("Checkpoint has no history: %s" % path)
    return history


def _metrics(history):
    values = history.get("_values", {})
    return {
        mode: sorted(list(metrics.keys()))
        for mode, metrics in values.items()
    }


def _series(history, mode, metric):
    values = history.get("_values", {})
    return list(values.get(mode, {}).get(metric, []))


def _print_series(series, tail):
    if not series:
        print("(empty)")
        return
    data = series[-tail:] if tail else series
    for step, value in data:
        print(f"{step}\t{value}")


def main():
    parser = argparse.ArgumentParser(
        description="Read Trax training metrics from checkpoints."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Checkpoint file (.pkl.gz) or output_dir containing checkpoints.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Mode to read (e.g. train/eval). Defaults to all.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Metric to read (e.g. metrics/loss). Defaults to all.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=10,
        help="How many last entries to show (0 = all).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available modes and metrics.",
    )
    args = parser.parse_args()

    path = _resolve_path(args.path)
    history = _load_history(path)

    metrics = _metrics(history)
    if args.list:
        for mode, metric_list in metrics.items():
            print(f"{mode}:")
            for name in metric_list:
                print(f"  - {name}")
        return

    modes = [args.mode] if args.mode else list(metrics.keys())
    for mode in modes:
        metric_list = [args.metric] if args.metric else metrics.get(mode, [])
        print(f"== {mode} ==")
        if not metric_list:
            print("(no metrics)")
            continue
        for metric in metric_list:
            print(f"-- {metric} --")
            series = _series(history, mode, metric)
            _print_series(series, args.tail)


if __name__ == "__main__":
    main()
