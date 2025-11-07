#!/usr/bin/env python3
"""
Sweep the dummy benchmark across entropy thresholds and reuse limits, then plot metrics.

Usage:
    PYTHONPATH=src python scripts/dummy_benchmark_sweep.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import tempfile
from pathlib import Path
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = REPO_ROOT / "scripts" / "benchmark_dummy_dit.py"


def parse_csv_floats(value: str) -> list[float]:
    return [float(token.strip()) for token in value.split(",") if token.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def run_benchmark(threshold: float, reuse_limit: int, gap: int, runs: int, prompt_length: int) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    cmd = [
        sys.executable,
        str(BENCH_SCRIPT),
        f"--entropy-threshold={threshold}",
        f"--dqar-max-gap={gap}",
        f"--dqar-max-reuse={reuse_limit}",
        f"--runs={runs}",
        f"--prompt-length={prompt_length}",
        f"--output={tmp_path}",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
    results = json.loads(tmp_path.read_text())
    tmp_path.unlink(missing_ok=True)
    return results


def collect_data(thresholds: list[float], reuse_limits: list[int], gap: int, runs: int, prompt_length: int):
    records = []
    for threshold, reuse_limit in itertools.product(thresholds, reuse_limits):
        summary = run_benchmark(threshold, reuse_limit, gap, runs, prompt_length)
        for entry in summary:
            records.append(
                {
                    "threshold": threshold,
                    "max_reuse": reuse_limit,
                    "scenario": entry["name"],
                    "avg_time_s": entry["avg_time_s"],
                    "avg_rss_mb": entry["avg_rss_mb"],
                    "avg_reuse": entry["avg_reuse"],
                }
            )
    return records


def plot_metrics(records: list[dict], thresholds: list[float], reuse_limits: list[int], output_path: Path) -> None:
    combos = list(itertools.product(thresholds, reuse_limits))
    combo_labels = [f"T{thr:g}|R{reuse}" for thr, reuse in combos]
    x = list(range(len(combos)))
    scenarios = ["baseline", "static", "dqar"]
    metrics = [
        ("avg_time_s", "Average Runtime (s)"),
        ("avg_rss_mb", "Average RSS (MB)"),
        ("avg_reuse", "Average Reuse Events"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for metric_idx, (metric_key, title) in enumerate(metrics):
        ax = axes[metric_idx]
        for scenario in scenarios:
            values = []
            for thr, reuse in combos:
                match = next(
                    (
                        rec
                        for rec in records
                        if rec["threshold"] == thr and rec["max_reuse"] == reuse and rec["scenario"] == scenario
                    ),
                    None,
                )
                values.append(match[metric_key] if match else float("nan"))
            ax.plot(x, values, marker="o", label=scenario.title())
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(combo_labels, rotation=45, ha="right")
        ax.grid(True, linestyle="--", alpha=0.3)
        if metric_idx == 2:
            ax.set_ylabel("Events")
        elif metric_idx == 1:
            ax.set_ylabel("MB")
        else:
            ax.set_ylabel("Seconds")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"[+] Saved sweep plot to {output_path}")


def save_records(records: list[dict], path: Path) -> None:
    path.write_text(json.dumps(records, indent=2))
    print(f"[+] Wrote sweep data to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dummy benchmark sweeps and plot metrics.")
    parser.add_argument("--thresholds", default="1.5,3.0,5.0", help="Comma-separated entropy thresholds.")
    parser.add_argument("--reuse-limits", default="2,4,6", help="Comma-separated DQAR max reuse values.")
    parser.add_argument("--dqar-max-gap", type=int, default=6, help="DQAR max gap (applied to all combos).")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per configuration.")
    parser.add_argument("--prompt-length", type=int, default=12)
    parser.add_argument("--plot-path", type=Path, default=Path("dummy_benchmark_sweep.png"))
    parser.add_argument("--data-path", type=Path, default=Path("dummy_benchmark_sweep.json"))
    args = parser.parse_args()

    thresholds = parse_csv_floats(args.thresholds)
    reuse_limits = parse_csv_ints(args.reuse_limits)
    records = collect_data(thresholds, reuse_limits, args.dqar_max_gap, args.runs, args.prompt_length)
    save_records(records, args.data_path)
    plot_metrics(records, thresholds, reuse_limits, args.plot_path)


if __name__ == "__main__":
    main()
