#!/usr/bin/env python3
"""
Benchmark DummyDiffusionTransformer with and without DQAR.

Usage:
    PYTHONPATH=src python scripts/benchmark_dummy_dit.py
"""

from __future__ import annotations

import argparse
import json
import time
import resource
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

from dqar import DQARConfig, DQARController
from dqar.dummy_dit import DummyDiffusionTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DummyDiffusionTransformer.")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--prompt-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--entropy-threshold", type=float, default=3.0)
    parser.add_argument("--min-prob", type=float, default=0.0)
    parser.add_argument("--static-max-gap", type=int, default=12, help="Max gap for static reuse scheduling.")
    parser.add_argument("--static-max-reuse", type=int, default=100, help="Max reuse per block for static scheduling.")
    parser.add_argument("--dqar-max-gap", type=int, default=6, help="Max gap for adaptive DQAR scheduling.")
    parser.add_argument("--dqar-max-reuse", type=int, default=6, help="Max reuse per block for adaptive scheduling.")
    parser.add_argument("--output", type=Path, default=Path("dummy_benchmark.json"))
    parser.add_argument("--plot", type=Path, default=Path("dummy_benchmark.png"), help="Output plot path.")
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep mode.")
    parser.add_argument("--sweep-plot", type=Path, default=Path("dummy_benchmark_sweep.png"), help="Sweep plot path.")
    return parser.parse_args()


def _rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS ru_maxrss is in bytes, on Linux it's kilobytes.
    if usage < 10_000_000:  # assume kilobytes if the number is small
        usage *= 1024
    return usage


def run_once(model: DummyDiffusionTransformer, controller: DQARController, args: argparse.Namespace):
    start = time.perf_counter()
    before_mem = _rss_bytes()
    output = model.generate(
        controller=controller,
        steps=args.steps,
        prompt_length=args.prompt_length,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - start
    after_mem = _rss_bytes()
    peak_mem = max(before_mem, after_mem)
    return elapsed, output, peak_mem


def summarize(name: str, timings: list[float], reuse_counts: list[int], mem_bytes: list[int]) -> dict:
    mem_mb = [m / (1024 * 1024) for m in mem_bytes]
    n = len(timings)
    return {
        "name": name,
        "samples": n,
        "avg_time_s": mean(timings),
        "std_time_s": pstdev(timings) if n > 1 else 0.0,
        "avg_reuse": mean(reuse_counts),
        "std_reuse": pstdev(reuse_counts) if n > 1 else 0.0,
        "avg_rss_mb": mean(mem_mb),
        "std_rss_mb": pstdev(mem_mb) if n > 1 else 0.0,
    }


def _benchmark_scenario(name: str, model: DummyDiffusionTransformer, controller_factory, args: argparse.Namespace) -> dict:
    timings = []
    reuse_counts = []
    mem_usages = []
    for _ in range(args.runs):
        controller = controller_factory()
        elapsed, output, mem = run_once(model, controller, args)
        timings.append(elapsed)
        reuse_counts.append(output.reuse_events)
        mem_usages.append(mem)
    return summarize(name, timings, reuse_counts, mem_usages)


def _make_baseline_config(args: argparse.Namespace) -> DQARConfig:
    """Config that prevents ALL reuse - true baseline with zero reuse events."""
    config = DQARConfig()
    config.gate.min_step = 9999  # Never start reusing (step will never reach this)
    config.gate.entropy_threshold = 0.0  # Impossible to pass (entropy is always >= 0)
    config.gate.min_probability = 1.0  # Policy always rejects
    config.gate.snr_range = (1e9, 1e9)  # Impossible SNR range
    config.gate.cooldown_steps = 9999  # Huge cooldown
    config.scheduler.max_gap = 0  # No gap allowed
    config.scheduler.max_reuse_per_block = 0  # No reuse budget
    return config


def _make_static_config(args: argparse.Namespace) -> DQARConfig:
    config = DQARConfig()
    config.gate.min_step = 0
    config.gate.entropy_threshold = 1e9
    config.gate.min_probability = 0.0
    config.gate.snr_range = (0.0, 1e9)
    config.gate.cooldown_steps = 0
    config.scheduler.max_gap = args.static_max_gap
    config.scheduler.max_reuse_per_block = args.static_max_reuse
    return config


def _make_dqar_config(args: argparse.Namespace) -> DQARConfig:
    config = DQARConfig()
    config.gate.entropy_threshold = args.entropy_threshold
    config.gate.min_probability = args.min_prob
    config.gate.min_step = 0
    config.gate.cooldown_steps = 0
    config.gate.snr_range = (0.0, 1e9)  # Permissive SNR range
    config.scheduler.max_gap = args.dqar_max_gap
    config.scheduler.max_reuse_per_block = args.dqar_max_reuse
    return config


def _make_dqar_config_with_threshold(args: argparse.Namespace, threshold: float) -> DQARConfig:
    """Create DQAR config with a specific entropy threshold."""
    config = DQARConfig()
    config.gate.entropy_threshold = threshold
    config.gate.min_probability = args.min_prob
    config.gate.min_step = 0
    config.gate.cooldown_steps = 0
    config.gate.snr_range = (0.0, 1e9)
    config.scheduler.max_gap = args.dqar_max_gap
    config.scheduler.max_reuse_per_block = args.dqar_max_reuse
    return config


def _run_threshold_sweep(model: DummyDiffusionTransformer, args: argparse.Namespace) -> list[dict]:
    """Sweep across different entropy thresholds."""
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    results = []

    for thresh in thresholds:
        result = _benchmark_scenario(
            f"thresh_{thresh}",
            model,
            controller_factory=lambda t=thresh: DQARController(
                num_layers=args.layers,
                config=_make_dqar_config_with_threshold(args, t),
            ),
            args=args,
        )
        result["threshold"] = thresh
        results.append(result)
        print(f"Threshold {thresh:.1f}: reuse={result['avg_reuse']:.0f}, time={result['avg_time_s']*1000:.2f}ms")

    return results


def _plot_sweep_results(results: list[dict], baseline: dict, static: dict, output_path: Path) -> None:
    """Plot threshold sweep results."""
    thresholds = [r["threshold"] for r in results]
    reuse_counts = [r["avg_reuse"] for r in results]
    times_ms = [r["avg_time_s"] * 1000 for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Threshold vs Reuse Rate
    ax1 = axes[0]
    ax1.plot(thresholds, reuse_counts, "o-", color="#C44E52", linewidth=2, markersize=8, label="DQAR")
    ax1.axhline(y=baseline["avg_reuse"], color="#4C72B0", linestyle="--", linewidth=2, label=f"Baseline ({baseline['avg_reuse']:.0f})")
    ax1.axhline(y=static["avg_reuse"], color="#55A868", linestyle="--", linewidth=2, label=f"Static ({static['avg_reuse']:.0f})")
    ax1.set_xlabel("Entropy Threshold", fontsize=12)
    ax1.set_ylabel("Reuse Events", fontsize=12)
    ax1.set_title("Entropy Threshold vs Reuse Count", fontsize=13)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Threshold vs Inference Time
    ax2 = axes[1]
    ax2.plot(thresholds, times_ms, "o-", color="#C44E52", linewidth=2, markersize=8, label="DQAR")
    ax2.axhline(y=baseline["avg_time_s"] * 1000, color="#4C72B0", linestyle="--", linewidth=2, label=f"Baseline ({baseline['avg_time_s']*1000:.2f}ms)")
    ax2.axhline(y=static["avg_time_s"] * 1000, color="#55A868", linestyle="--", linewidth=2, label=f"Static ({static['avg_time_s']*1000:.2f}ms)")
    ax2.set_xlabel("Entropy Threshold", fontsize=12)
    ax2.set_ylabel("Time (ms)", fontsize=12)
    ax2.set_title("Entropy Threshold vs Inference Time", fontsize=13)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("DQAR Entropy Threshold Sweep", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    model = DummyDiffusionTransformer(num_layers=args.layers)

    baseline = _benchmark_scenario(
        "baseline",
        model,
        controller_factory=lambda: DQARController(num_layers=args.layers, config=_make_baseline_config(args)),
        args=args,
    )
    static = _benchmark_scenario(
        "static",
        model,
        controller_factory=lambda: DQARController(num_layers=args.layers, config=_make_static_config(args)),
        args=args,
    )

    if args.sweep:
        # Run threshold sweep
        print("\n=== Running Entropy Threshold Sweep ===")
        sweep_results = _run_threshold_sweep(model, args)
        sweep_data = {
            "baseline": baseline,
            "static": static,
            "sweep": sweep_results,
        }
        args.output.write_text(json.dumps(sweep_data, indent=2))
        _plot_sweep_results(sweep_results, baseline, static, args.sweep_plot)
        print(f"\nSaved sweep plot to {args.sweep_plot}")
    else:
        dqar = _benchmark_scenario(
            "dqar",
            model,
            controller_factory=lambda: DQARController(num_layers=args.layers, config=_make_dqar_config(args)),
            args=args,
        )
        summary = [baseline, static, dqar]
        args.output.write_text(json.dumps(summary, indent=2))
        for entry in summary:
            print(f"{entry['name'].title()}:", entry)
        print(f"Wrote results to {args.output}")
        _plot_results(summary, args.plot)
        print(f"Saved plot to {args.plot}")


def _plot_results(results: list[dict], output_path: Path) -> None:
    """Generate a bar chart comparing benchmark scenarios."""
    names = [r["name"].title() for r in results]
    times = [r["avg_time_s"] * 1000 for r in results]  # Convert to ms
    time_errs = [r["std_time_s"] * 1000 for r in results]
    reuse = [r["avg_reuse"] for r in results]
    reuse_errs = [r["std_reuse"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Inference Time
    x = np.arange(len(names))
    bars1 = axes[0].bar(x, times, yerr=time_errs, capsize=5, color=["#4C72B0", "#55A868", "#C44E52"])
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_title("Inference Time")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].bar_label(bars1, fmt="%.2f")

    # Plot 2: Reuse Events
    bars2 = axes[1].bar(x, reuse, yerr=reuse_errs, capsize=5, color=["#4C72B0", "#55A868", "#C44E52"])
    axes[1].set_ylabel("Reuse Events")
    axes[1].set_title("Attention Reuse Count")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].bar_label(bars2, fmt="%.0f")

    fig.suptitle("DQAR Dummy DiT Benchmark", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
