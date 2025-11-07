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

from dqar import DQARConfig, DQARController
from dqar.dummy_dit import DummyDiffusionTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DummyDiffusionTransformer.")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--prompt-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--entropy-threshold", type=float, default=1.5)
    parser.add_argument("--min-prob", type=float, default=0.4)
    parser.add_argument("--output", type=Path, default=Path("dummy_benchmark.json"))
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
    return {
        "name": name,
        "samples": len(timings),
        "avg_time_s": mean(timings),
        "std_time_s": pstdev(timings) if len(timings) > 1 else 0.0,
        "avg_reuse": mean(reuse_counts),
        "avg_rss_mb": mean(mem_bytes) / (1024 * 1024),
    }


def main() -> None:
    args = parse_args()
    model = DummyDiffusionTransformer(num_layers=args.layers)

    baseline_times = []
    baseline_reuse = []
    baseline_mem = []
    for run_idx in range(args.runs):
        controller = DQARController(num_layers=args.layers)
        elapsed, output, mem = run_once(model, controller, args)
        baseline_times.append(elapsed)
        baseline_reuse.append(output.reuse_events)
        baseline_mem.append(mem)

    config = DQARConfig()
    config.gate.min_step = 0
    config.gate.entropy_threshold = 10.0  # effectively disable entropy gate
    config.gate.min_probability = 0.0
    config.gate.snr_range = (0.0, 1e6)
    config.gate.cooldown_steps = 0
    config.scheduler.max_gap = 12
    config.scheduler.max_reuse_per_block = 100
    dqar_times = []
    dqar_reuse = []
    dqar_mem = []
    for run_idx in range(args.runs):
        controller = DQARController(num_layers=args.layers, config=config)
        elapsed, output, mem = run_once(model, controller, args)
        dqar_times.append(elapsed)
        dqar_reuse.append(output.reuse_events)
        dqar_mem.append(mem)

    summary = [
        summarize("baseline", baseline_times, baseline_reuse, baseline_mem),
        summarize("dqar", dqar_times, dqar_reuse, dqar_mem),
    ]
    args.output.write_text(json.dumps(summary, indent=2))
    print("Baseline:", summary[0])
    print("DQAR:", summary[1])
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
