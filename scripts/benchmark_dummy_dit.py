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
    parser.add_argument("--min-prob", type=float, default=0.0)
    parser.add_argument("--static-max-gap", type=int, default=12, help="Max gap for static reuse scheduling.")
    parser.add_argument("--static-max-reuse", type=int, default=100, help="Max reuse per block for static scheduling.")
    parser.add_argument("--dqar-max-gap", type=int, default=6, help="Max gap for adaptive DQAR scheduling.")
    parser.add_argument("--dqar-max-reuse", type=int, default=6, help="Max reuse per block for adaptive scheduling.")
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
    config.scheduler.max_gap = args.dqar_max_gap
    config.scheduler.max_reuse_per_block = args.dqar_max_reuse
    return config


def main() -> None:
    args = parse_args()
    model = DummyDiffusionTransformer(num_layers=args.layers)

    baseline = _benchmark_scenario(
        "baseline",
        model,
        controller_factory=lambda: DQARController(num_layers=args.layers),
        args=args,
    )
    static = _benchmark_scenario(
        "static",
        model,
        controller_factory=lambda: DQARController(num_layers=args.layers, config=_make_static_config(args)),
        args=args,
    )
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


if __name__ == "__main__":
    main()
