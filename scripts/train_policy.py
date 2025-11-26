#!/usr/bin/env python3
"""
Policy training pipeline for DQAR.

This script:
1. Collects traces by running diffusion with and without reuse
2. Computes rewards based on quality preservation and compute savings
3. Trains the policy MLP on the collected traces

Usage:
    PYTHONPATH=src python scripts/train_policy.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Tuple

from dqar import DQARConfig, DQARController, DQARPolicy, PolicyConfig
from dqar.policy import PolicyFeatures
from dqar.dummy_dit import DummyDiffusionTransformer


@dataclass(slots=True)
class TraceEntry:
    """A single trace entry for policy training."""
    features: PolicyFeatures
    reused: bool
    latent_diff_norm: float  # Difference from baseline latent
    compute_saved: float  # Fraction of compute saved (1.0 if reused)


@dataclass
class TraceCollector:
    """Collects traces for policy training."""
    traces: List[TraceEntry] = field(default_factory=list)

    def add_trace(
        self,
        features: PolicyFeatures,
        reused: bool,
        latent_diff_norm: float,
        compute_saved: float,
    ) -> None:
        self.traces.append(TraceEntry(
            features=features,
            reused=reused,
            latent_diff_norm=latent_diff_norm,
            compute_saved=compute_saved,
        ))


def _l2_distance(a: List[float], b: List[float]) -> float:
    """Compute L2 distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _l2_norm(vec: List[float]) -> float:
    """Compute L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def run_baseline(
    model: DummyDiffusionTransformer,
    steps: int,
    prompt_length: int,
    seed: int,
) -> List[float]:
    """Run baseline (no reuse) to get reference latents."""
    # Create a controller that never reuses (very high entropy threshold)
    config = DQARConfig()
    config.gate.entropy_threshold = 1e9
    config.gate.min_step = 1000  # Never starts
    controller = DQARController(num_layers=model.num_layers, config=config)

    output = model.generate(
        controller=controller,
        steps=steps,
        prompt_length=prompt_length,
        seed=seed,
    )
    return output.latents


def collect_traces(
    model: DummyDiffusionTransformer,
    config: DQARConfig,
    baseline_latents: List[float],
    steps: int,
    prompt_length: int,
    seed: int,
    num_layers: int,
) -> TraceCollector:
    """
    Collect traces by running diffusion and recording decisions.

    For each layer at each step, we record:
    - The features at decision time
    - Whether reuse was attempted
    - The resulting quality (latent difference from baseline)
    """
    collector = TraceCollector()
    controller = DQARController(num_layers=num_layers, config=config)

    random.seed(seed)
    latents = [random.uniform(-1.0, 1.0) for _ in range(model.latent_dim)]

    total_compute = steps * num_layers
    compute_saved = 0

    for step in range(steps):
        latent_norm = math.sqrt(sum(v * v for v in latents))
        controller.begin_step(
            step_index=step,
            total_steps=steps,
            prompt_length=prompt_length,
            latent_norm=latent_norm,
        )

        snr = controller.get_estimated_snr()

        for layer_idx in range(num_layers):
            decision = controller.should_reuse(layer_idx, branch="cond")

            # Create features for this decision point
            features = PolicyFeatures(
                entropy=decision.entry.metrics.entropy if decision.entry else 0.0,
                snr=snr,
                latent_norm=latent_norm,
                step_index=step,
                total_steps=steps,
                prompt_length=prompt_length,
            )

            if decision.use_cache and decision.entry is not None:
                _, _, residual = controller.reuse(decision.entry)
                if residual is not None:
                    latents = [v + float(d) for v, d in zip(latents, residual)]
                compute_saved += 1

                # Record trace with current quality
                latent_diff = _l2_distance(latents, baseline_latents)
                collector.add_trace(
                    features=features,
                    reused=True,
                    latent_diff_norm=latent_diff,
                    compute_saved=1.0,
                )
            else:
                # Compute fresh attention (simplified simulation)
                attn_map = _make_attention(model.heads, model.tokens)
                keys = _make_matrix(model.tokens, model.latent_dim)
                values = _make_matrix(model.tokens, model.latent_dim)
                residual = [random.uniform(-0.05, 0.05) for _ in range(model.latent_dim)]

                controller.commit(
                    layer_id=layer_idx,
                    branch="cond",
                    attn_map=attn_map,
                    keys=keys,
                    values=values,
                    latent=latents,
                    residual=residual,
                    prompt_length=prompt_length,
                )
                latents = [v + d for v, d in zip(latents, residual)]

                # Record trace for non-reuse
                latent_diff = _l2_distance(latents, baseline_latents)
                collector.add_trace(
                    features=features,
                    reused=False,
                    latent_diff_norm=latent_diff,
                    compute_saved=0.0,
                )

    return collector


def _make_matrix(rows: int, cols: int) -> List[List[float]]:
    return [[random.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def _make_attention(heads: int, tokens: int) -> List[List[List[float]]]:
    attn = []
    for _ in range(heads):
        head_rows = []
        for _ in range(tokens):
            row = [random.random() for _ in range(tokens)]
            total = sum(row) or 1.0
            head_rows.append([v / total for v in row])
        attn.append(head_rows)
    return attn


def compute_reward(
    trace: TraceEntry,
    quality_weight: float = 0.7,
    efficiency_weight: float = 0.3,
    quality_threshold: float = 0.5,
) -> float:
    """
    Compute reward for a reuse decision.

    Reward formula:
    - If reused and quality preserved (latent_diff < threshold): high reward
    - If reused but quality degraded: negative reward
    - If not reused: small positive reward (conservative)
    """
    if trace.reused:
        # Quality penalty: exponential decay as latent differs from baseline
        quality_score = math.exp(-trace.latent_diff_norm / quality_threshold)
        # Efficiency bonus for reusing
        efficiency_score = trace.compute_saved
        reward = quality_weight * quality_score + efficiency_weight * efficiency_score
    else:
        # Conservative non-reuse: small reward
        reward = 0.3

    # Normalize to [0, 1]
    return max(0.0, min(1.0, reward))


def create_training_dataset(
    traces: List[TraceEntry],
    quality_weight: float = 0.7,
    efficiency_weight: float = 0.3,
) -> List[Tuple[PolicyFeatures, float]]:
    """Convert traces to training dataset with rewards as labels."""
    dataset = []
    for trace in traces:
        reward = compute_reward(
            trace,
            quality_weight=quality_weight,
            efficiency_weight=efficiency_weight,
        )
        dataset.append((trace.features, reward))
    return dataset


def evaluate_policy(
    policy: DQARPolicy,
    model: DummyDiffusionTransformer,
    baseline_latents: List[float],
    config: DQARConfig,
    steps: int,
    prompt_length: int,
    seed: int,
) -> dict:
    """Evaluate policy quality vs baseline."""
    config_copy = DQARConfig()
    config_copy.gate.entropy_threshold = config.gate.entropy_threshold
    config_copy.gate.min_probability = 0.5  # Use policy

    controller = DQARController(
        num_layers=model.num_layers,
        config=config_copy,
        policy=policy,
    )

    output = model.generate(
        controller=controller,
        steps=steps,
        prompt_length=prompt_length,
        seed=seed,
    )

    latent_diff = _l2_distance(output.latents, baseline_latents)
    baseline_norm = _l2_norm(baseline_latents)
    relative_error = latent_diff / max(baseline_norm, 1e-6)

    total_ops = steps * model.num_layers
    reuse_rate = output.reuse_events / total_ops

    return {
        "reuse_rate": reuse_rate,
        "reuse_events": output.reuse_events,
        "recompute_events": output.recompute_events,
        "latent_diff": latent_diff,
        "relative_error": relative_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQAR policy.")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--prompt-length", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-traces", type=int, default=5, help="Number of trace collection runs")
    parser.add_argument("--entropy-threshold", type=float, default=3.0)
    parser.add_argument("--quality-weight", type=float, default=0.7)
    parser.add_argument("--efficiency-weight", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=Path("policy_training.json"))
    args = parser.parse_args()

    print(f"[*] Training DQAR policy with {args.epochs} epochs")

    model = DummyDiffusionTransformer(num_layers=args.layers)

    # Initialize policy
    policy_config = PolicyConfig(hidden_dim=args.hidden_dim, num_hidden_layers=2)
    policy = DQARPolicy(policy_config)

    # DQAR config for trace collection
    dqar_config = DQARConfig()
    dqar_config.gate.entropy_threshold = args.entropy_threshold
    dqar_config.gate.min_step = 2

    all_traces = []
    baseline_cache = {}

    print(f"[*] Collecting traces from {args.num_traces} runs...")
    for run_idx in range(args.num_traces):
        seed = 42 + run_idx

        # Get or compute baseline
        if seed not in baseline_cache:
            baseline_cache[seed] = run_baseline(
                model, args.steps, args.prompt_length, seed
            )
        baseline_latents = baseline_cache[seed]

        # Collect traces
        collector = collect_traces(
            model=model,
            config=dqar_config,
            baseline_latents=baseline_latents,
            steps=args.steps,
            prompt_length=args.prompt_length,
            seed=seed,
            num_layers=args.layers,
        )
        all_traces.extend(collector.traces)
        print(f"  Run {run_idx + 1}: collected {len(collector.traces)} traces")

    print(f"[*] Total traces: {len(all_traces)}")

    # Create training dataset
    dataset = create_training_dataset(
        all_traces,
        quality_weight=args.quality_weight,
        efficiency_weight=args.efficiency_weight,
    )

    # Evaluate before training
    eval_seed = 99
    if eval_seed not in baseline_cache:
        baseline_cache[eval_seed] = run_baseline(
            model, args.steps, args.prompt_length, eval_seed
        )

    pre_eval = evaluate_policy(
        policy, model, baseline_cache[eval_seed],
        dqar_config, args.steps, args.prompt_length, eval_seed
    )
    print(f"[*] Pre-training eval: reuse_rate={pre_eval['reuse_rate']:.3f}, "
          f"relative_error={pre_eval['relative_error']:.4f}")

    # Train policy
    print(f"[*] Training policy for {args.epochs} epochs with lr={args.lr}...")
    for epoch in range(args.epochs):
        random.shuffle(dataset)
        policy.fit(dataset, epochs=1, lr=args.lr)

        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            eval_result = evaluate_policy(
                policy, model, baseline_cache[eval_seed],
                dqar_config, args.steps, args.prompt_length, eval_seed
            )
            print(f"  Epoch {epoch + 1}: reuse_rate={eval_result['reuse_rate']:.3f}, "
                  f"relative_error={eval_result['relative_error']:.4f}")

    # Final evaluation
    post_eval = evaluate_policy(
        policy, model, baseline_cache[eval_seed],
        dqar_config, args.steps, args.prompt_length, eval_seed
    )
    print(f"[*] Post-training eval: reuse_rate={post_eval['reuse_rate']:.3f}, "
          f"relative_error={post_eval['relative_error']:.4f}")

    # Save results
    results = {
        "config": {
            "layers": args.layers,
            "steps": args.steps,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "num_traces": args.num_traces,
            "entropy_threshold": args.entropy_threshold,
            "quality_weight": args.quality_weight,
            "efficiency_weight": args.efficiency_weight,
        },
        "num_traces": len(all_traces),
        "pre_training": pre_eval,
        "post_training": post_eval,
        "improvement": {
            "reuse_rate_delta": post_eval["reuse_rate"] - pre_eval["reuse_rate"],
            "error_delta": post_eval["relative_error"] - pre_eval["relative_error"],
        },
    }

    args.output.write_text(json.dumps(results, indent=2))
    print(f"[+] Saved training results to {args.output}")


if __name__ == "__main__":
    main()
