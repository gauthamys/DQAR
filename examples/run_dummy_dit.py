"""
Smoke test for the DummyDiffusionTransformer + DQAR controller.

Usage:
    PYTHONPATH=src python examples/run_dummy_dit.py
"""

from __future__ import annotations

import argparse

from dqar import DQARConfig, DQARController
from dqar.dummy_dit import DummyDiffusionTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dummy diffusion transformer with DQAR.")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--prompt-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--entropy-threshold", type=float, default=1.5)
    parser.add_argument("--min-prob", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DQARConfig()
    config.gate.entropy_threshold = args.entropy_threshold
    config.gate.min_probability = args.min_prob
    controller = DQARController(num_layers=args.layers, config=config)

    model = DummyDiffusionTransformer(num_layers=args.layers)
    output = model.generate(
        controller=controller,
        steps=args.steps,
        prompt_length=args.prompt_length,
        seed=args.seed,
    )

    print(f"Final latent norm: {sum(v * v for v in output.latents) ** 0.5:.4f}")
    print(f"Reuse events: {output.reuse_events}")
    print(f"Recompute events: {output.recompute_events}")


if __name__ == "__main__":
    main()
