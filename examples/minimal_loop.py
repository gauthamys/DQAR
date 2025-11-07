"""
Toy sampler loop that demonstrates how DQAR fits around a DiT attention stack.

This script keeps the math intentionally simple (pure Python lists) so it runs
on the bare environment that ships with the Codex CLI.
"""

from __future__ import annotations

import math
import random

from dqar import DQARController, DQARConfig

NUM_LAYERS = 6
TOTAL_STEPS = 12
TOKENS = 8
HEADS = 2
LATENT_DIM = 16


def fake_attention() -> list:
    attn = []
    for _ in range(HEADS):
        head = []
        for _ in range(TOKENS):
            row = [random.random() for _ in range(TOKENS)]
            row_sum = sum(row) or 1.0
            head.append([value / row_sum for value in row])
        attn.append(head)
    return attn


def fake_tensor(rows: int, cols: int) -> list:
    return [[random.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def fake_latent() -> list:
    return [random.uniform(-1.0, 1.0) for _ in range(LATENT_DIM)]


def main() -> None:
    config = DQARConfig()
    config.gate.min_step = 1
    config.gate.entropy_threshold = 2.0
    config.gate.snr_range = (0.0, 100.0)
    config.gate.min_probability = 0.0
    config.gate.cooldown_steps = 1
    config.scheduler.max_gap = 8
    controller = DQARController(num_layers=NUM_LAYERS, config=config)
    prompt = "city skyline at dusk with neon reflections"

    for step in range(TOTAL_STEPS):
        snr = math.exp(-step * 0.2) * 10.0
        controller.begin_step(
            step,
            total_steps=TOTAL_STEPS,
            snr=snr,
            prompt_length=len(prompt.split()),
        )
        reused_layers = 0
        for layer in range(NUM_LAYERS):
            decision = controller.should_reuse(layer, branch="cond")
            if decision.use_cache and decision.entry:
                controller.reuse(decision.entry)
                reused_layers += 1
                continue

            attn = fake_attention()
            keys = fake_tensor(TOKENS, LATENT_DIM)
            values = fake_tensor(TOKENS, LATENT_DIM)
            clean_latent = fake_latent()
            noisy_latent = [c + random.uniform(-0.1, 0.1) for c in clean_latent]
            controller.commit(
                layer,
                "cond",
                attn_map=attn,
                keys=keys,
                values=values,
                clean_latent=clean_latent,
                noisy_latent=noisy_latent,
            )
        print(f"Step {step:02d}: reused {reused_layers}/{NUM_LAYERS} layers")


if __name__ == "__main__":
    main()
