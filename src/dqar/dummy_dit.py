from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

from .controller import DQARController


def _make_matrix(rows: int, cols: int) -> List[List[float]]:
    """Create a random matrix."""
    return [[random.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def _make_attention(heads: int, tokens: int) -> List[List[List[float]]]:
    """Create a random attention map with valid softmax distribution."""
    attn: List[List[List[float]]] = []
    for _ in range(heads):
        head_rows: List[List[float]] = []
        for _ in range(tokens):
            row = [random.random() for _ in range(tokens)]
            total = sum(row) or 1.0
            head_rows.append([value / total for value in row])
        attn.append(head_rows)
    return attn


def _noisy(latent: List[float], noise_scale: float = 0.05) -> List[float]:
    """Add noise to latent."""
    return [value + random.uniform(-noise_scale, noise_scale) for value in latent]


@dataclass(slots=True)
class DummyDiffusionOutput:
    """Output from dummy diffusion generation."""
    latents: List[float]
    reuse_events: int
    recompute_events: int


class DummyDiffusionTransformer:
    """
    Small stand-in for a DiT that exercises the DQAR controller hooks.

    This class simulates a diffusion transformer for testing purposes,
    generating random attention maps and latents to exercise all DQAR
    controller code paths.
    """

    def __init__(
        self,
        num_layers: int = 6,
        tokens: int = 8,
        latent_dim: int = 16,
        heads: int = 2,
    ):
        """
        Initialize the dummy transformer.

        Args:
            num_layers: Number of transformer layers to simulate.
            tokens: Number of tokens per sequence.
            latent_dim: Dimensionality of the latent space.
            heads: Number of attention heads per layer.
        """
        self.num_layers = num_layers
        self.tokens = tokens
        self.latent_dim = latent_dim
        self.heads = heads

    def generate(
        self,
        *,
        controller: Optional[DQARController] = None,
        steps: int = 12,
        prompt_length: int = 12,
        seed: int = 7,
        branch: str = "cond",
    ) -> DummyDiffusionOutput:
        """
        Run dummy diffusion generation with DQAR controller.

        Args:
            controller: DQAR controller instance (created if None).
            steps: Number of diffusion steps.
            prompt_length: Simulated prompt length for adaptive thresholds.
            seed: Random seed for reproducibility.
            branch: CFG branch to use ("cond" or "uncond").

        Returns:
            DummyDiffusionOutput with final latents and reuse statistics.
        """
        random.seed(seed)
        controller = controller or DQARController(num_layers=self.num_layers)
        reuse_events = 0
        recompute_events = 0
        latents = [random.uniform(-1.0, 1.0) for _ in range(self.latent_dim)]

        for step in range(steps):
            # Begin step - SNR is now estimated from timestep automatically
            latent_norm = math.sqrt(sum(value * value for value in latents))
            controller.begin_step(
                step_index=step,
                total_steps=steps,
                prompt_length=prompt_length,
                latent_norm=latent_norm,
            )

            for layer_idx in range(self.num_layers):
                decision = controller.should_reuse(layer_idx, branch=branch)
                if decision.use_cache and decision.entry is not None:
                    _, _, residual = controller.reuse(decision.entry)
                    if residual is not None:
                        latents = [
                            value + float(delta)
                            for value, delta in zip(latents, residual)
                        ]
                    reuse_events += 1
                    continue

                # Compute fresh attention
                attn_map = _make_attention(self.heads, self.tokens)
                keys = _make_matrix(self.tokens, self.latent_dim)
                values = _make_matrix(self.tokens, self.latent_dim)
                residual = [random.uniform(-0.05, 0.05) for _ in range(self.latent_dim)]

                # Commit to cache using new API
                controller.commit(
                    layer_id=layer_idx,
                    branch=branch,
                    attn_map=attn_map,
                    keys=keys,
                    values=values,
                    latent=latents,  # Current latent for norm computation
                    residual=residual,
                    prompt_length=prompt_length,
                )
                recompute_events += 1

                # Update latent
                latents = [value + delta for value, delta in zip(latents, residual)]

        return DummyDiffusionOutput(
            latents=latents,
            reuse_events=reuse_events,
            recompute_events=recompute_events,
        )


__all__ = ["DummyDiffusionTransformer", "DummyDiffusionOutput"]
