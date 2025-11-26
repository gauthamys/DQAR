from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover - executed in bare envs
    torch = None

TensorLike = Union["torch.Tensor", Sequence[Sequence[float]]]


def _is_torch_tensor(value: object) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _flatten(nested: Sequence) -> Iterable[float]:
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield float(item)


def _l2_norm_sq(value: Union[TensorLike, Sequence[float]]) -> float:
    if _is_torch_tensor(value):
        tensor = value.float()
        return float(torch.sum(tensor * tensor).item())

    return sum(float(v) * float(v) for v in _flatten(value))


def compute_attention_entropy(attn_map: TensorLike, eps: float = 1e-6) -> float:
    """
    Shannon entropy (nats) averaged across all heads.

    Formula: H_t = -(1/H) * sum_h sum_{i,j} A^(h)_t(i,j) * log(A^(h)_t(i,j) + eps)

    Where H is the number of attention heads, and we sum over all query-key pairs
    within each head, then average across heads.

    Args:
        attn_map: Attention tensor of shape [num_heads, num_queries, num_keys]
                  or nested list with same structure. Values should be probabilities
                  (softmax outputs) summing to 1 along the key dimension.
        eps: Small constant for numerical stability.

    Returns:
        Average entropy across all heads (scalar).
    """
    if _is_torch_tensor(attn_map):
        # attn_map shape: [num_heads, num_queries, num_keys]
        tensor = attn_map.float().clamp(min=eps)
        # Compute -p * log(p) for each element
        entropy_elements = -tensor * tensor.log()
        # Sum over keys (dim=-1) to get per-query entropy, then mean over queries and heads
        # This gives: (1/H) * (1/Q) * sum_h sum_i sum_j (-p * log(p))
        per_query_entropy = entropy_elements.sum(dim=-1)  # [num_heads, num_queries]
        # Average over all queries and heads
        mean_entropy = per_query_entropy.mean()
        return float(mean_entropy.item())

    # Python fallback: attn_map -> [heads][query][key]
    total_entropy = 0.0
    num_heads = 0
    for head in attn_map:  # type: ignore[union-attr]
        head_entropy = 0.0
        num_queries = 0
        for row in head:  # type: ignore[union-attr]
            query_entropy = 0.0
            for prob in row:
                p = max(float(prob), eps)
                query_entropy += -p * math.log(p)
            head_entropy += query_entropy
            num_queries += 1
        # Average entropy for this head (across queries)
        if num_queries > 0:
            total_entropy += head_entropy / num_queries
        num_heads += 1

    # Average across heads
    return total_entropy / max(num_heads, 1)


def compute_snr(
    clean_latent: TensorLike,
    noisy_latent: TensorLike,
    eps: float = 1e-6
) -> float:
    """
    Compute SNR = ||x0||^2 / (||xt - x0||^2 + eps).

    NOTE: This requires access to the clean latent x0, which is only available
    during training or when using an estimator. For inference-time SNR,
    use `estimate_snr_from_timestep` or `estimate_snr_ddim` instead.

    Args:
        clean_latent: The clean/target latent x0.
        noisy_latent: The noisy latent xt at current timestep.
        eps: Small constant for numerical stability.

    Returns:
        Signal-to-noise ratio (scalar).
    """
    numerator = _l2_norm_sq(clean_latent)

    if _is_torch_tensor(clean_latent) and _is_torch_tensor(noisy_latent):
        residual = (noisy_latent - clean_latent).float()
        denom = float(torch.sum(residual * residual).item())
    else:
        if _is_torch_tensor(noisy_latent):
            noisy = noisy_latent.float().tolist()
        else:
            noisy = noisy_latent
        if _is_torch_tensor(clean_latent):
            clean = clean_latent.float().tolist()
        else:
            clean = clean_latent
        residual_sq = [
            (float(a) - float(b)) ** 2 for a, b in zip(_flatten(noisy), _flatten(clean))
        ]
        denom = sum(residual_sq)

    return numerator / max(denom, eps)


def estimate_snr_from_timestep(
    timestep: int,
    total_timesteps: int,
    schedule: str = "cosine",
    snr_max: float = 100.0,
    snr_min: float = 0.01,
) -> float:
    """
    Estimate SNR from the diffusion timestep using the noise schedule.

    This is the RECOMMENDED method for inference-time SNR estimation since
    it doesn't require access to the clean latent x0.

    For diffusion models, SNR at timestep t is defined as:
        SNR(t) = alpha_bar(t)^2 / (1 - alpha_bar(t))

    where alpha_bar(t) is the cumulative product of (1 - beta_t).

    Args:
        timestep: Current timestep (0 = clean, total_timesteps = pure noise).
        total_timesteps: Total number of diffusion timesteps.
        schedule: Noise schedule type ("cosine", "linear", "sqrt").
        snr_max: Maximum SNR (at t=0).
        snr_min: Minimum SNR (at t=T).

    Returns:
        Estimated SNR at the given timestep.
    """
    if total_timesteps <= 1:
        return snr_max

    # Normalized time in [0, 1] where 0 = clean, 1 = noisy
    t_normalized = timestep / max(total_timesteps - 1, 1)

    if schedule == "cosine":
        # Cosine schedule: SNR decreases smoothly following cosine curve
        # alpha_bar(t) = cos(pi * t / 2)^2 approximately
        alpha_bar = math.cos(t_normalized * math.pi / 2) ** 2
        alpha_bar = max(alpha_bar, 1e-8)  # Prevent division by zero
        snr = alpha_bar / (1 - alpha_bar + 1e-8)
    elif schedule == "linear":
        # Linear schedule: beta increases linearly
        beta_start, beta_end = 0.0001, 0.02
        beta = beta_start + t_normalized * (beta_end - beta_start)
        # Approximate alpha_bar for linear schedule
        alpha_bar = (1 - beta) ** (timestep + 1)
        alpha_bar = max(alpha_bar, 1e-8)
        snr = alpha_bar / (1 - alpha_bar + 1e-8)
    elif schedule == "sqrt":
        # Square root schedule
        alpha_bar = 1 - math.sqrt(t_normalized + 1e-8)
        alpha_bar = max(alpha_bar, 1e-8)
        snr = alpha_bar / (1 - alpha_bar + 1e-8)
    else:
        # Fallback: exponential decay
        snr = snr_max * math.exp(-5.0 * t_normalized)

    # Clamp to valid range
    return max(min(snr, snr_max), snr_min)


def estimate_snr_ddim(
    noisy_latent: TensorLike,
    predicted_noise: TensorLike,
    alpha_bar: float,
    eps: float = 1e-6,
) -> float:
    """
    Estimate SNR using DDIM's predicted x0.

    DDIM predicts x0 as: x0_hat = (xt - sqrt(1 - alpha_bar) * epsilon) / sqrt(alpha_bar)

    Then SNR can be estimated as: ||x0_hat||^2 / ||xt - x0_hat||^2

    This is useful when you have access to the model's noise prediction but
    not the actual clean latent.

    Args:
        noisy_latent: The noisy latent xt at current timestep.
        predicted_noise: The model's predicted noise epsilon.
        alpha_bar: The cumulative alpha value at current timestep.
        eps: Small constant for numerical stability.

    Returns:
        Estimated SNR based on DDIM prediction.
    """
    sqrt_alpha_bar = math.sqrt(max(alpha_bar, eps))
    sqrt_one_minus_alpha_bar = math.sqrt(max(1.0 - alpha_bar, eps))

    if _is_torch_tensor(noisy_latent) and _is_torch_tensor(predicted_noise):
        xt = noisy_latent.float()
        noise = predicted_noise.float()
        # DDIM prediction: x0_hat = (xt - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar)
        x0_hat = (xt - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar

        numerator = float(torch.sum(x0_hat * x0_hat).item())
        residual = xt - x0_hat
        denom = float(torch.sum(residual * residual).item())
    else:
        # Python fallback
        xt_flat = list(_flatten(noisy_latent))
        noise_flat = list(_flatten(predicted_noise))

        x0_hat = [
            (x - sqrt_one_minus_alpha_bar * n) / sqrt_alpha_bar
            for x, n in zip(xt_flat, noise_flat)
        ]

        numerator = sum(x * x for x in x0_hat)
        residual_sq = [(x - x0) ** 2 for x, x0 in zip(xt_flat, x0_hat)]
        denom = sum(residual_sq)

    return numerator / max(denom, eps)


def l2_norm(value: TensorLike) -> float:
    """Compute L2 norm of a tensor or nested list."""
    return math.sqrt(_l2_norm_sq(value))


@dataclass(slots=True)
class StepMetrics:
    """Metrics collected at each diffusion step for reuse decisions."""
    entropy: float
    snr: float
    latent_norm: float
    prompt_length: int
    step_index: int
    branch: str = "cond"


__all__ = [
    "compute_attention_entropy",
    "compute_snr",
    "estimate_snr_from_timestep",
    "estimate_snr_ddim",
    "l2_norm",
    "StepMetrics",
]
