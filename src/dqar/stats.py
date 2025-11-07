from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Union

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
        tensor = value.type(torch.float32)
        return float(torch.sum(tensor * tensor).item())

    return sum(float(v) * float(v) for v in _flatten(value))


def compute_attention_entropy(attn_map: TensorLike, eps: float = 1e-6) -> float:
    """Shannon entropy (nats) averaged across heads."""

    if _is_torch_tensor(attn_map):
        tensor = attn_map.clamp(min=eps)
        entropy = -(tensor * tensor.log()).sum(dim=-1).mean()
        return float(entropy.item())

    # Python fallback: attn_map -> [heads][query][key]
    total = 0.0
    count = 0
    for head in attn_map:  # type: ignore[assignment]
        for row in head:  # type: ignore[index]
            row_total = 0.0
            for prob in row:
                p = max(float(prob), eps)
                row_total += -p * math.log(p)
            total += row_total
            count += 1
    return total / max(count, 1)


def compute_snr(clean_latent: TensorLike, noisy_latent: TensorLike, eps: float = 1e-6) -> float:
    """Compute ||x0||^2 / (||xt - x0||^2 + eps)."""

    numerator = _l2_norm_sq(clean_latent)

    if _is_torch_tensor(clean_latent) and _is_torch_tensor(noisy_latent):
        residual = (noisy_latent - clean_latent).type(torch.float32)
        denom = float(torch.sum(residual * residual).item())
    else:
        if _is_torch_tensor(noisy_latent):
            noisy = noisy_latent.type(torch.float32).tolist()
        else:
            noisy = noisy_latent
        if _is_torch_tensor(clean_latent):
            clean = clean_latent.type(torch.float32).tolist()
        else:
            clean = clean_latent
        residual_sq = [
            (float(a) - float(b)) ** 2 for a, b in zip(_flatten(noisy), _flatten(clean))
        ]
        denom = sum(residual_sq)

    return numerator / max(denom, eps)


def l2_norm(value: TensorLike) -> float:
    return math.sqrt(_l2_norm_sq(value))


@dataclass(slots=True)
class StepMetrics:
    entropy: float
    snr: float
    latent_norm: float
    prompt_length: int
    step_index: int
    branch: str = "cond"


__all__ = ["compute_attention_entropy", "compute_snr", "l2_norm", "StepMetrics"]
