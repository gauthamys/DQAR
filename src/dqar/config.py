from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(slots=True)
class QuantizationConfig:
    """Configuration controlling how cached KV tensors are quantized."""

    num_bits: int = 8
    per_channel: bool = False
    symmetric: bool = True
    keep_k_precision: bool = False
    calibration_samples: int = 32
    salience_temperature: float = 0.2
    spearman_smoothing: float = 0.9
    eps: float = 1e-6


@dataclass(slots=True)
class ReuseGateConfig:
    """Heuristics and thresholds for entropy/SNR driven reuse."""

    entropy_threshold: float = 1.3
    prompt_length_scale: float = 0.012
    snr_range: Tuple[float, float] = (0.25, 64.0)
    min_step: int = 4
    cooldown_steps: int = 2
    min_probability: float = 0.55
    eps: float = 1e-6

    def adaptive_entropy_threshold(self, prompt_length: int) -> float:
        # Longer prompts typically push attention to be sharper; lower the bar slightly.
        return max(
            self.eps,
            self.entropy_threshold - self.prompt_length_scale * float(max(prompt_length - 16, 0)),
        )


@dataclass(slots=True)
class SchedulerConfig:
    """Controls which layers are eligible for reuse across the diffusion timeline."""

    early_ratio: float = 0.2
    late_ratio: float = 0.6
    shallow_layers: int = 4
    deep_layers: int = 12
    max_reuse_per_block: int = 3
    max_gap: int = 4


@dataclass(slots=True)
class PolicyConfig:
    """Lightweight MLP policy hyper-parameters."""

    hidden_dim: int = 32
    num_hidden_layers: int = 2
    dropout: float = 0.0
    seed: int = 7


@dataclass(slots=True)
class CacheConfig:
    """Settings specific to KV cache behaviour."""

    allow_cross_cfg: bool = True
    window_length: int = 2
    track_residuals: bool = True


@dataclass(slots=True)
class DQARConfig:
    """Top-level configuration wrapper used by the controller."""

    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    gate: ReuseGateConfig = field(default_factory=ReuseGateConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    total_steps: Optional[int] = None


__all__ = [
    "QuantizationConfig",
    "ReuseGateConfig",
    "SchedulerConfig",
    "PolicyConfig",
    "CacheConfig",
    "DQARConfig",
]
