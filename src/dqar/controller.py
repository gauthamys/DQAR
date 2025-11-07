from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .cache import CacheEntry, QuantizedKVCache
from .config import DQARConfig
from .policy import DQARPolicy, PolicyFeatures
from .quantization import Quantizer
from .scheduler import LayerScheduler
from .stats import StepMetrics, compute_attention_entropy, compute_snr, l2_norm


@dataclass(slots=True)
class ReuseDecision:
    use_cache: bool
    probability: float
    reason: str
    entry: Optional[CacheEntry] = None


class DQARController:
    """Coordinates entropy/SNR gating, quantized caching, and policy scores."""

    def __init__(
        self,
        num_layers: int,
        config: Optional[DQARConfig] = None,
        policy: Optional[DQARPolicy] = None,
    ):
        self.config = config or DQARConfig()
        self.policy = policy or DQARPolicy(self.config.policy)
        self.quantizer = Quantizer(self.config.quantization)
        self.cache = QuantizedKVCache(self.config.cache, self.quantizer)
        self.scheduler = LayerScheduler(num_layers=num_layers, config=self.config.scheduler)
        self.num_layers = num_layers
        self.current_step = 0
        self.total_steps = self.config.total_steps or 1
        self.prompt_length = 16
        self.pending_snr: Optional[float] = None
        self.pending_latent_norm: Optional[float] = None
        self._layer_window = self.scheduler.eligible_layers(0, self.total_steps)

    def begin_step(
        self,
        step_index: int,
        total_steps: Optional[int] = None,
        *,
        snr: Optional[float] = None,
        prompt_length: Optional[int] = None,
        latent_norm: Optional[float] = None,
    ) -> None:
        self.current_step = step_index
        if total_steps is not None:
            self.total_steps = total_steps
        self._layer_window = self.scheduler.eligible_layers(step_index, self.total_steps)
        if prompt_length is not None:
            self.prompt_length = prompt_length
        self.pending_snr = snr
        self.pending_latent_norm = latent_norm

    def should_reuse(self, layer_id: int, branch: str = "cond") -> ReuseDecision:
        gate_cfg = self.config.gate
        sched_cfg = self.config.scheduler

        if self.current_step < gate_cfg.min_step:
            return ReuseDecision(False, 0.0, "warmup")

        if not self.scheduler.can_reuse(layer_id, self._layer_window):
            return ReuseDecision(False, 0.0, "scheduler")

        entry = self.cache.lookup(layer_id, branch)
        if not entry:
            return ReuseDecision(False, 0.0, "miss")

        step_gap = self.current_step - entry.step
        if step_gap <= 0:
            return ReuseDecision(False, 0.0, "same-step")
        if step_gap < gate_cfg.cooldown_steps:
            return ReuseDecision(False, 0.0, "cooldown")
        if step_gap > sched_cfg.max_gap:
            return ReuseDecision(False, 0.0, "stale")
        if entry.reuse_count >= sched_cfg.max_reuse_per_block:
            return ReuseDecision(False, 0.0, "budget")

        snr = self.pending_snr or entry.metrics.snr
        snr_low, snr_high = gate_cfg.snr_range
        if snr < snr_low or snr > snr_high:
            return ReuseDecision(False, 0.0, "snr")

        entropy_limit = gate_cfg.adaptive_entropy_threshold(self.prompt_length)
        if entry.metrics.entropy > entropy_limit:
            return ReuseDecision(False, 0.0, "entropy")

        latent_norm = self.pending_latent_norm or entry.metrics.latent_norm
        features = PolicyFeatures(
            entropy=entry.metrics.entropy,
            snr=snr,
            latent_norm=latent_norm,
            step_index=self.current_step,
            total_steps=self.total_steps,
            prompt_length=self.prompt_length,
        )
        probability = self.policy.predict_proba(features)
        if probability < gate_cfg.min_probability:
            return ReuseDecision(False, probability, "policy")

        return ReuseDecision(True, probability, "reuse", entry=entry)

    def reuse(self, entry: CacheEntry) -> Tuple[Optional[object], object, Optional[object]]:
        self.cache.increment_reuse(entry)
        k = self.quantizer.dequantize(entry.k) if entry.k is not None else None
        v = self.quantizer.dequantize(entry.v)
        residual = self.quantizer.dequantize(entry.residual) if entry.residual else None
        return k, v, residual

    def commit(
        self,
        layer_id: int,
        branch: str,
        *,
        attn_map,
        keys,
        values,
        clean_latent,
        noisy_latent,
        residual=None,
        snr: Optional[float] = None,
        prompt_length: Optional[int] = None,
    ) -> CacheEntry:
        entropy = compute_attention_entropy(attn_map, self.config.gate.eps)
        latent_norm = l2_norm(clean_latent)
        snr_value = snr if snr is not None else compute_snr(clean_latent, noisy_latent)
        metrics = StepMetrics(
            entropy=entropy,
            snr=snr_value,
            latent_norm=latent_norm,
            prompt_length=prompt_length or self.prompt_length,
            step_index=self.current_step,
            branch=branch,
        )
        return self.cache.store(
            layer_id=layer_id,
            branch=branch,
            step=self.current_step,
            metrics=metrics,
            k=keys,
            v=values,
            residual=residual,
        )

    def clear(self) -> None:
        self.cache.clear()


__all__ = ["DQARController", "ReuseDecision"]
