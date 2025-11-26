from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .cache import CacheEntry, QuantizedKVCache
from .config import DQARConfig
from .policy import DQARPolicy, PolicyFeatures
from .quantization import Quantizer
from .scheduler import LayerScheduler
from .stats import (
    StepMetrics,
    compute_attention_entropy,
    compute_snr,
    estimate_snr_from_timestep,
    estimate_snr_ddim,
    l2_norm,
)


@dataclass(slots=True)
class ReuseDecision:
    """Result of a reuse decision with explanation."""
    use_cache: bool
    probability: float
    reason: str
    entry: Optional[CacheEntry] = None


class DQARController:
    """
    Coordinates entropy/SNR gating, quantized caching, and policy scores.

    The controller manages reuse decisions across diffusion steps by:
    1. Computing/estimating SNR from timestep (no clean latent required)
    2. Checking entropy thresholds on cached attention
    3. Consulting the learned policy network
    4. Managing quantized KV cache storage and retrieval
    """

    def __init__(
        self,
        num_layers: int,
        config: Optional[DQARConfig] = None,
        policy: Optional[DQARPolicy] = None,
        noise_schedule: str = "cosine",
    ):
        """
        Initialize the DQAR controller.

        Args:
            num_layers: Number of transformer layers in the model.
            config: Configuration for all DQAR components.
            policy: Pre-trained policy network (or None for default).
            noise_schedule: Type of noise schedule for SNR estimation
                           ("cosine", "linear", "sqrt").
        """
        self.config = config or DQARConfig()
        self.policy = policy or DQARPolicy(self.config.policy)
        self.quantizer = Quantizer(self.config.quantization)
        self.cache = QuantizedKVCache(self.config.cache, self.quantizer)
        self.scheduler = LayerScheduler(num_layers=num_layers, config=self.config.scheduler)
        self.num_layers = num_layers
        self.noise_schedule = noise_schedule

        # Step state
        self.current_step = 0
        self.total_steps = self.config.total_steps or 1
        self.prompt_length = 16

        # Pending values for current step (set via begin_step)
        self.pending_snr: Optional[float] = None
        self.pending_latent_norm: Optional[float] = None

        # Layer eligibility window
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
        """
        Begin a new diffusion step.

        Call this at the start of each sampling step to update controller state.

        Args:
            step_index: Current timestep index (0 = start of diffusion).
            total_steps: Total number of sampling steps.
            snr: Optional SNR value. If not provided, estimated from timestep.
            prompt_length: Length of the text prompt (for adaptive thresholds).
            latent_norm: L2 norm of current latent (optional).
        """
        self.current_step = step_index
        if total_steps is not None:
            self.total_steps = total_steps

        self._layer_window = self.scheduler.eligible_layers(step_index, self.total_steps)

        if prompt_length is not None:
            self.prompt_length = prompt_length

        # Use provided SNR or estimate from timestep
        if snr is not None:
            self.pending_snr = snr
        else:
            # Estimate SNR from timestep using noise schedule
            self.pending_snr = estimate_snr_from_timestep(
                timestep=step_index,
                total_timesteps=self.total_steps,
                schedule=self.noise_schedule,
            )

        self.pending_latent_norm = latent_norm

    def get_estimated_snr(self) -> float:
        """Get the current step's SNR (estimated or provided)."""
        if self.pending_snr is not None:
            return self.pending_snr
        return estimate_snr_from_timestep(
            timestep=self.current_step,
            total_timesteps=self.total_steps,
            schedule=self.noise_schedule,
        )

    def should_reuse(self, layer_id: int, branch: str = "cond") -> ReuseDecision:
        """
        Decide whether to reuse cached attention for a layer.

        Goes through a cascade of gates:
        1. Warmup check (skip early steps)
        2. Scheduler check (layer eligibility)
        3. Cache lookup (entry exists?)
        4. Staleness check (step gap within limit?)
        5. Budget check (reuse count within limit?)
        6. SNR gate (SNR in valid range?)
        7. Entropy gate (entropy below threshold?)
        8. Policy gate (learned policy approves?)

        Args:
            layer_id: Index of the transformer layer.
            branch: CFG branch ("cond" or "uncond").

        Returns:
            ReuseDecision with use_cache flag, probability, reason, and cache entry.
        """
        gate_cfg = self.config.gate
        sched_cfg = self.config.scheduler

        # Gate 1: Warmup - skip early steps entirely
        if self.current_step < gate_cfg.min_step:
            return ReuseDecision(False, 0.0, "warmup")

        # Gate 2: Scheduler - check layer eligibility at this timestep
        if not self.scheduler.can_reuse(layer_id, self._layer_window):
            return ReuseDecision(False, 0.0, "scheduler")

        # Gate 3: Cache lookup
        entry = self.cache.lookup(layer_id, branch)
        if not entry:
            return ReuseDecision(False, 0.0, "miss")

        # Gate 4: Same-step and cooldown checks
        step_gap = self.current_step - entry.step
        if step_gap <= 0:
            return ReuseDecision(False, 0.0, "same-step")
        if step_gap < gate_cfg.cooldown_steps:
            return ReuseDecision(False, 0.0, "cooldown")

        # Gate 5: Staleness check
        if step_gap > sched_cfg.max_gap:
            return ReuseDecision(False, 0.0, "stale")

        # Gate 6: Budget check
        if entry.reuse_count >= sched_cfg.max_reuse_per_block:
            return ReuseDecision(False, 0.0, "budget")

        # Gate 7: SNR gate - use estimated SNR
        snr = self.get_estimated_snr()
        snr_low, snr_high = gate_cfg.snr_range
        if snr < snr_low or snr > snr_high:
            return ReuseDecision(False, 0.0, "snr")

        # Gate 8: Entropy gate - check cached entry's entropy
        entropy_limit = gate_cfg.adaptive_entropy_threshold(self.prompt_length)
        if entry.metrics.entropy > entropy_limit:
            return ReuseDecision(False, 0.0, "entropy")

        # Gate 9: Policy gate - learned decision
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

        # All gates passed - approve reuse
        return ReuseDecision(True, probability, "reuse", entry=entry)

    def reuse(self, entry: CacheEntry) -> Tuple[Optional[object], object, Optional[object]]:
        """
        Retrieve and dequantize cached KV tensors.

        Args:
            entry: Cache entry to retrieve from.

        Returns:
            Tuple of (keys, values, residual) - dequantized tensors.
        """
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
        latent,
        residual=None,
        predicted_noise=None,
        alpha_bar: Optional[float] = None,
        prompt_length: Optional[int] = None,
    ) -> CacheEntry:
        """
        Compute metrics and store KV tensors in quantized cache.

        Args:
            layer_id: Index of the transformer layer.
            branch: CFG branch ("cond" or "uncond").
            attn_map: Attention weights tensor [num_heads, num_queries, num_keys].
            keys: Key tensor to cache.
            values: Value tensor to cache.
            latent: Current latent tensor (for norm computation).
            residual: Optional residual tensor to cache.
            predicted_noise: Model's noise prediction (for DDIM SNR estimation).
            alpha_bar: Cumulative alpha at current timestep (for DDIM SNR).
            prompt_length: Override prompt length for this entry.

        Returns:
            The created CacheEntry.
        """
        # Compute attention entropy
        entropy = compute_attention_entropy(attn_map, self.config.gate.eps)

        # Compute latent norm
        latent_norm = l2_norm(latent)

        # Determine SNR: use DDIM estimate if noise prediction available,
        # otherwise use timestep-based estimate
        if predicted_noise is not None and alpha_bar is not None:
            snr_value = estimate_snr_ddim(latent, predicted_noise, alpha_bar)
        else:
            snr_value = self.get_estimated_snr()

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
        """Clear all cached entries."""
        self.cache.clear()


__all__ = ["DQARController", "ReuseDecision"]
