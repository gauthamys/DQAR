from .config import (
    CacheConfig,
    DQARConfig,
    PolicyConfig,
    QuantizationConfig,
    ReuseGateConfig,
    SchedulerConfig,
)
from .controller import DQARController, ReuseDecision
from .policy import DQARPolicy, PolicyFeatures
from .quantization import Quantizer, QuantizedTensor, SalienceCalibrator, SalienceProfile
from .scheduler import LayerScheduler
from .stats import (
    StepMetrics,
    compute_attention_entropy,
    compute_snr,
    estimate_snr_from_timestep,
    estimate_snr_ddim,
    l2_norm,
)

# Optional DiT integration (requires torch/diffusers)
try:
    from .dit_wrapper import patch_dit_pipeline, get_dit_layer_count
except ImportError:
    patch_dit_pipeline = None  # type: ignore
    get_dit_layer_count = None  # type: ignore

__all__ = [
    "CacheConfig",
    "DQARConfig",
    "PolicyConfig",
    "QuantizationConfig",
    "ReuseGateConfig",
    "SchedulerConfig",
    "DQARController",
    "ReuseDecision",
    "DQARPolicy",
    "PolicyFeatures",
    "Quantizer",
    "QuantizedTensor",
    "SalienceCalibrator",
    "SalienceProfile",
    "LayerScheduler",
    "StepMetrics",
    "compute_attention_entropy",
    "compute_snr",
    "estimate_snr_from_timestep",
    "estimate_snr_ddim",
    "l2_norm",
    "patch_dit_pipeline",
    "get_dit_layer_count",
]
