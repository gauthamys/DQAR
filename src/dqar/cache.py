from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .config import CacheConfig
from .quantization import Quantizer, QuantizedTensor
from .stats import StepMetrics


@dataclass(slots=True)
class CacheEntry:
    layer_id: int
    branch: str
    step: int
    metrics: StepMetrics
    k: Optional[QuantizedTensor]
    v: QuantizedTensor
    residual: Optional[QuantizedTensor] = None
    reuse_count: int = 0


class QuantizedKVCache:
    def __init__(self, config: CacheConfig, quantizer: Quantizer):
        self.config = config
        self.quantizer = quantizer
        self._entries: Dict[Tuple[int, str], CacheEntry] = {}

    def store(
        self,
        layer_id: int,
        branch: str,
        step: int,
        metrics: StepMetrics,
        k,
        v,
        residual=None,
    ) -> CacheEntry:
        layer_name = f"layer_{layer_id}"
        qk = self.quantizer.quantize(k, layer=layer_name, kind="key") if k is not None else None
        qv = self.quantizer.quantize(v, layer=layer_name, kind="value")
        qres = self.quantizer.quantize(residual, layer=layer_name, kind="value") if residual is not None else None
        entry = CacheEntry(
            layer_id=layer_id,
            branch=branch,
            step=step,
            metrics=metrics,
            k=qk,
            v=qv,
            residual=qres,
        )
        self._entries[(layer_id, branch)] = entry
        return entry

    def lookup(self, layer_id: int, branch: str) -> Optional[CacheEntry]:
        entry = self._entries.get((layer_id, branch))
        if entry:
            return entry
        if self.config.allow_cross_cfg:
            alt_branch = "uncond" if branch == "cond" else "cond"
            return self._entries.get((layer_id, alt_branch))
        return None

    def increment_reuse(self, entry: CacheEntry) -> None:
        entry.reuse_count += 1

    def clear_layer(self, layer_id: int) -> None:
        to_delete = [key for key in self._entries if key[0] == layer_id]
        for key in to_delete:
            del self._entries[key]

    def clear(self) -> None:
        self._entries.clear()


__all__ = ["QuantizedKVCache", "CacheEntry"]
