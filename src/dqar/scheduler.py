from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from .config import SchedulerConfig


@dataclass(slots=True)
class LayerWindow:
    allowed_layers: Set[int]


class LayerScheduler:
    def __init__(self, num_layers: int, config: SchedulerConfig):
        self.num_layers = num_layers
        self.config = config

    def eligible_layers(self, step_index: int, total_steps: int) -> LayerWindow:
        progress = 0.0
        if total_steps > 1:
            progress = step_index / float(total_steps - 1)

        if progress < self.config.early_ratio:
            depth = min(self.config.shallow_layers, self.num_layers)
        elif progress < self.config.late_ratio:
            depth = min(self.config.deep_layers, self.num_layers)
        else:
            depth = self.num_layers

        return LayerWindow(allowed_layers=set(range(depth)))

    def can_reuse(self, layer_idx: int, window: LayerWindow) -> bool:
        return layer_idx in window.allowed_layers


__all__ = ["LayerScheduler", "LayerWindow"]
