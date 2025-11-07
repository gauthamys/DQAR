from __future__ import annotations

import itertools
import math
import random
from array import array
from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableSequence, Optional, Sequence, Tuple, Union

from .config import QuantizationConfig

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


TensorLike = Union["torch.Tensor", Sequence[Sequence[float]], Sequence[float]]


def _is_torch_tensor(value: object) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _flatten(value: Sequence) -> Tuple[List[float], Tuple[int, ...]]:
    flat: List[float] = []
    shape: List[int] = []

    def _walk(node: Sequence, level: int = 0) -> None:
        if isinstance(node, (list, tuple)):
            if len(shape) <= level:
                shape.append(len(node))
            for child in node:
                _walk(child, level + 1)
        else:
            flat.append(float(node))

    _walk(value, 0)
    return flat, tuple(shape)


def _reshape(values: Sequence[float], shape: Tuple[int, ...]) -> List:
    if not shape:
        return list(values)

    def _build(idx: int, dims: Tuple[int, ...]) -> Tuple[List, int]:
        if not dims:
            return float(values[idx]), idx + 1
        current_dim = dims[0]
        block: List = []
        cursor = idx
        for _ in range(current_dim):
            child, cursor = _build(cursor, dims[1:])
            block.append(child)
        return block, cursor

    data, _ = _build(0, shape)
    return data  # type: ignore[return-value]


def _max_abs(value: TensorLike, axis: Optional[int] = None) -> TensorLike:
    if _is_torch_tensor(value):
        tensor = value.abs()
        if axis is None:
            return tensor.max()
        return tensor.amax(dim=axis)

    flat, _ = _flatten(value) if isinstance(value, (list, tuple)) else ([float(value)], ())
    return max(abs(v) for v in flat)


def _max_per_channel(value: TensorLike) -> List[float]:
    if _is_torch_tensor(value):
        tensor = value.abs()
        if tensor.ndim == 1:
            return tensor.tolist()
        last_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, last_dim)
        return tensor.max(dim=0).values.tolist()

    rows = list(value)  # type: ignore[arg-type]
    maxima: Optional[List[float]] = None
    for row in rows:
        flattened = [abs(float(v)) for v in row]  # type: ignore[index]
        if maxima is None:
            maxima = flattened
        else:
            maxima = [max(a, b) for a, b in zip(maxima, flattened)]
    return maxima or []


@dataclass(slots=True)
class QuantizedTensor:
    data: Union["torch.Tensor", array, Sequence[float]]
    scale: Union["torch.Tensor", float, Sequence[float], None]
    shape: Tuple[int, ...]
    axis: Optional[int] = None
    zero_point: float = 0.0
    dtype: str = "int8"

    def is_quantized(self) -> bool:
        return isinstance(self.data, (array,)) or (
            _is_torch_tensor(self.data) and str(self.data.dtype).startswith("torch.int")
        )

    def dequantize(self) -> TensorLike:
        return dequantize_tensor(self)


@dataclass(slots=True)
class SalienceProfile:
    act_scale: List[float]
    weight_scale: List[float]
    channel_axis: int = -1


class SalienceCalibrator:
    """Collects activation/weight stats and produces balancing factors."""

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self._act: Dict[str, List[List[float]]] = {}
        self._wgt: Dict[str, List[List[float]]] = {}

    def observe(self, layer: str, activations: TensorLike, weights: TensorLike) -> None:
        self._act.setdefault(layer, []).append(_max_per_channel(activations))
        self._wgt.setdefault(layer, []).append(_max_per_channel(weights))

    def build(self) -> Dict[str, SalienceProfile]:
        profiles: Dict[str, SalienceProfile] = {}
        for layer in set(self._act) | set(self._wgt):
            act_samples = self._act.get(layer, [])
            wgt_samples = self._wgt.get(layer, [])
            act = self._mean_vector(act_samples)
            wgt = self._mean_vector(wgt_samples)
            profile = SalienceProfile(
                act_scale=self._balance(act),
                weight_scale=self._balance(wgt),
                channel_axis=-1,
            )
            profiles[layer] = profile
        return profiles

    @staticmethod
    def _mean_vector(samples: List[List[float]]) -> List[float]:
        if not samples:
            return []
        length = max(len(s) for s in samples)
        total = [0.0] * length
        for sample in samples:
            for idx, value in enumerate(sample):
                total[idx] += value
        return [v / len(samples) for v in total]

    def _balance(self, values: List[float]) -> List[float]:
        if not values:
            return []
        temperature = self.config.salience_temperature
        max_val = max(values) + self.config.eps
        return [
            math.pow((max_val - v + self.config.eps) / max_val, temperature) for v in values
        ]


class Quantizer:
    """Wrapper that exposes quantize/dequantize helpers with CSB-aware hooks."""

    def __init__(
        self,
        config: QuantizationConfig,
        profiles: Optional[Dict[str, SalienceProfile]] = None,
    ):
        self.config = config
        self.profiles = profiles or {}

    def quantize(
        self,
        tensor: TensorLike,
        layer: Optional[str] = None,
        kind: str = "value",
        axis: Optional[int] = -1,
    ) -> QuantizedTensor:
        if kind == "key" and self.config.keep_k_precision:
            # No quantization requested for keys.
            shape = tuple(tensor.shape) if _is_torch_tensor(tensor) else _flatten(tensor)[1]
            return QuantizedTensor(data=tensor, scale=None, shape=shape, dtype="fp16")

        if layer and layer in self.profiles:
            tensor = self._apply_profile(tensor, self.profiles[layer], kind)

        if _is_torch_tensor(tensor):
            return self._quantize_torch(tensor, axis=axis)

        if self.config.per_channel:
            raise ValueError("Per-channel quantization requires torch to be installed.")

        return self._quantize_python(tensor)

    def dequantize(self, qtensor: QuantizedTensor) -> TensorLike:
        return dequantize_tensor(qtensor)

    def _apply_profile(
        self, tensor: TensorLike, profile: SalienceProfile, kind: str
    ) -> TensorLike:
        if not _is_torch_tensor(tensor):
            return tensor  # python fallback keeps tensor untouched

        scales = profile.act_scale if kind == "value" else profile.weight_scale
        if not scales:
            return tensor
        device_tensor = tensor
        scale_tensor = torch.tensor(scales, device=device_tensor.device, dtype=device_tensor.dtype)
        while scale_tensor.ndim < device_tensor.ndim:
            scale_tensor = scale_tensor.unsqueeze(0)
        return device_tensor * scale_tensor

    def _quantize_torch(
        self, tensor: "torch.Tensor", axis: Optional[int] = None
    ) -> QuantizedTensor:
        bits = max(1, min(self.config.num_bits, 16))
        if self.config.symmetric:
            qmax = float((1 << (bits - 1)) - 1)
            scale = tensor.abs().amax(dim=axis, keepdim=True) / max(qmax, 1.0)
            scale = torch.clamp(scale, min=self.config.eps)
            q = torch.clamp(torch.round(tensor / scale), -qmax, qmax).to(torch.int8)
            shape = tuple(tensor.shape)
            return QuantizedTensor(data=q, scale=scale, shape=shape, axis=axis)

        raise NotImplementedError("Asymmetric quantization is not implemented.")

    def _quantize_python(self, tensor: TensorLike) -> QuantizedTensor:
        flat, shape = _flatten(tensor)  # type: ignore[arg-type]
        if not flat:
            return QuantizedTensor(data=array("b"), scale=1.0, shape=shape)

        bits = max(1, min(self.config.num_bits, 8))
        qmax = float((1 << (bits - 1)) - 1)
        max_val = max(abs(v) for v in flat)
        scale = max(max_val / qmax, self.config.eps)

        ints = array(
            "b",
            (
                int(max(-qmax, min(qmax, round(value / scale))))
                for value in flat
            ),
        )
        return QuantizedTensor(data=ints, scale=scale, shape=shape)


def dequantize_tensor(qtensor: QuantizedTensor) -> TensorLike:
    if _is_torch_tensor(qtensor.data):
        if qtensor.scale is None:
            return qtensor.data
        if not _is_torch_tensor(qtensor.scale):
            raise ValueError("Torch tensors must carry tensor scales.")
        scale = qtensor.scale
        data = qtensor.data.to(scale.dtype) * scale
        return data.reshape(qtensor.shape)

    if isinstance(qtensor.data, array):
        floats = [float(v) * float(qtensor.scale or 1.0) for v in qtensor.data]
        return _reshape(floats, qtensor.shape)

    return qtensor.data


__all__ = [
    "QuantizedTensor",
    "Quantizer",
    "SalienceCalibrator",
    "SalienceProfile",
    "dequantize_tensor",
]
