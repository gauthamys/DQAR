from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .config import PolicyConfig


@dataclass(slots=True)
class PolicyFeatures:
    entropy: float
    snr: float
    latent_norm: float
    step_index: int
    total_steps: int
    prompt_length: int


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class _DenseLayer:
    def __init__(self, in_dim: int, out_dim: int, rng_seed: int):
        rng_seed = (rng_seed * 1315423911 + in_dim * 31 + out_dim) & 0xFFFFFFFF
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = [
            [_rand_uniform(rng_seed + i * in_dim + j) for j in range(in_dim)]
            for i in range(out_dim)
        ]
        self.bias = [0.0 for _ in range(out_dim)]


def _rand_uniform(seed: int, limit: float = 0.1) -> float:
    seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
    return (float(seed) / float(0x7FFFFFFF) * 2.0 - 1.0) * limit


class DQARPolicy:
    """Tiny MLP that can be trained without external deps."""

    def __init__(self, config: PolicyConfig):
        self.config = config
        input_dim = 5  # entropy, snr, latent norm, step progress, prompt length
        self.layers: List[_DenseLayer] = []
        current_dim = input_dim
        for _ in range(config.num_hidden_layers):
            layer = _DenseLayer(current_dim, config.hidden_dim, config.seed)
            self.layers.append(layer)
            current_dim = config.hidden_dim
        self.output = _DenseLayer(current_dim, 1, config.seed + 17)

    def _forward(self, inputs: Sequence[float], retain: bool = False):
        activations = [list(inputs)]
        pre_activations: List[List[float]] = []
        current = list(inputs)
        for layer in self.layers:
            z = self._dense_forward(layer, current)
            pre_activations.append(z)
            current = [math.tanh(value) for value in z]
            activations.append(list(current))

        out_z = self._dense_forward(self.output, current)[0]
        if retain:
            return out_z, activations, pre_activations
        return out_z

    @staticmethod
    def _dense_forward(layer: _DenseLayer, inputs: Sequence[float]) -> List[float]:
        outputs = []
        for weights, bias in zip(layer.weights, layer.bias):
            value = sum(w * x for w, x in zip(weights, inputs)) + bias
            outputs.append(value)
        return outputs

    def predict_proba(self, features: PolicyFeatures) -> float:
        inputs = self._normalize(features)
        logit = self._forward(inputs)
        return _sigmoid(logit)

    def should_reuse(self, features: PolicyFeatures, threshold: float = 0.5) -> bool:
        return self.predict_proba(features) >= threshold

    def fit(
        self,
        dataset: Iterable[Tuple[PolicyFeatures, float]],
        epochs: int = 3,
        lr: float = 1e-2,
    ) -> None:
        for _ in range(epochs):
            for features, label in dataset:
                self._train_step(features, label, lr)

    def _train_step(self, features: PolicyFeatures, label: float, lr: float) -> None:
        inputs = self._normalize(features)
        logit, activations, pre_activations = self._forward(inputs, retain=True)
        prob = _sigmoid(logit)
        delta = prob - label

        # Update output layer
        output_weights_snapshot = [row[:] for row in self.output.weights]
        for idx in range(len(self.output.weights[0])):
            grad = delta * activations[-1][idx]
            self.output.weights[0][idx] -= lr * grad
        self.output.bias[0] -= lr * delta

        # Backpropagate to hidden layers
        grad_hidden = [
            output_weights_snapshot[0][i] * delta for i in range(len(output_weights_snapshot[0]))
        ]
        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]
            z = pre_activations[layer_idx]
            prev_activation = activations[layer_idx]
            # apply tanh' = 1 - tanh^2
            grad_hidden = [
                grad_hidden[j] * (1.0 - math.tanh(z[j]) ** 2) for j in range(len(z))
            ]
            weight_snapshot = [row[:] for row in layer.weights]
            for neuron_idx in range(layer.out_dim):
                for in_idx in range(layer.in_dim):
                    grad = grad_hidden[neuron_idx] * prev_activation[in_idx]
                    layer.weights[neuron_idx][in_idx] -= lr * grad
                layer.bias[neuron_idx] -= lr * grad_hidden[neuron_idx]

            if layer_idx == 0:
                break

            next_grad = []
            for in_idx in range(layer.in_dim):
                acc = 0.0
                for neuron_idx in range(layer.out_dim):
                    acc += weight_snapshot[neuron_idx][in_idx] * grad_hidden[neuron_idx]
                next_grad.append(acc)
            grad_hidden = next_grad

    @staticmethod
    def _normalize(features: PolicyFeatures) -> List[float]:
        step_progress = 0.0
        if features.total_steps > 0:
            step_progress = features.step_index / max(1.0, float(features.total_steps - 1))
        return [
            features.entropy,
            math.log(features.snr + 1e-8),
            math.log(features.latent_norm + 1e-8),
            step_progress,
            min(features.prompt_length, 128) / 128.0,
        ]


__all__ = ["DQARPolicy", "PolicyFeatures"]
