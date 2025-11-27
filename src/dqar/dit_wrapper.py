"""
DiT Integration Wrapper for DQAR.

This module provides patches and wrappers to integrate DQAR's attention reuse
into HuggingFace DiT pipelines. It hooks into the transformer attention layers
via custom AttentionProcessors to capture real attention weights and enable
actual computation skipping.

Usage:
    from dqar.dit_wrapper import patch_dit_pipeline

    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
    pipe = patch_dit_pipeline(pipe)  # Now supports DQAR controller

    controller = DQARController(num_layers=28)
    output = pipe(class_labels=[207], controller=controller)
"""

from __future__ import annotations

import math
import weakref
from functools import wraps
from typing import TYPE_CHECKING, Callable, Optional, Any

if TYPE_CHECKING:
    from .controller import DQARController

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from diffusers import DiTPipeline
    from diffusers.models.attention import Attention
    from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None
    DiTPipeline = None
    Attention = None
    AttnProcessor = None
    AttnProcessor2_0 = None


class DQARAttentionProcessor:
    """
    Custom AttentionProcessor that enables DQAR caching with REAL attention weights.

    This processor:
    1. Computes Q, K, V projections
    2. Computes attention_probs (captures REAL attention weights for entropy)
    3. Caches the full output tensor for reuse
    4. On reuse decision, returns cached output (ACTUAL computation skip)
    """

    def __init__(
        self,
        layer_idx: int,
        original_processor: Any = None,
    ):
        self.layer_idx = layer_idx
        self.original_processor = original_processor
        self._controller_ref: Optional[weakref.ref] = None
        self._branch: str = "cond"

        # Cache for output reuse (stores full output tensor)
        self._cached_output: Optional[torch.Tensor] = None
        self._cached_step: int = -1

    def set_controller(self, controller: Optional["DQARController"], branch: str = "cond"):
        """Set the active DQAR controller."""
        if controller is not None:
            self._controller_ref = weakref.ref(controller)
        else:
            self._controller_ref = None
        self._branch = branch

    def _get_controller(self) -> Optional["DQARController"]:
        """Get controller from weak reference."""
        if self._controller_ref is None:
            return None
        return self._controller_ref()

    def __call__(
        self,
        attn: "Attention",
        hidden_states: "torch.Tensor",
        encoder_hidden_states: Optional["torch.Tensor"] = None,
        attention_mask: Optional["torch.Tensor"] = None,
        temb: Optional["torch.Tensor"] = None,
        *args,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Process attention with DQAR caching.

        When controller approves reuse, returns cached output directly.
        Otherwise, computes fresh attention and caches results.
        """
        controller = self._get_controller()

        # No controller - use original processor or compute normally
        if controller is None:
            if self.original_processor is not None:
                return self.original_processor(
                    attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs
                )
            return self._compute_attention(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

        # Check if we should reuse cached output
        decision = controller.should_reuse(self.layer_idx, branch=self._branch)

        if decision.use_cache and decision.entry is not None:
            # ACTUAL COMPUTATION SKIP - return cached output directly
            cached_output = getattr(decision.entry, '_cached_output', None)
            if cached_output is not None and cached_output.shape == hidden_states.shape:
                controller._total_reuse_count += 1
                return cached_output.to(hidden_states.device, hidden_states.dtype)

        # Compute fresh attention with real attention weights capture
        output, attention_probs = self._compute_attention_with_weights(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb
        )

        # Cache results with REAL attention weights
        self._cache_attention_output(controller, hidden_states, output, attention_probs, attn.heads)

        return output

    def _compute_attention(
        self,
        attn: "Attention",
        hidden_states: "torch.Tensor",
        encoder_hidden_states: Optional["torch.Tensor"],
        attention_mask: Optional["torch.Tensor"],
        temb: Optional["torch.Tensor"],
    ) -> "torch.Tensor":
        """Compute attention without weight capture (fallback path)."""
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Use scaled_dot_product_attention if available (PyTorch 2.0+)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def _compute_attention_with_weights(
        self,
        attn: "Attention",
        hidden_states: "torch.Tensor",
        encoder_hidden_states: Optional["torch.Tensor"],
        attention_mask: Optional["torch.Tensor"],
        temb: Optional["torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Compute attention AND capture attention weights for entropy calculation.

        Returns:
            Tuple of (output, attention_probs) where attention_probs has shape
            [batch_size, num_heads, seq_len, seq_len]
        """
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Compute attention scores manually to capture attention_probs
        scale = 1.0 / math.sqrt(head_dim)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # REAL attention probabilities - this is what we need for entropy!
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Compute attention output
        hidden_states = torch.matmul(attention_probs, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states, attention_probs

    def _cache_attention_output(
        self,
        controller: "DQARController",
        hidden_states: "torch.Tensor",
        output: "torch.Tensor",
        attention_probs: "torch.Tensor",
        num_heads: int,
    ) -> None:
        """
        Cache attention output with REAL entropy from attention_probs.

        Args:
            controller: DQAR controller
            hidden_states: Input tensor
            output: Output tensor to cache
            attention_probs: Real attention weights [batch, heads, seq, seq]
            num_heads: Number of attention heads
        """
        # Compute REAL entropy from actual attention probabilities
        # attention_probs shape: [batch_size, num_heads, seq_len, seq_len]
        # Average over batch dimension for entropy calculation
        attn_for_entropy = attention_probs.mean(dim=0)  # [heads, seq, seq]

        # Convert to list format for controller
        attn_list = attn_for_entropy.detach().cpu().tolist()

        # Compute residual
        residual = (output - hidden_states).detach()

        # Get sequence length
        seq_len = attention_probs.shape[-1]

        # Create minimal K, V placeholders (we cache full output instead)
        k_list = [[0.0] * 16 for _ in range(min(seq_len, 64))]
        v_list = [[0.0] * 16 for _ in range(min(seq_len, 64))]

        latent_list = hidden_states.mean(dim=0).flatten()[:256].cpu().tolist()
        residual_list = residual.mean(dim=0).flatten()[:256].cpu().tolist()

        # Commit to controller cache
        entry = controller.commit(
            layer_id=self.layer_idx,
            branch=self._branch,
            attn_map=attn_list,
            keys=k_list,
            values=v_list,
            latent=latent_list,
            residual=residual_list,
        )

        # Store full output tensor on the cache entry for direct reuse
        entry._cached_output = output.detach().clone()


class DQARPipelineWrapper:
    """
    Wrapper class for DiTPipeline that adds DQAR controller support.

    This wrapper intercepts calls to the pipeline and manages the DQAR
    controller lifecycle, setting it on attention processors before inference
    and clearing it afterwards.
    """

    def __init__(self, pipe: "DiTPipeline", processors: list):
        self._pipe = pipe
        self._dqar_processors = processors
        self._dqar_num_layers = len(processors)
        self._active_controller = None

        # Copy commonly accessed attributes from the wrapped pipeline
        self.transformer = pipe.transformer
        self.scheduler = pipe.scheduler
        self.vae = getattr(pipe, 'vae', None)
        self.config = getattr(pipe, 'config', None)

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped pipeline."""
        return getattr(self._pipe, name)

    def __call__(
        self,
        *args,
        controller: Optional["DQARController"] = None,
        **kwargs,
    ):
        """
        Call the pipeline with optional DQAR controller.

        Args:
            *args: Positional arguments passed to the pipeline.
            controller: Optional DQARController for attention reuse.
            **kwargs: Keyword arguments passed to the pipeline.

        Returns:
            Pipeline output (typically images).
        """
        # Extract DQAR-specific kwargs
        branch = kwargs.pop('dqar_branch', 'cond')

        # Set controller on all processors
        for processor in self._dqar_processors:
            processor.set_controller(controller, branch=branch)

        self._active_controller = controller

        try:
            # Add callback for step updates if controller provided
            if controller is not None:
                original_callback = kwargs.get('callback_on_step_end')

                def step_callback(pipe, step_idx, timestep, callback_kwargs):
                    if self._active_controller:
                        num_steps = kwargs.get('num_inference_steps', 50)
                        # For class-conditional models, use a default prompt length
                        prompt_length = 16
                        if 'prompt' in kwargs and kwargs['prompt']:
                            prompt_length = len(str(kwargs['prompt']).split())
                        self._active_controller.begin_step(
                            step_index=step_idx,
                            total_steps=num_steps,
                            prompt_length=prompt_length,
                        )
                    if original_callback:
                        return original_callback(pipe, step_idx, timestep, callback_kwargs)
                    return callback_kwargs

                kwargs['callback_on_step_end'] = step_callback

            # Call the underlying pipeline
            return self._pipe(*args, **kwargs)
        finally:
            # Clear controller references
            for processor in self._dqar_processors:
                processor.set_controller(None)
            self._active_controller = None

    def to(self, device):
        """Move pipeline to device."""
        self._pipe = self._pipe.to(device)
        return self

    def set_progress_bar_config(self, **kwargs):
        """Configure progress bar."""
        self._pipe.set_progress_bar_config(**kwargs)


def patch_dit_pipeline(pipe: "DiTPipeline") -> "DQARPipelineWrapper":
    """
    Patch a DiTPipeline to support DQAR controller integration.

    This function:
    1. Replaces attention processors with DQARAttentionProcessor
    2. Returns a wrapper that accepts a controller argument
    3. Hooks step callbacks to update controller state

    Args:
        pipe: A DiTPipeline instance from HuggingFace diffusers.

    Returns:
        A DQARPipelineWrapper that wraps the original pipeline.

    Raises:
        ImportError: If torch/diffusers are not installed.
        ValueError: If the pipeline structure is not recognized.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch and diffusers are required for DiT integration. "
            "Install with: pip install torch diffusers"
        )

    if not hasattr(pipe, 'transformer'):
        raise ValueError("Pipeline does not have a 'transformer' attribute")

    transformer = pipe.transformer
    blocks = getattr(transformer, 'transformer_blocks', None) or getattr(transformer, 'blocks', [])

    if not blocks:
        raise ValueError("Could not find transformer blocks in the model")

    # Replace attention processors with DQAR processors
    processors: list[DQARAttentionProcessor] = []

    for idx, block in enumerate(blocks):
        attn_module = None

        if hasattr(block, 'attn1'):
            attn_module = block.attn1
        elif hasattr(block, 'attn'):
            attn_module = block.attn

        if attn_module is not None:
            # Get original processor
            original_processor = attn_module.processor if hasattr(attn_module, 'processor') else None

            # Create DQAR processor
            dqar_processor = DQARAttentionProcessor(
                layer_idx=idx,
                original_processor=original_processor,
            )
            processors.append(dqar_processor)

            # Replace the processor
            attn_module.processor = dqar_processor

            # Store reference to original
            block._dqar_original_processor = original_processor

    if not processors:
        raise ValueError("No attention layers found to patch")

    # Create and return wrapper
    wrapped = DQARPipelineWrapper(pipe, processors)
    print(f"[DQAR] Patched pipeline with {len(processors)} attention processors")
    return wrapped


def get_dit_layer_count(pipe) -> int:
    """
    Get the number of transformer layers in a DiT pipeline.

    Works with both original DiTPipeline and DQARPipelineWrapper.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch and diffusers required")

    # Check if it's a DQAR wrapper
    if isinstance(pipe, DQARPipelineWrapper):
        return pipe._dqar_num_layers

    if hasattr(pipe, '_dqar_num_layers'):
        return pipe._dqar_num_layers

    transformer = getattr(pipe, 'transformer', None)
    if transformer is None:
        raise ValueError("Pipeline has no transformer")

    blocks = getattr(transformer, 'transformer_blocks', None) or getattr(transformer, 'blocks', [])
    return len(blocks)


__all__ = ["patch_dit_pipeline", "get_dit_layer_count", "DQARAttentionProcessor", "DQARPipelineWrapper"]
