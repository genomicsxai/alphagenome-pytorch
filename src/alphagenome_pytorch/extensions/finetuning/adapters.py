"""Parameter-Efficient Fine-Tuning (PEFT) adapters for AlphaGenome.

Ported from Baskerville's TensorFlow implementation:
https://github.com/calico/baskerville/blob/main/src/baskerville/adapters.py

Supports:
- LoRA: Low-Rank Adaptation for Dense/Linear layers
- Locon: LoRA for Conv1D layers
- IA3: Input/output scaling adapters
- Houlsby: Classic bottleneck adapters
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRA(nn.Module):
    """Low-Rank Adaptation for Linear layers.
    
    Wraps an existing Linear layer and adds trainable low-rank matrices.
    The original layer is frozen while only the LoRA matrices are trained.
    
    Reference: https://arxiv.org/abs/2106.09685
    
    Args:
        original_layer: The Linear layer to wrap.
        rank: Rank of the low-rank decomposition (default: 8).
        alpha: Scaling factor (default: 16).
        
    Example:
        >>> linear = nn.Linear(768, 768)
        >>> lora_linear = LoRA(linear, rank=8, alpha=16)
        >>> output = lora_linear(input)
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        if rank > self.out_features:
            raise ValueError(
                f"LoRA rank {rank} must be <= output features {self.out_features}"
            )
        
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA matrices: down (A) and up (B)
        # A: (in_features, rank) - He initialization
        # B: (rank, out_features) - Zero initialization
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output (frozen)
        original_output = self.original_layer(x)
        
        # LoRA output: scale * B(A(x))
        lora_output = self.lora_B(self.lora_A(x)) * self.scale
        
        return original_output + lora_output
    
    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into original layer for efficient inference.
        
        Returns:
            New Linear layer with merged weights.
        """
        # W' = W + scale * B @ A
        merged_weight = (
            self.original_layer.weight.data +
            self.scale * self.lora_B.weight.data @ self.lora_A.weight.data
        )
        
        merged_layer = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.original_layer.bias is not None,
        )
        merged_layer.weight.data = merged_weight
        if self.original_layer.bias is not None:
            merged_layer.bias.data = self.original_layer.bias.data.clone()
        
        return merged_layer


class Locon(nn.Module):
    """LoRA for Conv1D layers.
    
    Applies low-rank adaptation to convolutional layers.
    
    Reference: https://arxiv.org/pdf/2309.14859
    
    Args:
        original_layer: The Conv1d layer to wrap.
        rank: Rank of the low-rank decomposition (default: 4).
        alpha: Scaling factor (default: 1).
    """
    
    def __init__(
        self,
        original_layer: nn.Conv1d,
        rank: int = 4,
        alpha: int = 1,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size[0]
        self.stride = original_layer.stride[0]
        self.dilation = original_layer.dilation[0]
        self.padding = original_layer.padding[0]
        
        if rank > self.out_channels:
            raise ValueError(
                f"Locon rank {rank} must be <= output channels {self.out_channels}"
            )
        
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Locon: down conv (maintains spatial dims) + up conv (1x1)
        self.locon_down = nn.Conv1d(
            self.in_channels,
            rank,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=False,
        )
        
        self.locon_up = nn.Conv1d(
            rank,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        
        # Initialize
        nn.init.kaiming_uniform_(self.locon_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.locon_up.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output (frozen)
        original_output = self.original_layer(x)
        
        # Locon output
        lora_output = self.locon_up(self.locon_down(x)) * self.scale
        
        return original_output + lora_output
    
    def merge_weights(self) -> nn.Conv1d:
        """Merge Locon weights into original layer for efficient inference.
        
        Note: This is an approximation that works for 1x1 up-projection.
        For full correctness, the merged conv may need adjustment.
        
        Returns:
            New Conv1d layer with merged weights.
        """
        # For 1x1 up-conv, we can merge: W' = W + scale * up @ down
        # down: (rank, in_channels, kernel_size)
        # up: (out_channels, rank, 1)
        # Result should be (out_channels, in_channels, kernel_size)
        
        with torch.no_grad():
            # Reshape for matmul: up (out, rank) @ down (rank, in*k) -> (out, in*k)
            down_w = self.locon_down.weight.data  # (rank, in_channels, kernel_size)
            up_w = self.locon_up.weight.data.squeeze(-1)  # (out_channels, rank)
            
            # Merge: (out, rank) @ (rank, in, k) reshaped
            rank, in_ch, k = down_w.shape
            down_flat = down_w.view(rank, -1)  # (rank, in*k)
            merged_delta = (up_w @ down_flat).view(self.out_channels, in_ch, k)
            
            merged_weight = self.original_layer.weight.data + self.scale * merged_delta
        
        merged_layer = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=self.original_layer.bias is not None,
        )
        merged_layer.weight.data = merged_weight
        if self.original_layer.bias is not None:
            merged_layer.bias.data = self.original_layer.bias.data.clone()
        
        return merged_layer


class IA3(nn.Module):
    """IA3 adapter for Linear layers - scales output.
    
    Learns a multiplicative scaling vector for the layer output.
    Extremely parameter-efficient (only output_dim parameters).
    
    Reference: https://arxiv.org/pdf/2205.05638
    
    Args:
        original_layer: The Linear layer to wrap.
    """
    
    def __init__(self, original_layer: nn.Linear):
        super().__init__()
        
        self.original_layer = original_layer
        self.out_features = original_layer.out_features
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Learnable scaling vector (initialized to 1)
        self.scale = nn.Parameter(torch.ones(self.out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original_layer(x)
        return original_output * self.scale


class IA3_FF(nn.Module):
    """IA3 adapter for feed-forward layers - scales input.
    
    Used for down-projection layers where we scale the input.
    
    Reference: https://arxiv.org/pdf/2205.05638
    
    Args:
        original_layer: The Linear layer to wrap.
    """
    
    def __init__(self, original_layer: nn.Linear):
        super().__init__()
        
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Learnable scaling vector (initialized to 1)
        self.scale = nn.Parameter(torch.ones(self.in_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_input = x * self.scale
        return self.original_layer(scaled_input)


class AdapterHoulsby(nn.Module):
    """Houlsby adapter (bottleneck).
    
    A classic adapter architecture with down-projection, activation, and up-projection.
    Inserted after existing layers with a residual connection.
    
    Reference: https://arxiv.org/abs/1902.00751
    
    Args:
        input_dim: Input/output dimension of the adapter.
        latent_dim: Bottleneck dimension (default: 8).
        activation: Activation function (default: ReLU).
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation or nn.ReLU()
        
        # Down-project, activate, up-project
        self.down_project = nn.Linear(input_dim, latent_dim)
        self.up_project = nn.Linear(latent_dim, input_dim)
        
        # Initialize with small values
        nn.init.trunc_normal_(self.down_project.weight, std=1e-3)
        nn.init.zeros_(self.down_project.bias)
        nn.init.trunc_normal_(self.up_project.weight, std=1e-3)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottleneck with residual
        hidden = self.down_project(x)
        hidden = self.activation(hidden)
        output = self.up_project(hidden)
        return output + x  # Residual connection


class HoulsbyWrapper(nn.Module):
    """Wraps a Linear layer with a Houlsby bottleneck adapter.

    Freezes the original layer and applies AdapterHoulsby to its output.
    This is the counterpart of LoRA/IA3 wrappers for the Houlsby pattern.

    Args:
        original_layer: The Linear layer to wrap.
        latent_dim: Bottleneck dimension (default: 8).
        activation: Activation function (default: ReLU).
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        latent_dim: int = 8,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        self.original_layer = original_layer

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        self.adapter = AdapterHoulsby(
            input_dim=original_layer.out_features,
            latent_dim=latent_dim,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.original_layer(x))


class HoulsbyBlockWrapper(nn.Module):
    """Wraps a transformer sub-layer (MHA or MLP) with a Houlsby adapter.

    This implements Baskerville-style placement where adapters are inserted
    at transformer block boundaries, after each sub-layer output but before
    the residual add. This matches the original Houlsby paper placement.

    The adapter output is what gets added to the residual connection in the
    tower. Since AdapterHoulsby has an internal residual, the total computation
    for a wrapped MHA block becomes:
        x_new = x + adapter(mha(x))
              = x + mha(x) + bottleneck(mha(x))

    This adds a learnable non-linear adaptation term to each residual path.

    Reference:
        - Houlsby et al. (2019): https://arxiv.org/abs/1902.00751
        - Baskerville: https://github.com/calico/baskerville/blob/main/src/baskerville/transfer.py

    Args:
        block: The MHA or MLP block to wrap.
        d_model: Model dimension for the adapter (default: 1536).
        latent_dim: Bottleneck dimension (default: 8).
        activation: Activation function (default: ReLU).
    """

    def __init__(
        self,
        block: nn.Module,
        d_model: int = 1536,
        latent_dim: int = 8,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        self.block = block

        # Freeze the original block
        for param in self.block.parameters():
            param.requires_grad = False

        self.adapter = AdapterHoulsby(
            input_dim=d_model,
            latent_dim=latent_dim,
            activation=activation,
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass: block output -> adapter (with internal residual)."""
        block_output = self.block(*args, **kwargs)
        return self.adapter(block_output)


# =============================================================================
# Utility functions for applying adapters to models
# =============================================================================

def _is_adapter_internal(name: str, parent: nn.Module) -> bool:
    """Check if a module is internal to an existing adapter wrapper.

    This prevents wrapping modules like `q_proj.original_layer` when composing
    multiple adapter types (e.g., LoRA + Houlsby on the same targets).
    """
    # Skip modules inside adapter wrappers (e.g., "q_proj.original_layer")
    if ".original_layer" in name or ".adapter" in name:
        return True
    # Skip if parent is already an adapter wrapper
    adapter_parents = (LoRA, Locon, IA3, IA3_FF, HoulsbyWrapper)
    if isinstance(parent, adapter_parents):
        return True
    return False


def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 8,
    alpha: int = 16,
) -> nn.Module:
    """Apply LoRA to Linear layers matching target module names.

    Args:
        model: The model to modify (in-place).
        target_modules: List of substrings to match in module names.
        rank: LoRA rank (default: 8).
        alpha: LoRA alpha scaling factor (default: 16).

    Returns:
        The modified model.

    Example:
        >>> model = apply_lora(model, ['to_q', 'to_v'], rank=8)
    """
    for name, module in list(model.named_modules()):
        if not any(target in name for target in target_modules):
            continue
        if isinstance(module, nn.Linear):
            parent_name, attr_name = _get_parent_and_attr(name)
            parent = _get_module_by_name(model, parent_name)
            if _is_adapter_internal(name, parent):
                continue
            lora_module = LoRA(module, rank=rank, alpha=alpha)
            setattr(parent, attr_name, lora_module)

    return model


def apply_locon(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 4,
    alpha: int = 1,
) -> nn.Module:
    """Apply Locon (LoRA for Conv1D) to layers matching target module names.

    Args:
        model: The model to modify (in-place).
        target_modules: List of substrings to match in module names.
        rank: Locon rank (default: 4).
        alpha: Locon alpha scaling factor (default: 1).

    Returns:
        The modified model.

    Example:
        >>> model = apply_locon(model, ['conv1', 'conv2'], rank=4)
    """
    for name, module in list(model.named_modules()):
        if not any(target in name for target in target_modules):
            continue
        if isinstance(module, nn.Conv1d):
            parent_name, attr_name = _get_parent_and_attr(name)
            parent = _get_module_by_name(model, parent_name)
            if _is_adapter_internal(name, parent):
                continue
            locon_module = Locon(module, rank=rank, alpha=alpha)
            setattr(parent, attr_name, locon_module)

    return model


def apply_ia3(
    model: nn.Module,
    target_modules: list[str],
    ff_modules: list[str] | None = None,
) -> nn.Module:
    """Apply IA3 adapters to Linear layers matching target module names.

    Args:
        model: The model to modify (in-place).
        target_modules: List of substrings for output-scaling (IA3).
        ff_modules: Optional list of substrings for input-scaling (IA3_FF).
            Typically used for down-projection in feed-forward layers.

    Returns:
        The modified model.

    Example:
        >>> model = apply_ia3(model, ['to_k', 'to_v'], ff_modules=['fc2'])
    """
    ff_modules = ff_modules or []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        parent_name, attr_name = _get_parent_and_attr(name)
        parent = _get_module_by_name(model, parent_name)
        if _is_adapter_internal(name, parent):
            continue

        if any(target in name for target in ff_modules):
            # Input-scaling for feed-forward down-projection
            ia3_module = IA3_FF(module)
            setattr(parent, attr_name, ia3_module)
        elif any(target in name for target in target_modules):
            # Output-scaling for attention
            ia3_module = IA3(module)
            setattr(parent, attr_name, ia3_module)

    return model


def apply_houlsby(
    model: nn.Module,
    target_modules: list[str],
    latent_dim: int = 8,
) -> nn.Module:
    """Apply Houlsby bottleneck adapters to Linear layers matching target module names.

    This is the "linear" placement mode - adapters wrap individual Linear layers.
    For Baskerville-style block-level placement, use apply_houlsby_baskerville().

    Note:
        This function skips modules that are internal to existing adapter wrappers
        (e.g., `.original_layer` inside LoRA). This prevents breaking adapter
        composition when multiple adapter types target the same modules.

    Args:
        model: The model to modify (in-place).
        target_modules: List of substrings to match in module names.
        latent_dim: Bottleneck dimension (default: 8).

    Returns:
        The modified model.

    Example:
        >>> model = apply_houlsby(model, ['q_proj', 'v_proj'], latent_dim=8)
    """
    for name, module in list(model.named_modules()):
        if not any(target in name for target in target_modules):
            continue
        if isinstance(module, nn.Linear):
            parent_name, attr_name = _get_parent_and_attr(name)
            parent = _get_module_by_name(model, parent_name)
            if _is_adapter_internal(name, parent):
                continue
            houlsby_module = HoulsbyWrapper(module, latent_dim=latent_dim)
            setattr(parent, attr_name, houlsby_module)

    return model


def apply_houlsby_baskerville(
    model: nn.Module,
    latent_dim: int = 8,
    d_model: int = 1536,
    target_blocks: list[str] | None = None,
) -> nn.Module:
    """Apply Houlsby adapters at transformer block boundaries (Baskerville-style).

    This implements the original Houlsby paper placement where adapters are
    inserted after each transformer sub-layer (MHA and MLP), before the
    residual add. This matches the Baskerville TensorFlow implementation.

    The adapter wraps the sub-layer output so the residual computation becomes:
        x = x + adapter(sublayer(x))
          = x + sublayer(x) + bottleneck(sublayer(x))

    Reference:
        - Houlsby et al. (2019): https://arxiv.org/abs/1902.00751
        - Baskerville: https://github.com/calico/baskerville/blob/main/src/baskerville/transfer.py

    Args:
        model: The AlphaGenome model to modify (in-place).
        latent_dim: Bottleneck dimension (default: 8).
        d_model: Model dimension (default: 1536).
        target_blocks: Which sub-layers to wrap. Default ['mha', 'mlp'] wraps both.
            Use ['mha'] or ['mlp'] to wrap only one type.

    Returns:
        The modified model.

    Example:
        >>> model = apply_houlsby_baskerville(model, latent_dim=8)
    """
    if target_blocks is None:
        target_blocks = ['mha', 'mlp']

    # Find the transformer tower
    tower = None
    for name, module in model.named_modules():
        if name.endswith('tower') or type(module).__name__ == 'TransformerTower':
            tower = module
            break

    if tower is None:
        raise ValueError(
            "Could not find TransformerTower in model. "
            "apply_houlsby_baskerville() requires an AlphaGenome model."
        )

    # Wrap MHA and/or MLP blocks in each transformer block
    adapters_added = 0
    for block_idx, block in enumerate(tower.blocks):
        if 'mha' in target_blocks and 'mha' in block:
            original_mha = block['mha']
            block['mha'] = HoulsbyBlockWrapper(
                original_mha,
                d_model=d_model,
                latent_dim=latent_dim,
            )
            adapters_added += 1

        if 'mlp' in target_blocks and 'mlp' in block:
            original_mlp = block['mlp']
            block['mlp'] = HoulsbyBlockWrapper(
                original_mlp,
                d_model=d_model,
                latent_dim=latent_dim,
            )
            adapters_added += 1

    if adapters_added == 0:
        raise ValueError(
            f"No blocks matched target_blocks={target_blocks}. "
            "Available block keys: mha, mlp"
        )

    return model


def unfreeze_norm_layers(model: nn.Module) -> int:
    """Unfreeze normalization layers (LayerNorm, RMSBatchNorm, etc.).

    This is a common technique when using adapters, as normalization layers
    have few parameters but significantly influence feature distribution.
    Baskerville unfreezes LayerNorm layers alongside Houlsby adapters.

    Args:
        model: The model to modify (in-place).

    Returns:
        Number of parameters unfrozen.

    Example:
        >>> unfrozen = unfreeze_norm_layers(model)
        >>> print(f"Unfroze {unfrozen} normalization parameters")
    """
    norm_layer_types = (
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.GroupNorm,
    )

    # Also match custom norm layers by name pattern
    norm_name_patterns = ['norm', 'layernorm', 'batchnorm', 'rmsnorm', 'rmsbatchnorm']

    params_unfrozen = 0

    for name, module in model.named_modules():
        is_norm = isinstance(module, norm_layer_types)
        if not is_norm:
            # Check by class name for custom norm layers
            class_name = type(module).__name__.lower()
            is_norm = any(pattern in class_name for pattern in norm_name_patterns)

        if is_norm:
            for param in module.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    params_unfrozen += param.numel()

    return params_unfrozen


def merge_adapters(model: nn.Module) -> nn.Module:
    """Merge all adapter weights into base layers for efficient inference.
    
    Handles LoRA and Locon adapters. After merging, adapters are replaced
    with standard Linear/Conv1d layers containing the merged weights.
    
    Args:
        model: Model with adapters.
        
    Returns:
        Model with merged weights (adapters replaced with base layers).
        
    Example:
        >>> model = merge_adapters(model)
        >>> model.save('merged_model.pt')
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRA):
            parent_name, attr_name = _get_parent_and_attr(name)
            parent = _get_module_by_name(model, parent_name)
            merged = module.merge_weights()
            setattr(parent, attr_name, merged)
        elif isinstance(module, Locon):
            parent_name, attr_name = _get_parent_and_attr(name)
            parent = _get_module_by_name(model, parent_name)
            merged = module.merge_weights()
            setattr(parent, attr_name, merged)
    
    return model


def get_adapter_params(model: nn.Module) -> list:
    """Get only the trainable adapter parameters for optimizer.
    
    Returns parameters from LoRA, Locon, IA3, and Houlsby adapters.
    Useful for creating optimizers that only update adapter weights.
    
    Args:
        model: Model with adapters applied.
        
    Returns:
        List of trainable parameters from adapters.
        
    Example:
        >>> optimizer = torch.optim.AdamW(get_adapter_params(model), lr=1e-4)
    """
    adapter_types = (LoRA, Locon, IA3, IA3_FF, AdapterHoulsby, HoulsbyWrapper)
    params = []
    
    for module in model.modules():
        if isinstance(module, adapter_types):
            for param in module.parameters():
                if param.requires_grad:
                    params.append(param)
    
    return params


# Keep for backwards compatibility
def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Deprecated: Use merge_adapters() instead."""
    return merge_adapters(model)


def _get_parent_and_attr(name: str) -> tuple[str, str]:
    """Split module name into parent path and attribute name."""
    parts = name.rsplit('.', 1)
    if len(parts) == 1:
        return '', parts[0]
    return parts[0], parts[1]


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Get a module by its dotted name path."""
    if not name:
        return model
    module = model
    for part in name.split('.'):
        module = getattr(module, part)
    return module


__all__ = [
    # Adapter classes
    'LoRA',
    'Locon',
    'IA3',
    'IA3_FF',
    'AdapterHoulsby',
    'HoulsbyWrapper',
    'HoulsbyBlockWrapper',
    # Apply functions
    'apply_lora',
    'apply_locon',
    'apply_ia3',
    'apply_houlsby',
    'apply_houlsby_baskerville',
    'unfreeze_norm_layers',
    # Utilities
    'merge_adapters',
    'get_adapter_params',
    # Deprecated
    'merge_lora_weights',
]

