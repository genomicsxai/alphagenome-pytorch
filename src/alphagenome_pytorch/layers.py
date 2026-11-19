import torch
import torch.nn as nn
import torch.nn.functional as F

class QuickGELU(nn.Module):
    """GELU using sigmoid-gated approximation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coef = torch.tensor(1.702, dtype=x.dtype, device=x.device)
        return torch.sigmoid(coef * x) * x

class Log1p(nn.Module):
    """Hookable torch.log1p wrapper."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(x)

class UndoSoftClip(nn.Module):
    """Hookable inverse of soft clipping used when unscaling genomic track predictions."""
    def __init__(self, soft_clip_value: float):
        super().__init__()
        self.soft_clip_value = soft_clip_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x > self.soft_clip_value,
            (x + self.soft_clip_value) ** 2 / (4 * self.soft_clip_value),
            x,
        )

class UndoSquashing(nn.Module):
    """Undo the power-law squashing used for RNA-seq targets."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(x, 1.0 / 0.75)

# Similar to tangermeme.deep_lift_shap.BilinearOp; defined locally to avoid
# making tangermeme a required dependency.
class BilinearOp(nn.Module):
    """Bilinear contraction of two tensors.

    Args:
        equation:
            If None, computes ``torch.matmul(left, right)``.
            If a string, uses it as the equation in ``torch.einsum(equation, left, right)``.

    The operands are used exactly as passed; any transpose, reshape, cast,
    or scaling should happen outside this module.
    """
    def __init__(self, equation: str | None = None):
        super().__init__()
        self.equation = equation

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.equation is None:
            return torch.matmul(left, right)
        return torch.einsum(self.equation, left, right)

class Pool1d(nn.Module):
    """1D pooling with SAME padding. Expects NCL input (B, C, S).

    Matches JAX: alphagenome_research.model.layers.pool
    JAX uses padding='SAME' which pads input to ensure output_size = ceil(input_size / stride).
    """
    def __init__(self, kernel_size: int, stride: int = None, method: str = 'max'):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.method = method
        if self.method == 'max':
            self.pool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride)
        elif self.method in ['avg', 'mean']:
            self.pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride)
        else:
            self.pool = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S) - NCL format, no transpose needed
        input_size = x.shape[-1]
        output_size = (input_size + self.stride - 1) // self.stride  # ceil division
        pad_total = max((output_size - 1) * self.stride + self.kernel_size - input_size, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        if pad_total > 0:
            x = F.pad(x, (pad_left, pad_right))

        if self.pool is None:
            raise NotImplementedError(f"Pooling method {self.method} not implemented")
        return self.pool(x)

class RMSBatchNorm(nn.Module):
    """RMS Batch Normalization supporting both channels-first and channels-last formats.

    Normalizes over the channel dimension using stored running statistics.
    Matches JAX: alphagenome_research.model.layers.RMSBatchNorm

    Args:
        num_features: Number of channels.
        channels: Alias for num_features.
        eps: Small constant for numerical stability.
        channels_last: If True, expects (B, S, C) format. If False, expects (B, C, S).
                       Default False (channels-first, matching PyTorch conv conventions).
    """
    def __init__(self, num_features: int = 0, channels: int = 0, eps: float = 1e-5, channels_last: bool = False):
        super().__init__()
        num_features = num_features or channels
        if num_features == 0:
            raise ValueError("Must provide num_features or channels")
        self.num_features = num_features
        self.eps = eps
        self.channels_last = channels_last

        # Always store parameters as (C,) - standard PyTorch convention
        # Reshape for broadcasting happens in forward()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # JAX casts the inverse std dev to input dtype BEFORE multiplying by scale
        inv = self.weight * torch.rsqrt(self.running_var + self.eps).to(x.dtype)
        if self.channels_last:
            # NLC format (B, S, C) - parameters broadcast from the right
            return x * inv + self.bias
        else:
            # NCL format (B, C, S) - reshape for broadcasting
            return x * inv.view(1, -1, 1) + self.bias.view(1, -1, 1)

class LayerNorm(nn.Module):
    """Layer Normalization with optional RMSNorm mode (centering=False).

    Expects NLC format (B, S, C) - used by TransformerTower.
    Normalizes over the last dimension(s).

    Matches JAX: alphagenome_research.model.layers.LayerNorm
    JAX computes variance in float32 for numerical stability, then casts back.
    Here, we compute the entire normalization in float32 to use standard
    DeepLIFT rules, then cast back to the input dtype before applying weight
    and bias.
    """
    def __init__(
        self,
        normalized_shape, eps: float = 1e-5,
        elementwise_affine: bool = True,
        rms_norm: bool = False
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = getattr(normalized_shape, 'tuple', lambda: normalized_shape)()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.rms_norm = rms_norm

        if self.rms_norm:
            # RMSNorm: x / sqrt(mean(x^2) + eps)
            self.norm = nn.RMSNorm(self.normalized_shape, eps=self.eps, elementwise_affine=False)
        else:
            # Standard LayerNorm with centering
            self.norm = nn.LayerNorm(
                self.normalized_shape, eps=self.eps, elementwise_affine=False, bias=False
            )

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_norm = self.norm(x.float()).to(input_dtype)

        if self.elementwise_affine:
            return x_norm * self.weight + self.bias
        return x_norm
