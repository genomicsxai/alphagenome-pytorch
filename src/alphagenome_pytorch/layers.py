import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def gelu(x):
    """GELU using JAX's custom approximation: sigmoid(1.702 * x) * x

    Matches JAX: alphagenome_research.model.layers.gelu
    JAX explicitly converts coefficient to match input dtype.
    """
    coef = torch.tensor(1.702, dtype=x.dtype, device=x.device)
    return torch.sigmoid(coef * x) * x

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S) - NCL format, no transpose needed
        input_size = x.shape[-1]
        output_size = (input_size + self.stride - 1) // self.stride  # ceil division
        pad_total = max((output_size - 1) * self.stride + self.kernel_size - input_size, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        if pad_total > 0:
            x = F.pad(x, (pad_left, pad_right))

        if self.method == 'max':
            return F.max_pool1d(x, kernel_size=self.kernel_size, stride=self.stride)
        elif self.method in ['avg', 'mean']:
            return F.avg_pool1d(x, kernel_size=self.kernel_size, stride=self.stride)
        else:
            raise NotImplementedError(f"Pooling method {self.method} not implemented")

class RMSBatchNorm(nn.Module):
    """RMS Batch Normalization supporting both channels-first and channels-last formats.

    Normalizes over the channel dimension by the per-channel second moment
    (RMS, i.e. mean of x**2 with no mean subtraction).

    In eval mode this uses the stored ``running_var`` buffer, matching JAX
    (alphagenome_research.model.layers.RMSBatchNorm), which only ever runs
    inference. In train mode it normalizes by the current batch second moment
    and updates ``running_var`` via an exponential moving average, so the layer
    can be trained from scratch.

    Args:
        num_features: Number of channels.
        channels: Alias for num_features.
        eps: Small constant for numerical stability.
        channels_last: If True, expects (B, S, C) format. If False, expects (B, C, S).
                       Default False (channels-first, matching PyTorch conv conventions).
        momentum: EMA factor for the running second moment in train mode.
        track_running_stats: If False, train mode skips the running_var update
                             (and from-scratch training is not supported).
        sync_stats: If True, train mode computes the batch second moment across
                    all ranks in process_group (SyncBN). Needed for correct,
                    rank-invariant stats under sequence parallelism, where each
                    rank otherwise only sees its local slice of the sequence.
        process_group: Process group for sync_stats. None uses the default group.
    """
    def __init__(self, num_features: int = 0, channels: int = 0, eps: float = 1e-5,
                 channels_last: bool = False, momentum: float = 0.1,
                 track_running_stats: bool = True, sync_stats: bool = False,
                 process_group=None):
        super().__init__()
        num_features = num_features or channels
        if num_features == 0:
            raise ValueError("Must provide num_features or channels")
        self.num_features = num_features
        self.eps = eps
        self.channels_last = channels_last
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.sync_stats = sync_stats
        self.process_group = process_group

        # Always store parameters as (C,) - standard PyTorch convention
        # Reshape for broadcasting happens in forward()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def _second_moment(self, x: torch.Tensor, dims) -> torch.Tensor:
        """Per-channel second moment (mean of x**2), optionally synced across ranks."""
        sq = x.float() ** 2
        sync = (self.sync_stats and dist.is_available() and dist.is_initialized()
                and dist.get_world_size(self.process_group) > 1)
        if not sync:
            return sq.mean(dim=dims)
        # SyncBN: global mean = sum(x**2) over all ranks / total element count.
        # The sum is reduced with a differentiable all-reduce so each rank's
        # local activations receive gradient from every rank's normalization.
        import torch.distributed.nn.functional as dist_fn
        group = self.process_group if self.process_group is not None else dist.group.WORLD
        sum_sq = dist_fn.all_reduce(sq.sum(dim=dims), op=dist.ReduceOp.SUM, group=group)
        count = torch.tensor(sq.numel() / self.num_features, device=sq.device, dtype=sum_sq.dtype)
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group)
        return sum_sq / count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.track_running_stats:
            # Reduce over batch + sequence, keeping per-channel stats. Second
            # moment only (no centering) to match the RMS / running_var semantics.
            # JAX computes the statistic in float32 for stability.
            dims = tuple(range(x.ndim - 1)) if self.channels_last else (0,) + tuple(range(2, x.ndim))
            stat = self._second_moment(x, dims)
            with torch.no_grad():
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * stat.detach())
        else:
            stat = self.running_var
        # JAX casts the inverse std dev to input dtype BEFORE multiplying by scale
        inv = self.weight * torch.rsqrt(stat + self.eps).to(x.dtype)
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
    """
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, rms_norm: bool = False):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = getattr(normalized_shape, 'tuple', lambda: normalized_shape)()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.rms_norm = rms_norm

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))

        if self.rms_norm:
            # RMSNorm: x / sqrt(mean(x^2) + eps)
            # JAX computes variance in float32 for stability
            variance = torch.mean(x.float() ** 2, dim=dims, keepdim=True)
            inv = torch.rsqrt(variance + self.eps).to(input_dtype)
            x_norm = x * inv
        else:
            # Standard LayerNorm with centering
            # JAX: mean and variance both computed in float32
            mean = torch.mean(x.float(), dim=dims, keepdim=True)
            x_centered = x - mean.to(input_dtype)
            variance = torch.mean(x_centered.float() ** 2, dim=dims, keepdim=True)
            inv = torch.rsqrt(variance + self.eps).to(input_dtype)
            x_norm = x_centered * inv

        if self.elementwise_affine:
            return x_norm * self.weight + self.bias
        return x_norm


def set_sync_batchnorm(module: nn.Module, enabled: bool = True, process_group=None) -> int:
    """Toggle cross-rank SyncBN on every RMSBatchNorm in a module tree.

    Use when training the trunk (full fine-tuning or from-scratch) under
    sequence parallelism, so batch statistics are computed over the whole
    sequence rather than each rank's local slice. Returns the number of
    RMSBatchNorm layers updated. No effect in eval mode or when the world
    size is 1.
    """
    count = 0
    for m in module.modules():
        if isinstance(m, RMSBatchNorm):
            m.sync_stats = enabled
            m.process_group = process_group
            count += 1
    return count
