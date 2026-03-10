import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers

class StandardizedConv1d(nn.Conv1d):
    """1D Convolution with weight standardization and learned scaling.

    Expects NCL format (B, C, S).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        # JAX uses padding='SAME'. PyTorch 'same' padding requires specific setup or manual padding.
        # We will handle padding in forward to match 'SAME' behavior roughly.
        self.pad_mode = padding
        
        # Scale parameter
        self.scale = nn.Parameter(torch.ones(out_channels, 1, 1))

    def forward(self, x):
        # x: (B, C, S) - NCL format
        
        # Weight standardization
        # JAX: w -= mean(w, axis=(0, 1)) -> (kernel_width, input_channels)
        # PyTorch weight: (out_channels, in_channels, kernel_width)
        # We want to standardize over (in_channels, kernel_width) corresponding to fan-in?
        # JAX shape: (width, input_channels, output_channels). Mean axis (0, 1) means mean over width and input_channels.
        # PyTorch equivalent: mean over (1, 2).
        
        w = self.weight
        mean = w.mean(dim=(1, 2), keepdim=True)
        var = w.var(dim=(1, 2), keepdim=True, unbiased=False) 
        
        fan_in = self.in_channels * self.kernel_size[0]
        scale_factor = torch.rsqrt(torch.maximum(var * fan_in, torch.tensor(1e-4, device=w.device, dtype=w.dtype))) * self.scale
        
        w_standardized = (w - mean) * scale_factor
        
        # Padding 'SAME' manually if needed, or use functional
        # For even kernel sizes, 'same' padding is asymmetric. JAX/TF usually pad more on the right.
        if self.pad_mode == 'same':
            # Padding formulation:
            # Note: this formula is valid for stride=1 only (the only stride used in this model).
            pad_total = self.kernel_size[0] - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x = F.pad(x, (pad_left, pad_right))
            
        return F.conv1d(x, w_standardized, self.bias, self.stride, 0, self.dilation, self.groups)

class ConvBlock(nn.Module):
    """Convolution block operating on NCL format (B, C, S)."""

    def __init__(self, in_channels, out_channels, kernel_size, name=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.norm = layers.RMSBatchNorm(in_channels)

        if kernel_size == 1:
            # Use Conv1d(k=1) instead of Linear - same math, native NCL
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv = StandardizedConv1d(in_channels, out_channels, kernel_size, padding='same')

    def forward(self, x):
        # x: (B, C, S) - NCL format, no transposes needed
        return self.conv(layers.gelu(self.norm(x)))

class DnaEmbedder(nn.Module):
    """Embeds one-hot DNA to feature space. Expects NCL format (B, 4, S)."""

    def __init__(self):
        super().__init__()
        # JAX: Conv1D(768, 15) -> Input 4 channels (one-hot)
        # Then + ConvBlock(768, 5)
        self.conv1 = nn.Conv1d(4, 768, kernel_size=15, padding='same')
        self.block = ConvBlock(768, 768, kernel_size=5)

    def forward(self, x):
        # x: (B, 4, S) - NCL format, no transposes needed
        out = self.conv1(x)
        return out + self.block(out)

class DownResBlock(nn.Module):
    """Downsampling residual block. Expects NCL format (B, C, S)."""

    def __init__(self, in_channels, name=None):
        super().__init__()
        self.out_channels_int = in_channels + 128
        self.block1 = ConvBlock(in_channels, self.out_channels_int, kernel_size=5)
        self.block2 = ConvBlock(self.out_channels_int, self.out_channels_int, kernel_size=5)

    def forward(self, x):
        # x: (B, C, S) - NCL format
        out = self.block1(x)

        # Residual connection with channel padding
        # F.pad pads from last dim backwards: (left_S, right_S, left_C, right_C)
        # We want to pad channels (dim 1), so: (0, 0, 0, 128)
        x_padded = F.pad(x, (0, 0, 0, 128))

        out = out + x_padded
        return out + self.block2(out)

class UpResBlock(nn.Module):
    """Upsampling residual block with skip connection. Expects NCL format (B, C, S)."""

    def __init__(self, in_channels, skip_channels):
        super().__init__()
        self.conv_in = ConvBlock(in_channels, skip_channels, kernel_size=5)
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.pointwise = ConvBlock(skip_channels, skip_channels, kernel_size=1)
        self.conv_out = ConvBlock(skip_channels, skip_channels, kernel_size=5)

    def forward(self, x, unet_skip):
        # x: (B, C, S) - NCL format
        # unet_skip: (B, C_skip, S*2) - skip has 2x sequence length

        # 1. First block + slice channels to match skip
        # Channels are dim 1 in NCL: x[:, :skip_channels, :]
        out = self.conv_in(x) + x[:, :unet_skip.shape[1], :]

        # 2. Upsample sequence (dim 2 in NCL)
        out = torch.repeat_interleave(out, repeats=2, dim=2)

        out = out * self.residual_scale

        # 3. Add skip connection
        out = out + self.pointwise(unet_skip)

        # 4. Final block
        return out + self.conv_out(out)
