# AlphaGenome JAX vs PyTorch Architecture Comparison

This document provides a detailed comparison between the original JAX implementation and the PyTorch port, identifying potential sources of numerical discrepancies.

## Table of Contents

1. [Overall Architecture](#overall-architecture)
2. [Resolution Changes (1bp -> 128bp -> 1bp)](#resolution-changes)
3. [Scaling and Unscaling Operations](#scaling-and-unscaling)
4. [Precision / Dtype Handling](#precision-dtype-handling)
5. [Component-by-Component Comparison](#component-comparison)
6. [Potential Sources of Discrepancy](#potential-discrepancies)

---

## 1. Overall Architecture <a name="overall-architecture"></a>

### Data Flow

```
DNA Sequence (B, S, 4) - One-hot encoded
         |
         v
[SequenceEncoder] - DNA embedding + 6 DownResBlocks + pooling
         |
         v
Trunk Embeddings (B, S/128, 1536) @ 128bp resolution
         |
         +-- Add organism embedding (1536 dims)
         |
         v
[TransformerTower] - 9 blocks with PairUpdate, AttentionBias, MHA, MLP
         |
         +----------------------+
         |                      |
         v                      v
Trunk (B, S/128, 1536)    Pair Acts (B, S/2048, S/2048, 128)
         |                      |
         v                      |
[SequenceDecoder] - 7 UpResBlocks with U-Net skips
         |                      |
         v                      |
Decoded (B, S, 768) @ 1bp       |
         |                      |
         v                      v
[OutputEmbedders]           [OutputPair]
         |                      |
         v                      v
Embeddings 128bp (3072)     Pair Embeddings (128)
Embeddings 1bp (1536)           |
         |                      |
         v                      v
[Genome Track Heads]       [Contact Maps Head]
```

---

## 2. Resolution Changes <a name="resolution-changes"></a>

### Encoding Path (1bp -> 128bp)

| Stage | Resolution | JAX Shape | PyTorch Shape | Channels |
|-------|-----------|-----------|---------------|----------|
| Input | 1bp | (B, S, 4) | (B, S, 4) | 4 |
| DnaEmbedder | 1bp | (B, S, 768) | (B, S, 768) | 768 |
| Pool | 2bp | (B, S/2, 768) | (B, S/2, 768) | 768 |
| DownResBlock 0 | 2bp | (B, S/2, 896) | (B, S/2, 896) | +128 |
| Pool | 4bp | (B, S/4, 896) | (B, S/4, 896) | 896 |
| DownResBlock 1 | 4bp | (B, S/4, 1024) | (B, S/4, 1024) | +128 |
| Pool | 8bp | (B, S/8, 1024) | (B, S/8, 1024) | 1024 |
| DownResBlock 2 | 8bp | (B, S/8, 1152) | (B, S/8, 1152) | +128 |
| Pool | 16bp | (B, S/16, 1152) | (B, S/16, 1152) | 1152 |
| DownResBlock 3 | 16bp | (B, S/16, 1280) | (B, S/16, 1280) | +128 |
| Pool | 32bp | (B, S/32, 1280) | (B, S/32, 1280) | 1280 |
| DownResBlock 4 | 32bp | (B, S/32, 1408) | (B, S/32, 1408) | +128 |
| Pool | 64bp | (B, S/64, 1408) | (B, S/64, 1408) | 1408 |
| DownResBlock 5 | 64bp | (B, S/64, 1536) | (B, S/64, 1536) | +128 |
| Pool | 128bp | (B, S/128, 1536) | (B, S/128, 1536) | 1536 |

**Key Operations:**
- **Pooling**: JAX uses `hk.MaxPool(window_shape=(2,1), strides=(2,1), padding='SAME')`
- **PyTorch**: Custom `Pool1d` with SAME padding calculation

### Decoding Path (128bp -> 1bp)

| Stage | Input Resolution | Output Resolution | Skip From | Output Channels |
|-------|-----------------|-------------------|-----------|-----------------|
| UpResBlock 0 | 128bp | 64bp | bin_size_64 | 1536 |
| UpResBlock 1 | 64bp | 32bp | bin_size_32 | 1408 |
| UpResBlock 2 | 32bp | 16bp | bin_size_16 | 1280 |
| UpResBlock 3 | 16bp | 8bp | bin_size_8 | 1152 |
| UpResBlock 4 | 8bp | 4bp | bin_size_4 | 1024 |
| UpResBlock 5 | 4bp | 2bp | bin_size_2 | 896 |
| UpResBlock 6 | 2bp | 1bp | bin_size_1 | 768 |

**Key Operations:**
- **Upsampling**: JAX uses `jnp.repeat(out, 2, axis=1)`
- **PyTorch**: `torch.repeat_interleave(out, repeats=2, dim=1)`

### Intermediates Dictionary

```python
intermediates = {
    'bin_size_1': 768ch,    # Original 1bp embeddings
    'bin_size_2': 896ch,    # After DownResBlock 0
    'bin_size_4': 1024ch,   # After DownResBlock 1
    'bin_size_8': 1152ch,   # After DownResBlock 2
    'bin_size_16': 1280ch,  # After DownResBlock 3
    'bin_size_32': 1408ch,  # After DownResBlock 4
    'bin_size_64': 1536ch,  # After DownResBlock 5
}
```

---

## 3. Scaling and Unscaling Operations <a name="scaling-and-unscaling"></a>

### Prediction Scaling (Model Space -> Experimental Space)

**Used for:** Converting raw model outputs to experimental data scale

```python
# JAX: alphagenome_research.model.heads.predictions_scaling
# PyTorch: alphagenome_pytorch.heads.predictions_scaling

def predictions_scaling(x, track_means, resolution, apply_squashing):
    # Step 1: Soft Clipping
    # Where x > 10.0, apply quadratic expansion
    soft_clip_value = 10.0
    x = where(x > soft_clip_value,
              (x + soft_clip_value)**2 / (4 * soft_clip_value),  # = (x+10)^2/40
              x)

    # Step 2: Squashing Inverse (only for RNA-seq)
    if apply_squashing:
        x = x ** (1.0 / 0.75)  # Power law expansion, ~= x^1.333

    # Step 3: Scale by track means and resolution
    x = x * (track_means * resolution)

    return x
```

### Target Scaling (Experimental Space -> Model Space)

**Used for:** Converting experimental targets to model prediction space for loss computation

```python
# JAX: alphagenome_research.model.heads.targets_scaling

def targets_scaling(targets, track_means, resolution, apply_squashing):
    # Step 1: Normalize by track means
    targets = targets / (track_means * resolution)

    # Step 2: Squashing (only for RNA-seq)
    if apply_squashing:
        targets = targets ** 0.75  # Power law compression

    # Step 3: Soft Clipping
    soft_clip_value = 10.0
    targets = where(targets > soft_clip_value,
                    2 * sqrt(targets * soft_clip_value) - soft_clip_value,
                    targets)

    return targets
```

### Track Means

**Source**: Bundled into model weights by `convert_weights.py` (stored as `heads.{head_name}.track_means`)
**Shape**: `(num_organisms=2, num_tracks)`
**Purpose**: Per-organism, per-track scaling factors from training data statistics

| Head | Num Tracks | Apply Squashing |
|------|-----------|-----------------|
| ATAC | 256 | False |
| DNASE | 384 | False |
| PROCAP | 128 | False |
| CAGE | 640 | False |
| RNA_SEQ | 768 | **True** |
| CHIP_TF | 1664 | False |
| CHIP_HISTONE | 1152 | False |

### Head Prediction Flow

```python
# In GenomeTracksHead._predict():

# 1. Linear transformation (organism-specific)
x = MultiOrganismLinear(x, organism_index)  # (B, S, num_tracks)

# 2. Get learned residual scale
residual_scale = learnt_scale[organism_index]  # (B, num_tracks)

# 3. Apply softplus to both predictions and scale
output = softplus(x) * softplus(residual_scale)

# 4. Unscale to experimental data space
output = predictions_scaling(output, track_means, resolution, apply_squashing)
```

---

## 4. Precision / Dtype Handling <a name="precision-dtype-handling"></a>

### JAX Mixed Precision Policy

**Location**: `alphagenome_research.model.dna_model.py:1038`

```python
jmp_policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')
```

### PyTorch Precision Configuration

The PyTorch implementation supports configurable precision via the `dtype_policy` constructor argument:

```python
import torch
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy

# Load model (track_means are bundled in weights from convert_weights.py)
model = AlphaGenome()
model.load_state_dict(torch.load('model.pth'))

# For JAX-matching mode (params=f32, compute/output=bf16):
model = AlphaGenome(dtype_policy=DtypePolicy.mixed_precision())
model.load_state_dict(torch.load('model.pth'))
```

| Aspect | JAX | PyTorch (bfloat16 mode) | PyTorch (float32 mode) |
|--------|-----|------------------------|------------------------|
| **Parameters** | float32 | float32 | float32 |
| **Computation** | bfloat16 | bfloat16 | float32 |
| **Accumulation** | float32 | float32 | float32 |
| **Outputs** | bfloat16 | bfloat16 | float32 |

**Key implementation details:**
- Attention logits always computed in float32 for stability
- LayerNorm variance always computed in float32
- Softmax always in float32
- Results cast back to compute_dtype after accumulation

### Precision in Attention Operations

**JAX** (attention.py):
```python
# Attention logits computation
attention_logits = jnp.einsum(
    'bshc,bS1c->bhsS', q, k,
    precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,  # Compute in BF16, accumulate in F32
    preferred_element_type=jnp.float32,  # Output type
)

# Value projection
y = jnp.einsum(
    'bhsS,bS1c->bshc', attention_weights, v,
    precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
).astype(q.dtype)  # Cast back to input dtype
```

**PyTorch** (attention.py):
```python
# Standard matmul (no precision control)
att = torch.matmul(q_t, k_t.transpose(-2, -1))
att = att / math.sqrt(128.0)

y = torch.matmul(attn_weights, v_t)
```

### Sum Pooling Precision

**JAX** (heads.py:255):
```python
return x.reshape(...).sum(axis=-2, dtype=jnp.float32)  # Explicit float32 for stability
```

**PyTorch**: Standard sum (uses input dtype)

### Final Output Upcasting

**JAX** (dna_model.py):
```python
def _upcast_single_batch_predictions(x, *, transfer_to_host: bool = True):
    # Convert from bfloat16 back to float32 for output
    x = jax.tree.map(lambda x: tensor_utils.upcast_floating(x[0]), x)
    return jax.device_put(x, jax.memory.Space.Host)
```

---

## 5. Component-by-Component Comparison <a name="component-comparison"></a>

### 5.1 GELU Activation

| Aspect | JAX | PyTorch |
|--------|-----|---------|
| **Formula** | `sigmoid(1.702 * x) * x` | `sigmoid(1.702 * x) * x` |
| **Coefficient** | `jax.lax.convert_element_type(1.702, x.dtype)` | `torch.tensor(1.702, dtype=x.dtype)` |

**Status**: MATCHED

### 5.2 RMSBatchNorm

**JAX** (layers.py):
```python
# Uses exponential moving average of variance (var_ema)
variance = hk.get_state('var_ema', param_shape, dtype=jnp.float32, init=jnp.ones)
scale = hk.get_parameter('scale', param_shape, dtype=x.dtype, init=jnp.ones)
offset = hk.get_parameter('offset', param_shape, dtype=x.dtype, init=jnp.zeros)
inv = scale * jax.lax.rsqrt(variance + 1e-5).astype(x.dtype)
return x * inv + offset
```

**PyTorch** (layers.py):
```python
# Uses running_var buffer
var = self.running_var
inv = self.weight * torch.rsqrt(var + self.eps)
return x * inv + self.bias
```

**Differences**:
- JAX uses separate `scale` and `var_ema` (loaded from state)
- PyTorch combines into `weight` and `running_var`
- Epsilon: Both use 1e-5

**Status**: SHOULD MATCH (verify weight conversion)

### 5.3 LayerNorm

**JAX** (layers.py):
```python
# With optional RMS mode (centering=False)
if not self._rms_norm:
    mean = jnp.mean(x, axis=self._axis, dtype=jnp.float32, keepdims=True)
    x = x - mean

variance = jnp.mean(jnp.square(x), axis=self._axis, dtype=jnp.float32, keepdims=True)
inv = scale * jax.lax.rsqrt(variance + 1e-5).astype(dtype)
return inv * x + offset
```

**PyTorch** (layers.py):
```python
if self.rms_norm:
    dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
    variance = torch.mean(x ** 2, dim=dims, keepdim=True)
    x_norm = x * torch.rsqrt(variance + self.eps)
else:
    x_norm = F.layer_norm(x, self.normalized_shape, eps=self.eps)

if self.elementwise_affine:
    return x_norm * self.weight + self.bias
```

**Differences**:
- JAX computes variance in float32 explicitly
- PyTorch uses input dtype for variance computation
- **POTENTIAL DISCREPANCY**: Intermediate precision

### 5.4 Pooling (SAME Padding)

**JAX** (layers.py):
```python
hk.MaxPool(window_shape=(by, 1), strides=(by, 1), padding='SAME')(x)
# Input shape: (B, S, D)
```

**PyTorch** (layers.py):
```python
class Pool1d(nn.Module):
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, D, S)

        # SAME padding calculation
        input_size = x.shape[-1]
        output_size = (input_size + self.stride - 1) // self.stride
        pad_total = max((output_size - 1) * self.stride + self.kernel_size - input_size, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        if pad_total > 0:
            x = F.pad(x, (pad_left, pad_right))

        x = F.max_pool1d(x, kernel_size=self.kernel_size, stride=self.stride)
        x = x.transpose(1, 2)
        return x
```

**Differences**:
- JAX operates on (B, S, D) directly
- PyTorch requires transpose to (B, D, S)
- SAME padding should match for even kernel sizes

**Status**: SHOULD MATCH (verify edge cases)

### 5.5 StandardizedConv1D

**JAX** (convolutions.py):
```python
# Weight shape: (width, input_channels, output_channels)
w -= jnp.mean(w, axis=(0, 1), keepdims=True)  # Mean over kernel and input channels
var_w = jnp.var(w, axis=(0, 1), keepdims=True)
scale = scale * jax.lax.rsqrt(jnp.maximum(fan_in * var_w, 1e-4))
w_standardized = w * scale

out = jax.lax.conv_general_dilated(
    lhs=x, rhs=w_standardized,
    window_strides=[1], padding='SAME',
    dimension_numbers=jax.lax.ConvDimensionNumbers(
        lhs_spec=(0, 2, 1),  # (batch, channel, spatial)
        rhs_spec=(2, 1, 0),  # (out, in, spatial)
        out_spec=(0, 2, 1)
    ),
)
```

**PyTorch** (convolutions.py):
```python
# Weight shape: (out_channels, in_channels, kernel_width)
w = self.weight
mean = w.mean(dim=(1, 2), keepdim=True)  # Mean over in_channels and kernel
var = w.var(dim=(1, 2), keepdim=True, unbiased=False)

fan_in = self.in_channels * self.kernel_size[0]
scale_factor = torch.rsqrt(torch.maximum(var * fan_in, torch.tensor(1e-4))) * self.scale

w_standardized = (w - mean) * scale_factor
```

**Differences**:
- Weight layout differs (requires careful conversion)
- JAX: `(width, in, out)` vs PyTorch: `(out, in, width)`
- Mean/var computed over different axes due to layout

**POTENTIAL DISCREPANCY**: Weight transpose during conversion

### 5.6 RoPE (Rotary Position Embeddings)

**JAX** (attention.py):
```python
inv_freq = 1.0 / (
    jnp.arange(num_freq) + jnp.geomspace(1, max_position - num_freq + 1, num_freq)
).astype(x.dtype)
theta = jnp.einsum('bs,f->bsf', positions, inv_freq)
theta = jnp.repeat(theta, 2, axis=-1)[..., None, :]
x_rotated = jnp.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(x.shape)
return x * jnp.cos(theta) + x_rotated * jnp.sin(theta)
```

**PyTorch** (attention.py):
```python
base_freqs = torch.logspace(math.log10(1), math.log10(max_position - num_freq + 1),
                            steps=num_freq, base=10, device=x.device)
denom = torch.arange(num_freq, device=x.device) + base_freqs
inv_freq = 1.0 / denom

theta = torch.einsum('bs,f->bsf', positions, inv_freq)
theta = torch.repeat_interleave(theta, 2, dim=-1).unsqueeze(2)

x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(start_dim=-2)
return x * torch.cos(theta) + x_rotated * torch.sin(theta)
```

**Differences**:
- `jnp.geomspace` vs `torch.logspace` - slight numerical differences possible
- `jnp.repeat` vs `torch.repeat_interleave` - should be identical

**POTENTIAL DISCREPANCY**: Geomspace implementation

### 5.7 Attention Softcap

**Both**:
```python
logits_soft_cap = 5.0
attention_logits = tanh(attention_logits / logits_soft_cap) * logits_soft_cap
```

**Status**: MATCHED

### 5.8 Central Mask Features

**JAX** (attention.py):
```python
center_widths = jnp.arange(feature_size) + jnp.geomspace(
    1, seq_length - feature_size + 1, feature_size, endpoint=False
)
outputs = (center_widths > distances[..., None]).astype(distances.dtype)
```

**PyTorch** (attention.py):
```python
log_start = math.log(1)
log_end = math.log(seq_length - feature_size + 1)
log_step = (log_end - log_start) / steps  # endpoint=False equivalent

exponents = torch.arange(steps) * log_step
center_widths = torch.exp(log_start + exponents)
# Note: Missing the + torch.arange(feature_size) term!

return (center_widths > distances[..., None]).float()
```

**DISCREPANCY**: PyTorch implementation is missing the `+ arange(feature_size)` term!

---

## 6. Potential Sources of Discrepancy <a name="potential-discrepancies"></a>

### HIGH Priority (Likely causes of significant differences)

#### 6.1 Central Mask Features - Missing Term
**Location**: `attention.py:_central_mask_features`
**Issue**: PyTorch implementation missing `+ arange(feature_size)`
```python
# JAX:
center_widths = jnp.arange(feature_size) + jnp.geomspace(...)

# PyTorch (BUGGY):
center_widths = torch.exp(log_start + exponents)
# MISSING: + torch.arange(feature_size)
```
**Impact**: Affects pair representation in SequenceToPairBlock

#### 6.2 Precision in Attention Einsum
**Location**: `attention.py:MHABlock`, `RowAttentionBlock`
**Issue**: JAX uses `precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32`
```python
# JAX computes attention in BF16 with F32 accumulator
# PyTorch uses standard float32 throughout
```
**Impact**: Numerical differences accumulate through 9 transformer layers

#### 6.3 Variance Computation Dtype
**Location**: `layers.py:LayerNorm`
**Issue**: JAX explicitly computes variance in float32
```python
# JAX:
variance = jnp.mean(jnp.square(x), axis=..., dtype=jnp.float32, keepdims=True)

# PyTorch:
variance = torch.mean(x ** 2, dim=..., keepdim=True)  # Uses input dtype
```
**Impact**: Normalization layers may differ slightly

### MEDIUM Priority (May cause subtle differences)

#### 6.4 Geomspace vs Logspace
**Location**: `attention.py:apply_rope`, `_central_mask_features`
**Issue**: Different implementations of geometric spacing
```python
# JAX: jnp.geomspace(1, max_val, n)
# PyTorch: torch.logspace(log10(1), log10(max_val), n, base=10)
```
**Impact**: Small floating-point differences in RoPE frequencies

#### 6.5 GELU Approximation
**Location**: `layers.py:gelu`
**Issue**: Both use sigmoid approximation, but dtype handling may differ
```python
# JAX:
coef = jax.lax.convert_element_type(1.702, x.dtype)

# PyTorch:
coef = torch.tensor(1.702, dtype=x.dtype, device=x.device)
```
**Impact**: Usually minimal, but compounds through network

#### 6.6 OutputEmbedder Order of Operations
**Location**: `embeddings.py:OutputEmbedder`
**JAX Order**:
```python
return layers.gelu(layers.RMSBatchNorm()(x) + organism_embedding)
```
**PyTorch Order**:
```python
out = self.norm(x_proj)
out = out + emb.unsqueeze(1)
out = layers.gelu(out)
```
**Status**: Should be equivalent, verify norm placement

### LOW Priority (Unlikely to cause significant issues)

#### 6.7 Softplus Implementation
**Both use standard softplus**: `log(1 + exp(x))`
**Status**: Should match

#### 6.8 Repeat vs Repeat_Interleave
```python
# JAX: jnp.repeat(x, 2, axis=1)
# PyTorch: torch.repeat_interleave(x, 2, dim=1)
```
**Status**: Equivalent behavior

---

## Summary of Required Fixes

### Critical Fixes

1. **`attention.py:_central_mask_features`** - Add missing `+ arange` term
2. Consider adding explicit `dtype=torch.float32` to variance computations in LayerNorm

### Recommended Improvements

1. Add option to run in bfloat16 to match JAX precision behavior
2. Verify weight conversion for StandardizedConv1d (axis ordering)
3. Add precision control to attention einsum operations

### Testing Recommendations

1. Compare intermediate outputs at each stage (encoder, tower, decoder)
2. Test with bfloat16 inputs to match JAX compute precision
3. Verify scaling/unscaling outputs match for each head type
4. Test edge cases for SAME padding in pooling layers

---

## 7. Verified Matching Components <a name="verified-matching"></a>

This section documents components that have been verified to match between JAX and PyTorch implementations.

### 7.1 Activation Functions

#### GELU
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Formula | `sigmoid(1.702 * x) * x` | `sigmoid(1.702 * x) * x` | MATCHED |
| Coefficient dtype | `jax.lax.convert_element_type(1.702, x.dtype)` | `torch.tensor(1.702, dtype=x.dtype)` | MATCHED |
| Location | `layers.py:24-27` | `layers.py:5-12` | - |

#### Softplus
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Formula | `jax.nn.softplus(x)` = `log(1 + exp(x))` | `F.softplus(x)` = `log(1 + exp(x))` | MATCHED |
| Usage | Heads prediction scaling | Heads prediction scaling | MATCHED |

### 7.2 Normalization Layers

#### RMSBatchNorm
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Formula | `x * (scale / sqrt(var + eps)) + offset` | `x * (weight * rsqrt(var + eps)) + bias` | MATCHED |
| Epsilon | `1e-5` | `1e-5` (default) | MATCHED |
| Variance source | `var_ema` state | `running_var` buffer | MATCHED |
| Location | `layers.py:55-80` | `layers.py:51-73` | - |

#### LayerNorm (RMS mode)
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Formula (RMS) | `x / sqrt(mean(x^2) + eps) * scale + offset` | `x * rsqrt(mean(x^2) + eps) * weight + bias` | MATCHED |
| Centering | Optional (`rms_norm=False` centers) | Optional (`rms_norm=True` skips centering) | MATCHED |
| Epsilon | `1e-5` | `1e-5` | MATCHED |
| Location | `layers.py:83-123` | `layers.py:75-107` | - |

### 7.3 Convolutional Components

#### DnaEmbedder
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| First conv | `hk.Conv1D(768, kernel=15)` | `nn.Conv1d(4, 768, kernel=15, padding='same')` | MATCHED |
| Residual block | `ConvBlock(768, width=5)` | `ConvBlock(768, 768, kernel=5)` | MATCHED |
| Output | `conv(x) + ConvBlock(conv(x))` | `conv(x) + block(conv(x))` | MATCHED |
| Location | `convolutions.py:104-112` | `convolutions.py:76-94` | - |

#### DownResBlock
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Channel increase | +128 per block | +128 per block | MATCHED |
| Structure | `ConvBlock + pad(x) + ConvBlock` | `ConvBlock + pad(x) + ConvBlock` | MATCHED |
| Padding | `jnp.pad(x, [(0,0), (0,0), (0, 128)])` | `F.pad(x, (0, 128))` | MATCHED |
| Location | `convolutions.py:115-125` | `convolutions.py:96-113` | - |

#### UpResBlock
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Structure | conv_in + slice → upsample → scale → skip → conv_out | Same | MATCHED |
| Upsampling | `jnp.repeat(out, 2, axis=1)` | `repeat_interleave(out, 2, dim=1)` | MATCHED |
| Residual scale | Learned scalar parameter | Learned scalar parameter | MATCHED |
| Location | `convolutions.py:128-152` | `convolutions.py:115-149` | - |

#### StandardizedConv1D
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Weight standardization | `w -= mean(w); w *= scale/sqrt(var*fan_in)` | Same formula | MATCHED |
| Fan-in | `width * input_channels` | `in_channels * kernel_size[0]` | MATCHED |
| Padding | `'SAME'` | Manual SAME padding | MATCHED |
| Location | `convolutions.py:54-101` | `convolutions.py:6-47` | - |

### 7.4 Attention Components

#### MHABlock (Multi-Head Attention)
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Num heads | 8 | 8 | MATCHED |
| Q dim | 8 × 128 = 1024 | 8 × 128 = 1024 | MATCHED |
| K dim | 1 × 128 = 128 | 1 × 128 = 128 | MATCHED |
| V dim | 1 × 192 = 192 | 1 × 192 = 192 | MATCHED |
| RoPE | Applied to Q and K | Applied to Q and K | MATCHED |
| Softcap | `tanh(logits/5) * 5` | `tanh(logits/5) * 5` | MATCHED |
| Location | `attention.py:109-161` | `attention.py:77-107` | - |

#### MLPBlock
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Hidden dim | `2 * d_model` | `2 * d_model` | MATCHED |
| Activation | ReLU | ReLU | MATCHED |
| Norms | RMSBatchNorm (pre and post) | RMSBatchNorm (pre and post) | MATCHED |
| Location | `attention.py:82-91` | `attention.py:109-121` | - |

#### PairUpdateBlock
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Components | SequenceToPair + RowAttention + PairMLP | Same | MATCHED |
| Residual connections | `x + y` for each component | Same | MATCHED |
| Location | `attention.py:266-279` | `attention.py:237-258` | - |

#### RowAttentionBlock
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Q/K/V dim | 128 | 128 | MATCHED |
| Attention formula | `softmax(QK^T / sqrt(128)) @ V` | Same | MATCHED |
| Norm | LayerNorm (RMS mode) | LayerNorm (RMS mode) | MATCHED |
| Location | `attention.py:177-202` | `attention.py:203-221` | - |

#### AttentionBiasBlock
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Output heads | 8 (matches MHA heads) | 8 | MATCHED |
| Upsampling | `repeat(16)` on both spatial dims | `repeat_interleave(16)` on both | MATCHED |
| Activation | GELU before projection | GELU before projection | MATCHED |
| Location | `attention.py:164-174` | `attention.py:123-136` | - |

#### RoPE (Rotary Position Embeddings)
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Max position | 8192 | 8192 | MATCHED |
| Frequency formula | `1/(arange + geomspace)` | `1/(arange + logspace)` | MATCHED |
| Rotation | `x*cos(θ) + rotate(x)*sin(θ)` | Same | MATCHED |
| Location | `attention.py:61-79` | `attention.py:9-25` | - |

### 7.5 Scaling Operations

#### Predictions Scaling (Model → Experimental)
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Soft clip value | 10.0 | 10.0 | MATCHED |
| Soft clip formula | `(x + 10)^2 / 40` when `x > 10` | Same | MATCHED |
| Squashing inverse | `x^(1/0.75)` for RNA-seq | Same | MATCHED |
| Scale factor | `track_means * resolution` | Same | MATCHED |
| Location | `heads.py:305-333` | `heads.py:9-44` | - |

#### Targets Scaling (Experimental → Model)
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Normalize | `targets / (track_means * resolution)` | Same | MATCHED |
| Squashing | `targets^0.75` for RNA-seq | Same | MATCHED |
| Soft clip | `2*sqrt(x*10) - 10` when `x > 10` | Same | MATCHED |
| Location | `heads.py:336-367` | N/A (inference only) | - |

### 7.6 Head Components

#### GenomeTracksHead
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Linear layer | `_MultiOrganismLinear` | `MultiOrganismLinear` | MATCHED |
| Residual scale | `softplus(learnt_scale)` | `softplus(residual_scales)` | MATCHED |
| Output formula | `softplus(linear(x)) * softplus(scale)` | Same | MATCHED |
| Unscaling | `predictions_scaling()` | Same | MATCHED |
| Location | `heads.py:424-627` | `heads.py:96-177` | - |

#### ContactMapsHead
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Input | Pair embeddings (B, S, S, 128) | Same | MATCHED |
| Linear | `_MultiOrganismLinear` to 28 tracks | Same | MATCHED |
| Location | `heads.py:630-687` | `heads.py:180-205` | - |

#### MultiOrganismLinear
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Weight shape | `(num_organisms, in, out)` | Same | MATCHED |
| Bias shape | `(num_organisms, out)` | Same | MATCHED |
| Forward | `einsum('b...i,bij->b...j', x, w[org]) + b[org]` | `bmm(x, w[org]) + b[org]` | MATCHED |
| Location | `heads.py:274-302` | `heads.py:47-94` | - |

### 7.7 Output Embedders

#### OutputEmbedder
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Projection | `Linear(2 * in_dim)` | `Linear(in, out)` | MATCHED |
| Skip handling | `Linear(out_dim, bias=False) + repeat` | Same | MATCHED |
| Organism embed | Added after norm | Added after norm | MATCHED |
| Final activation | `gelu(norm(x) + org_emb)` | Same | MATCHED |
| Location | `embeddings.py:43-75` | `embeddings.py:6-105` | - |

#### OutputPair
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Symmetrization | `(x + x.T) / 2` | `(x + x.transpose(1,2)) / 2` | MATCHED |
| Norm | LayerNorm (RMS mode) | LayerNorm (RMS mode) | MATCHED |
| Organism embed | 128 dim | 128 dim | MATCHED |
| Location | `embeddings.py:78-101` | `embeddings.py:107-127` | - |

### 7.8 Model Architecture

#### SequenceEncoder
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Blocks | DnaEmbedder + 6 DownResBlocks | Same | MATCHED |
| Bin sizes | [2, 4, 8, 16, 32, 64] | Same | MATCHED |
| Pooling | Max pool by 2 between blocks | Same | MATCHED |
| Channel progression | 768 → 896 → ... → 1536 | Same | MATCHED |
| Location | `model.py:37-52` | `model.py:5-33` | - |

#### SequenceDecoder
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Blocks | 7 UpResBlocks | Same | MATCHED |
| Bin sizes | [64, 32, 16, 8, 4, 2, 1] | Same | MATCHED |
| Skip connections | From encoder intermediates | Same | MATCHED |
| Location | `model.py:55-64` | `model.py:35-83` | - |

#### TransformerTower
| Aspect | JAX | PyTorch | Status |
|--------|-----|---------|--------|
| Num blocks | 9 | 9 | MATCHED |
| PairUpdate | Even blocks (0, 2, 4, 6, 8) | Same | MATCHED |
| Block structure | PairUpdate? → AttnBias → MHA → MLP | Same | MATCHED |
| Residual | `x += MHA(...)` and `x += MLP(...)` | Same | MATCHED |
| Location | `model.py:67-81` | `model.py:85-118` | - |

### 7.9 Head Configurations

| Head | Tracks | Resolutions | Squashing | Status |
|------|--------|-------------|-----------|--------|
| ATAC | 256 | [1, 128] | False | MATCHED |
| DNASE | 384 | [1, 128] | False | MATCHED |
| PROCAP | 128 | [1, 128] | False | MATCHED |
| CAGE | 640 | [1, 128] | False | MATCHED |
| RNA_SEQ | 768 | [1, 128] | **True** | MATCHED |
| CHIP_TF | 1664 | [128] | False | MATCHED |
| CHIP_HISTONE | 1152 | [128] | False | MATCHED |
| CONTACT_MAPS | 28 | N/A | N/A | MATCHED |

---

## 8. Change Log

### 2025-01-13: Initial Comparison
- Created comprehensive architecture comparison document
- Documented all matching components between JAX and PyTorch implementations
- Verified `_central_mask_features` implementation in `attention.py:44-75`
- Added configurable precision via `dtype_policy` constructor argument
  - `DtypePolicy.full_float32()` for stable mode (default)
  - `DtypePolicy.mixed_precision()` for JAX-matching mode (params=f32, compute/output=bf16)
- Updated LayerNorm to compute variance in float32
- Updated MHABlock and RowAttentionBlock for float32 accumulation
