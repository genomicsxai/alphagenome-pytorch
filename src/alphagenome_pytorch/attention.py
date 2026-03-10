import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers

_MAX_RELATIVE_DISTANCE = 8192


def _apply_rope_inplace(x, cos_theta, sin_theta):
    """Memory-efficient in-place RoPE application.

    Only allocates 0.5x extra memory (vs 2x in standard implementation).
    Modifies x in-place and returns it.

    Args:
        x: Input tensor (B, S, H, C)
        cos_theta: Cosine of rotation angles (B, S, 1, C)
        sin_theta: Sine of rotation angles (B, S, 1, C)

    Returns:
        x modified in-place with RoPE applied
    """
    # Clone even positions before overwriting (0.5x memory overhead)
    x_even = x[..., ::2].clone()

    # Compute and write new even values in-place
    # RoPE formula for even indices: x_even * cos - x_odd * sin
    x[..., ::2] = x_even * cos_theta[..., ::2] - x[..., 1::2] * sin_theta[..., ::2]

    # Compute and write new odd values in-place (uses saved x_even)
    # RoPE formula for odd indices: x_even * sin + x_odd * cos
    x[..., 1::2] = x_even * sin_theta[..., 1::2] + x[..., 1::2] * cos_theta[..., 1::2]

    return x


def apply_rope(x, positions=None, max_position=_MAX_RELATIVE_DISTANCE, inplace=False):
    """Applies Rotary Position Embeddings to the input tensor.

    Matches JAX: alphagenome_research.model.attention.apply_rope

    All computations use the input dtype (x.dtype), matching JAX behavior.
    When using DtypePolicy.mixed_precision(), this means RoPE computes in bfloat16.
    When using DtypePolicy.full_float32(), this means RoPE computes in float32.

    Args:
        x: Input tensor (B, S, H, C)
        positions: Optional position indices (B, S)
        max_position: Maximum position for frequency calculation
        inplace: If True, use memory-efficient in-place implementation.
                 Reduces memory overhead from ~2x to ~0.5x but modifies x in-place.

    Returns:
        Tensor with RoPE applied. If inplace=True, returns the same tensor (modified).
    """
    # x: (B, S, H, C)
    B, S, H, C = x.shape
    compute_dtype = x.dtype  # Match JAX: use input dtype for all RoPE ops

    if positions is None:
        positions = torch.arange(S, device=x.device, dtype=compute_dtype).unsqueeze(0)  # (1, S)
    elif positions.dtype != compute_dtype:
        positions = positions.to(compute_dtype)

    num_freq = C // 2
    # JAX geomspace equivalent: geomspace(1, max_position - num_freq + 1, num_freq)
    base_freqs = torch.logspace(
        math.log10(1), math.log10(max_position - num_freq + 1),
        steps=num_freq, base=10, device=x.device, dtype=compute_dtype
    )
    denom = torch.arange(num_freq, device=x.device, dtype=compute_dtype) + base_freqs
    inv_freq = 1.0 / denom

    theta = torch.einsum('bs,f->bsf', positions, inv_freq)
    theta = torch.repeat_interleave(theta, 2, dim=-1).unsqueeze(2)  # (B, S, 1, C)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    if inplace:
        return _apply_rope_inplace(x, cos_theta, sin_theta)
    else:
        x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(start_dim=-2)
        return x * cos_theta + x_rotated * sin_theta

def _shift(x, query_length, key_length):
    """Shifts the diagonal of a 2D array, PyTorch equivalent."""
    # x: (..., query_length, query_length + key_length)
    shape = x.shape
    batch_shape = shape[:-2]
    n_rows = shape[-2]
    n_diags = shape[-1]
    
    # Reshape to (..., n_diags, n_rows)
    x = x.view(*batch_shape, n_diags, n_rows)
    # Drop first row (originally first diag)
    x = x[..., 1:, :] 
    # Reshape back to (..., n_rows, n_diags - 1)
    x = x.view(*batch_shape, n_rows, n_diags - 1)
    # Return first key_length columns
    return x[..., :key_length]

def _central_mask_features(distances, feature_size, seq_length):
    """Positional features using exponentially-spaced central mask.

    Matches JAX: alphagenome_research.model.attention._central_mask_features

    JAX formula:
        center_widths = jnp.arange(feature_size) + jnp.geomspace(
            1, seq_length - feature_size + 1, feature_size, endpoint=False
        )
    """
    device = distances.device
    dtype = torch.float32

    # Compute geomspace(1, seq_length - feature_size + 1, feature_size, endpoint=False)
    # geomspace with endpoint=False: values[i] = start * (end/start)^(i/n)
    start = 1.0
    end = float(seq_length - feature_size + 1)

    log_start = math.log(start)
    log_end = math.log(end)
    log_step = (log_end - log_start) / feature_size  # endpoint=False

    exponents = torch.arange(feature_size, device=device, dtype=dtype) * log_step
    geomspace_values = torch.exp(torch.tensor(log_start, device=device, dtype=dtype) + exponents)

    # JAX: center_widths = jnp.arange(feature_size) + jnp.geomspace(...)
    center_widths = torch.arange(feature_size, device=device, dtype=dtype) + geomspace_values

    # center_widths: (feature_size,)
    # distances: (...)
    # Output: (..., feature_size)
    return (center_widths > distances.unsqueeze(-1)).to(dtype)

class MHABlock(nn.Module):
    """Multi-Head Attention block.

    Matches JAX: alphagenome_research.model.attention.MHABlock

    JAX uses precision=BF16_BF16_F32 for attention, meaning:
    - Inputs in bfloat16
    - Accumulation in float32
    - Output cast back to input dtype
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm = layers.RMSBatchNorm(d_model, channels_last=True)
        self.q_proj = nn.Linear(d_model, 8 * 128, bias=False)
        self.norm_q = layers.LayerNorm(128)
        self.k_proj = nn.Linear(d_model, 128, bias=False)
        self.norm_k = layers.LayerNorm(128)
        self.v_proj = nn.Linear(d_model, 192, bias=False)
        self.norm_v = layers.LayerNorm(192)
        self.final_norm = layers.RMSBatchNorm(d_model, channels_last=True)
        self.linear_embedding = nn.Linear(8 * 192, d_model)

    def forward(self, x, attention_bias, compute_dtype=None):
        B, S, D = x.shape
        if compute_dtype is None:
            compute_dtype = x.dtype

        # Cast to compute dtype
        x = x.to(compute_dtype)

        h = self.norm(x)

        q = self.norm_q(self.q_proj(h).view(B, S, 8, 128))
        k = self.norm_k(self.k_proj(h).view(B, S, 1, 128))
        v = self.norm_v(self.v_proj(h).view(B, S, 1, 192))

        q = apply_rope(q, inplace=True)
        k = apply_rope(k, inplace=True)

        q_t = q.permute(0, 2, 1, 3)  # (B, 8, S, C)
        k_t = k.permute(0, 2, 1, 3)  # (B, 1, S, C)

        # Attention logits: bf16 matmul then cast to f32 (matches JAX BF16_BF16_F32)
        # JAX uses precision=BF16_BF16_F32: bf16 inputs, f32 accumulation, f32 output
        att = torch.matmul(q_t, k_t.transpose(-2, -1)).float()  # (B, 8, S, S)
        att = att / math.sqrt(128.0)

        if attention_bias is not None:
            att = att + attention_bias.float()

        logits_soft_cap = 5.0
        att = torch.tanh(att / logits_soft_cap) * logits_soft_cap

        attn_weights = F.softmax(att, dim=-1)

        # Value projection: bf16 matmul then cast back to compute dtype
        v_t = v.permute(0, 2, 1, 3)
        y = torch.matmul(attn_weights.to(compute_dtype), v_t).float()  # (B, 8, S, 192)
        y = y.to(compute_dtype)
        y = y.permute(0, 2, 1, 3).reshape(B, S, -1)

        y = self.linear_embedding(y)
        return self.final_norm(y)

class MLPBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = layers.RMSBatchNorm(d_model, channels_last=True)
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.final_norm = layers.RMSBatchNorm(d_model, channels_last=True)

    def forward(self, x):
        h = self.norm(x)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return self.final_norm(h)

class AttentionBiasBlock(nn.Module):
    def __init__(self, pair_dim):
        super().__init__()
        self.norm = layers.RMSBatchNorm(pair_dim, channels_last=True)
        self.proj = nn.Linear(pair_dim, 8, bias=False)

    def forward(self, x):
        # x: (B, s, s, D)
        h = F.gelu(self.norm(x))
        h = self.proj(h) # (B, s, s, 8)
        # Repeat 16x16
        h = torch.repeat_interleave(h, 16, dim=1)
        h = torch.repeat_interleave(h, 16, dim=2)
        return h.permute(0, 3, 1, 2) # (B, 8, S, S)

class SequenceToPairBlock(nn.Module):
    def __init__(self, d_model, pair_dim=128):
        super().__init__()
        self.d_model = d_model
        
        # 32 heads * 128 dim = 4096 params for q/k internal?
        # JAX uses hardcoded 32*128.
        self.num_heads = 32
        self.head_dim = 128
        
        self.pool = layers.Pool1d(kernel_size=16, stride=16, method='mean')
        self.norm_seq2pair = layers.LayerNorm(d_model, rms_norm=True)
        
        self.linear_q = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False)
        self.linear_k = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False)
        
        # Relative positions features -> 2*32 -> ...
        self.linear_pos_features = nn.Linear(2 * self.num_heads, self.num_heads * self.head_dim)
        
        self.q_r_bias = nn.Parameter(torch.zeros(1, 1, self.num_heads, self.head_dim))
        self.k_r_bias = nn.Parameter(torch.zeros(1, 1, self.num_heads, self.head_dim))
        
        self.linear_y_q = nn.Linear(d_model, self.head_dim, bias=False)
        self.linear_y_k = nn.Linear(d_model, self.head_dim, bias=False)
        
        self.linear_pair = nn.Linear(self.num_heads, self.head_dim) 

    def forward(self, x):
        # x: (B, S, D) - NLC format
        # Pool1d expects NCL, so transpose around pool call
        x_pooled = self.pool(x.transpose(1, 2)).transpose(1, 2)
        x_norm = self.norm_seq2pair(x_pooled)
        
        B, S_prime, _ = x_norm.shape
        
        q = self.linear_q(x_norm).view(B, S_prime, self.num_heads, self.head_dim)
        k = self.linear_k(x_norm).view(B, S_prime, self.num_heads, self.head_dim)
        
        # Relative positions (computed in float32 for precision, then cast to model dtype)
        range_vec = torch.arange(-S_prime, S_prime, device=x.device, dtype=torch.float32)
        pos_feat = _central_mask_features(torch.abs(range_vec), self.num_heads, _MAX_RELATIVE_DISTANCE // 16)
        sign = torch.sign(range_vec).unsqueeze(-1)
        pos_feat = torch.cat([pos_feat, sign * pos_feat], dim=-1) # (2S', 64)
        pos_feat = pos_feat.to(x.dtype)  # Match model dtype

        pos_encoding = self.linear_pos_features(pos_feat).view(2 * S_prime, self.num_heads, self.head_dim)
        
        term_q = torch.einsum('bqhc,phc->bhqp', q + self.q_r_bias, pos_encoding)
        term_k = torch.einsum('bkhc,phc->bhkp', k + self.k_r_bias, pos_encoding)
        
        rel_q_a = _shift(term_q, S_prime, S_prime)
        rel_k_a = _shift(term_k, S_prime, S_prime)
        
        rel_q_a = rel_q_a.permute(0, 2, 3, 1) # (B, S', S', H)
        rel_k_a = rel_k_a.permute(0, 3, 2, 1) # (B, S', S', H) from bhkp -> bpkh logic
        
        a = torch.einsum('bqhc,bkhc->bqkh', q, k) # (B, S', S', H)
        a = a + 0.5 * (rel_q_a + rel_k_a)
        
        # y branches
        x_gelu = F.gelu(x_norm)
        y_q = self.linear_y_q(x_gelu)
        y_k = self.linear_y_k(x_gelu)
        
        pair_act = self.linear_pair(a) + y_q.unsqueeze(2) + y_k.unsqueeze(1)
        return pair_act

class RowAttentionBlock(nn.Module):
    """Self-attention block applied along rows of pairwise representations.

    Matches JAX: alphagenome_research.model.attention.RowAttentionBlock

    JAX uses precision=BF16_BF16_F32 for einsum operations.
    """
    def __init__(self, pair_dim=128):
        super().__init__()
        self.norm = layers.LayerNorm(pair_dim, rms_norm=True)
        self.linear_q = nn.Linear(pair_dim, pair_dim, bias=False)
        self.linear_k = nn.Linear(pair_dim, pair_dim, bias=False)
        self.linear_v = nn.Linear(pair_dim, pair_dim)

    def forward(self, x, compute_dtype=None):
        if compute_dtype is None:
            compute_dtype = x.dtype
        x = x.to(compute_dtype)

        h = self.norm(x)
        q = self.linear_q(h)
        k = self.linear_k(h)
        v = self.linear_v(h)

        # Attention: bf16 einsum then cast to f32 (matches JAX BF16_BF16_F32)
        scale = 1.0 / math.sqrt(128.0)
        attn = torch.einsum('bpqf,bpkf->bpqk', q, k).float() * scale
        attn = F.softmax(attn, dim=-1)

        # Value projection: bf16 einsum then cast back
        out = torch.einsum('bpqk,bpkf->bpqf', attn.to(compute_dtype), v).float()
        return out.to(compute_dtype)

class PairMLPBlock(nn.Module):
    def __init__(self, pair_dim=128):
        super().__init__()
        self.norm = layers.LayerNorm(pair_dim, rms_norm=True)
        self.linear1 = nn.Linear(pair_dim, 2 * pair_dim)
        self.linear2 = nn.Linear(2 * pair_dim, pair_dim)
        
    def forward(self, x):
        h = self.norm(x)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.linear2(h)
        return h

class PairUpdateBlock(nn.Module):
    def __init__(self, d_model, pair_dim=128):
        super().__init__()
        self.seq2pair = SequenceToPairBlock(d_model, pair_dim)
        self.row_attn = RowAttentionBlock(pair_dim)
        self.pair_mlp = PairMLPBlock(pair_dim)

    def forward(self, x, pair_rep, compute_dtype=None):
        # x: (B, S, D)
        # pair_rep: (B, S/16, S/16, F)

        y = self.seq2pair(x)

        if pair_rep is None:
            pair_rep = y
        else:
            pair_rep = pair_rep + y

        pair_rep = pair_rep + self.row_attn(pair_rep, compute_dtype=compute_dtype)
        pair_rep = pair_rep + self.pair_mlp(pair_rep)

        return pair_rep
