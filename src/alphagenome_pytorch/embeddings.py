import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from . import layers

def _parse_embedder_chunk_size():
    value = os.getenv("ALPHAGENOME_EMBEDDER_ACT_CHUNK_SIZE")
    if value is None or value == "":
        return 524288
    parsed = int(value)
    return max(parsed, 0)


def _gelu_inplace_chunked_ncl(x, chunk_size):
    """In-place GELU approximation on NCL tensor with optional sequence chunking."""
    coef = 1.702
    if chunk_size <= 0 or chunk_size >= x.shape[2]:
        x.mul_(torch.sigmoid(x * coef))
        return x

    for start in range(0, x.shape[2], chunk_size):
        end = min(start + chunk_size, x.shape[2])
        chunk = x[:, :, start:end]
        chunk.mul_(torch.sigmoid(chunk * coef))
    return x


class OutputEmbedder(nn.Module):
    """Output embedder using Conv1d for NCL format (B, C, S).

    Matches JAX `alphagenome_research.model.embeddings.OutputEmbedder`.

    Logic:
    1. Conv1d projection to output channels.
    2. Optional skip connection addition (with projection if needed).
    3. Add Organism Embedding.
    4. Norm + GELU.
    """

    def __init__(self, in_channels, out_channels, num_organisms=2):
        super().__init__()
        self.num_organisms = num_organisms
        self.out_channels = out_channels

        # Use Conv1d(k=1) instead of Linear - same math, native NCL
        # For 128bp: Input 1536 -> Output 3072
        # For 1bp: Input 768 -> Output 1536
        self.project_in = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # Skip projection - set externally if needed (e.g., for 1bp embedder)
        self.project_skip = None

        self.organism_embed = nn.Embedding(num_organisms, out_channels)
        self.norm = layers.RMSBatchNorm(channels=out_channels)
        self.inference_act_chunk_size = _parse_embedder_chunk_size()

    def forward(self, x, organism_index, skip_x=None, channels_last=False):
        # x: (B, C, S) - NCL format

        # Project main input
        x_proj = self.project_in(x)

        if skip_x is not None and self.project_skip is not None:
            # skip_x: (B, C_skip, S_skip)
            s_proj = self.project_skip(skip_x)

            # Upsample sequence if needed (dim 2 in NCL)
            repeat_factor = x_proj.shape[2] // s_proj.shape[2]
            if repeat_factor > 1:
                if torch.is_grad_enabled():
                    s_proj = s_proj.repeat_interleave(repeat_factor, dim=2)
                    x_proj = x_proj + s_proj
                else:
                    # Inference path: avoid materializing full repeated skip tensor.
                    # Add each low-res position directly to its repeated span in x_proj.
                    s_len = s_proj.shape[2]
                    for i in range(s_len):
                        start = i * repeat_factor
                        end = min(start + repeat_factor, x_proj.shape[2])
                        x_proj[:, :, start:end].add_(s_proj[:, :, i:i + 1])
            else:
                if torch.is_grad_enabled():
                    x_proj = x_proj + s_proj
                else:
                    x_proj.add_(s_proj)

        # Apply norm (use in-place ops during inference to reduce peak memory)
        if torch.is_grad_enabled():
            out = self.norm(x_proj)
        else:
            inv = self.norm.weight * torch.rsqrt(self.norm.running_var + self.norm.eps).to(x_proj.dtype)
            x_proj.mul_(inv.view(1, -1, 1)).add_(self.norm.bias.view(1, -1, 1))
            out = x_proj

        # Add organism embedding: (B, C) → (B, C, 1) for NCL broadcast
        emb = self.organism_embed(organism_index).unsqueeze(2)
        if torch.is_grad_enabled():
            out = out + emb
            out = layers.gelu(out)
        else:
            out.add_(emb)
            _gelu_inplace_chunked_ncl(out, self.inference_act_chunk_size)

        if channels_last:
            # (B, C, S) -> (B, S, C)
            out = out.transpose(1, 2)
            
        return out

class OutputPair(nn.Module):
    """Output embedder for pair activations (B, S, S, D).

    Note: Pair activations use a different format than sequence data.
    LayerNorm operates over the last dimension (features).
    """

    def __init__(self, dim=128, num_organisms=2):
        super().__init__()
        self.num_organisms = num_organisms
        self.organism_embed = nn.Embedding(num_organisms, dim)
        self.norm = layers.LayerNorm(normalized_shape=dim, rms_norm=True)

    def forward(self, x, organism_index):
        # x: (B, S, S, D) - pair activations
        # Symmetrize
        x = (x + x.transpose(1, 2)) / 2.0

        # Apply norm, then add organism embedding, then gelu
        x = self.norm(x)

        emb = self.organism_embed(organism_index)  # (B, D)
        x = x + emb[:, None, None, :]

        return layers.gelu(x)
