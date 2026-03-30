from typing import Literal
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from alphagenome_pytorch.attention import apply_rope

_SOFT_CLIP_VALUE = 10.0
TRUNK_DIM = 1536
EMBEDDING_128BP_DIM = 3072
DECODER_DIM = 768
PAIR_EMBEDDING_DIM = 128
CONTACT_MAPS_OUTPUT_TRACKS = 28
NUM_SPLICE_TISSUES = 367
SPLICE_USAGE_OUTPUT_TRACKS = NUM_SPLICE_TISSUES * 2

def _parse_head_conv_chunk_size():
    value = os.getenv("ALPHAGENOME_HEAD_CONV_CHUNK_SIZE")
    if value is None or value == "":
        return 524288
    parsed = int(value)
    return max(parsed, 0)


_HEAD_CONV_CHUNK_SIZE = _parse_head_conv_chunk_size()

def predictions_scaling(
    x: torch.Tensor,
    track_means: torch.Tensor,
    resolution: int,
    apply_squashing: bool,
    soft_clip_value: float = _SOFT_CLIP_VALUE,
    channels_last: bool = True,
) -> torch.Tensor:
    """Scales predictions to experimental data scale.

    Matches JAX: alphagenome_research.model.heads.predictions_scaling

    Args:
        x: Model predictions - NLC (B, S, C) if channels_last else NCL (B, C, S)
        track_means: Mean values per track (B, C)
        resolution: Bin resolution (1 or 128)
        apply_squashing: Whether to apply power law expansion (for RNA-seq)
        soft_clip_value: Value for soft clipping
        channels_last: If True, x is NLC. If False, x is NCL.

    Returns:
        Scaled predictions in experimental data space (same format as input)
    """
    if torch.is_grad_enabled():
        # Training path: keep autograd-compatible out-of-place ops.
        x = torch.where(
            x > soft_clip_value,
            (x + soft_clip_value) ** 2 / (4 * soft_clip_value),
            x,
        )
        if apply_squashing:
            x = torch.pow(x, 1.0 / 0.75)
        if channels_last:
            x = x * (track_means[:, None, :] * resolution)
        else:
            x = x * (track_means[:, :, None] * resolution)
        return x

    mask = x > soft_clip_value
    if mask.any():
        clipped = x[mask] + soft_clip_value   # 1-D, len = nnz(mask)
        clipped.pow_(2).div_(4 * soft_clip_value)
        x[mask] = clipped

    # Power-law expansion for RNA-seq (in-place).
    if apply_squashing:
        x.pow_(1.0 / 0.75)

    # Scale by track means × resolution.
    # track_means[:, None/None, :] is (B,1,C) or (B,C,1) — tiny.
    # mul_ writes back into x with no extra allocation.
    if channels_last:
        x.mul_(track_means[:, None, :] * resolution)
    else:
        x.mul_(track_means[:, :, None] * resolution)

    return x


def targets_scaling(
    x: torch.Tensor,
    track_means: torch.Tensor,
    resolution: int,
    apply_squashing: bool,
    soft_clip_value: float = _SOFT_CLIP_VALUE,
    channels_last: bool = True,
) -> torch.Tensor:
    """Scales targets from experimental data to model prediction space.

    Inverse of predictions_scaling. Used to scale targets before loss computation.
    Matches JAX: alphagenome_research.model.heads.targets_scaling

    Args:
        x: Targets in experimental space - NLC (B, S, C) if channels_last else NCL (B, C, S)
        track_means: Per-track scaling factors (B, C)
        resolution: Resolution multiplier (1 or 128)
        apply_squashing: Apply power law compression (only for RNA-seq)
        soft_clip_value: Value for soft clipping
        channels_last: If True, x is NLC. If False, x is NCL.

    Returns:
        Targets in model space (same format as input)
    """
    # Step 1: Normalize by track means and resolution
    if channels_last:
        # NLC: track_means (B, C) → (B, 1, C)
        x = x / (track_means[:, None, :] * resolution + 1e-8)
    else:
        # NCL: track_means (B, C) → (B, C, 1)
        x = x / (track_means[:, :, None] * resolution + 1e-8)

    # Step 2: Apply power law compression (RNA-seq only)
    if apply_squashing:
        x = torch.pow(x, 0.75)

    # Step 3: Soft clipping (inverse of quadratic expansion)
    x = torch.where(
        x > soft_clip_value,
        2.0 * torch.sqrt(x * soft_clip_value) - soft_clip_value,
        x,
    )

    return x


class MultiOrganismLinear(nn.Module):
    """Linear layer with organism-specific weights. Expects NLC format (B, S, C).

    Used for non-sequence operations like ContactMapsHead on pair activations.
    JAX: alphagenome_research.model.heads._MultiOrganismLinear
    """
    def __init__(
        self,
        in_features,
        out_features,
        num_organisms=2,
        init_scheme: Literal['truncated_normal', 'uniform'] = 'truncated_normal',
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_organisms = num_organisms
        self._init_scheme = init_scheme

        # We store weights as (num_organisms, in_features, out_features)
        # Note: PyTorch nn.Linear stores (out_features, in_features).
        # But here we are doing custom einsum logic anyway.
        # Let's stick to JAX shape for easier mapping, then transpose if needed for efficiency.
        # JAX shape: (num_organisms, in, out).
        self.weight = nn.Parameter(torch.empty(num_organisms, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_organisms, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_features)

        if self._init_scheme == 'truncated_normal':
            # Match JAX: TruncatedNormal for weights, zeros for bias
            nn.init.trunc_normal_(self.weight, std=stdv)
            nn.init.zeros_(self.bias)
        else:  # 'uniform'
            # Legacy PyTorch-style uniform initialization
            self.weight.data.uniform_(-stdv, stdv)
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, organism_index):
        # x: (B, S, in_features) - NLC format
        w = self.weight[organism_index]  # (B, In, Out)
        b = self.bias[organism_index]    # (B, Out)

        input_dtype = x.dtype
        out = torch.bmm(x.float(), w.float()).to(input_dtype)
        if torch.is_grad_enabled():
            return out + b.unsqueeze(1)
        out.add_(b.unsqueeze(1))
        return out


class MultiOrganismConv1d(nn.Module):
    """Organism-specific 1x1 conv for NCL format (B, C, S).

    Equivalent to JAX _MultiOrganismLinear which operates on NLC format.
    Using Conv1d avoids transpose overhead when data is already NCL.
    """

    def __init__(self, in_channels, out_channels, num_organisms=2,
                 init_scheme: Literal['truncated_normal', 'uniform'] = 'truncated_normal'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_organisms = num_organisms
        self._init_scheme = init_scheme

        # Weight: (num_organisms, out_channels, in_channels) - Conv1d convention
        self.weight = nn.Parameter(torch.empty(num_organisms, out_channels, in_channels))
        self.bias = nn.Parameter(torch.empty(num_organisms, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_channels)

        if self._init_scheme == 'truncated_normal':
            nn.init.trunc_normal_(self.weight, std=stdv)
            nn.init.zeros_(self.bias)
        else:  # 'uniform'
            self.weight.data.uniform_(-stdv, stdv)
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, organism_index):
        # x: (B, C_in, S) - NCL format
        w = self.weight[organism_index]  # (B, C_out, C_in)
        b = self.bias[organism_index]    # (B, C_out)

        # Batched 1x1 conv via einsum: (B, C_in, S) @ (B, C_out, C_in).T -> (B, C_out, S)
        input_dtype = x.dtype
        if (not torch.is_grad_enabled()) and _HEAD_CONV_CHUNK_SIZE > 0 and x.shape[-1] > _HEAD_CONV_CHUNK_SIZE:
            B, _, S = x.shape
            out = torch.empty(
                B, self.out_channels, S, device=x.device, dtype=input_dtype
            )
            w_float = w.float()
            for start in range(0, S, _HEAD_CONV_CHUNK_SIZE):
                end = min(start + _HEAD_CONV_CHUNK_SIZE, S)
                out_chunk = torch.einsum(
                    'bcs,boc->bos',
                    x[:, :, start:end].float(),
                    w_float,
                ).to(input_dtype)
                out_chunk.add_(b.unsqueeze(2))
                out[:, :, start:end] = out_chunk
            return out

        out = torch.einsum('bcs,boc->bos', x.float(), w.float()).to(input_dtype)
        if torch.is_grad_enabled():
            return out + b.unsqueeze(2)  # bias: (B, C_out, 1)
        out.add_(b.unsqueeze(2))
        return out

class GenomeTracksHead(nn.Module):
    """Predicts genome tracks at multiple resolutions.

    Internal computation is NCL for Conv1d efficiency.
    Outputs NLC format (B, S, T) to match JAX reference.

    Matches JAX: alphagenome_research.model.heads.GenomeTracksHead
    """

    def __init__(
        self,
        in_channels,
        num_tracks,
        resolutions=(1, 128),
        num_organisms=2,
        apply_squashing=False,
        track_means=None,
        init_scheme: Literal['truncated_normal', 'uniform'] = 'truncated_normal',
    ):
        super().__init__()
        self.num_tracks = num_tracks
        self.resolutions = sorted(resolutions)
        self.num_organisms = num_organisms
        self.apply_squashing = apply_squashing

        # in_channels controls input dim per requested resolution:
        # - None: default AlphaGenome dims (1bp->TRUNK_DIM, 128bp->EMBEDDING_128BP_DIM)
        # - int: same dim for all requested resolutions (e.g., encoder_only=TRUNK_DIM)
        # - dict[int, int]: explicit per-resolution override
        # - tuple/list: positional dims aligned with sorted resolutions
        if in_channels is None:
            resolved_in_channels = {1: TRUNK_DIM, 128: EMBEDDING_128BP_DIM}
        elif isinstance(in_channels, int):
            resolved_in_channels = {res: in_channels for res in self.resolutions}
        elif isinstance(in_channels, dict):
            missing_resolutions = [res for res in self.resolutions if res not in in_channels]
            if missing_resolutions:
                raise ValueError(
                    f"in_channels dict missing resolutions: {missing_resolutions}; "
                    f"required resolutions={self.resolutions}"
                )
            resolved_in_channels = {
                res: int(in_channels[res]) for res in self.resolutions
            }
        elif isinstance(in_channels, (tuple, list)):
            if len(in_channels) != len(self.resolutions):
                raise ValueError(
                    f"in_channels tuple/list length ({len(in_channels)}) must match "
                    f"resolutions length ({len(self.resolutions)})"
                )
            resolved_in_channels = {
                res: int(ch) for res, ch in zip(self.resolutions, in_channels)
            }
        else:
            raise TypeError(
                "in_channels must be None, int, dict[int, int], tuple[int, ...], "
                f"or list[int], got {type(in_channels).__name__}"
            )

        # Track means: (num_organisms, num_tracks)
        # Replace NaN with 0 to prevent gradient poisoning.
        if track_means is not None:
            sanitized_means = torch.nan_to_num(track_means, nan=0.0)
            self.register_buffer('track_means', sanitized_means)
        else:
            self.register_buffer('track_means', torch.ones(num_organisms, num_tracks))

        self.convs = nn.ModuleDict()
        self.residual_scales = nn.ParameterDict()

        for res in self.resolutions:
            res_str = str(res)
            dim = resolved_in_channels[res]

            self.convs[res_str] = MultiOrganismConv1d(dim, num_tracks, num_organisms, init_scheme=init_scheme)

            # learnt_scale: (num_organisms, num_tracks)
            self.residual_scales[res_str] = nn.Parameter(torch.ones(num_organisms, num_tracks))

    def _predict(self, x, organism_index, res_str):
        """Raw model prediction (in model space)."""
        # x: (B, C, S) - NCL format
        x = self.convs[res_str](x, organism_index)  # (B, T, S)

        # Residual Scale: (B, T) → (B, T, 1) for NCL broadcast
        scale = self.residual_scales[res_str][organism_index]

        if torch.is_grad_enabled():
            # Training path: standard ops, autograd needs full tensors in memory.
            # Softplus: softplus(x) * softplus(scale)
            x = F.softplus(x) * F.softplus(scale.unsqueeze(2))
        else:
            # 1. Compute scale factor once — tiny (B, T, 1), negligible memory.
            scale_factor = F.softplus(scale.unsqueeze(2))

            # 2. Apply softplus to x in chunks along S, writing results back into
            #    the same buffer.  Peak extra allocation = one chunk slice, not a
            #    full duplicate of x.
            chunk_size = int(os.getenv("ALPHAGENOME_SOFTPLUS_CHUNK_SIZE", "262144"))
            S = x.shape[2]
            if chunk_size <= 0 or chunk_size >= S:
                x = F.softplus(x)
            else:
                for start in range(0, S, chunk_size):
                    end = min(start + chunk_size, S)
                    x[:, :, start:end] = F.softplus(x[:, :, start:end])

            # 3. Multiply by scale in-place — no extra allocation.
            x.mul_(scale_factor)

        return x

    def unscale(self, x, organism_index, resolution, channels_last=True):
        """Unscales predictions to experimental data scale."""
        track_means = self.track_means[organism_index]  # (B, num_tracks)
        return predictions_scaling(
            x,
            track_means=track_means,
            resolution=resolution,
            apply_squashing=self.apply_squashing,
            channels_last=channels_last,
        )

    def scale(self, x, organism_index, resolution, channels_last=True):
        """Scales targets from experimental to model prediction space.

        Args:
            x: Targets in experimental space.
               NLC (B, S, T) if channels_last else NCL (B, T, S)
            organism_index: Organism indices (B,)
            resolution: Resolution (1 or 128)
            channels_last: If True, x is NLC. If False, x is NCL.

        Returns:
            Targets in model space (same format as input)
        """
        track_means = self.track_means[organism_index]  # (B, num_tracks)
        return targets_scaling(
            x,
            track_means=track_means,
            resolution=resolution,
            apply_squashing=self.apply_squashing,
            channels_last=channels_last,
        )

    def forward(self, embeddings_dict, organism_index, return_scaled=False, channels_last=True):
        """Returns predictions in experimental or model scale.

        Args:
            embeddings_dict: Dict mapping resolution to embeddings (B, C, S) NCL
            organism_index: Organism indices (B,)
            return_scaled: If True, return model space (for loss).
                           If False, return experimental space (for inference).
            channels_last: Output format.
                - True (default): NLC format (B, S, T) - user-friendly, matches JAX
                - False: NCL format (B, T, S) - for training efficiency (0 transposes)

        Returns:
            Dict mapping resolution to predictions in specified format
        """
        outputs = {}
        for res in self.resolutions:
            if res not in embeddings_dict:
                continue
            res_str = str(res)
            emb = embeddings_dict[res]  # (B, C, S) NCL

            # Get raw predictions (model space) - NCL internally
            scaled_pred = self._predict(emb, organism_index, res_str)  # (B, T, S) NCL

            # Transpose to NLC if channels_last
            if channels_last:
                scaled_pred = scaled_pred.transpose(1, 2)  # (B, S, T)

            if return_scaled:
                outputs[res] = scaled_pred.cpu()
            else:
                outputs[res] = self.unscale(scaled_pred, organism_index, res, channels_last).cpu()

        return outputs


class ContactMapsHead(nn.Module):
    """
    Predicts contact maps from pairwise embeddings.
    JAX: alphagenome_research.model.heads.ContactMapsHead
    """
    def __init__(
        self,
        in_features=PAIR_EMBEDDING_DIM,
        num_tracks=CONTACT_MAPS_OUTPUT_TRACKS,
        num_organisms=2,
    ):
        super().__init__()
        self.num_tracks = num_tracks
        self.num_organisms = num_organisms
        self.linear = MultiOrganismLinear(in_features, num_tracks, num_organisms)

    def forward(self, pair_embeddings, organism_index, channels_last=True):
        # pair_embeddings: (B, S, S, D) where D=128
        # organism_index: (B,)
        B, S1, S2, D = pair_embeddings.shape

        # Reshape for MultiOrganismLinear: (B, S*S, D)
        x = pair_embeddings.view(B, S1 * S2, D)

        # Apply linear: (B, S*S, num_tracks)
        x = self.linear(x, organism_index)

        # Reshape back: (B, S, S, num_tracks)
        x = x.view(B, S1, S2, self.num_tracks)

        if not channels_last:
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, T, S, S)

        return x.cpu()

class SpliceSitesClassificationHead(nn.Module):
    """Predicts splice site classification.

    Internal computation is NCL for Conv1d efficiency.
    Outputs NLC format (B, S, 5) to match JAX reference.

    Classes: Donor+, Acceptor+, Donor-, Acceptor-, Not a splice site
    JAX: alphagenome_research.model.heads.SpliceSitesClassificationHead
    """

    def __init__(self, in_channels=TRUNK_DIM, num_organisms=2):
        super().__init__()
        self.num_organisms = num_organisms
        self.conv = MultiOrganismConv1d(
            in_channels=in_channels,
            out_channels=5,  # 5 classes
            num_organisms=num_organisms
        )

    def forward(self, embeddings_1bp, organism_index, channels_last=True):
        # embeddings_1bp: (B, C, S) - NCL format (internal)
        logits_ncl = self.conv(embeddings_1bp, organism_index)  # (B, 5, S)

        if channels_last:
            # Transpose to NLC: (B, 5, S) -> (B, S, 5)
            logits = logits_ncl.transpose(1, 2)
            # Softmax over classes (dim=-1 in NLC)
            probs = F.softmax(logits, dim=-1)
        else:
            logits = logits_ncl
            # Softmax over classes (dim=1 in NCL)
            probs = F.softmax(logits, dim=1)

        return {
            "logits": logits.cpu(),
            "probs": probs.cpu(),
        }

class SpliceSitesUsageHead(nn.Module):
    """Predicts splice site usage.

    Internal computation is NCL for Conv1d efficiency.
    Outputs NLC format (B, S, T) to match JAX reference.

    Outputs proportion of RNA using each splice site.
    JAX: alphagenome_research.model.heads.SpliceSitesUsageHead
    """

    def __init__(
        self,
        in_channels=TRUNK_DIM,
        num_output_tracks=SPLICE_USAGE_OUTPUT_TRACKS,
        num_organisms=2,
        num_tracks_per_organism=None,
    ):
        super().__init__()
        self.num_organisms = num_organisms
        self.num_output_tracks = num_output_tracks

        # keep a fixed output width and use per-organism masks to
        # ignore padded channels in loss/metrics.
        if num_tracks_per_organism is None:
            num_tracks_per_organism = [num_output_tracks] * num_organisms
        if len(num_tracks_per_organism) != num_organisms:
            raise ValueError(
                f"num_tracks_per_organism length ({len(num_tracks_per_organism)}) "
                f"must equal num_organisms ({num_organisms})"
            )

        for org_idx, tracks in enumerate(num_tracks_per_organism):
            if tracks < 0 or tracks > num_output_tracks:
                raise ValueError(
                    f"num_tracks_per_organism[{org_idx}]={tracks} must be in "
                    f"[0, {num_output_tracks}]"
                )

        # Computed on-the-fly from config, not learned - exclude from state_dict
        track_mask = torch.arange(num_output_tracks)[None, :] < torch.tensor(
            list(num_tracks_per_organism),
            dtype=torch.long,
        )[:, None]
        self.register_buffer('track_mask', track_mask, persistent=False)

        self.conv = MultiOrganismConv1d(
            in_channels=in_channels,
            out_channels=num_output_tracks,  # NUM_SPLICE_TISSUES * 2 strands
            num_organisms=num_organisms
        )

    def forward(self, embeddings_1bp, organism_index, channels_last=True):
        # embeddings_1bp: (B, C, S) - NCL format (internal)
        logits_ncl = self.conv(embeddings_1bp, organism_index)  # (B, T, S)

        if channels_last:
            # Transpose to NLC: (B, T, S) -> (B, S, T)
            logits = logits_ncl.transpose(1, 2)
            mask = self.track_mask[organism_index][:, None, :]
            chunk_dim = 1  # chunk over S
        else:
            logits = logits_ncl
            mask = self.track_mask[organism_index][:, :, None]
            chunk_dim = 2  # chunk over S

        if torch.is_grad_enabled() or _HEAD_CONV_CHUNK_SIZE <= 0 or logits.size(chunk_dim) <= _HEAD_CONV_CHUNK_SIZE:
            predictions = torch.sigmoid(logits).cpu()
        else:
            chunks = []
            for start in range(0, logits.size(chunk_dim), _HEAD_CONV_CHUNK_SIZE):
                end = start + _HEAD_CONV_CHUNK_SIZE
                chunk = logits.narrow(chunk_dim, start, min(_HEAD_CONV_CHUNK_SIZE, logits.size(chunk_dim) - start))
                chunks.append(torch.sigmoid(chunk).cpu())
            predictions = torch.cat(chunks, dim=chunk_dim)

        return {
            "logits": logits.cpu(),
            "predictions": predictions.cpu(),
            "track_mask": mask.cpu(),
        }

class SpliceSitesJunctionHead(nn.Module):
    """Predicts splice junction read counts. Expects NCL format (B, C, S).

    JAX: alphagenome_research.model.heads.SpliceSitesJunctionHead
    """
    def __init__(
        self,
        in_channels=TRUNK_DIM,
        hidden_dim=DECODER_DIM,
        num_tissues=NUM_SPLICE_TISSUES,
        num_organisms=2,
        num_tracks_per_organism=None,
    ):
        super().__init__()
        self._num_organisms = num_organisms
        self._num_tissues = num_tissues
        self._in_channels = in_channels
        self._hidden_dim = hidden_dim
        self._max_position_encoding_distance = int(2**20)

        # Precompute tissue mask per organism (matches JAX get_multi_organism_track_mask).
        # If num_tracks_per_organism is not specified, all organisms use num_tissues (no masking).
        if num_tracks_per_organism is None:
            num_tracks_per_organism = [num_tissues] * num_organisms
        if len(num_tracks_per_organism) != num_organisms:
            raise ValueError(
                f"num_tracks_per_organism length ({len(num_tracks_per_organism)}) "
                f"must equal num_organisms ({num_organisms})"
            )
        for org_idx, tracks in enumerate(num_tracks_per_organism):
            if tracks < 0 or tracks > num_tissues:
                raise ValueError(
                    f"num_tracks_per_organism[{org_idx}]={tracks} must be in "
                    f"[0, {num_tissues}]"
                )

        # Computed on-the-fly from config, not learned - exclude from state_dict
        tissue_mask = torch.arange(num_tissues)[None, :] < torch.tensor(
            list(num_tracks_per_organism),
            dtype=torch.long,
        )[:, None]
        self.register_buffer('tissue_mask', tissue_mask, persistent=False)

        self.conv = MultiOrganismConv1d(
            in_channels=self._in_channels,
            out_channels=self._hidden_dim,
            num_organisms=self._num_organisms
        )

        def make_rope_params():
            return nn.Parameter(torch.zeros(
                self._num_organisms, 2, self._num_tissues, self._hidden_dim
            ))

        self.rope_params = nn.ParameterDict({
            "pos_donor": make_rope_params(),
            "pos_acceptor": make_rope_params(),
            "neg_donor": make_rope_params(),
            "neg_acceptor": make_rope_params(),
        })

    def forward(self, embeddings_1bp, organism_index, channels_last=True, **kwargs):
        """
        Args:
            embeddings_1bp: (B, C, S) - NCL format
            organism_index: (B,)
            splice_site_positions: (B, 4, P) - required kwarg

        Returns:
            Dict with pred_counts (B, P, P, 2*T), positions, mask
        """
        splice_site_positions = kwargs.get("splice_site_positions", None)
        if splice_site_positions is None:
            raise ValueError("splice_site_positions is required")

        def _predict(embeddings_1bp, splice_site_positions, organism_index):
            # embeddings_1bp: (B, C, S), splice_site_positions: (B, 4, P)
            assert splice_site_positions.shape[1] == 4
            pos_donor_idx = splice_site_positions[:, 0, :]
            pos_acceptor_idx = splice_site_positions[:, 1, :]
            neg_donor_idx = splice_site_positions[:, 2, :]
            neg_acceptor_idx = splice_site_positions[:, 3, :]

            # Project: (B, C, S) → (B, H, S)
            splice_site_logits = self.conv(embeddings_1bp, organism_index)

            def _index_embeddings(embedding, indices):
                """Select embeddings at positions. embedding: (B, H, S), indices: (B, P)"""
                B, H, S = embedding.shape
                batch_idx = torch.arange(B, device=embedding.device).unsqueeze(1)
                # Index along S dimension: embedding[b, :, indices[b, p]] → (B, P, H)
                # PyTorch advanced indexing: broadcast indices give leading dims, : gives trailing
                return embedding[batch_idx, :, indices]  # (B, P, H)

            def _apply_rope(embedding, indices, params, organism_index):
                x = _index_embeddings(embedding, indices)  # (B, P, H)
                batch_params = params[organism_index]  # (B, 2, T, H)
                scale = batch_params[:, [0], :, :]
                offset = batch_params[:, [1], :, :]
                x = scale * x[:, :, None, :] + offset  # (B, P, T, H)
                return apply_rope(
                    x, indices,
                    max_position=self._max_position_encoding_distance,
                    inplace=True,
                )

            pos_donor_logits = _apply_rope(
                splice_site_logits, pos_donor_idx,
                self.rope_params["pos_donor"], organism_index
            )
            pos_acceptor_logits = _apply_rope(
                splice_site_logits, pos_acceptor_idx,
                self.rope_params["pos_acceptor"], organism_index
            )
            neg_donor_logits = _apply_rope(
                splice_site_logits, neg_donor_idx,
                self.rope_params["neg_donor"], organism_index
            )
            neg_acceptor_logits = _apply_rope(
                splice_site_logits, neg_acceptor_idx,
                self.rope_params["neg_acceptor"], organism_index
            )

            pos_counts = F.softplus(torch.einsum(
                "bdth,bath->bdat", pos_donor_logits, pos_acceptor_logits
            ))
            neg_counts = F.softplus(torch.einsum(
                "bdth,bath->bdat", neg_donor_logits, neg_acceptor_logits
            ))

            pos_mask = torch.einsum("bd,ba->bda", pos_donor_idx >= 0, pos_acceptor_idx >= 0)
            neg_mask = torch.einsum("bd,ba->bda", neg_donor_idx >= 0, neg_acceptor_idx >= 0)

            tissue_mask = self.tissue_mask[organism_index]

            pos_mask = pos_mask[:, :, :, None] * tissue_mask[:, None, None, :]
            neg_mask = neg_mask[:, :, :, None] * tissue_mask[:, None, None, :]

            splice_junction_mask = torch.cat([pos_mask, neg_mask], dim=-1)
            pred_counts = torch.cat([pos_counts, neg_counts], dim=-1)
            pred_counts = torch.where(splice_junction_mask, pred_counts, 0.0)

            return pred_counts, splice_junction_mask

        pred_counts, splice_junction_mask = _predict(
            embeddings_1bp, splice_site_positions, organism_index
        )
        return {
            "pred_counts": pred_counts.cpu(),
            "splice_site_positions": splice_site_positions.cpu(),
            "splice_junction_mask": splice_junction_mask.cpu(),
        }
