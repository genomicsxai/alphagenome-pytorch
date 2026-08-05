from typing import Literal

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
    # Soft clip: where x > soft_clip_value, apply quadratic expansion
    x = torch.where(
        x > soft_clip_value,
        (x + soft_clip_value) ** 2 / (4 * soft_clip_value),
        x,
    )

    # Apply squashing inverse (power law expansion) for RNA-seq type heads
    if apply_squashing:
        x = torch.pow(x, 1.0 / 0.75)

    # Scale by track means and resolution
    if channels_last:
        # NLC: track_means (B, C) → (B, 1, C)
        x = x * (track_means[:, None, :] * resolution)
    else:
        # NCL: track_means (B, C) → (B, C, 1)
        x = x * (track_means[:, :, None] * resolution)

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
        return out + b.unsqueeze(1)


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
        out = torch.einsum('bcs,boc->bos', x.float(), w.float()).to(input_dtype)
        return out + b.unsqueeze(2)  # bias: (B, C_out, 1)

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

    def _organism_slot(self, organism_index):
        """Resolve which organism weight slot this head uses.

        A single-organism fine-tuned head (``num_organisms == 1``) has exactly
        one slot and is organism-agnostic: it uses slot 0 regardless of which
        organism was selected for the *trunk* embedding (e.g. mouse data
        forwards the trunk at index 1, but this head only has slot 0). A
        multi-organism head uses the index as given — an out-of-range index
        raises on indexing as usual, we do not mask it.
        """
        if self.num_organisms == 1:
            return torch.zeros_like(organism_index) if torch.is_tensor(organism_index) else 0
        return organism_index

    def _predict(self, x, organism_index, res_str):
        """Raw model prediction (in model space)."""
        # x: (B, C, S) - NCL format
        x = self.convs[res_str](x, organism_index)  # (B, T, S)

        # Residual Scale: (B, T) → (B, T, 1) for NCL broadcast
        scale = self.residual_scales[res_str][organism_index]

        # Softplus: softplus(x) * softplus(scale)
        x = F.softplus(x) * F.softplus(scale.unsqueeze(2))

        return x

    def unscale(self, x, organism_index, resolution, channels_last=True):
        """Unscales predictions to experimental data scale."""
        organism_index = self._organism_slot(organism_index)
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
        organism_index = self._organism_slot(organism_index)
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
        organism_index = self._organism_slot(organism_index)
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
                outputs[res] = scaled_pred
            else:
                outputs[res] = self.unscale(scaled_pred, organism_index, res, channels_last)

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

        return x

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
            "logits": logits,
            "probs": probs,
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
        else:
            logits = logits_ncl
            mask = self.track_mask[organism_index][:, :, None]

        predictions = torch.sigmoid(logits)

        return {
            "logits": logits,
            "predictions": predictions,
            "track_mask": mask,
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
        rope_init: str = "truncated_normal",
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
            shape = (self._num_organisms, 2, self._num_tissues, self._hidden_dim)
            if rope_init == "zeros":
                # Original (buggy) JAX init: zeros block gradient flow during finetuning.
                # Retained only for ablation experiments.
                params = torch.zeros(shape)
            else:
                # TruncatedNormal(0.1) matches the JAX reference init and the pretrained
                # weight distribution. See alphagenome_research commit e2acbfae.
                std = 0.1
                params = torch.nn.init.trunc_normal_(
                    torch.empty(shape), mean=0.0, std=std, a=-2 * std, b=2 * std,
                )
            return nn.Parameter(params)

        self.rope_params = nn.ParameterDict({
            "pos_donor": make_rope_params(),
            "pos_acceptor": make_rope_params(),
            "neg_donor": make_rope_params(),
            "neg_acceptor": make_rope_params(),
        })

    def _predict(self, embeddings_1bp, splice_site_positions, organism_index):
        """Core junction prediction. embeddings_1bp: (B, C, S) NCL format.

        Args:
            embeddings_1bp: (B, C, S)
            splice_site_positions: (B, 4, P) — [pos_donors, pos_acceptors, neg_donors, neg_acceptors]
            organism_index: (B,)

        Returns:
            pred_counts: (B, D, A, 2*T)
            splice_junction_mask: (B, D, A, 2*T) bool
        """
        splice_site_logits = self.conv(embeddings_1bp, organism_index)  # (B, H, S)

        def _index_embeddings(embedding, indices):
            # PyTorch non-contiguous advanced indexing: embedding[batch_idx, :, indices]
            # returns (B, P, H) because the sliced dim is placed after advanced dims.
            # Transpose to (B, H, P) so the result matches the (B, H, K) contract
            # expected by _predict_from_sparse_logits.
            Bsz = embedding.shape[0]
            batch_idx = torch.arange(Bsz, device=embedding.device).unsqueeze(1)
            return embedding[batch_idx, :, indices].transpose(1, 2)  # (B, P, H) → (B, H, P)

        assert splice_site_positions.shape[1] == 4
        pos_donor_idx    = splice_site_positions[:, 0, :]
        pos_acceptor_idx = splice_site_positions[:, 1, :]
        neg_donor_idx    = splice_site_positions[:, 2, :]
        neg_acceptor_idx = splice_site_positions[:, 3, :]

        pos_donor_logits    = _index_embeddings(splice_site_logits, pos_donor_idx)
        pos_acceptor_logits = _index_embeddings(splice_site_logits, pos_acceptor_idx)
        neg_donor_logits    = _index_embeddings(splice_site_logits, neg_donor_idx)
        neg_acceptor_logits = _index_embeddings(splice_site_logits, neg_acceptor_idx)

        return self._predict_from_sparse_logits(
            pos_donor_logits, pos_acceptor_logits,
            neg_donor_logits, neg_acceptor_logits,
            splice_site_positions, organism_index,
        )

    def _apply_rope_sparse(self, logits_nhk, global_positions, rope_key, organism_index):
        """Apply scale/offset + RoPE to pre-extracted sparse logits.

        Args:
            logits_nhk: (B, H, K) — logits already extracted at K positions.
            global_positions: (B, K) — global sequence coordinates used for RoPE frequencies.
            rope_key: Key into self.rope_params.
            organism_index: (B,)

        Returns:
            (B, K, T, H) tensor after RoPE.
        """
        x = logits_nhk.transpose(1, 2)                      # (B, K, H)
        batch_params = self.rope_params[rope_key][organism_index]  # (B, 2, T, H)
        scale  = batch_params[:, [0], :, :]                  # (B, 1, T, H)
        offset = batch_params[:, [1], :, :]
        x = scale * x[:, :, None, :] + offset                # (B, K, T, H)
        return apply_rope(
            x, global_positions,
            max_position=self._max_position_encoding_distance,
            inplace=True,
        )

    def _predict_from_sparse_logits(
        self,
        pos_donor_logits,    # (B, H, K)
        pos_acceptor_logits, # (B, H, K)
        neg_donor_logits,    # (B, H, K)
        neg_acceptor_logits, # (B, H, K)
        splice_site_positions,  # (B, 4, K) — global positions for RoPE + valid masking
        organism_index,
    ):
        """Compute junction predictions from pre-extracted per-position logits.

        Used by the sequence-parallel training path where each rank runs the 1x1
        conv locally and the K sparse position logits are gathered across ranks,
        avoiding a full-sequence all-gather of the 1bp embeddings.

        Args:
            pos_donor_logits: (B, H, K) — conv output already extracted at each position.
            pos_acceptor_logits: (B, H, K)
            neg_donor_logits: (B, H, K)
            neg_acceptor_logits: (B, H, K)
            splice_site_positions: (B, 4, K) — global positions; -1 = padding.
            organism_index: (B,)

        Returns:
            pred_counts: (B, D, A, 2*T)
            splice_junction_mask: (B, D, A, 2*T) bool
        """
        assert splice_site_positions.shape[1] == 4
        pos_donor_idx    = splice_site_positions[:, 0, :]
        pos_acceptor_idx = splice_site_positions[:, 1, :]
        neg_donor_idx    = splice_site_positions[:, 2, :]
        neg_acceptor_idx = splice_site_positions[:, 3, :]

        pd = self._apply_rope_sparse(pos_donor_logits,    pos_donor_idx,    "pos_donor",    organism_index)
        pa = self._apply_rope_sparse(pos_acceptor_logits, pos_acceptor_idx, "pos_acceptor", organism_index)
        nd = self._apply_rope_sparse(neg_donor_logits,    neg_donor_idx,    "neg_donor",    organism_index)
        na = self._apply_rope_sparse(neg_acceptor_logits, neg_acceptor_idx, "neg_acceptor", organism_index)

        pos_counts = F.softplus(torch.einsum("bdth,bath->bdat", pd, pa))
        neg_counts = F.softplus(torch.einsum("bdth,bath->bdat", nd, na))

        pos_mask = torch.einsum("bd,ba->bda", pos_donor_idx >= 0, pos_acceptor_idx >= 0)
        neg_mask = torch.einsum("bd,ba->bda", neg_donor_idx >= 0, neg_acceptor_idx >= 0)

        tissue_mask = self.tissue_mask[organism_index]
        pos_mask = pos_mask[:, :, :, None] * tissue_mask[:, None, None, :]
        neg_mask = neg_mask[:, :, :, None] * tissue_mask[:, None, None, :]

        splice_junction_mask = torch.cat([pos_mask, neg_mask], dim=-1)
        pred_counts = torch.cat([pos_counts, neg_counts], dim=-1)
        pred_counts = torch.where(splice_junction_mask, pred_counts, 0.0)

        return pred_counts, splice_junction_mask

    def forward(self, embeddings_1bp, organism_index, splice_site_positions=None, channels_last=True):
        """
        Args:
            embeddings_1bp: (B, S, C) if channels_last=True (NLC), or (B, C, S) if channels_last=False (NCL)
            organism_index: (B,)
            splice_site_positions: (B, 4, P) — required
            channels_last: whether embeddings_1bp is in NLC format (default True)

        Returns:
            Dict with pred_counts (B, D, A, 2*T), splice_site_positions, splice_junction_mask.
        """
        if splice_site_positions is None:
            raise ValueError("splice_site_positions is required")

        if channels_last:
            embeddings_1bp = embeddings_1bp.transpose(1, 2)  # (B, S, C) → (B, C, S)

        pred_counts, splice_junction_mask = self._predict(
            embeddings_1bp, splice_site_positions, organism_index
        )
        return {
            "pred_counts": pred_counts,
            "splice_site_positions": splice_site_positions,
            "splice_junction_mask": splice_junction_mask,
        }
