"""Sequence parallelism for AlphaGenome training.

Splits the input DNA sequence across GPUs instead of splitting the batch (DDP),
allowing longer sequences to be processed with constant per-GPU memory.

Architecture:
    LOCAL (per GPU):
        shard → encoder (unet-down) → all-gather → scatter (unet-up) → local heads

    GLOBAL (all GPUs):
        all-gather trunk → add organism emb → transformer → scatter back

    SPARSE (specific positions):
        gather donor/acceptor positions → splice junction head
"""

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_fn
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import AlphaGenome


class SequenceParallelism:
    """Sequence-parallel forward/backward for AlphaGenome.

    Splits the input sequence across GPUs to enable training on longer sequences
    and to reduce per-GPU memory usage by distributing activations.

    Args:
        overlap_highres: Context tokens shared between adjacent 1bp shards.
        overlap_lowres: Context tokens shared between adjacent 128bp transformer shards.
    """

    def __init__(
        self,
        overlap_highres: int = 1024,
        overlap_lowres: int = 8,
    ):
        """Initialize SequenceParallelism."""
        self.overlap_highres = overlap_highres
        self.overlap_lowres = overlap_lowres

    @property
    def world_size(self) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 1
        return dist.get_world_size()

    @property
    def rank(self) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 0
        return dist.get_rank()

    def shard_sequence(
        self,
        x: torch.Tensor,
        overlap: int,
        return_bounds: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[int, int]]:
        """Split the sequence dimension across ranks with symmetric overlap.

        Operates on the last dimension of x (NCL format: last dim = sequence).

        Args:
            x: Input tensor (..., L).
            overlap: Number of positions to include from adjacent shards on each side.
            return_bounds: If True, also return the (start, end) indices used.

        Returns:
            Local shard (..., L_local), or tuple of (shard, (start, end)) when
            return_bounds is True.
        """
        world = self.world_size
        rank = self.rank

        # Determine sequence length (last dimension)
        L = x.shape[-1]
        base = L // world

        # Calculate local bounds with overlap
        start = max(0, rank * base - overlap)
        end = min(L, (rank + 1) * base + overlap)

        if return_bounds:
            return x[..., start:end], (start, end)

        return x[..., start:end]

    def gather_full(
        self,
        x_local: torch.Tensor,
        overlap: int,
        expected_len: Optional[int] = None,
    ) -> torch.Tensor:
        """All-gather shards from every rank and trim the overlap regions.

        Args:
            x_local: Local shard (..., L_local).
            overlap: Overlap that was used in shard_sequence.
            expected_len: If provided, asserts that the reconstructed length matches.

        Returns:
            Full tensor (..., L) with overlap regions removed and shards concatenated.
        """
        world = self.world_size
        device = x_local.device
        
        # Gather all raw (overlapped) shards        
        # consider that shards can have different shapes
        # Use differentiable all_gather when grad is enabled (for encoder training via transformer)
        # Gather actual lengths from all ranks (may differ due to overlap on boundary ranks)
        local_len = torch.tensor(x_local.shape[-1], device=device, dtype=torch.long)
        shape_list = [torch.zeros_like(local_len) for _ in range(world)]
        dist.all_gather(shape_list, local_len)
        shapes = [int(s.item()) for s in shape_list]
        max_len = max(shapes)
        *leading_dims, _ = x_local.shape

        if torch.is_grad_enabled() and x_local.requires_grad:
            # dist_fn.all_gather is differentiable but requires equal-sized tensors.
            # Pad to max_len, gather, then unpad using the collected shapes.
            pad = max_len - x_local.shape[-1]
            x_padded = torch.nn.functional.pad(x_local, (0, pad))
            gathered_padded = list(dist_fn.all_gather(x_padded))
            shards = [g[..., :shapes[i]] for i, g in enumerate(gathered_padded)]
        else:
            shards = [
                torch.zeros(*leading_dims, shapes[i], device=device, dtype=x_local.dtype)
                for i in range(world)
            ]
            dist.all_gather(shards, x_local)
            
        # Trim overlaps per rank
        trimmed = []
        for rank_idx, shard in enumerate(shards):
            # Left trim except rank 0
            left = overlap if rank_idx > 0 else 0
            # Right trim except last rank
            right = shard.shape[-1] - overlap if rank_idx < (world - 1) else shard.shape[-1]
            
            shard_trimmed = shard[..., left:right]
            
            trimmed.append(shard_trimmed)

        # Concatenate into final full tensor
        full = torch.cat(trimmed, dim=-1)
        
        # Optional validation
        if expected_len is not None:
            assert full.shape[-1] == expected_len, (
                f"Expected length {expected_len}, got {full.shape[-1]}"
            )

        return full

    def subset_global_positions_locally(
        self,
        x_local: torch.Tensor,
        overlap: int,
        global_length: int,
        global_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the subset of global positions that reside on this rank's shard.

        Args:
            x_local: Local shard (B, C, L_local).
            overlap: Overlap used in shard_sequence.
            global_length: Total full-sequence length.
            global_indices: 1-D tensor of global position indices to extract.

        Returns:
            Tensor (B, C, K) where K is the number of global_indices that fall
            within this rank's non-overlapping region.  K=0 if none match.
        """
        rank = self.rank
        world = self.world_size

        device = x_local.device
        global_indices = global_indices.to(device)

        # Global coords of local region with overlap
        base = global_length // world
        region_start = max(0, rank * base - overlap)
        region_end = min(global_length, (rank + 1) * base + overlap)

        # Trim amounts based on rank position
        left_trim = overlap if rank > 0 else 0
        right_trim = overlap if rank < (world - 1) else 0

        region_start = region_start + left_trim
        region_end = region_end - right_trim  # non-inclusive

        # Mask global positions that fall in this region
        mask = (global_indices >= region_start) & (global_indices < region_end)

        if not mask.any():
            # No relevant positions on this rank → return empty slice
            B, C = x_local.shape[:2]
            return x_local.new_zeros(B, C, 0)

        positions_here = global_indices[mask]

        # Convert global to local indices
        local_positions = (positions_here - region_start).long()

        # Extract from local shard
        subset = x_local.index_select(dim=-1, index=local_positions)

        return subset

    def concat_across_ranks(self, x_local: torch.Tensor) -> torch.Tensor:
        """Concatenate variable-length tensors from all ranks.

        Handles the case where each rank contributes a different number of
        positions (e.g. sparse splice-site positions) by padding to the
        maximum local size before all_gather and then trimming.

        Args:
            x_local: Local tensor (B, C, K_local) where K_local may differ
                per rank.

        Returns:
            Concatenated tensor (B, C, sum(K_local across all ranks)).
        """
        world = self.world_size
        device = x_local.device

        # Get last dim size
        K_local = x_local.shape[-1]
        K_tensor = torch.tensor([K_local], device=device)

        # Gather sizes
        counts = [torch.zeros_like(K_tensor) for _ in range(world)]
        dist.all_gather(counts, K_tensor)
        counts = [int(c.item()) for c in counts]

        # Pad to max size for all_gather
        K_max = max(counts)
        B, C = x_local.shape[:2]

        padded = torch.zeros(B, C, K_max, device=device)
        padded[..., :K_local] = x_local

        # Gather padded tensors
        gathered = [torch.zeros_like(padded) for _ in range(world)]
        dist.all_gather(gathered, padded)

        # Remove padding and concatenate
        out = []
        for g, k in zip(gathered, counts):
            if k > 0:
                out.append(g[..., :k])

        return torch.cat(out, dim=-1)

    def gather_positions(
        self,
        x_local: torch.Tensor,
        overlap: int,
        global_length: int,
        global_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather embeddings at specific global positions from all ranks.

        Combines subset_global_positions_locally (each rank picks its own
        relevant positions) with concat_across_ranks (all ranks collect the
        full set).  Useful for sparse splice-site or donor/acceptor queries.

        Args:
            x_local: Local shard (B, C, L_local).
            overlap: Overlap used in shard_sequence.
            global_length: Total full-sequence length.
            global_indices: 1-D tensor of global positions to collect.

        Returns:
            Tensor (B, C, len(global_indices)) with embeddings for every
            requested position, ordered by global_indices.
        """
        subset_local = self.subset_global_positions_locally(
            x_local, overlap, global_length, global_indices
        )
        subset_global = self.concat_across_ranks(subset_local)

        return subset_global

    def forward(
        self,
        model: "AlphaGenome",
        sequence: torch.Tensor,
        organism_index: torch.Tensor,
        resolutions: Optional[Tuple[int, ...]] = (1, 128),
        original_length: Optional[int] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Any]:
        """Run a sequence-parallel forward pass through AlphaGenome.

        Mirrors _compute_embeddings_ncl but distributes the sequence across GPUs:
        the encoder and decoder run locally on each rank's shard; the transformer
        tower runs globally after an all-gather of the low-resolution trunk.

        Returns embeddings in NCL format, identical to model.encode(channels_last=False).

        Args:
            model: AlphaGenome model instance.
            sequence: One-hot encoded DNA (B, S, 4) in NLC format.
            organism_index: Organism index per sample (B,). 0=human, 1=mouse.
            resolutions: Resolutions to compute.  When 1 is absent, the decoder
                is skipped for efficiency.
            original_length: If the sequence was padded before this call, pass the
                original (unpadded) length here.  The output embeddings will be
                trimmed so each rank covers exactly original_length // world_size
                positions, discarding the padding-derived positions.

        Returns:
            Tuple of (embeddings_dict, pair_activations) where embeddings_dict
            maps resolution to NCL embeddings:
            {1: (B, 1536, S_local), 128: (B, 3072, S_local//128)}.
        """
        # Cast to compute dtype (mirrors _compute_embeddings_ncl)
        sequence = model.dtype_policy.cast_to_compute(sequence)
        
        # sequence input is NLC (B, S, 4); convert to NCL (B, 4, S) for last-dim sharding
        sequence_ncl = sequence.transpose(1, 2)  # (B, 4, S)
        L_full = sequence_ncl.shape[-1]

        # ===== LOCAL: Encoder =====
        # Shard along sequence dim (last dim in NCL)
        sequence_local_ncl = self.shard_sequence(sequence_ncl, self.overlap_highres)
        
        # Encoder expects NLC (B, S, 4) - transpose back
        sequence_local_nlc = sequence_local_ncl.transpose(1, 2)
        trunk_local, intermediates_local = model.encoder(sequence_local_nlc)
        # trunk_local: (B, 1536, S_local//128) NCL
        
        # ===== GLOBAL: Transformer =====
        # All-gather trunk in NCL format (last dim = sequence positions)
        expected_lowres_len = L_full // 128
        trunk_global = self.gather_full(
            trunk_local, self.overlap_lowres, expected_len=expected_lowres_len
        )
        # trunk_global: (B, 1536, S_full//128) NCL

        # Transpose to NLC for transformer (NLC is native for attention)
        trunk_global_nlc = trunk_global.transpose(1, 2)  # (B, S_full//128, 1536)

        # Add organism embedding (mirrors _compute_embeddings_ncl)
        org_emb = model.organism_embed(organism_index).unsqueeze(1)  # (B, 1, 1536)
        trunk_global_nlc = trunk_global_nlc + org_emb
        
        # Run transformer globally
        trunk_global_nlc, pair_activations = model.tower(
            trunk_global_nlc, compute_dtype=model.dtype_policy.compute_dtype
        )
        # trunk_global_nlc: (B, S_full//128, 1536) NLC

        # ===== LOCAL: Decoder =====
        # Transpose back to NCL for scatter (last dim = sequence positions)
        trunk_global_ncl = trunk_global_nlc.transpose(1, 2)  # (B, 1536, S_full//128) NCL

        # Scatter back to local ranks
        trunk_local_ncl = self.shard_sequence(trunk_global_ncl, self.overlap_lowres)
        # trunk_local_ncl: (B, 1536, S_local//128) NCL

        # 128bp embedder (always computed, mirrors _compute_embeddings_ncl)
        embeddings_128bp = model.embedder_128bp(
            trunk_local_ncl, organism_index, channels_last=False
        )  # (B, 3072, S_local//128) NCL

        # 1bp embedder (only if needed, mirrors _compute_embeddings_ncl)
        need_1bp = resolutions is None or 1 in resolutions
        if need_1bp:
            decoded_x = model.decoder(trunk_local_ncl, intermediates_local)
            embeddings_1bp = model.embedder_1bp(
                decoded_x, organism_index, skip_x=embeddings_128bp, channels_last=False
            )  # (B, 1536, S_local) NCL
        else:
            embeddings_1bp = None
            del intermediates_local

        # Trim overlap from output embeddings to match non-overlapping region this rank owns
        world = self.world_size
        rank = self.rank
        left_lo = self.overlap_lowres if rank > 0 else 0
        right_lo = self.overlap_lowres if rank < world - 1 else 0

        if left_lo or right_lo:
            r_end = embeddings_128bp.shape[-1] - right_lo if right_lo else embeddings_128bp.shape[-1]
            embeddings_128bp = embeddings_128bp[..., left_lo:r_end]

        if need_1bp and embeddings_1bp is not None:
            left_hi = left_lo * 128
            right_hi = right_lo * 128
            r_end = embeddings_1bp.shape[-1] - right_hi if right_hi else embeddings_1bp.shape[-1]
            embeddings_1bp = embeddings_1bp[..., left_hi:r_end]

        # If the sequence was padded before this call, trim output embeddings back
        # to the per-rank size that corresponds to the original (unpadded) length.
        if original_length is not None and original_length < L_full:
            world = self.world_size
            local_orig_lo = (original_length // world) // 128
            embeddings_128bp = embeddings_128bp[..., :local_orig_lo]
            if embeddings_1bp is not None:
                local_orig = original_length // world
                embeddings_1bp = embeddings_1bp[..., :local_orig]
            
        # Pair Embeddings (B, S, S, D) - different format, not NCL
        embeddings_pair = model.embedder_pair(pair_activations, organism_index)
        
        return embeddings_1bp, embeddings_128bp, embeddings_pair, need_1bp


def create_sequence_parallel_strategy(
    overlap_highres: int = 1024,
    overlap_lowres: int = 32,
) -> SequenceParallelism:
    """Create a SequenceParallelism instance with the given overlap sizes.

    Args:
        overlap_highres: Context tokens shared between adjacent 1bp shards.
        overlap_lowres: Context tokens shared between adjacent 128bp shards.

    Returns:
        Configured SequenceParallelism instance.
    """
    return SequenceParallelism(
        overlap_highres=overlap_highres,
        overlap_lowres=overlap_lowres,
    )


__all__ = [
    'SequenceParallelism',
    'create_sequence_parallel_strategy',
]