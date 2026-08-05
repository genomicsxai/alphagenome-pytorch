"""Unified training utilities for AlphaGenome fine-tuning.

Provides common training functions for both RNA-seq and ATAC-seq modalities.
Includes enhanced versions with DDP support, profiling, and Pearson R metrics.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm

from alphagenome_pytorch import losses
from alphagenome_pytorch.losses import (
    multinomial_loss,
    cross_entropy_loss,
    cross_entropy_loss_normalized,
    cross_entropy_loss_from_logits,
    binary_crossentropy_from_logits,
    poisson_loss,
)
from alphagenome_pytorch.heads import (
    SpliceSitesClassificationHead,
    SpliceSitesUsageHead,
    SpliceSitesJunctionHead,
)

# Number of segments for multinomial loss computation.
# AlphaGenome divides sequences into 8 equal segments for numerical stability.
NUM_SEGMENTS = 8

# Tuple of all splice head types for isinstance checks
SPLICE_HEAD_TYPES = (SpliceSitesClassificationHead, SpliceSitesUsageHead, SpliceSitesJunctionHead)

if TYPE_CHECKING:
    from torch.optim import Optimizer


def collate_genomic(
    batch: list[tuple[Tensor, dict[int, Tensor]]],
) -> tuple[Tensor, dict[int, Tensor]]:
    """Collate function for genomic fine-tuning datasets.

    Args:
        batch: List of (sequence, targets_dict) tuples from dataset.

    Returns:
        Tuple of (sequences, targets_dict) where:
            - sequences: Stacked sequences tensor (batch, seq_len, 4)
            - targets_dict: Dict mapping resolution to targets (batch, out_len, n_tracks)

    Example:
        >>> batch = [(seq1, {1: t1_1bp, 128: t1_128bp}), (seq2, {1: t2_1bp, 128: t2_128bp})]
        >>> sequences, targets = collate_genomic(batch)
        >>> targets[1].shape, targets[128].shape
    """
    sequences = torch.stack([item[0] for item in batch])

    # Targets are always a dict
    first_targets = batch[0][1]
    targets_dict: dict[int, Tensor] = {}
    for res in first_targets.keys():
        targets_dict[res] = torch.stack([item[1][res] for item in batch])

    return sequences, targets_dict


def _top_k_positions_from_logits(logits_ncl: torch.Tensor, top_k: int) -> torch.Tensor:
    """Derive splice-site positions from classification head logits via top-k selection.

    Args:
        logits_ncl: (B, 5, S) — NCL logits from SpliceSitesClassificationHead
                    (channels: 0=Donor+, 1=Acceptor+, 2=Donor-, 3=Acceptor-).
        top_k: Maximum number of positions to select per role per batch item.

    Returns:
        positions: (B, 4, top_k) int32 tensor — [pos_donors, pos_acceptors,
                   neg_donors, neg_acceptors], padded with -1 where fewer than
                   top_k sites exist.
    """
    B, C, S = logits_ncl.shape
    device = logits_ncl.device
    positions = torch.full((B, 4, top_k), -1, dtype=torch.int32, device=device)
    k = min(top_k, S)
    for role_idx in range(4):  # Donor+, Acceptor+, Donor-, Acceptor-
        scores = logits_ncl[:, role_idx, :]  # (B, S)
        topk_idx = torch.topk(scores, k, dim=-1, sorted=True).indices
        # Sort positions in ascending genomic order so RoPE distances are stable.
        sorted_idx, _ = topk_idx.sort(dim=-1)
        positions[:, role_idx, :k] = sorted_idx.to(torch.int32)
    return positions


def _gather_sparse_logits(
    logits_local: torch.Tensor,
    positions_bk: torch.Tensor,
    sequence_parallel: Any,
    global_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Gather 1x1-conv logits at sparse global positions across SP ranks.

    Each rank holds a shard of the 1bp conv output (B, H, S_local). This
    function collects, for every batch item, the logit vectors at K requested
    global positions by having each rank contribute the positions that fall
    within its shard, then all-gathering the results.

    Args:
        logits_local: (B, H, S_local) — local shard of conv output.
        positions_bk: (B, K) — global positions; -1 = padding.
        sequence_parallel: SequenceParallelism instance.
        global_length: Total (unpadded) sequence length S.
        device: Target device.

    Returns:
        (B, H, K) tensor; columns corresponding to -1 positions are zeros.
    """
    B, H, K = logits_local.shape[0], logits_local.shape[1], positions_bk.shape[1]
    result = torch.zeros(B, H, K, device=device, dtype=logits_local.dtype)

    for b in range(B):
        pos_b = positions_bk[b]           # (K,)
        valid_mask = pos_b >= 0           # (K,) bool
        valid_pos = pos_b[valid_mask]     # (K_valid,)

        if valid_pos.numel() == 0:
            continue

        # Sort so gather_positions returns embeddings in ascending-position order
        # (each rank owns a contiguous range; rank-order concat = global order
        # only when positions are sorted).
        sorted_pos, sort_idx = torch.sort(valid_pos.long())
        gathered = sequence_parallel.gather_positions(
            logits_local[b : b + 1],      # (1, H, S_local)
            overlap=0,                    # overlaps already stripped from emb_1bp
            global_length=global_length,
            global_indices=sorted_pos,
        )                                 # (1, H, K_valid) in sorted-position order

        # Unsort to restore original position ordering, then scatter into result.
        unsort_idx = torch.argsort(sort_idx)
        result[b : b + 1, :, valid_mask] = gathered[:, :, unsort_idx]

    return result


def _call_splice_junction_head_sp(
    head,
    emb_local: torch.Tensor,
    annotated_positions,
    organism_idx: torch.Tensor,
    sequence_parallel: Any,
    global_length: int,
    junction_top_k: int | None,
    cls_head,
    device: torch.device,
) -> dict:
    """Sequence-parallel forward pass for SpliceSitesJunctionHead.

    Avoids the O(B * C * S) memory cost of a full 1bp-embedding all-gather by
    exploiting the fact that the junction head's first layer is a 1x1 conv
    (MultiOrganismConv1d): each rank runs the conv locally on its shard, then
    only the K sparse per-position logits are gathered across ranks.

    Args:
        head: SpliceSitesJunctionHead (unwrapped from DDP).
        emb_local: (B, C, S_local) — local 1bp embedding shard (overlaps stripped).
        annotated_positions: (B, 4, K) global positions from targets_dict, or None
            when junction_top_k is set (predicted mode).
        organism_idx: (B,) or (B, 1).
        sequence_parallel: SequenceParallelism instance.
        global_length: Total (unpadded) sequence length.
        junction_top_k: If set, derive positions from classification head (predicted
            mode); otherwise use annotated_positions (annotated mode).
        cls_head: SpliceSitesClassificationHead (required when junction_top_k set).
        device: Target device.

    Returns:
        Dict compatible with _compute_splice_loss: {pos_counts, neg_counts, positions}.
    """
    org = organism_idx[:, 0] if organism_idx.ndim > 1 else organism_idx
    org = torch.zeros_like(org)

    # ── Predicted mode: derive global positions from classification head ─────
    if junction_top_k is not None:
        if cls_head is None:
            raise ValueError(
                "junction_top_k requires cls_head for predicted-mode SP forward."
            )
        # cls head is also 1x1 conv → run locally, then gather the (B, 5, S) logits.
        # 5*S floats per sample (~5 MB for S=1M) — cheap compared to 1536*S.
        cls_logits_local = cls_head.conv(emb_local, org)   # (B, 5, S_local)
        cls_logits_full  = sequence_parallel.gather_full(
            cls_logits_local, overlap=0,
        )                                                   # (B, 5, S)
        positions = _top_k_positions_from_logits(cls_logits_full, junction_top_k)  # (B, 4, K)
    else:
        if annotated_positions is None:
            return {}
        positions = annotated_positions.to(device)

    # Clamp -1 padding to 0 (safe dummy index; masked out in loss anyway).
    positions_clamped = positions.clamp(min=0)

    # ── Apply junction head's 1x1 conv locally ───────────────────────────────
    logits_local = head.conv(emb_local, org)   # (B, H, S_local)

    # ── Gather sparse logits at the K positions across ranks ─────────────────
    gathered = []
    for role in range(4):   # pos_donor, pos_acceptor, neg_donor, neg_acceptor
        gathered.append(
            _gather_sparse_logits(
                logits_local,
                positions_clamped[:, role, :],   # (B, K)
                sequence_parallel,
                global_length,
                device,
            )
        )   # each: (B, H, K)

    # ── Compute predictions using pre-extracted sparse logits ────────────────
    pred_counts, splice_junction_mask = head._predict_from_sparse_logits(
        *gathered, positions, org,
    )

    n_tissues = head._num_tissues
    return {
        "pos_counts": pred_counts[..., :n_tissues],
        "neg_counts": pred_counts[..., n_tissues:],
        "positions":  positions,
    }


def _call_splice_head(
    head,
    embeddings_dict,
    organism_idx,
    positions,
    channels_last,
    cls_head=None,
    junction_top_k: int | None = None,
):
    """Call a splice head with the training-loop's embeddings_dict interface.

    Unwraps embeddings_dict[1] and calls the correct forward signature per head type.

    Args:
        head: SpliceSitesClassificationHead, SpliceSitesUsageHead, or SpliceSitesJunctionHead.
        embeddings_dict: Dict with key 1 → embeddings tensor (B, C, S) or (B, S, C).
        organism_idx: Organism indices, shape (B,) or (B, 1).
        positions: Annotated splice-site positions (B, 4, K) or None.
            Ignored for SpliceSitesJunctionHead when junction_top_k is set.
        channels_last: If True, embeddings are (B, S, C); if False, (B, C, S).
        cls_head: SpliceSitesClassificationHead used to derive positions when
            junction_top_k is not None. Required when junction_top_k is set.
        junction_top_k: If set, positions for SpliceSitesJunctionHead are derived
            from the top-k scoring sites predicted by cls_head rather than from
            the annotated positions tensor.

    Returns:
        Dict compatible with _compute_splice_loss():
        - {1: logits} for classification/usage heads
        - {pos_counts: ..., neg_counts: ...} for junction head
        - {} if junction head and no positions available
    """
    if 1 not in embeddings_dict:
        available_keys = list(embeddings_dict.keys())
        raise ValueError(
            f"embeddings_dict missing key 1 for splice heads. Available: {available_keys}. "
            f"Make sure resolutions include 1bp for splice modalities."
        )
    emb = embeddings_dict[1]
    org = organism_idx[:, 0] if organism_idx.ndim > 1 else organism_idx
    org = torch.zeros_like(org)

    if emb.ndim != 3:
        raise ValueError(f"Expected 3D embeddings for splice heads, got shape {emb.shape}")

    if isinstance(head, SpliceSitesJunctionHead):
        if junction_top_k is not None:
            if cls_head is None:
                raise ValueError(
                    "junction_top_k requires cls_head (SpliceSitesClassificationHead) "
                    "to be passed to _call_splice_head."
                )
            # Run classification head to get per-position scores; always NCL internally.
            emb_for_cls = emb if not channels_last else emb.transpose(1, 2)
            cls_out = cls_head(emb_for_cls, org, channels_last=False)
            positions = _top_k_positions_from_logits(cls_out["logits"], junction_top_k)
        if positions is None:
            return {}
        # Clamp -1 padding to 0 to avoid PyTorch negative indexing wrapping.
        # Padded positions use -1, but negative indices wrap to last position in PyTorch.
        # Clamping to 0 ensures a safe dummy index; output predictions are masked anyway.
        positions_clamped = positions.clamp(min=0)
        out = head(emb, org, splice_site_positions=positions_clamped, channels_last=channels_last)
        n_tissues = head._num_tissues
        return {
            "pos_counts": out["pred_counts"][..., :n_tissues],
            "neg_counts": out["pred_counts"][..., n_tissues:],
            "positions":  positions,  # (B, 4, K) — present only in predicted mode
        }
    else:
        out = head(emb, org, channels_last=channels_last)
        logits = out["logits"]  # (B, S, C) if channels_last, else (B, C, S)
        # Always transpose to NLC (B, S, C) for training loop compatibility
        if channels_last:
            # Already NLC, no transpose needed
            pass
        else:
            # NCL to NLC: (B, C, S) → (B, S, C)
            logits = logits.transpose(1, 2)
        return {1: logits}


def _ce_loss_with_smoothing(pred: torch.Tensor, target: torch.Tensor, label_smoothing: float, n_classes: int) -> torch.Tensor:
    target_smooth = (1.0 - label_smoothing) * target.float() + label_smoothing / n_classes
    mask = target.any(dim=-1, keepdim=True).expand_as(pred)
    return cross_entropy_loss_from_logits(y_pred_logits=pred, y_true=target_smooth, mask=mask, axis=-1)


def _partitioned_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_partitions: int,
    loss_fn,
    mask_fn,
    device: torch.device,
) -> torch.Tensor:
    """Compute loss as sum of per-partition losses along the sequence dimension (dim=1).

    Splits pred and target into num_partitions equal chunks along dim=1. Partitions
    that have no valid positions (mask is all False) are skipped. The final loss is
    the sum over non-empty partitions, upweighting the signal relative to a global mean.
    """
    seq_len = pred.shape[1]
    chunk_size = seq_len // num_partitions
    chunk_losses = []
    for i in range(num_partitions):
        start = i * chunk_size
        end = start + chunk_size if i < num_partitions - 1 else seq_len
        p_chunk = pred[:, start:end, :]
        t_chunk = target[:, start:end, :]
        # Skip partitions with no valid positions
        if not mask_fn(t_chunk).any():
            continue
        chunk_losses.append(loss_fn(p_chunk, t_chunk))
    if not chunk_losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(chunk_losses).sum()


def _soft_clip_counts(counts: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    return torch.where(counts > clip, 2.0 * torch.sqrt(counts * clip) - clip, counts)


def _compute_junction_strand_loss(pred_counts, target_counts, donor_pos, accept_pos, device,
                                   junction_loss: str = "original"):
    """Strand-specific junction loss matching JAX SpliceSitesJunctionHead.loss.

    loss = 0.2 * (CE(axis=donor) + CE(axis=acceptor)) + 0.04 * (Poisson(axis=donor) + Poisson(axis=acceptor))

    pairs_mask[b,d,a,s] = (donor_pos[b,d] >= 0) & (accept_pos[b,a] >= 0)

    Args:
        junction_loss: "original" uses cross_entropy_loss (JAX pre-de264f5);
                       "normalized" uses cross_entropy_loss_normalized (JAX post-de264f5,
                       both targets and predictions normalized to ratios within mask).
    """
    assert pred_counts.shape == target_counts.shape, (
        f"pred_counts {tuple(pred_counts.shape)} and target_counts {tuple(target_counts.shape)} must match"
    )
    assert donor_pos.shape[-1] == pred_counts.shape[1], (
        f"donor positions K={donor_pos.shape[-1]} must equal pred_counts dim-1={pred_counts.shape[1]}"
    )
    assert accept_pos.shape[-1] == pred_counts.shape[2], (
        f"acceptor positions K={accept_pos.shape[-1]} must equal pred_counts dim-2={pred_counts.shape[2]}"
    )
    valid_d = (donor_pos >= 0).float()
    valid_a = (accept_pos >= 0).float()
    pairs_mask = torch.einsum('bd,ba->bda', valid_d, valid_a).bool()
    pairs_mask = pairs_mask.unsqueeze(-1).expand_as(pred_counts)

    if not pairs_mask.any():
        return torch.tensor(0.0, device=device, dtype=pred_counts.dtype)

    target = torch.where(pairs_mask, target_counts, torch.zeros_like(target_counts))
    pred   = torch.where(pairs_mask, pred_counts,   torch.zeros_like(pred_counts))

    # Skip intervals with no observed junction counts — the JAX cross-entropy
    # goes negative when targets are all zero, polluting the loss with noise.
    if not (target > 0).any():
        return torch.tensor(0.0, device=device, dtype=pred_counts.dtype)

    sum_pred_d = pred.sum(dim=1)
    sum_tgt_d  = _soft_clip_counts(target.sum(dim=1))
    sum_pred_a = pred.sum(dim=2)
    sum_tgt_a  = _soft_clip_counts(target.sum(dim=2))

    if junction_loss == "sparse":
        # Restrict CE and Poisson to donors/acceptors that have observed junction
        # counts. The standard loss applies Poisson to all 512 positions; for sparse
        # targets (~45 real junctions) this creates a ~25:1 suppression gradient from
        # zero-target positions that overwhelms the signal from true junctions.
        has_d = (sum_tgt_d > 0) & pairs_mask.any(dim=1)  # (B, A, S) acceptors with incoming reads
        has_a = (sum_tgt_a > 0) & pairs_mask.any(dim=2)  # (B, D, S) donors with outgoing reads
        ce_mask_d = pairs_mask & has_d.unsqueeze(1).expand_as(pairs_mask)
        ce_mask_a = pairs_mask & has_a.unsqueeze(2).expand_as(pairs_mask)
        donor_ratios_loss    = cross_entropy_loss(y_true=target, y_pred=pred, mask=ce_mask_d, axis=1)
        acceptor_ratios_loss = cross_entropy_loss(y_true=target, y_pred=pred, mask=ce_mask_a, axis=2)
        donor_total_loss  = poisson_loss(y_true=sum_tgt_d, y_pred=sum_pred_d, mask=has_d)
        accept_total_loss = poisson_loss(y_true=sum_tgt_a, y_pred=sum_pred_a, mask=has_a)
    else:
        _ce = cross_entropy_loss_normalized if junction_loss == "normalized" else cross_entropy_loss
        donor_ratios_loss    = _ce(y_true=target, y_pred=pred, mask=pairs_mask, axis=1)
        acceptor_ratios_loss = _ce(y_true=target, y_pred=pred, mask=pairs_mask, axis=2)
        donor_total_loss  = poisson_loss(y_true=sum_tgt_d, y_pred=sum_pred_d, mask=pairs_mask.any(dim=1))
        accept_total_loss = poisson_loss(y_true=sum_tgt_a, y_pred=sum_pred_a, mask=pairs_mask.any(dim=2))

    return 0.2 * (donor_ratios_loss + acceptor_ratios_loss) + 0.04 * (donor_total_loss + accept_total_loss)


def _get_junction_targets(predictions, targets_dict, device):
    """Return (junc_matrix, positions) aligned to the current predictions.

    In predicted mode (``"positions"`` key present in predictions): builds the
    junction matrix on-the-fly from pre-filtered DataFrames in
    ``targets_dict["all_junctions"]`` and the predicted splice-site positions.
    In annotated mode: uses the pre-built tensors from ``targets_dict``.
    """
    if "positions" in predictions:
        from alphagenome_pytorch.extensions.finetuning.star_junctions import (
            junctions_to_junction_matrix,
        )
        pred_pos = predictions["positions"]          # (B, 4, K)
        assert pred_pos.ndim == 3 and pred_pos.shape[1] == 4, (
            f"Expected positions shape (B, 4, K), got {tuple(pred_pos.shape)}"
        )
        assert (pred_pos >= -1).all(), "positions must be -1 (padding) or a valid 0-based index"
        pred_pos_np = pred_pos.cpu().numpy()
        all_juncs_batch = targets_dict["all_junctions"]  # list[B] of list[DataFrame]
        max_splice_sites = pred_pos_np.shape[-1]
        mats = []
        for b in range(pred_pos_np.shape[0]):
            _, mat = junctions_to_junction_matrix(
                all_juncs_batch[b],
                max_splice_sites=max_splice_sites,
                positions=pred_pos_np[b],
            )
            mats.append(torch.from_numpy(mat))
        junc_matrix = torch.stack(mats).to(device)
        assert junc_matrix.shape[1] == junc_matrix.shape[2] == pred_pos.shape[2], (
            f"K mismatch after building junction matrix: "
            f"junc_matrix {tuple(junc_matrix.shape)} vs positions K={pred_pos.shape[2]}"
        )
        return junc_matrix, pred_pos
    else:
        junc_matrix = targets_dict["junction_matrix"].to(device)
        positions   = targets_dict["junction_positions"].to(device)
        assert junc_matrix.shape[1] == junc_matrix.shape[2] == positions.shape[-1], (
            f"K mismatch in annotated targets: "
            f"junc_matrix {tuple(junc_matrix.shape)} vs positions {tuple(positions.shape)}"
        )
        return junc_matrix, positions


def _compute_splice_loss(head, predictions, targets_dict, device, num_segments: int = 1,
                         junction_loss: str = "original", min_alpha_juncs: int = 5):
    """Compute loss for any of the three splice head types.

    Args:
        head: SpliceSitesClassificationHead, SpliceSitesUsageHead, or SpliceSitesJunctionHead.
        predictions: Dict returned by _call_splice_head.
        targets_dict: Dict with string keys: 'probs', 'usage',
            'junction_positions', 'junction_matrix'.
        device: Torch device.
        num_segments: Number of equal-length partitions to split the sequence
            into before computing loss. Each partition's loss is computed independently
            and the results are averaged. Values > 1 upweight sequence regions with
            fewer splice sites relative to a global mean, since each partition
            contributes equally regardless of how many valid positions it contains.
            Defaults to 1 (standard global mean, unchanged behaviour).
        junction_loss: Cross-entropy variant for the junction head. "original" matches
            JAX pre-de264f5; "normalized" matches JAX post-de264f5 (ratio CE).
        min_alpha_juncs: Minimum junction read depth (alpha) for a splice site to
            contribute to the SSU loss.  Positions with 0 < alpha < min_alpha_juncs
            are excluded to avoid training on low-confidence SSU estimates.
            Background positions (alpha == 0) are also excluded so the model only
            trains on well-supported observations.  Set to 0 to include all positions.
            Defaults to 5.  Requires 'usage_alpha' key in targets_dict; falls back
            to ones_like mask when absent.

    Returns:
        (loss_tensor, components_dict) where components_dict has keys like
        'cls_loss', 'usage_loss', 'junction_pos_loss', 'junction_neg_loss'.
    """
    N_CLASSES = 5
    label_smoothing = 1e-7

    if isinstance(head, SpliceSitesClassificationHead):
        pred = predictions[1]
        target = targets_dict["probs"].to(device)
        if num_segments > 1:
            loss = _partitioned_loss(
                pred, target,
                num_partitions=num_segments,
                loss_fn=lambda p, t: _ce_loss_with_smoothing(p, t, label_smoothing, N_CLASSES),
                mask_fn=lambda t: t.any(dim=-1, keepdim=True).expand_as(t),
                device=device,
            )
        else:
            target_smooth = (1.0 - label_smoothing) * target.float() + label_smoothing / N_CLASSES
            mask = target.any(dim=-1, keepdim=True).expand_as(pred)
            loss = cross_entropy_loss_from_logits(
                y_pred_logits=pred,
                y_true=target_smooth,
                mask=mask,
                axis=-1,
            )
        return loss, {"cls_loss": loss.item()}

    elif isinstance(head, SpliceSitesUsageHead):
        pred = predictions[1]
        target = targets_dict["usage"].to(device)
        # Coverage-based mask: exclude only low-confidence splice sites (0 <= alpha < threshold).
        # Background positions (alpha == -1) are kept — they anchor the head to predict 0
        # at non-splice sites and provide the contrastive signal for the sparse SSU distribution.
        # alpha == -1 is the sentinel for "not a splice site"; actual splice sites have alpha >= 0.
        # When the alpha mask is active, _partitioned_loss is not used — it would need
        # the mask to be sliced in sync with pred/target chunks (unsupported by the
        # current interface).
        if "usage_alpha" in targets_dict and min_alpha_juncs > 0:
            alpha = targets_dict["usage_alpha"].to(device)
            usage_mask = (alpha < 0) | (alpha >= min_alpha_juncs)
            loss = binary_crossentropy_from_logits(
                y_pred=pred,
                y_true=target.float().clamp(1e-7, 1.0 - 1e-7),
                mask=usage_mask,
            )
        elif num_segments > 1:
            loss = _partitioned_loss(
                pred, target,
                num_partitions=num_segments,
                loss_fn=lambda p, t: binary_crossentropy_from_logits(
                    y_pred=p,
                    y_true=t.float().clamp(1e-7, 1.0 - 1e-7),
                    mask=torch.ones_like(p, dtype=torch.bool),
                ),
                mask_fn=lambda t: torch.ones_like(t, dtype=torch.bool),
                device=device,
            )
        else:
            loss = binary_crossentropy_from_logits(
                y_pred=pred,
                y_true=target.float().clamp(1e-7, 1.0 - 1e-7),
                mask=torch.ones_like(pred, dtype=torch.bool),
            )
        return loss, {"usage_loss": loss.item()}

    elif isinstance(head, SpliceSitesJunctionHead):
        if "junction_matrix" not in targets_dict or "pos_counts" not in predictions:
            return torch.tensor(0.0, device=device), {}
        junc_matrix, positions = _get_junction_targets(predictions, targets_dict, device)
        n_s = head._num_tissues
        pos_loss = _compute_junction_strand_loss(
            predictions["pos_counts"], junc_matrix[..., :n_s],
            positions[:, 0, :].long(), positions[:, 1, :].long(), device,
            junction_loss=junction_loss,
        )
        neg_loss = _compute_junction_strand_loss(
            predictions["neg_counts"], junc_matrix[..., n_s:],
            positions[:, 2, :].long(), positions[:, 3, :].long(), device,
            junction_loss=junction_loss,
        )
        loss = pos_loss + neg_loss
        return loss, {"junction_pos_loss": pos_loss.item(), "junction_neg_loss": neg_loss.item()}

    return torch.tensor(0.0, device=device), {}


def _extract_junction_pearson_per_sample(predictions, targets_dict, device):
    """Per-biological-sample (pred, true) tensors for junction Pearson.

    Combines pos+neg strand data for each biological sample.
    Returns list of n_s dicts {"full": (pred, true), "nonzero": (pred[nz], true[nz]) | None},
    or None if data unavailable.
    """
    if "junction_matrix" not in targets_dict or "pos_counts" not in predictions:
        return None

    junc_matrix, positions = _get_junction_targets(predictions, targets_dict, device)
    n_s = junc_matrix.shape[-1] // 2

    per_sample_pred = [[] for _ in range(n_s)]
    per_sample_true = [[] for _ in range(n_s)]

    for pred_key, tgt_slice, donor_row, accept_row in [
        ("pos_counts", slice(None, n_s),  0, 1),
        ("neg_counts", slice(n_s, None),  2, 3),
    ]:
        pred_strand = predictions[pred_key]
        tgt_strand  = junc_matrix[:, :, :, tgt_slice]
        donor_pos   = positions[:, donor_row,  :].long()
        accept_pos  = positions[:, accept_row, :].long()
        pairs_mask  = torch.einsum(
            "bd,ba->bda",
            (donor_pos >= 0).float(),
            (accept_pos >= 0).float(),
        ).bool()
        for s in range(n_s):
            per_sample_pred[s].append(pred_strand[:, :, :, s][pairs_mask].float().cpu())
            per_sample_true[s].append(tgt_strand[:, :, :, s][pairs_mask].float().cpu())

    result = []
    for s in range(n_s):
        pred_s = torch.cat(per_sample_pred[s])
        true_s = torch.cat(per_sample_true[s])
        nz = true_s > 0
        result.append({
            "full":    (pred_s, true_s),
            "nonzero": (pred_s[nz], true_s[nz]) if nz.any() else None,
        })
    return result  # length n_s


def _extract_usage_pearson_per_sample(predictions, targets_dict, device):
    """Per-sample (pred, true) tensors for splice usage Pearson.

    Returns list of n_s (pred_flat, true_flat) tuples, or None per empty sample.
    """
    if 1 not in predictions:
        return None
    pred   = torch.sigmoid(predictions[1])        # (B, S, n_s)
    target = targets_dict["usage"].to(device)     # (B, S, n_s)
    n_s    = pred.shape[-1]

    result = []
    for s in range(n_s):
        mask_s = target[:, :, s] > 0
        if not mask_s.any():
            result.append(None)
        else:
            result.append((
                pred[:, :, s][mask_s].float().cpu(),
                target[:, :, s][mask_s].float().cpu(),
            ))
    return result  # length n_s


def _extract_splice_pearson_pairs(
    head, predictions, targets_dict, device, min_alpha_juncs: int = 0
):
    """Extract flat (N,) pred and true tensors over valid positions for Pearson R.

    For SpliceSitesUsageHead: returns a dict with variant "full" (all target>0 positions)
        and, when min_alpha_juncs>0 and "usage_alpha" is present, variant "alpha"
        (positions with alpha >= min_alpha_juncs only — the high-confidence subset).
    For SpliceSitesJunctionHead: returns dict with variants:
        - "full": all valid (donor, acceptor) cells — log1p Pearson
        - "nonzero": valid cells with target > 0 — log1p Pearson
        - "psi5": PSI5 = n(D,A)/Σ_A' n(D,A'), restricted to donors with reads — Pearson
        - "psi3": PSI3 = n(D,A)/Σ_D' n(D',A), restricted to acceptors with reads — Pearson
        - "binary_cls": all valid pairs, pred=counts (scores), true=binary nonzero — auPRC
        Each value is a dict with "pred" and "true" keys, or None if empty.

    Returns dict or (None, None) if no valid entries.
    """
    if isinstance(head, SpliceSitesUsageHead):
        if 1 not in predictions:
            return None, None
        pred = torch.sigmoid(predictions[1])          # (B, S, n_samples)
        target = targets_dict["usage"].to(device)     # (B, S, n_samples)
        mask = (target > 0).any(dim=-1)                # (B, S)
        if not mask.any():
            return None, None
        variants = {"full": (pred[mask].reshape(-1), target[mask].reshape(-1))}
        if min_alpha_juncs > 0 and "usage_alpha" in targets_dict:
            alpha = targets_dict["usage_alpha"].to(device)   # (B, S) or (B, S, n_samples)
            alpha_cond = alpha >= min_alpha_juncs
            if alpha_cond.dim() > 2:
                alpha_cond = alpha_cond.any(dim=-1)          # (B, S)
            alpha_mask = mask & alpha_cond
            if alpha_mask.any():
                variants["alpha"] = (pred[alpha_mask].reshape(-1), target[alpha_mask].reshape(-1))
        return variants

    elif isinstance(head, SpliceSitesJunctionHead):
        if "junction_matrix" not in targets_dict or "pos_counts" not in predictions:
            return None, None
        junc_matrix, positions = _get_junction_targets(predictions, targets_dict, device)
        n_s = head._num_tissues

        variants = {"full": {}, "nonzero": {}, "psi5": {}, "psi3": {}, "binary_cls": {}}
        all_pred_full, all_true_full = [], []
        all_pred_nz, all_true_nz = [], []
        all_pred_psi5, all_true_psi5 = [], []
        all_pred_psi3, all_true_psi3 = [], []
        all_pred_bincls, all_true_bincls = [], []

        _EPS = 1e-8

        for pred_key, tgt_slice, donor_row, accept_row in [
            ("pos_counts", (slice(None), slice(None), slice(None), slice(None, n_s)),   0, 1),
            ("neg_counts", (slice(None), slice(None), slice(None), slice(n_s, None)),   2, 3),
        ]:
            pred_counts = predictions[pred_key]           # (B, D, A, n_s)
            tgt_counts = junc_matrix[tgt_slice]           # (B, D, A, n_s)
            donor_pos  = positions[:, donor_row,  :].long()
            accept_pos = positions[:, accept_row, :].long()
            valid_d = (donor_pos >= 0).float()
            valid_a = (accept_pos >= 0).float()
            pairs_mask = torch.einsum('bd,ba->bda', valid_d, valid_a).bool()
            pairs_mask4 = pairs_mask.unsqueeze(-1).expand_as(pred_counts)

            # Full variant: all valid pairs
            if pairs_mask4.any():
                all_pred_full.append(pred_counts[pairs_mask4].float().cpu())
                all_true_full.append(tgt_counts[pairs_mask4].float().cpu())

            # Nonzero variant: valid pairs with target > 0
            nonzero_mask = pairs_mask4 & (tgt_counts > 0)
            if nonzero_mask.any():
                all_pred_nz.append(pred_counts[nonzero_mask].float().cpu())
                all_true_nz.append(tgt_counts[nonzero_mask].float().cpu())

            # PSI5: n(D,A)/Σ_A' n(D,A') — only where donor has reads in ground truth
            true_d_total = tgt_counts.sum(dim=2, keepdim=True)      # (B, D, 1, n_s)
            pred_d_total = pred_counts.sum(dim=2, keepdim=True)
            psi5_valid = pairs_mask4 & (true_d_total > 0).expand_as(pairs_mask4)
            if psi5_valid.any():
                pred_psi5 = pred_counts / (pred_d_total + _EPS)
                true_psi5 = tgt_counts.float() / (true_d_total.float() + _EPS)
                all_pred_psi5.append(pred_psi5[psi5_valid].float().cpu())
                all_true_psi5.append(true_psi5[psi5_valid].float().cpu())

            # PSI3: n(D,A)/Σ_D' n(D',A) — only where acceptor has reads in ground truth
            true_a_total = tgt_counts.sum(dim=1, keepdim=True)      # (B, 1, A, n_s)
            pred_a_total = pred_counts.sum(dim=1, keepdim=True)
            psi3_valid = pairs_mask4 & (true_a_total > 0).expand_as(pairs_mask4)
            if psi3_valid.any():
                pred_psi3 = pred_counts / (pred_a_total + _EPS)
                true_psi3 = tgt_counts.float() / (true_a_total.float() + _EPS)
                all_pred_psi3.append(pred_psi3[psi3_valid].float().cpu())
                all_true_psi3.append(true_psi3[psi3_valid].float().cpu())

            # binary_cls: predicted counts as scores vs binary existence label — for auPRC
            if pairs_mask4.any():
                all_pred_bincls.append(pred_counts[pairs_mask4].float().cpu())
                all_true_bincls.append((tgt_counts[pairs_mask4] > 0).float().cpu())

        # Aggregate variants
        if all_pred_full:
            variants["full"] = {"pred": torch.cat(all_pred_full), "true": torch.cat(all_true_full)}
        if all_pred_nz:
            variants["nonzero"] = {"pred": torch.cat(all_pred_nz), "true": torch.cat(all_true_nz)}
        if all_pred_psi5:
            variants["psi5"] = {"pred": torch.cat(all_pred_psi5), "true": torch.cat(all_true_psi5)}
        if all_pred_psi3:
            variants["psi3"] = {"pred": torch.cat(all_pred_psi3), "true": torch.cat(all_true_psi3)}
        if all_pred_bincls:
            variants["binary_cls"] = {"pred": torch.cat(all_pred_bincls), "true": torch.cat(all_true_bincls)}

        if not variants["full"]:
            return None, None
        return variants

    return None, None


@dataclass
class ModalityConfig:
    """Configuration for a fine-tuning modality.

    Attributes:
        name: Modality name ('rnaseq' or 'atac').
        resolutions: Tuple of output resolutions (e.g., (1, 128) or (128,)).
        default_resolution_weights: Default weights for each resolution.
        embedding_dim: Embedding dimension for ATAC (None for RNA-seq).
        positions_arg: CLI argument name for positions ('positions' or 'peaks').
    """

    name: str
    resolutions: tuple[int, ...]
    default_resolution_weights: dict[int, float]
    embedding_dim: int | None
    positions_arg: str


# Registry of modality configurations
MODALITY_CONFIGS: dict[str, ModalityConfig] = {
    "rna_seq": ModalityConfig(
        name="rna_seq",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "atac": ModalityConfig(
        name="atac",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "dnase": ModalityConfig(
        name="dnase",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "procap": ModalityConfig(
        name="procap",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "cage": ModalityConfig(
        name="cage",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "chip_tf": ModalityConfig(
        name="chip_tf",
        resolutions=(128,),
        default_resolution_weights={128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "chip_histone": ModalityConfig(
        name="chip_histone",
        resolutions=(128,),
        default_resolution_weights={128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "splice": ModalityConfig(
        name="splice",
        resolutions=(1,),
        default_resolution_weights={1: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "splice_junction": ModalityConfig(
        name="splice_junction",
        resolutions=(1,),
        default_resolution_weights={1: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
}


def create_lr_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    schedule: str = "cosine",
) -> LambdaLR:
    """Create learning rate scheduler with optional warmup.

    Args:
        optimizer: Optimizer to schedule.
        warmup_steps: Number of warmup steps (linear ramp from 0 to lr).
        total_steps: Total number of training steps.
        schedule: Schedule type after warmup. Options:
            - "cosine": Cosine decay to 0 (default)
            - "constant": Constant learning rate

    Returns:
        LambdaLR scheduler.

    Examples:
        # Warmup + cosine decay (default)
        scheduler = create_lr_scheduler(opt, warmup_steps=500, total_steps=10000)

        # Constant learning rate (no warmup, no decay)
        scheduler = create_lr_scheduler(opt, warmup_steps=0, total_steps=10000, schedule="constant")

        # Warmup then constant
        scheduler = create_lr_scheduler(opt, warmup_steps=500, total_steps=10000, schedule="constant")
    """
    if schedule not in ("cosine", "constant"):
        raise ValueError(f"Unknown schedule: {schedule}. Must be 'cosine' or 'constant'.")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if schedule == "constant":
            return 1.0
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def compute_finetuning_loss(
    predictions: dict[int, Tensor],
    targets: dict[int, Tensor],
    resolution_weights: dict[int, float],
    positional_weight: float,
    device: torch.device,
    channels_last: bool = True,
    *,
    gene_mask: Tensor | None = None,
    gene_loss_weight: float = 0.0,
    gene_cross_track_weight: float = 5.0,
    strand_channel_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute combined loss across resolutions.

    Uses dynamic multinomial_resolution = seq_len // 8 for consistent loss
    granularity across different sequence lengths.

    Optionally adds the cross-track gene LFC term (Decima-style; see
    `losses.gene_lfc_loss`) at 1bp resolution when:
      - `gene_loss_weight > 0`
      - `gene_mask` and `strand_channel_mask` are both provided
      - `predictions` / `targets` contain key `1`
    The gene LFC contribution is `resolution_weights[1] * gene_loss_weight *
    gene_lfc_term` so it scales with the existing 1bp resolution weight.

    Args:
        predictions: Dict mapping resolution to prediction tensors.
        targets: Dict mapping resolution to target tensors.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        device: Torch device.
        channels_last: If True, assumes (B, S, C). If False, assumes (B, C, S).
        gene_mask: Optional `[B, S, 2, G]` gene-body mask for the gene LFC
            term. Ignored if `gene_loss_weight <= 0`.
        gene_loss_weight: Outer weight on the gene LFC term (paper: 0.1).
            Default 0.0 disables the term entirely (no behavioral change vs.
            the pre-B3.2 loss path).
        gene_cross_track_weight: Inner weight on the multinomial component
            of the gene LFC term (paper default: 5.0).
        strand_channel_mask: Optional `[2, 1, C]` track strand-compatibility
            mask, required when `gene_loss_weight > 0`.

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains per-resolution
        losses and other metrics.
    """
    total_loss = torch.tensor(0.0, device=device)
    loss_dict: dict[str, Tensor] = {}

    for res, weight in resolution_weights.items():
        if res not in predictions:
            continue

        pred = predictions[res]
        target = targets[res]

        # Detect dimensions based on format
        if channels_last:
            # (B, S, C)
            current_seq_len = pred.shape[-2]
            num_channels = pred.shape[-1]
            mask_shape = (pred.shape[0], 1, num_channels)
        else:
            # (B, C, S)
            current_seq_len = pred.shape[-1]
            num_channels = pred.shape[-2]
            mask_shape = (pred.shape[0], num_channels, 1)

        # Use multinomial_resolution matching JAX for 1Mb sequences (2^17 // res),
        # but allow for fewer segments if the sequence is shorter.
        # This ensures segments are at least 131k bp (at 1bp) and that
        # multinomial_resolution always divides current_seq_len.
        num_segments = max(1, min(8, current_seq_len // (131072 // res)))
        multinomial_resolution = current_seq_len // num_segments

        # Create mask (all True)
        mask = torch.ones(*mask_shape, dtype=torch.bool, device=device)

        res_loss_dict = multinomial_loss(
            y_pred=pred,
            y_true=target,
            mask=mask,
            multinomial_resolution=multinomial_resolution,
            positional_weight=positional_weight,
            channels_last=channels_last,
        )

        res_loss = res_loss_dict["loss"]

        # Add gene LFC term at 1bp resolution when enabled. Mirrors upstream
        # which only threads gene_mask through the resolution=1 head path.
        if res == 1 and gene_loss_weight > 0:
            # Fail loud on a wiring error: when the gene term is enabled, the
            # dataset always yields a (possibly all-zero) gene_mask tensor and
            # strand_channel_mask is required. A None here means a dependency
            # was never threaded through, so silently skipping would train
            # without the intended term.
            missing_args = []
            if gene_mask is None:
                missing_args.append("gene_mask")
            if strand_channel_mask is None:
                missing_args.append("strand_channel_mask")
            if missing_args:
                raise ValueError(
                    "gene_loss_weight > 0 requires "
                    f"{', '.join(missing_args)} to be provided for the gene "
                    "LFC loss at 1bp resolution."
                )
            # gene_lfc_loss expects channels-last; transpose if needed.
            if channels_last:
                pred_nlc, target_nlc, mask_nlc = pred, target, mask
            else:
                pred_nlc = pred.transpose(-1, -2).contiguous()
                target_nlc = target.transpose(-1, -2).contiguous()
                mask_nlc = mask.transpose(-1, -2).contiguous()
            gene_loss, gene_aux = losses.gene_lfc_loss(
                predictions=pred_nlc,
                targets=target_nlc,
                targets_mask=mask_nlc,
                gene_mask=gene_mask,
                strand_channel_mask=strand_channel_mask,
                gene_cross_track_weight=gene_cross_track_weight,
            )
            res_loss = res_loss + gene_loss_weight * gene_loss
            loss_dict["loss_gene_lfc"] = gene_loss
            loss_dict["loss_gene_total_count"] = gene_aux["gene_loss_total_count"]
            loss_dict["loss_gene_positional"] = gene_aux["gene_loss_positional"]

        total_loss = total_loss + weight * res_loss
        loss_dict[f"loss_{res}bp"] = res_loss

    loss_dict["loss"] = total_loss
    return total_loss, loss_dict


def train_epoch(
    model: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    resolution_weights: dict[int, float],
    positional_weight: float,
    epoch: int,
    log_every: int,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    resolutions: tuple[int, ...] | None = None,
    *,
    gene_loss_weight: float = 0.0,
    gene_cross_track_weight: float = 5.0,
    strand_channel_mask: Tensor | None = None,
    organism: int = 0,
) -> float:
    """Train for one epoch.

    Args:
        model: AlphaGenome trunk model.
        head: Output head module.
        train_loader: Training data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Torch device.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        epoch: Current epoch number.
        log_every: Log frequency in steps.
        use_amp: Whether to use automatic mixed precision (default: True).
        accumulation_steps: Number of batches to accumulate gradients over
            before performing an optimizer step. Useful for simulating larger
            batch sizes when GPU memory is limited (default: 1, no accumulation).
        resolutions: Tuple of resolutions to train on (e.g., (1,), (128,), or (1, 128)).
            If None, inferred from resolution_weights keys. Training on 1bp resolution
            requires significantly more memory than 128bp.
        gene_loss_weight: Outer weight on the gene LFC term (paper: 0.1 for
            RNA-seq). Default 0.0 disables. Requires `train_loader` to yield
            3-tuples (sequence, targets, gene_mask) and `strand_channel_mask`
            to be set.
        gene_cross_track_weight: Inner multinomial weight inside the gene LFC
            term (paper default: 5.0).
        strand_channel_mask: `[2, 1, C]` track strand-compatibility mask,
            required when gene_loss_weight > 0.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    head.train()

    total_loss = 0.0
    n_batches = 0

    # Determine which resolutions to use
    if resolutions is None:
        resolutions = tuple(resolution_weights.keys())
    if invalid := (set(resolutions) - {1, 128}):
        raise ValueError(f"Invalid resolutions {invalid}, must be 1 or 128")

    if gene_loss_weight > 0 and strand_channel_mask is None:
        raise ValueError(
            "gene_loss_weight > 0 requires strand_channel_mask to be set."
        )

    # Set up autocast context (bfloat16 on CUDA, no-op on CPU)
    if use_amp and device.type == "cuda":
        amp_context = autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_context = nullcontext()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch_data in enumerate(pbar):
        # Dataset returns a 3-tuple when gene_mask is configured, else 2-tuple.
        if len(batch_data) == 3:
            sequences, targets_dict, gene_mask = batch_data
            gene_mask = gene_mask.to(device)
        else:
            sequences, targets_dict = batch_data
            gene_mask = None
        sequences = sequences.to(device)
        targets_dict = {k: v.to(device) for k, v in targets_dict.items() if k in resolutions}

        # Organism index for this fine-tune (0=human, 1=mouse); the forward
        # uses the matching organism embedding + head slot.
        organism_idx = torch.full((sequences.shape[0],), organism, dtype=torch.long, device=device)

        with amp_context:
            # Forward through trunk
            outputs = model(sequences, organism_idx, return_embeddings=True, channels_last=False)

            # Only get embeddings for requested resolutions (1bp is 128x larger than 128bp)
            embeddings_dict = _extract_embeddings(outputs, resolutions)

            # Forward through head
            predictions = head(embeddings_dict, organism_idx)

            # Compute loss
            loss, _ = compute_finetuning_loss(
                predictions=predictions,
                targets=targets_dict,
                resolution_weights=resolution_weights,
                positional_weight=positional_weight,
                device=device,
                channels_last=True,
                gene_mask=gene_mask,
                gene_loss_weight=gene_loss_weight,
                gene_cross_track_weight=gene_cross_track_weight,
                strand_channel_mask=strand_channel_mask,
            )

        # Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        # Optimizer step every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % log_every == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

    # Handle remaining gradients if dataset size is not divisible by accumulation_steps
    if n_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate(
    model: nn.Module,
    head: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    resolution_weights: dict[int, float],
    positional_weight: float,
    use_amp: bool = True,
    resolutions: tuple[int, ...] | None = None,
    organism: int = 0,
) -> float:
    """Validate the model.

    Args:
        model: AlphaGenome trunk model.
        head: Output head module.
        val_loader: Validation data loader.
        device: Torch device.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        use_amp: Whether to use automatic mixed precision (default: True).
        resolutions: Tuple of resolutions to validate on (e.g., (1,), (128,), or (1, 128)).
            If None, inferred from resolution_weights keys.

    Returns:
        Average validation loss.
    """
    model.eval()
    head.eval()

    total_loss = 0.0
    n_batches = 0

    # Determine which resolutions to use
    if resolutions is None:
        resolutions = tuple(resolution_weights.keys())
    if invalid := (set(resolutions) - {1, 128}):
        raise ValueError(f"Invalid resolutions {invalid}, must be 1 or 128")

    # Set up autocast context (bfloat16 on CUDA, no-op on CPU)
    if use_amp and device.type == "cuda":
        amp_context = autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_context = nullcontext()

    with torch.no_grad():
        for sequences, targets_dict in tqdm(val_loader, desc="Validation"):
            sequences = sequences.to(device)
            targets_dict = {k: v.to(device) for k, v in targets_dict.items() if k in resolutions}
            organism_idx = torch.full((sequences.shape[0],), organism, dtype=torch.long, device=device)

            with amp_context:
                outputs = model(sequences, organism_idx, return_embeddings=True, channels_last=False)

                # Only get embeddings for requested resolutions
                embeddings_dict = _extract_embeddings(outputs, resolutions)

                predictions = head(embeddings_dict, organism_idx)

                loss, _ = compute_finetuning_loss(
                    predictions=predictions,
                    targets=targets_dict,
                    resolution_weights=resolution_weights,
                    positional_weight=positional_weight,
                    device=device,
                    channels_last=True,
                )

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)


# Re-export save_checkpoint from checkpointing module for backward compatibility
from alphagenome_pytorch.extensions.finetuning.checkpointing import save_checkpoint


# =============================================================================
# Profiling utilities
# =============================================================================


class ProfilingStats:
    """Collect timing statistics for profiling training batches.

    Example:
        >>> stats = ProfilingStats()
        >>> t0 = time.perf_counter()
        >>> # ... some operation ...
        >>> stats.add("forward", time.perf_counter() - t0)
        >>> print(stats.report(n_batches=10))
    """

    def __init__(self) -> None:
        self.times: dict[str, list[float]] = defaultdict(list)

    def add(self, name: str, elapsed: float) -> None:
        """Add a timing measurement.

        Args:
            name: Name of the operation (e.g., "forward", "backward").
            elapsed: Elapsed time in seconds.
        """
        self.times[name].append(elapsed)

    def report(self, n_batches: int) -> str:
        """Generate a profiling report.

        Args:
            n_batches: Number of batches profiled.

        Returns:
            Formatted report string with timing breakdowns.
        """
        import numpy as np

        lines = ["\n" + "=" * 70, "PROFILING REPORT", "=" * 70]
        total_time = 0.0

        for name, times in sorted(self.times.items()):
            arr = np.array(times)
            total_time += arr.sum()
            lines.append(
                f"\n{name}:\n"
                f"  Mean:  {arr.mean()*1000:.2f} ms (+/- {arr.std()*1000:.2f} ms)\n"
                f"  Total: {arr.sum():.2f} s ({len(times)} samples)"
            )

        lines.append(f"\n{'=' * 70}")
        lines.append(f"TOTAL TIME: {total_time:.2f} s for {n_batches} batches")
        lines.append(f"AVG TIME PER BATCH: {total_time/n_batches*1000:.2f} ms")

        # Breakdown percentages
        lines.append(f"\nBREAKDOWN:")
        for name, times in sorted(self.times.items()):
            pct = np.sum(times) / total_time * 100
            lines.append(f"  {name}: {pct:.1f}%")

        lines.append("=" * 70)
        return "\n".join(lines)

    def estimated_epoch_time(self, total_batches: int) -> float:
        """Estimate total epoch time based on profiled batches.

        Args:
            total_batches: Total number of batches in the epoch.

        Returns:
            Estimated epoch time in seconds.
        """
        import numpy as np

        n_profiled = len(next(iter(self.times.values()))) if self.times else 0
        if n_profiled == 0:
            return 0.0

        total_profiled_time = sum(np.sum(t) for t in self.times.values())
        avg_batch_time = total_profiled_time / n_profiled
        return avg_batch_time * total_batches


# =============================================================================
# Enhanced training functions with DDP and profiling support
# =============================================================================


def _extract_embeddings(
    outputs: dict, resolutions: tuple[int, ...], frozen_backbone: bool = False
) -> dict[int, Any]:
    """Extract per-resolution embeddings from model output dict."""
    result = {}
    for res in resolutions:
        emb = outputs.get(f"embeddings_{res}bp")
        if emb is not None:
            result[res] = emb.detach() if frozen_backbone else emb
    return result


def _run_head(
    head,
    head_module,
    modality: str,
    embeddings_dict: dict,
    organism_idx,
    targets_dict: dict,
    device: torch.device,
    junction_top_k: int | None,
    heads: dict,
    embeddings_pair=None,
    return_scaled: bool = True,
):
    """Forward pass through a head, dispatching by type (splice, contact_maps, genomic)."""
    if embeddings_pair is not None and modality == "contact_maps":
        return head(embeddings_pair, organism_idx, channels_last=True)
    if isinstance(head_module, SPLICE_HEAD_TYPES):
        _positions = targets_dict.get("junction_positions")
        if _positions is not None:
            _positions = _positions.to(device)
        _cls_head = heads.get("splice_site") if junction_top_k is not None else None
        if _cls_head is not None:
            _cls_head = _cls_head.module if hasattr(_cls_head, "module") else _cls_head
        return _call_splice_head(
            head_module, embeddings_dict, organism_idx, _positions, channels_last=False,
            cls_head=_cls_head, junction_top_k=junction_top_k,
        )
    return head(embeddings_dict, organism_idx, return_scaled=return_scaled, channels_last=True)


def _cuda_sync(device: torch.device) -> None:
    """Synchronize CUDA if on GPU (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _unpack_batch(batch_data) -> tuple:
    """Unpack a ``collate_multimodal`` batch into ``(sequences, targets, extras)``.

    ``collate_multimodal`` yields a 2-tuple ``(sequences, modality_targets)`` or a
    3-tuple with a trailing ``extras`` dict (``{"gene_mask", "coords"}``). This
    normalizes both to a 3-tuple with ``extras`` defaulting to an empty dict.
    """
    if len(batch_data) == 3:
        sequences, modality_targets, extras = batch_data
    else:
        sequences, modality_targets = batch_data
        extras = {}
    return sequences, modality_targets, extras


def _accumulate_gene_expr_windows(
    windows: list,
    *,
    pred_unscaled: Tensor,
    targets: Tensor,
    coords: list,
    annotation: Any,
    track_strands: list[str] | None,
    window_cache: dict | None = None,
) -> None:
    """Append per-window ``(gene_ids, pred[G,C], obs[G,C])`` for the val metric.

    For each window in the batch, build exon masks (≥50%-exon rule, strand-matched
    to ``track_strands``) and aggregate log-mean exon coverage for the predicted
    and observed RNA-seq signal. Windows with no qualifying gene are skipped.
    ``window_cache`` (a plain dict reused across batches and epochs) memoizes the
    per-window annotation lookup so it runs once per unique window, and lets the
    pred and obs calls of a window share it.
    """
    from alphagenome_pytorch.aggregation import gene_expression_values

    batch_size = pred_unscaled.shape[0]
    for b in range(batch_size):
        interval = tuple(coords[b])
        pred_vals, gene_ids, _ = gene_expression_values(
            pred_unscaled[b], annotation, interval,
            log="log1p", track_strands=track_strands, window_cache=window_cache,
        )
        if pred_vals.numel() == 0:
            continue
        obs_vals, _, _ = gene_expression_values(
            targets[b], annotation, interval,
            log="log1p", track_strands=track_strands, window_cache=window_cache,
        )
        windows.append((gene_ids, pred_vals.float().cpu(), obs_vals.float().cpu()))


def _gene_expr_metrics(
    gene_expr_windows: list,
    *,
    modality: str,
    world_size: int = 1,
) -> dict[str, float]:
    """Reduce accumulated per-window ``(gene_ids, pred, obs)`` to gene-expression correlations.

    Gathers windows across ranks (DDP), deduplicates genes, and returns the three
    correlation flavors plus the gene count under ``{modality}_gene_log_expr_*``
    keys. Pure given ``gene_expr_windows`` (the ``world_size == 1`` path needs no
    distributed context), so it is unit-testable without a model.
    """
    from alphagenome_pytorch.aggregation import combine_gene_expression

    all_windows = gene_expr_windows
    if world_size > 1:
        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, gene_expr_windows)
        all_windows = [w for part in gathered if part for w in part]

    ge = combine_gene_expression(all_windows)
    prefix = f"{modality}_gene_log_expr_pearson"
    return {
        f"{prefix}_across_genes": ge["across_genes"],
        f"{prefix}_across_genes_norm": ge["across_genes_norm"],
        f"{prefix}_across_tracks_norm": ge["across_tracks_norm"],
        f"{modality}_gene_log_expr_n_genes": ge["n_genes"],
    }


def _compute_multinomial_resolution(
    seq_len: int,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
) -> int:
    """Compute positions per segment for multinomial loss.

    Args:
        seq_len: Total sequence length (number of positions).
        num_segments: Target number of segments.
        min_segment_size: Minimum positions per segment (optional).

    Returns:
        Resolution (positions per segment).
    """
    resolution = max(1, seq_len // num_segments)

    if min_segment_size is not None:
        resolution = max(resolution, min_segment_size)

    return resolution


def train_epoch_ddp(
    model: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    resolution_weights: dict[int, float],
    positional_weight: float,
    count_weight: float,
    epoch: int,
    log_every: int,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    frozen_backbone: bool = False,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    train_sampler: DistributedSampler | None = None,
    rank: int = 0,
    world_size: int = 1,
    max_grad_norm: float = 1.0,
    profile_batches: int = 0,
    log_fn: Any | None = None,
    encoder_only: bool = False,
    organism: int = 0,
) -> float:
    """Train for one epoch with DDP and profiling support.

    This is the enhanced version of train_epoch() with:
    - Distributed Data Parallel (DDP) support
    - Optional profiling of first N batches
    - Gradient accumulation
    - Frozen backbone optimization (memory saving when no LoRA)

    Args:
        model: AlphaGenome trunk model (may be DDP-wrapped).
        head: Output head module.
        train_loader: Training data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Torch device.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        count_weight: Weight for count component of multinomial loss.
        epoch: Current epoch number.
        log_every: Log frequency in steps.
        use_amp: Whether to use automatic mixed precision.
        accumulation_steps: Number of batches to accumulate gradients over.
        frozen_backbone: If True, use torch.no_grad() for backbone (memory optimization).
        num_segments: Number of segments for multinomial loss.
        min_segment_size: Minimum positions per segment.
        train_sampler: DistributedSampler for DDP (set epoch for shuffling).
        rank: Process rank for DDP.
        world_size: Total number of processes.
        max_grad_norm: Maximum gradient norm for clipping.
        profile_batches: Number of batches to profile (0 to disable).
        log_fn: Optional function to call for step logging: log_fn(metrics_dict).
        encoder_only: If True, run only the CNN encoder (skip transformer and decoder)
            and pass the raw encoder output (B, S//128, 1536) to the head as resolution
            128. The backbone is always frozen in encoder-only mode. Requires the head
            to have been created with ``create_finetuning_head(..., encoder_only=True)``.

    Returns:
        Average training loss for the epoch (synchronized across ranks).
    """
    from alphagenome_pytorch.extensions.finetuning.distributed import (
        is_main_process,
        reduce_tensor,
    )

    model.train()
    head.train()

    # Set epoch for distributed sampler (important for shuffling)
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    total_loss = 0.0
    n_batches = 0

    # Profiling (only on rank 0)
    do_profile = profile_batches > 0 and is_main_process(rank)
    profile_stats = ProfilingStats() if do_profile else None

    if do_profile:
        print(f"\n*** PROFILING ENABLED for first {profile_batches} batches ***\n")

    # Only show progress bar on rank 0
    if is_main_process(rank):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader

    t_batch_start = time.perf_counter()
    running_loss = 0.0
    accumulated_batches = 0

    for batch_idx, (sequences, targets_dict) in enumerate(pbar):
        is_profiling = do_profile and batch_idx < profile_batches

        # --- Data loading time (time since last batch ended) ---
        if is_profiling and batch_idx > 0:
            _cuda_sync(device)
            t_data_load = time.perf_counter() - t_batch_start
            profile_stats.add("1_data_loading", t_data_load)

        # --- Transfer to GPU ---
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        sequences = sequences.to(device)
        organism_idx = torch.full((sequences.shape[0],), organism, dtype=torch.long, device=device)

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("2_to_device", time.perf_counter() - t0)

        # --- Forward pass ---
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        # When backbone is frozen (no LoRA), we can save memory by not building
        # the computation graph for the backbone forward pass.
        resolutions = tuple(resolution_weights.keys())

        if encoder_only:
            # Run only the CNN encoder; skip transformer, decoder, OutputEmbedders.
            # Backbone is always frozen in encoder-only mode.
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx, encoder_only=True)
            embeddings_dict = {128: outputs["encoder_output"].detach()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                predictions = head(
                    embeddings_dict, organism_idx, return_scaled=True, channels_last=True
                )
        else:
            backbone_ctx = torch.no_grad() if frozen_backbone else nullcontext()
            with backbone_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)

            embeddings_dict = _extract_embeddings(outputs, resolutions, frozen_backbone)
            predictions = head(embeddings_dict, organism_idx, return_scaled=True, channels_last=True)

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("3_forward", time.perf_counter() - t0)

        # --- Loss computation ---
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        loss = torch.tensor(0.0, device=device)
        loss_components: dict[str, float] = {}

        for res, weight in resolution_weights.items():
            if res not in predictions or res not in targets_dict:
                continue

            pred = predictions[res]
            targets = targets_dict[res].to(device)

            # Scale targets from experimental space to model space
            head_module = head.module if hasattr(head, "module") else head
            targets = head_module.scale(targets, organism_idx, resolution=res, channels_last=True)
            mask = torch.ones(pred.shape[0], 1, pred.shape[-1], dtype=torch.bool, device=device)

            # Compute multinomial loss
            current_seq_len = pred.shape[-2]
            multinomial_res = _compute_multinomial_resolution(
                current_seq_len, num_segments, min_segment_size
            )

            loss_dict = multinomial_loss(
                y_pred=pred,
                y_true=targets,
                mask=mask,
                multinomial_resolution=multinomial_res,
                positional_weight=positional_weight,
                count_weight=count_weight,
                channels_last=True,
            )

            res_loss = loss_dict["loss"] * weight
            loss = loss + res_loss
            loss_components[f"loss_{res}bp"] = res_loss.item()
            # Log raw (unweighted) losses for comparability across runs
            loss_components[f"loss_{res}bp_count"] = loss_dict["loss_total"].item()
            loss_components[f"loss_{res}bp_positional"] = loss_dict["loss_positional"].item()

        # Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("4_loss", time.perf_counter() - t0)

        # --- Backward pass ---
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        # --- Optimizer step (only every accumulation_steps batches) ---
        is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch = batch_idx == len(train_loader) - 1

        # Skip DDP gradient sync on intermediate accumulation steps
        no_sync = (
            accumulation_steps > 1
            and not is_accumulation_step
            and not is_last_batch
            and hasattr(model, "no_sync")
        )
        with model.no_sync() if no_sync else nullcontext():
            scaled_loss.backward()

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("5_backward", time.perf_counter() - t0)

        if is_accumulation_step or is_last_batch:
            if is_profiling:
                _cuda_sync(device)
                t0 = time.perf_counter()

            # Get trainable parameters for gradient clipping
            trainable_params = [p for p in head.parameters() if p.requires_grad]
            trainable_params += [p for p in model.parameters() if p.requires_grad]

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if is_profiling:
                _cuda_sync(device)
                profile_stats.add("6_optimizer", time.perf_counter() - t0)

        # Update totals
        raw_loss = loss.item()
        total_loss += raw_loss
        n_batches += 1

        # Update running loss
        running_loss += raw_loss
        accumulated_batches += 1

        current_lr = scheduler.get_last_lr()[0]

        # Logging (only on rank 0)
        if is_main_process(rank) and batch_idx % log_every == 0:
            avg_running_loss = running_loss / accumulated_batches
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "loss": f"{raw_loss:.4f}",
                    "run_loss": f"{avg_running_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

            if log_fn is not None:
                step_metrics = {
                    "batch": batch_idx,
                    "epoch": epoch,
                    "loss": raw_loss,
                    "running_loss": avg_running_loss,
                    "learning_rate": current_lr,
                    **loss_components,
                }
                log_fn(step_metrics)

            # Reset running loss after logging
            running_loss = 0.0
            accumulated_batches = 0

        # Print profiling report after profiling is done
        if do_profile and batch_idx == profile_batches - 1:
            print(profile_stats.report(profile_batches))

            # Estimate epoch time
            estimated_time = profile_stats.estimated_epoch_time(len(train_loader))
            print(f"\nESTIMATED EPOCH TIME: {estimated_time/60:.1f} minutes ({estimated_time/3600:.2f} hours)")
            print(f"  Based on {profile_batches} profiled batches, {len(train_loader)} total batches")
            print()

        # Mark end of batch for next iteration's data loading measurement
        if is_profiling:
            _cuda_sync(device)
        t_batch_start = time.perf_counter()

    # Reduce loss across all processes
    avg_loss = total_loss / max(1, n_batches)
    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()

    return avg_loss


@torch.no_grad()
def validate_ddp(
    model: nn.Module,
    head: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    resolution_weights: dict[int, float],
    positional_weight: float,
    count_weight: float,
    use_amp: bool = True,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    compute_pearson: bool = True,
    rank: int = 0,
    world_size: int = 1,
    encoder_only: bool = False,
    organism: int = 0,
) -> tuple[float, dict[str, Any]]:
    """Validate the model with DDP support and Pearson R metrics.

    This is the enhanced version of validate() with:
    - Distributed Data Parallel (DDP) support with proper tensor gathering
    - Optional Pearson R computation (profile and count correlations)

    Args:
        model: AlphaGenome trunk model (may be DDP-wrapped).
        head: Output head module.
        val_loader: Validation data loader.
        device: Torch device.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        count_weight: Weight for count component of multinomial loss.
        use_amp: Whether to use automatic mixed precision.
        num_segments: Number of segments for multinomial loss.
        min_segment_size: Minimum positions per segment.
        compute_pearson: Whether to compute Pearson R metrics.
        rank: Process rank for DDP.
        world_size: Total number of processes.
        encoder_only: If True, run only the CNN encoder and pass raw encoder output
            (B, S//128, 1536) to the head as resolution 128. Must match the setting
            used during training.

    Returns:
        Tuple of (avg_loss, metrics_dict) where metrics_dict contains:
        - Per-resolution losses (e.g., "1bp", "128bp")
        - Pearson R metrics if compute_pearson=True (profile_pearson_r_mean, count_pearson_r, etc.)
    """
    from alphagenome_pytorch.extensions.finetuning.distributed import (
        gather_tensors,
        is_main_process,
        reduce_tensor,
    )
    from alphagenome_pytorch.metrics import pearson_r, profile_pearson_r

    model.eval()
    head.eval()

    total_loss = 0.0
    n_batches = 0
    loss_by_resolution: dict[str, float] = defaultdict(float)

    # For Pearson R computation - accumulate across ALL batches
    accumulated_profile_r: dict[int, list[Tensor]] = defaultdict(list)
    accumulated_pred_counts: dict[int, list[Tensor]] = defaultdict(list)
    accumulated_true_counts: dict[int, list[Tensor]] = defaultdict(list)

    # Only show progress bar on rank 0
    if is_main_process(rank):
        pbar = tqdm(val_loader, desc="Validation")
    else:
        pbar = val_loader

    with torch.no_grad():
        for sequences, targets_dict in pbar:
            sequences = sequences.to(device)
            organism_idx = torch.full((sequences.shape[0],), organism, dtype=torch.long, device=device)
            resolutions = tuple(resolution_weights.keys())

            if encoder_only:
                outputs = model(sequences, organism_idx, encoder_only=True)
                embeddings_dict = {128: outputs["encoder_output"]}
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)

                embeddings_dict = _extract_embeddings(outputs, resolutions)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                # Get predictions in MODEL space for loss computation
                head_module = head.module if hasattr(head, "module") else head
                predictions_scaled = head(
                    embeddings_dict, organism_idx, return_scaled=True, channels_last=True
                )

                # Get predictions in EXPERIMENTAL space for Pearson R
                if compute_pearson:
                    predictions_unscaled = head(
                        embeddings_dict, organism_idx, return_scaled=False, channels_last=True
                    )

            loss = torch.tensor(0.0, device=device)

            for res, weight in resolution_weights.items():
                if res not in predictions_scaled or res not in targets_dict:
                    continue

                pred_scaled = predictions_scaled[res]
                targets = targets_dict[res].to(device)

                # Scale targets from experimental space to model space for loss
                targets_scaled = head_module.scale(
                    targets, organism_idx, resolution=res, channels_last=True
                )
                mask = torch.ones(
                    pred_scaled.shape[0], 1, pred_scaled.shape[-1], dtype=torch.bool, device=device
                )

                # Compute multinomial loss
                current_seq_len = pred_scaled.shape[-2]
                multinomial_res = _compute_multinomial_resolution(
                    current_seq_len, num_segments, min_segment_size
                )

                loss_dict = multinomial_loss(
                    y_pred=pred_scaled,
                    y_true=targets_scaled,
                    mask=mask,
                    multinomial_resolution=multinomial_res,
                    positional_weight=positional_weight,
                    count_weight=count_weight,
                    channels_last=True,
                )

                res_loss = loss_dict["loss"] * weight
                loss = loss + res_loss
                loss_by_resolution[f"{res}bp"] += res_loss.item()
                # Log raw (unweighted) losses for comparability across runs
                loss_by_resolution[f"{res}bp_count"] += loss_dict["loss_total"].item()
                loss_by_resolution[f"{res}bp_positional"] += loss_dict["loss_positional"].item()

                # Accumulate for Pearson R (in experimental space)
                if compute_pearson:
                    pred_unscaled = predictions_unscaled[res]

                    # Profile Pearson R: compute per-region correlation on-the-fly, store scalars
                    batch_profile_r = profile_pearson_r(pred_unscaled, targets)  # (batch, tracks)
                    accumulated_profile_r[res].append(batch_profile_r.float().cpu())

                    # Count Pearson R: store total counts per region (tiny memory)
                    accumulated_pred_counts[res].append(pred_unscaled.sum(dim=1).float().cpu())  # (batch, tracks)
                    accumulated_true_counts[res].append(targets.sum(dim=1).float().cpu())

            total_loss += loss.item()
            n_batches += 1

    # Reduce across all processes
    avg_loss = total_loss / max(1, n_batches)
    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()

    # Compute per-resolution loss metrics (synchronized across ranks)
    metrics: dict[str, Any] = {}
    for k, v in loss_by_resolution.items():
        res_avg = v / max(1, n_batches)
        if world_size > 1:
            res_tensor = torch.tensor(res_avg, device=device)
            res_tensor = reduce_tensor(res_tensor, world_size)
            metrics[k] = res_tensor.item()
        else:
            metrics[k] = res_avg

    # Compute Pearson R metrics (must gather across all DDP ranks)
    if compute_pearson:
        for res in resolution_weights.keys():
            # Profile Pearson R (from accumulated per-region correlations)
            if res in accumulated_profile_r and accumulated_profile_r[res]:
                all_profile_r = torch.cat(accumulated_profile_r[res], dim=0)  # (N_local, tracks)

                # Gather profile correlations from all ranks
                if world_size > 1:
                    all_profile_r = gather_tensors(all_profile_r, world_size, device)

                metrics[f"{res}bp_profile_pearson_r_mean"] = all_profile_r.mean().item()
                metrics[f"{res}bp_profile_pearson_r_std"] = all_profile_r.std().item()
                # Store full distribution for wandb histogram
                metrics[f"{res}bp_profile_pearson_r_values"] = all_profile_r.flatten().tolist()

            # Count Pearson R (from accumulated counts)
            if res in accumulated_pred_counts and accumulated_pred_counts[res]:
                all_pred_counts = torch.cat(accumulated_pred_counts[res], dim=0)  # (N_local, tracks)
                all_true_counts = torch.cat(accumulated_true_counts[res], dim=0)

                # Gather counts from all ranks
                if world_size > 1:
                    all_pred_counts = gather_tensors(all_pred_counts, world_size, device)
                    all_true_counts = gather_tensors(all_true_counts, world_size, device)

                if all_pred_counts.shape[0] > 1:
                    count_r = pearson_r(all_pred_counts, all_true_counts, dim=0)  # (tracks,)
                    metrics[f"{res}bp_count_pearson_r"] = count_r.mean().item()
                else:
                    metrics[f"{res}bp_count_pearson_r"] = float("nan")

    return avg_loss, metrics


def train_epoch_multihead(
    model: nn.Module,
    heads: dict[str, nn.Module],
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    modality_weights: dict[str, float],
    resolution_weights: dict[str, dict[int, float]],
    positional_weight: float,
    count_weight: float,
    epoch: int,
    log_every: int,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    frozen_backbone: bool = False,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    train_sampler: DistributedSampler | None = None,
    rank: int = 0,
    world_size: int = 1,
    max_grad_norm: float = 1.0,
    profile_batches: int = 0,
    log_fn: Any | None = None,
    encoder_only: bool = False,
    *,
    save_every_steps: int | None = None,
    save_fn: Any | None = None,
    global_step_offset: int = 0,
    skip_batches: int = 0,
    save_state: dict | None = None,
    organism_idx: int = 0,
    junction_top_k: int | None = None,
    junction_loss: str = "original",
    sequence_parallel: Any | None = None,
    min_alpha_juncs: int = 5,
    handler: Any | None = None,
    gene_loss_weights: dict[str, float] | None = None,
    gene_cross_track_weight: float = 5.0,
    strand_channel_masks: dict[str, Tensor] | None = None,
) -> tuple[float, dict[str, float]]:
    """Train for one epoch with multiple modality heads.

    This extends train_epoch_ddp to support multi-modality training where
    each modality has its own head and weights.

    Args:
        model: AlphaGenome trunk model (may be DDP-wrapped).
        heads: Dict mapping modality name to output head module.
        train_loader: Training data loader (yields sequences, modality_targets dict).
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Torch device.
        modality_weights: Weight for each modality's loss (e.g., {"atac": 1.0, "rna_seq": 0.5}).
        resolution_weights: Per-modality resolution weights (e.g., {"atac": {1: 1.0, 128: 1.0}}).
        positional_weight: Weight for positional component of multinomial loss.
        count_weight: Weight for count component of multinomial loss.
        epoch: Current epoch number.
        log_every: Log frequency in steps.
        use_amp: Whether to use automatic mixed precision.
        accumulation_steps: Number of batches to accumulate gradients over.
        frozen_backbone: If True, use torch.no_grad() for backbone.
        num_segments: Number of segments for multinomial loss.
        min_segment_size: Minimum positions per segment.
        num_segments: Number of sequence segments for both multinomial count loss and splice losses.
        train_sampler: DistributedSampler for DDP.
        rank: Process rank for DDP.
        world_size: Total number of processes.
        max_grad_norm: Maximum gradient norm for clipping.
        profile_batches: Number of batches to profile.
        log_fn: Optional function for step logging.
        encoder_only: If True, run only the CNN encoder and pass raw encoder output
            (B, S//128, 1536) to all heads as resolution 128. Backbone is always
            frozen in encoder-only mode.

    Returns:
        Tuple of (avg_total_loss, per_modality_losses).
    """
    from alphagenome_pytorch.extensions.finetuning.distributed import (
        is_main_process,
        reduce_tensor,
    )

    model.train()
    for head in heads.values():
        head.train()

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    total_loss_accum = 0.0
    modality_loss_accum: dict[str, float] = {m: 0.0 for m in heads}
    n_batches = 0

    # Profiling (only on rank 0)
    do_profile = profile_batches > 0 and is_main_process(rank)
    profile_stats = ProfilingStats() if do_profile else None

    if do_profile:
        print(f"\n*** PROFILING ENABLED for first {profile_batches} batches ***\n")

    if is_main_process(rank):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader

    t_batch_start = time.perf_counter()
    running_loss = 0.0
    accumulated_batches = 0
    opt_step = 0

    gene_loss_weights = gene_loss_weights or {}

    for batch_idx, batch_data in enumerate(pbar):
        if batch_idx < skip_batches:
            continue

        # Kept current so an async preemption signal (which can fire between any
        # two batches, not just on save_every_steps boundaries) always saves an
        # accurate resume position instead of the stale value from the last
        # periodic checkpoint.
        if save_state is not None:
            save_state["batch_idx"] = batch_idx

        # Break as soon as a preemption signal is seen instead of only at
        # epoch boundaries. The signal handler's own save runs in a background
        # thread concurrently with this loop, which can race and capture a
        # stale batch_idx/model state; stopping here promptly shrinks that
        # race window and lets the caller's post-loop save_and_exit() (which
        # joins the background thread) capture a consistent, current state.
        if handler is not None and handler.preempted:
            break

        # collate_multimodal yields an optional extras dict (gene_mask/coords).
        sequences, modality_targets, extras = _unpack_batch(batch_data)
        gene_mask = extras.get("gene_mask")
        if gene_mask is not None:
            gene_mask = gene_mask.to(device)

        is_profiling = do_profile and batch_idx < profile_batches

        if is_profiling and batch_idx > 0:
            _cuda_sync(device)
            t_data_load = time.perf_counter() - t_batch_start
            profile_stats.add("1_data_loading", t_data_load)

        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        sequences = sequences.to(device)
        organism_idx = torch.full((sequences.shape[0],), organism_idx, dtype=torch.long, device=device)

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("2_to_device", time.perf_counter() - t0)

        # Forward through backbone
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        # Collect all needed resolutions across all modalities
        all_resolutions = set()
        for modality in heads:
            all_resolutions.update(resolution_weights.get(modality, {}).keys())
        resolutions = tuple(all_resolutions)

        embeddings_pair = None  # only populated in sequence-parallel mode (contact_maps)
        original_length = None

        if sequence_parallel is not None:
            # Align sequence length so each rank's shard is divisible by 128.
            pad_multiple = world_size * 128 * 16
            seq_len = sequences.shape[1]
            original_length = seq_len
            padded_len = ((seq_len + pad_multiple - 1) // pad_multiple) * pad_multiple
            if padded_len > seq_len:
                n_pad = padded_len - seq_len
                if rank == 0:
                    import warnings
                    warnings.warn(
                        f"Sequence length {seq_len} not divisible by {pad_multiple}. "
                        f"Padding to {padded_len} (+{n_pad} bp) for sequence parallelism.",
                        stacklevel=2,
                    )
                sequences = torch.nn.functional.pad(sequences, (0, 0, 0, n_pad))

            model_module = model.module if hasattr(model, "module") else model
            model_module.train()
            if frozen_backbone:
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        emb_1bp, emb_128bp, embeddings_pair, need_1bp = sequence_parallel.forward(
                            model=model_module, sequence=sequences, organism_index=organism_idx,
                            resolutions=resolutions, original_length=original_length,
                        )
                emb_1bp = emb_1bp.detach() if emb_1bp is not None else None
                emb_128bp = emb_128bp.detach()
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    emb_1bp, emb_128bp, embeddings_pair, need_1bp = sequence_parallel.forward(
                        model=model_module, sequence=sequences, organism_index=organism_idx,
                        resolutions=resolutions, original_length=original_length,
                    )
            embeddings_dict = {128: emb_128bp}
            if need_1bp and emb_1bp is not None:
                embeddings_dict[1] = emb_1bp
        elif encoder_only:
            # Run only the CNN encoder; backbone is always frozen in encoder-only mode.
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx, encoder_only=True)
            embeddings_dict = {128: outputs["encoder_output"].detach()}
        else:
            backbone_ctx = torch.no_grad() if frozen_backbone else nullcontext()
            with backbone_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)

            embeddings_dict = _extract_embeddings(outputs, resolutions, frozen_backbone)

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("3_forward_backbone", time.perf_counter() - t0)

        # Forward through each head and compute losses
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        loss = torch.tensor(0.0, device=device)
        loss_components: dict[str, float] = {}

        for modality, head in heads.items():
            if modality not in modality_targets:
                continue

            modality_weight = modality_weights.get(modality, 1.0)
            res_weights = resolution_weights.get(modality, {})
            targets_dict = modality_targets[modality]
            head_module = head.module if hasattr(head, "module") else head

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                # Embeddings are NCL (channels-first); splice heads handle channels_last=False internally.
                #
                # Junction head under sequence parallelism: positions are global but
                # embeddings_dict[1] is only the local shard.  Use the sparse gather
                # path to avoid a full O(B*C*S) all-gather of the 1bp embedding.
                if (
                    sequence_parallel is not None
                    and isinstance(head_module, SpliceSitesJunctionHead)
                    and 1 in embeddings_dict
                ):
                    _cls_head = heads.get("splice_site")
                    _cls_head = _cls_head.module if hasattr(_cls_head, "module") else _cls_head
                    predictions = _call_splice_junction_head_sp(
                        head_module,
                        embeddings_dict[1],
                        targets_dict.get("junction_positions"),
                        organism_idx,
                        sequence_parallel,
                        original_length,
                        junction_top_k,
                        _cls_head,
                        device,
                    )
                else:
                    predictions = _run_head(
                        head, head_module, modality, embeddings_dict, organism_idx,
                        targets_dict, device, junction_top_k, heads,
                        embeddings_pair=embeddings_pair,
                    )

            modality_loss = torch.tensor(0.0, device=device)

            if isinstance(head_module, SPLICE_HEAD_TYPES):
                # In SP mode, slice sequence-aligned targets ("probs", "usage") to the local
                # shard. Sparse junction targets ("junction_matrix", "junction_positions",
                # "all_junctions") must NOT be sliced — they use global positions and are
                # handled by _call_splice_junction_head_sp.
                if sequence_parallel is not None:
                    _SEQ_KEYS = ("probs", "usage")
                    full_len = next(
                        (tgt.shape[1] for k, tgt in targets_dict.items()
                         if k in _SEQ_KEYS and isinstance(tgt, torch.Tensor) and tgt.ndim >= 2),
                        None,
                    )
                    if full_len is not None:
                        local_len = full_len // world_size
                        t_start = rank * local_len
                        splice_targets = {
                            k: (tgt.to(device)[:, t_start:t_start + local_len, :]
                                if k in _SEQ_KEYS and isinstance(tgt, torch.Tensor)
                                and tgt.ndim >= 2 and tgt.shape[1] == full_len
                                else (tgt.to(device) if isinstance(tgt, torch.Tensor) else tgt))
                            for k, tgt in targets_dict.items()
                        }
                    else:
                        splice_targets = {
                            k: (tgt.to(device) if isinstance(tgt, torch.Tensor) else tgt)
                            for k, tgt in targets_dict.items()
                        }
                else:
                    splice_targets = targets_dict
                modality_loss, splice_components = _compute_splice_loss(
                    head_module, predictions, splice_targets, device,
                    num_segments=num_segments,
                    junction_loss=junction_loss,
                    min_alpha_juncs=min_alpha_juncs,
                )
                # In SP mode, SpliceSitesJunctionHead all-gathers logits at every
                # rank so all ranks compute the identical full junction loss.  After
                # dist.all_reduce(SUM) the gradients would be world_size× too large
                # relative to heads that process only their local sequence shard.
                # Divide here so the effective gradient matches singlegpu / DDP.
                if (
                    sequence_parallel is not None
                    and world_size > 1
                    and isinstance(head_module, SpliceSitesJunctionHead)
                ):
                    modality_loss = modality_loss / world_size
                for k, v in splice_components.items():
                    loss_components[f"{modality}_{k}"] = v
            else:
                for res, weight in res_weights.items():
                    if res not in predictions or res not in targets_dict:
                        continue

                    pred = predictions[res]
                    targets = targets_dict[res].to(device)

                    # In SP mode, slice targets to match local shard for this rank.
                    if sequence_parallel is not None:
                        local_len = targets.shape[1] // world_size
                        t_start = rank * local_len
                        targets = targets[:, t_start:t_start + local_len, :]

                    targets = head_module.scale(
                        targets, organism_idx, resolution=res, channels_last=True
                    )
                    mask = torch.ones(pred.shape[0], 1, pred.shape[-1], dtype=torch.bool, device=device)

                    current_seq_len = pred.shape[-2]
                    multinomial_res = _compute_multinomial_resolution(
                        current_seq_len, num_segments, min_segment_size
                    )

                    loss_dict = multinomial_loss(
                        y_pred=pred,
                        y_true=targets,
                        mask=mask,
                        multinomial_resolution=multinomial_res,
                        positional_weight=positional_weight,
                        count_weight=count_weight,
                        channels_last=True,
                    )

                    res_loss = loss_dict["loss"] * weight

                    # Optional gene LFC term (Decima-style cross-track loss).
                    # Only applies at 1bp resolution to the head whose modality
                    # has a non-zero entry in `gene_loss_weights`. Mirrors
                    # upstream which threads gene_mask only through res=1.
                    gene_w = gene_loss_weights.get(modality, 0.0)
                    if (
                        res == 1
                        and gene_w > 0
                        and gene_mask is not None
                        and strand_channel_masks is not None
                        and modality in strand_channel_masks
                    ):
                        gene_loss, gene_aux = losses.gene_lfc_loss(
                            predictions=pred,
                            targets=targets,
                            targets_mask=mask,
                            gene_mask=gene_mask,
                            strand_channel_mask=strand_channel_masks[modality],
                            gene_cross_track_weight=gene_cross_track_weight,
                        )
                        res_loss = res_loss + weight * gene_w * gene_loss
                        loss_components[f"{modality}_gene_lfc"] = gene_loss.item()
                        loss_components[f"{modality}_gene_total_count"] = gene_aux["gene_loss_total_count"].item()
                        loss_components[f"{modality}_gene_positional"] = gene_aux["gene_loss_positional"].item()

                    modality_loss = modality_loss + res_loss
                    loss_components[f"{modality}_loss_{res}bp"] = res_loss.item()
                    loss_components[f"{modality}_loss_{res}bp_count"] = loss_dict["loss_total"].item()
                    loss_components[f"{modality}_loss_{res}bp_positional"] = loss_dict["loss_positional"].item()

            weighted_modality_loss = modality_loss * modality_weight
            loss = loss + weighted_modality_loss
            loss_components[f"{modality}_loss"] = modality_loss.item()
            modality_loss_accum[modality] += modality_loss.item()

        scaled_loss = loss / accumulation_steps

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("4_heads_and_loss", time.perf_counter() - t0)

        # Backward
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        # Optimizer step
        is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch = batch_idx == len(train_loader) - 1

        # Skip DDP gradient sync on intermediate accumulation steps
        no_sync = (
            accumulation_steps > 1
            and not is_accumulation_step
            and not is_last_batch
            and hasattr(model, "no_sync")
        )
        with model.no_sync() if no_sync else nullcontext():
            scaled_loss.backward()

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("5_backward", time.perf_counter() - t0)

        if is_accumulation_step or is_last_batch:
            if is_profiling:
                _cuda_sync(device)
                t0 = time.perf_counter()

            trainable_params = []
            for head in heads.values():
                trainable_params.extend([p for p in head.parameters() if p.requires_grad])
            trainable_params.extend([p for p in model.parameters() if p.requires_grad])

            # SP bypasses DDP's allreduce hook — sum gradients across ranks.
            # Each rank holds complementary sequence shards (not data-parallel copies),
            # so gradients must be summed (not averaged) to reconstruct the full gradient.
            if sequence_parallel is not None and world_size > 1:
                for p in trainable_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_step += 1

            if is_profiling:
                _cuda_sync(device)
                profile_stats.add("6_optimizer", time.perf_counter() - t0)

            if save_every_steps is not None and save_fn is not None:
                global_step = global_step_offset + opt_step
                if global_step % save_every_steps == 0:
                    if save_state is not None:
                        save_state["batch_idx"] = batch_idx + 1
                    save_fn()

        raw_loss = loss.item()
        total_loss_accum += raw_loss
        n_batches += 1

        running_loss += raw_loss
        accumulated_batches += 1

        current_lr = scheduler.get_last_lr()[0]

        if is_main_process(rank) and batch_idx % log_every == 0:
            avg_running_loss = running_loss / accumulated_batches
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "loss": f"{raw_loss:.4f}",
                    "run_loss": f"{avg_running_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

            if log_fn is not None:
                step_metrics = {
                    "batch": batch_idx,
                    "epoch": epoch,
                    "loss": raw_loss,
                    "running_loss": avg_running_loss,
                    "learning_rate": current_lr,
                    **loss_components,
                }
                log_fn(step_metrics)

            running_loss = 0.0
            accumulated_batches = 0

        if do_profile and batch_idx == profile_batches - 1:
            print(profile_stats.report(profile_batches))
            estimated_time = profile_stats.estimated_epoch_time(len(train_loader))
            print(f"\nESTIMATED EPOCH TIME: {estimated_time/60:.1f} minutes ({estimated_time/3600:.2f} hours)")
            print()

        if is_profiling:
            _cuda_sync(device)
        t_batch_start = time.perf_counter()

    # Reduce across processes
    avg_loss = total_loss_accum / max(1, n_batches)
    per_modality_loss = {m: v / max(1, n_batches) for m, v in modality_loss_accum.items()}

    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()

        for m in per_modality_loss:
            m_tensor = torch.tensor(per_modality_loss[m], device=device)
            m_tensor = reduce_tensor(m_tensor, world_size)
            per_modality_loss[m] = m_tensor.item()

    return avg_loss, per_modality_loss


@torch.no_grad()
def validate_multihead(
    model: nn.Module,
    heads: dict[str, nn.Module],
    val_loader: DataLoader,
    device: torch.device,
    modality_weights: dict[str, float],
    resolution_weights: dict[str, dict[int, float]],
    positional_weight: float,
    count_weight: float,
    use_amp: bool = True,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    compute_pearson: bool = True,
    rank: int = 0,
    world_size: int = 1,
    encoder_only: bool = False,
    organism_idx: int = 0,
    junction_top_k: int | None = None,
    junction_loss: str = "original",
    compute_per_sample: bool = False,
    min_alpha_juncs: int = 5,
    gene_annotation: Any = None,
    gene_expr_track_strands: list[str] | None = None,
    gene_expr_modality: str = "rna_seq",
    gene_expr_resolution: int | None = None,
    gene_expr_window_cache: dict | None = None,
) -> tuple[float, dict[str, Any]]:
    """Validate model with multiple modality heads.

    Args:
        model: AlphaGenome trunk model.
        heads: Dict mapping modality name to output head module.
        val_loader: Validation data loader.
        device: Torch device.
        modality_weights: Weight for each modality's loss.
        resolution_weights: Per-modality resolution weights.
        positional_weight: Weight for positional component.
        count_weight: Weight for count component.
        use_amp: Whether to use automatic mixed precision.
        num_segments: Number of segments for multinomial loss.
        min_segment_size: Minimum positions per segment.
        compute_pearson: Whether to compute Pearson R metrics.
        rank: Process rank for DDP.
        world_size: Total number of processes.
        encoder_only: If True, run only the CNN encoder and pass raw encoder output
            (B, S//128, 1536) to all heads as resolution 128.
        gene_annotation: Optional ``GeneAnnotation`` (with exon rows) enabling the
            gene-expression validation metric for
            ``gene_expr_modality``. When set (and ``compute_pearson``), per-window
            log-mean exon coverage is aggregated for predictions and observed
            targets, deduplicated across windows, and three Pearson correlations
            are emitted: ``{modality}_gene_log_expr_pearson_{across_genes,
            across_genes_norm, across_tracks_norm}``.
        gene_expr_track_strands: Per-track strand chars (``'+'/'-'/'.'``) for
            ``gene_expr_modality``, used for sense-strand matching. Required for a
            correct metric when tracks are stranded.
        gene_expr_modality: Modality the gene-expression metric applies to
            (default ``"rna_seq"``).
        gene_expr_resolution: Resolution of the head output the metric reads.
            Defaults to 128 under ``encoder_only`` (those heads emit 128bp only)
            and 1 otherwise. The metric itself is resolution-agnostic — it
            derives bin size from the interval width — so this only selects
            *which* head output to consume. It must be a resolution the head
            actually emits, or no windows accumulate and the metric is NaN.
        gene_expr_window_cache: Optional dict reused across epochs to memoize the
            per-window exon-mask lookup (the only pandas-heavy step). Create once
            in the training driver and pass it every epoch so each validation
            window's gene selection is built exactly once for the whole run.

    Returns:
        Tuple of (avg_total_loss, metrics_dict).
    """
    from alphagenome_pytorch.extensions.finetuning.distributed import (
        gather_tensors,
        is_main_process,
        reduce_tensor,
    )
    from alphagenome_pytorch.metrics import pearson_r, profile_pearson_r

    model.eval()
    for head in heads.values():
        head.eval()

    total_loss_accum = 0.0
    modality_loss_accum: dict[str, float] = {m: 0.0 for m in heads}
    n_batches = 0

    # For Pearson R - per modality and resolution
    accumulated_profile_r: dict[str, dict[int, list[Tensor]]] = {m: defaultdict(list) for m in heads}
    accumulated_pred_counts: dict[str, dict[int, list[Tensor]]] = {m: defaultdict(list) for m in heads}
    accumulated_true_counts: dict[str, dict[int, list[Tensor]]] = {m: defaultdict(list) for m in heads}

    # For splice Pearson R - per variant (full, nonzero, psi5, psi3) + binary_cls for auPRC
    # For junction head: dict[modality][variant] = {"pred": [], "true": []}
    # For usage head: dict[modality]["full"] = {"pred": [], "true": []}
    accumulated_splice: dict[str, dict[str, dict[str, list[Tensor]]]] = {
        m: {"full": {"pred": [], "true": []}} for m in heads
    }

    # For classification head auPRC: accumulate logits and one-hot targets
    accumulated_cls: dict[str, dict[str, list[Tensor]]] = {
        m: {"logits": [], "true": []} for m in heads
    }

    # For per-sample Pearson (only populated when compute_per_sample=True)
    # accumulated_junc_ps_pearson[modality] = list[n_s dicts], each {"full":..., "nonzero":...}
    accumulated_junc_ps_pearson: dict[str, list] = {m: [] for m in heads}
    # accumulated_usage_ps_pearson[modality] = list[n_s dicts], each {"pred":[], "true":[]}
    accumulated_usage_ps_pearson: dict[str, list] = {m: [] for m in heads}

    # For the exon-based gene-expression metric: per-window (gene_ids, pred, obs).
    gene_expr_enabled = (
        gene_annotation is not None
        and compute_pearson
        and gene_expr_modality in heads
    )
    # Encoder-only heads emit 128bp only, so the 1bp default would match no
    # output and the metric would silently report NaN every epoch.
    if gene_expr_resolution is None:
        gene_expr_resolution = 128 if encoder_only else 1
    if gene_expr_enabled:
        # The metric only reads the head output at `gene_expr_resolution`. If the
        # modality never emits it, no window ever accumulates and every epoch
        # reports n_genes=0 with NaN correlations and no error. Fail instead.
        available = sorted(resolution_weights.get(gene_expr_modality, {}))
        if gene_expr_resolution not in available:
            raise ValueError(
                f"gene-expression metric requested at {gene_expr_resolution}bp, but "
                f"'{gene_expr_modality}' emits {available or 'no resolutions'}"
                + (" (encoder_only forces 128bp)" if encoder_only else "")
                + ". Pass gene_expr_resolution matching an emitted resolution."
            )
    gene_expr_windows: list[tuple[list[str], Tensor, Tensor]] = []

    if is_main_process(rank):
        pbar = tqdm(val_loader, desc="Validation")
    else:
        pbar = val_loader

    # @torch.no_grad() already wraps this whole function; the nested
    # `with torch.no_grad():` here is redundant but kept to match this
    # function's original indentation depth without a large reflow.
    with torch.no_grad():
        for batch_data in pbar:
            sequences, modality_targets, extras = _unpack_batch(batch_data)
            coords = extras.get("coords")
            sequences = sequences.to(device)
            organism_idx = torch.full((sequences.shape[0],), organism_idx, dtype=torch.long, device=device)

            # Collect all resolutions
            all_resolutions = set()
            for modality in heads:
                all_resolutions.update(resolution_weights.get(modality, {}).keys())
            resolutions = tuple(all_resolutions)

            if encoder_only:
                outputs = model(sequences, organism_idx, encoder_only=True)
                embeddings_dict = {128: outputs["encoder_output"]}
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)

                embeddings_dict = _extract_embeddings(outputs, resolutions)

            loss = torch.tensor(0.0, device=device)

            for modality, head in heads.items():
                if modality not in modality_targets:
                    continue

                modality_weight = modality_weights.get(modality, 1.0)
                res_weights = resolution_weights.get(modality, {})
                targets_dict = modality_targets[modality]

                head_module = head.module if hasattr(head, "module") else head

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    predictions_scaled = _run_head(
                        head, head_module, modality, embeddings_dict, organism_idx,
                        targets_dict, device, junction_top_k, heads,
                    )
                    if compute_pearson and not isinstance(head_module, SPLICE_HEAD_TYPES):
                        predictions_unscaled = head(
                            embeddings_dict, organism_idx, return_scaled=False, channels_last=True
                        )

                modality_loss = torch.tensor(0.0, device=device)

                if isinstance(head_module, SPLICE_HEAD_TYPES):
                    modality_loss, _ = _compute_splice_loss(
                        head_module, predictions_scaled, targets_dict, device,
                        num_segments=num_segments,
                        junction_loss=junction_loss,
                        min_alpha_juncs=min_alpha_juncs,
                    )
                    if compute_pearson:
                        # Accumulate logits + one-hot targets for auPRC (classification head only).
                        # Keep all active (splice-site) positions + an equal-sized random subsample
                        # of background positions so auPRC includes true negatives without
                        # accumulating the full (B, S, 5) tensors at 1bp resolution (OOM).
                        if isinstance(head_module, SpliceSitesClassificationHead):
                            if 1 in predictions_scaled and "probs" in targets_dict:
                                _logits_flat = predictions_scaled[1].float().reshape(-1, 5)  # (B*S, 5)
                                _true_flat   = targets_dict["probs"].float().reshape(-1, 5)  # (B*S, 5)
                                _active_flat = _true_flat.any(dim=-1)                        # (B*S,)
                                n_active = int(_active_flat.sum().item())
                                parts_logits, parts_true = [], []
                                if n_active > 0:
                                    parts_logits.append(_logits_flat[_active_flat])
                                    parts_true.append(_true_flat[_active_flat])
                                    # Subsample background at 1:1 ratio with active positions
                                    _bg_idx = (~_active_flat).nonzero(as_tuple=True)[0]
                                    n_bg = _bg_idx.shape[0]
                                    if n_bg > 0:
                                        n_sample = min(n_active, n_bg)
                                        perm = torch.randperm(n_bg, device=_logits_flat.device)[:n_sample]
                                        parts_logits.append(_logits_flat[_bg_idx[perm]])
                                        parts_true.append(_true_flat[_bg_idx[perm]])
                                if parts_logits:
                                    accumulated_cls[modality]["logits"].append(torch.cat(parts_logits).cpu())
                                    accumulated_cls[modality]["true"].append(torch.cat(parts_true).cpu())

                        result = _extract_splice_pearson_pairs(
                            head_module, predictions_scaled, targets_dict, device,
                            min_alpha_juncs=min_alpha_juncs,
                        )
                        if result is not None and result != (None, None):
                            # For junction head: result is a dict of variants
                            if isinstance(head_module, SpliceSitesJunctionHead):
                                for variant_name, variant_data in result.items():
                                    if variant_name not in accumulated_splice[modality]:
                                        accumulated_splice[modality][variant_name] = {"pred": [], "true": []}
                                    if variant_data:
                                        accumulated_splice[modality][variant_name]["pred"].append(variant_data["pred"].float().cpu())
                                        accumulated_splice[modality][variant_name]["true"].append(variant_data["true"].float().cpu())
                                if compute_per_sample:
                                    ps_junc = _extract_junction_pearson_per_sample(
                                        predictions_scaled, targets_dict, device
                                    )
                                    if ps_junc is not None:
                                        if not accumulated_junc_ps_pearson[modality]:
                                            accumulated_junc_ps_pearson[modality] = [
                                                {"full": {"pred": [], "true": []}, "nonzero": {"pred": [], "true": []}}
                                                for _ in range(len(ps_junc))
                                            ]
                                        for s, data in enumerate(ps_junc):
                                            for variant in ("full", "nonzero"):
                                                if data[variant] is not None:
                                                    p, t = data[variant]
                                                    accumulated_junc_ps_pearson[modality][s][variant]["pred"].append(p)
                                                    accumulated_junc_ps_pearson[modality][s][variant]["true"].append(t)
                            # For usage head: result is a dict of variants (always includes "full",
                            # optionally "alpha" when min_alpha_juncs > 0)
                            else:
                                for variant_name, (vp, vt) in result.items():
                                    if variant_name not in accumulated_splice[modality]:
                                        accumulated_splice[modality][variant_name] = {"pred": [], "true": []}
                                    accumulated_splice[modality][variant_name]["pred"].append(vp.float().cpu())
                                    accumulated_splice[modality][variant_name]["true"].append(vt.float().cpu())
                                if compute_per_sample:
                                    ps_usage = _extract_usage_pearson_per_sample(
                                        predictions_scaled, targets_dict, device
                                    )
                                    if ps_usage is not None:
                                        if not accumulated_usage_ps_pearson[modality]:
                                            accumulated_usage_ps_pearson[modality] = [
                                                {"pred": [], "true": []} for _ in range(len(ps_usage))
                                            ]
                                        for s, item in enumerate(ps_usage):
                                            if item is not None:
                                                p, t = item
                                                accumulated_usage_ps_pearson[modality][s]["pred"].append(p)
                                                accumulated_usage_ps_pearson[modality][s]["true"].append(t)
                else:
                    for res, weight in res_weights.items():
                        if res not in predictions_scaled or res not in targets_dict:
                            continue

                        pred_scaled = predictions_scaled[res]
                        targets = targets_dict[res].to(device)
                        targets_scaled = head_module.scale(
                            targets, organism_idx, resolution=res, channels_last=True
                        )
                        mask = torch.ones(
                            pred_scaled.shape[0], 1, pred_scaled.shape[-1], dtype=torch.bool, device=device
                        )

                        current_seq_len = pred_scaled.shape[-2]
                        multinomial_res = _compute_multinomial_resolution(
                            current_seq_len, num_segments, min_segment_size
                        )

                        loss_dict = multinomial_loss(
                            y_pred=pred_scaled,
                            y_true=targets_scaled,
                            mask=mask,
                            multinomial_resolution=multinomial_res,
                            positional_weight=positional_weight,
                            count_weight=count_weight,
                            channels_last=True,
                        )

                        res_loss = loss_dict["loss"] * weight
                        modality_loss = modality_loss + res_loss

                        # Accumulate for Pearson R
                        if compute_pearson:
                            pred_unscaled = predictions_unscaled[res]
                            batch_profile_r = profile_pearson_r(pred_unscaled, targets)
                            accumulated_profile_r[modality][res].append(batch_profile_r.float().cpu())
                            accumulated_pred_counts[modality][res].append(pred_unscaled.sum(dim=1).float().cpu())
                            accumulated_true_counts[modality][res].append(targets.sum(dim=1).float().cpu())

                            # Exon-based gene-expression metric.
                            if (
                                gene_expr_enabled
                                and modality == gene_expr_modality
                                and res == gene_expr_resolution
                                and coords is not None
                            ):
                                _accumulate_gene_expr_windows(
                                    gene_expr_windows,
                                    pred_unscaled=pred_unscaled,
                                    targets=targets,
                                    coords=coords,
                                    annotation=gene_annotation,
                                    track_strands=gene_expr_track_strands,
                                    window_cache=gene_expr_window_cache,
                                )

                weighted_modality_loss = modality_loss * modality_weight
                loss = loss + weighted_modality_loss
                modality_loss_accum[modality] += modality_loss.item()

            total_loss_accum += loss.item()
            n_batches += 1

    # Reduce across processes
    avg_loss = total_loss_accum / max(1, n_batches)
    per_modality_loss = {m: v / max(1, n_batches) for m, v in modality_loss_accum.items()}

    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()

        for m in per_modality_loss:
            m_tensor = torch.tensor(per_modality_loss[m], device=device)
            m_tensor = reduce_tensor(m_tensor, world_size)
            per_modality_loss[m] = m_tensor.item()

    # Build metrics dict
    metrics: dict[str, Any] = {}
    for m, v in per_modality_loss.items():
        metrics[f"{m}_loss"] = v

    # Compute Pearson R
    if compute_pearson:
        for modality in heads:
            for res in resolution_weights.get(modality, {}).keys():
                if res in accumulated_profile_r[modality] and accumulated_profile_r[modality][res]:
                    all_profile_r = torch.cat(accumulated_profile_r[modality][res], dim=0)
                    if world_size > 1:
                        all_profile_r = gather_tensors(all_profile_r, world_size, device)
                    metrics[f"{modality}_{res}bp_profile_pearson_r_mean"] = all_profile_r.mean().item()
                    metrics[f"{modality}_{res}bp_profile_pearson_r_std"] = all_profile_r.std().item()
                    metrics[f"{modality}_{res}bp_profile_pearson_r_values"] = all_profile_r.flatten().tolist()

                if res in accumulated_pred_counts[modality] and accumulated_pred_counts[modality][res]:
                    all_pred_counts = torch.cat(accumulated_pred_counts[modality][res], dim=0)
                    all_true_counts = torch.cat(accumulated_true_counts[modality][res], dim=0)
                    if world_size > 1:
                        all_pred_counts = gather_tensors(all_pred_counts, world_size, device)
                        all_true_counts = gather_tensors(all_true_counts, world_size, device)
                    if all_pred_counts.shape[0] > 1:
                        count_r = pearson_r(all_pred_counts, all_true_counts, dim=0)
                        metrics[f"{modality}_{res}bp_count_pearson_r"] = count_r.mean().item()
                    else:
                        metrics[f"{modality}_{res}bp_count_pearson_r"] = float("nan")

    # Compute splice Pearson R for each variant
    # Variants that are count-scale (apply log1p before Pearson)
    _COUNT_VARIANTS = {"full", "nonzero"}
    # Variants handled separately (auPRC, not Pearson)
    _AUPRC_VARIANTS = {"binary_cls"}

    if compute_pearson:
        for modality in heads:
            head_module = heads[modality].module if hasattr(heads[modality], "module") else heads[modality]
            is_junction = isinstance(head_module, SpliceSitesJunctionHead)

            for variant_name, variant_data in accumulated_splice[modality].items():
                if variant_name in _AUPRC_VARIANTS:
                    continue
                if not variant_data["pred"]:
                    continue
                all_pred = torch.cat(variant_data["pred"], dim=0)
                all_true = torch.cat(variant_data["true"], dim=0)
                if world_size > 1:
                    all_pred = gather_tensors(all_pred, world_size, device)
                    all_true = gather_tensors(all_true, world_size, device)
                if all_pred.shape[0] > 1:
                    # log1p only for raw count variants; PSI variants are already in [0,1]
                    use_log1p = is_junction and variant_name in _COUNT_VARIANTS
                    _pred_for_r = torch.log1p(all_pred) if use_log1p else all_pred
                    _true_for_r = torch.log1p(all_true) if use_log1p else all_true
                    r = pearson_r(_pred_for_r.unsqueeze(0), _true_for_r.unsqueeze(0), dim=1)
                    metric_key = f"{modality}_pearson_r"
                    if variant_name != "full":
                        metric_key += f"_{variant_name}"
                    metrics[metric_key] = r.item()

            # Add target nonzero fraction diagnostic for junction head
            if is_junction and "full" in accumulated_splice[modality]:
                full_true = accumulated_splice[modality]["full"]["true"]
                if full_true:
                    all_true = torch.cat(full_true, dim=0)
                    nonzero_frac = (all_true > 0).float().mean().item()
                    metrics[f"{modality}_target_nonzero_frac"] = nonzero_frac

    # Compute junction auPRC (binary classification: true junction vs background)
    if compute_pearson:
        from sklearn.metrics import average_precision_score as _avg_prec

        for modality in heads:
            head_module = heads[modality].module if hasattr(heads[modality], "module") else heads[modality]
            if not isinstance(head_module, SpliceSitesJunctionHead):
                continue
            bincls = accumulated_splice[modality].get("binary_cls", {})
            if not bincls.get("pred"):
                continue
            all_pred = torch.cat(bincls["pred"], dim=0)
            all_true = torch.cat(bincls["true"], dim=0)
            if world_size > 1:
                all_pred = gather_tensors(all_pred, world_size, device)
                all_true = gather_tensors(all_true, world_size, device)
            if all_true.sum() > 0 and all_pred.shape[0] > 1:
                metrics[f"{modality}_auprc_junction"] = _avg_prec(
                    all_true.cpu().numpy(), all_pred.cpu().numpy()
                )

    # Compute per-sample Pearson rows (only when compute_per_sample=True)
    if compute_pearson:
        import numpy as np

        if compute_per_sample:
            # Infer n_s from junction accumulation, fallback to usage
            n_s = None
            junc_modality = None
            for m in heads:
                if accumulated_junc_ps_pearson[m]:
                    n_s = len(accumulated_junc_ps_pearson[m])
                    junc_modality = m
                    break
            if n_s is None:
                for m in heads:
                    if accumulated_usage_ps_pearson[m]:
                        n_s = len(accumulated_usage_ps_pearson[m])
                        break

            if n_s is not None:
                per_sample_metrics = [{} for _ in range(n_s)]

                # Junction Pearson per sample (log1p)
                for modality in heads:
                    hm = heads[modality].module if hasattr(heads[modality], "module") else heads[modality]
                    if not isinstance(hm, SpliceSitesJunctionHead):
                        continue
                    for s in range(n_s):
                        if not accumulated_junc_ps_pearson[modality] or s >= len(accumulated_junc_ps_pearson[modality]):
                            continue
                        for variant, metric_key in [
                            ("full",    f"{modality}_pearson_r"),
                            ("nonzero", f"{modality}_pearson_r_nonzero"),
                        ]:
                            data = accumulated_junc_ps_pearson[modality][s][variant]
                            if data["pred"]:
                                all_pred = torch.log1p(torch.cat(data["pred"]))
                                all_true = torch.log1p(torch.cat(data["true"]))
                                if all_pred.shape[0] > 1:
                                    r = pearson_r(all_pred.unsqueeze(0), all_true.unsqueeze(0), dim=1)
                                    per_sample_metrics[s][metric_key] = r.item()

                # Usage Pearson per sample
                for modality in heads:
                    hm = heads[modality].module if hasattr(heads[modality], "module") else heads[modality]
                    if not isinstance(hm, SpliceSitesUsageHead):
                        continue
                    for s in range(n_s):
                        if not accumulated_usage_ps_pearson[modality] or s >= len(accumulated_usage_ps_pearson[modality]):
                            continue
                        data = accumulated_usage_ps_pearson[modality][s]
                        if data["pred"]:
                            all_pred = torch.cat(data["pred"])
                            all_true = torch.cat(data["true"])
                            if all_pred.shape[0] > 1:
                                r = pearson_r(all_pred.unsqueeze(0), all_true.unsqueeze(0), dim=1)
                                per_sample_metrics[s][f"{modality}_pearson_r"] = r.item()

                metrics["_per_sample"] = per_sample_metrics

    # Compute auPRC for classification head
    if compute_pearson:
        import torch.nn.functional as F
        from sklearn.metrics import average_precision_score

        _CLS_NAMES = ["donor_pos", "acceptor_pos", "donor_neg", "acceptor_neg", "no_site"]

        for modality in heads:
            head_module = heads[modality].module if hasattr(heads[modality], "module") else heads[modality]
            if not isinstance(head_module, SpliceSitesClassificationHead):
                continue
            if not accumulated_cls[modality]["logits"]:
                continue

            # Tensors contain active positions + subsampled background (already (N, 5) shaped).
            all_logits = torch.cat(accumulated_cls[modality]["logits"], dim=0)  # (N, 5)
            all_true   = torch.cat(accumulated_cls[modality]["true"],   dim=0)  # (N, 5)
            probs = F.softmax(all_logits, dim=-1)
            probs_flat = probs.cpu().numpy()
            true_flat  = all_true.cpu().numpy()

            if true_flat.shape[0] == 0:
                continue

            for i, cls_name in enumerate(_CLS_NAMES):
                if true_flat[:, i].sum() > 0:
                    ap = average_precision_score(true_flat[:, i], probs_flat[:, i])
                    metrics[f"{modality}_auprc_{cls_name}"] = ap

            # Macro and weighted average over splice-site classes only (exclude no_site)
            splice_cols = [i for i, n in enumerate(_CLS_NAMES) if n != "no_site"
                           and true_flat[:, i].sum() > 0]
            if splice_cols:
                per_class_aps = [metrics[f"{modality}_auprc_{_CLS_NAMES[i]}"] for i in splice_cols]
                metrics[f"{modality}_auprc_macro"] = sum(per_class_aps) / len(per_class_aps)
                weights = [true_flat[:, i].sum() for i in splice_cols]
                total = sum(weights)
                metrics[f"{modality}_auprc_weighted"] = sum(ap * w / total for ap, w in zip(per_class_aps, weights))

    # Exon-based gene-expression metric: gather per-window
    # (gene_ids, pred, obs) across ranks, dedup genes, and emit three Pearsons.
    if gene_expr_enabled:
        metrics.update(_gene_expr_metrics(
            gene_expr_windows, modality=gene_expr_modality, world_size=world_size,
        ))

    return avg_loss, metrics


def train_epoch_sequence_parallel(
    model: nn.Module,
    heads: dict[str, nn.Module],
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    modality_weights: dict[str, float],
    resolution_weights: dict[str, dict[int, float]],
    positional_weight: float,
    count_weight: float,
    sequence_parallel: Any,
    epoch: int,
    log_every: int,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    frozen_backbone: bool = False,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    train_sampler: DistributedSampler | None = None,
    rank: int = 0,
    world_size: int = 1,
    max_grad_norm: float = 1.0,
    profile_batches: int = 0,
    log_fn: Any | None = None,
    encoder_only: bool = False,
    *,
    save_every_steps: int | None = None,
    save_fn: Any | None = None,
    global_step_offset: int = 0,
    skip_batches: int = 0,
    save_state: dict | None = None,
    junction_top_k: int | None = None,
    junction_loss: str = "original",
    gene_loss_weights: dict[str, float] | None = None,
    gene_cross_track_weight: float = 5.0,
    strand_channel_masks: dict[str, Tensor] | None = None,
    organism_idx: int = 0,
) -> tuple[float, dict[str, float]]:  # noqa: D103 — thin wrapper
    """Thin wrapper — delegates to train_epoch_multihead with sequence_parallel enabled."""
    from alphagenome_pytorch.sequence_parallel import SequenceParallelism
    if not isinstance(sequence_parallel, SequenceParallelism):
        raise ValueError("sequence_parallel must be a SequenceParallelism instance")
    return train_epoch_multihead(
        model=model, heads=heads, train_loader=train_loader, optimizer=optimizer,
        scheduler=scheduler, device=device, modality_weights=modality_weights,
        resolution_weights=resolution_weights, positional_weight=positional_weight,
        count_weight=count_weight, epoch=epoch, log_every=log_every, use_amp=use_amp,
        accumulation_steps=accumulation_steps, frozen_backbone=frozen_backbone,
        num_segments=num_segments, min_segment_size=min_segment_size,
        train_sampler=train_sampler, rank=rank, world_size=world_size,
        max_grad_norm=max_grad_norm, profile_batches=profile_batches, log_fn=log_fn,
        encoder_only=encoder_only, save_every_steps=save_every_steps, save_fn=save_fn,
        global_step_offset=global_step_offset, skip_batches=skip_batches,
        save_state=save_state, junction_top_k=junction_top_k, junction_loss=junction_loss,
        sequence_parallel=sequence_parallel, gene_loss_weights=gene_loss_weights,
        gene_cross_track_weight=gene_cross_track_weight,
        strand_channel_masks=strand_channel_masks, organism_idx=organism_idx,
    )


__all__ = [
    "collate_genomic",
    "ModalityConfig",
    "MODALITY_CONFIGS",
    "create_lr_scheduler",
    "compute_finetuning_loss",
    "train_epoch",
    "validate",
    "save_checkpoint",
    # Enhanced versions with DDP support
    "ProfilingStats",
    "train_epoch_ddp",
    "validate_ddp",
    # Multi-head training
    "train_epoch_multihead",
    "validate_multihead",
    # Sequence parallel training
    "train_epoch_sequence_parallel",
]
