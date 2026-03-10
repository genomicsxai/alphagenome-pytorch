"""
Utility functions for splicing prediction heads.
"""
import torch

def _top_k_splice_sites(
    splice_site_classifications: torch.Tensor,
    k: int,
    pad_to_length: int,
    threshold: float,
) -> torch.Tensor:
    """
    Returns the top k splice sites from the predictions.
    We use B for batch, S for sequence length.

    Args:
        x: A tensor of shape [B, S, 5] containing splice
        site predictions.
        k: The number of splice sites to return.
        pad_to_length: Pad the output to this length.
        threshold: Threshold to filter out low confidecne splice sites.

    Returns:
        A tensor of shape [B, 4, K] containing the top k of each
        splice site type.
    """
    batch_size = splice_site_classifications.shape[0]
    values, positions = torch.topk(
        splice_site_classifications[..., :4], # Remove "other" class
        k=k,
        dim=1,
    )
    # Apply thresholding, marking invalid positions with a large integer
    INVALID_INT_POSITION = torch.iinfo(torch.long).max
    if threshold > 0:
        positions = torch.where(values < threshold, INVALID_INT_POSITION, positions)
    positions, _ = torch.sort(positions, dim=1, descending=False) # Need to sort for RoPE
    if threshold > 0:
        positions = torch.where(positions == INVALID_INT_POSITION, -1, positions)
    # Positions is now shape [B, 4 (+ve/-ve donors and acceptors), k]
    positions = positions.swapaxes(1, 2)
    if positions.shape[2] < pad_to_length:
        padding_shape = (batch_size, 4, pad_to_length - positions.shape[2])
        positions = torch.cat(
            [positions, torch.full(padding_shape, -1)], dim=2
        )
    return positions

def generate_splice_site_positions(
    ref: torch.Tensor,
    alt: torch.Tensor | None,
    true_splice_sites: torch.Tensor | None,
    *,
    k: int,
    pad_to_length: int,
    threshold: float,
) -> torch.Tensor:
    """
    Returns the top k splice sites from a sequence and (optionally)
    a comparison alt sequence. If the alt is provided, we include
    the maximum prediction between the ref and alt for each splice site
    class.

    If true_splice_sites is provided, we override any location with
    the true splice site information.

    Args:
        ref: A tensor of shape [B, S, 5] containing the reference sequence.
        alt: A tensor of shape [B, S, 5] containing the alternative sequence.
        true_splice_sites: A tensor of shape [B, S, 5] containing true splice site
            classifications. If provided, we override any location with the true
            splice site information.
        k: The number of splice sites to return.
        pad_to_length: Pad the output to this length.
        threshold: Threshold to filter out low confidence splice sites.

    Returns:
        A tensor of shape [B, 4, K] containing the top k of each
        splice site type.
    """
    if alt is not None:
        ref = torch.max(ref, alt)
    if true_splice_sites is not None:
        ref = torch.max(ref, true_splice_sites)
    return _top_k_splice_sites(ref, k, pad_to_length, threshold)


def unstack_junction_predictions(
    splice_junction_prediction: torch.Tensor,
    splice_site_positions: torch.Tensor,
    interval_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unstack splice junction predictions to list of junctions.

    Args:
        splice_junction_prediction: [B, P, P, 2*T] (pos/neg concatenated)
        splice_site_positions: [B, 4, P] (pos_donor, pos_acceptor, neg_donor, neg_acceptor)
        interval_start: Start position of the interval to offset coordinates.

    Returns:
        tuple of (scores [B, N, T], starts [B, N], ends [B, N], strands [B, N])
        where N is the number of valid junctions.
        Strands are returned as integers (0 for +, 1 for -).
    """
    B, P, _, TwoT = splice_junction_prediction.shape
    num_tissues = TwoT // 2

    # Separate pos and neg predictions
    # JAX model concats [pos_mask, neg_mask] at axis -1
    # prediction shape is [B, P, P, 2*T]
    # first T channels are positive strand, next T are negative strand
    pred_pos = splice_junction_prediction[..., :num_tissues]  # [B, P, P, T]
    pred_neg = splice_junction_prediction[..., num_tissues:]  # [B, P, P, T]

    # Positions: [B, 4, P]
    # 0: pos_donor, 1: pos_acceptor, 2: neg_donor, 3: neg_acceptor
    pos_donors = splice_site_positions[:, 0, :]    # [B, P]
    pos_acceptors = splice_site_positions[:, 1, :] # [B, P]
    neg_donors = splice_site_positions[:, 2, :]    # [B, P]
    neg_acceptors = splice_site_positions[:, 3, :] # [B, P]

    # We need to flatten PxP grid to a list of junctions
    # For positive strand: donor (rows) -> acceptor (cols)
    # Start = donor + 1, End = acceptor

    # Create meshgrids of indices
    # We want [B, P, P] of donor positions and acceptor positions
    # pos_donors is [B, P]. Expand to [B, P, 1] then title to [B, P, P]
    p_d = pos_donors.unsqueeze(2).expand(-1, -1, P)
    p_a = pos_acceptors.unsqueeze(1).expand(-1, P, -1)
    
    # Same for negative strand
    n_d = neg_donors.unsqueeze(2).expand(-1, -1, P)
    n_a = neg_acceptors.unsqueeze(1).expand(-1, P, -1)

    # Flatten
    # [B, P*P]
    p_d_flat = p_d.reshape(B, -1)
    p_a_flat = p_a.reshape(B, -1)
    n_d_flat = n_d.reshape(B, -1)
    n_a_flat = n_a.reshape(B, -1)
    
    pred_pos_flat = pred_pos.reshape(B, -1, num_tissues)
    pred_neg_flat = pred_neg.reshape(B, -1, num_tissues)

    # Valid masks (pad value is -1)
    # Positive: need valid donor AND valid acceptor
    mask_pos = (p_d_flat != -1) & (p_a_flat != -1)
    # Negative: need valid donor AND valid acceptor
    mask_neg = (n_d_flat != -1) & (n_a_flat != -1)

    # Calculate coordinates
    # Positive strand: Start = donor + 1, End = acceptor
    # Negative strand: Start = acceptor + 1, End = donor
    # (Note: JAX unstack code: starts = pos_donors + 1 (concatenated with neg_acceptors + 1))
    
    starts_pos = p_d_flat + 1 + interval_start
    ends_pos = p_a_flat + interval_start
    
    starts_neg = n_a_flat + 1 + interval_start
    ends_neg = n_d_flat + interval_start

    # Filter invalid order (Start < End) and > 0
    mask_pos = mask_pos & (starts_pos < ends_pos) & (starts_pos > 0)
    mask_neg = mask_neg & (starts_neg < ends_neg) & (starts_neg > 0)

    # We return everything, scorer can filter/select max
    # Concatenate pos and neg
    # To keep batch dimension aligned, we might simply cat everything and return list of tensors 
    # But usually B=1 in scoring.
    # Let's flatten batch for simplicity as Scorer handles one variant usually?
    # No, scorer handles batch. But for simplicity let's return flattened valid junctions + batch_idx
    
    # Actually, SpliceJunctionScorer in JAX does per-gene max.
    # We will return the raw tensors and let scorer handle selection.
    
    return (
        torch.cat([pred_pos_flat, pred_neg_flat], dim=1), # Scores [B, 2*P*P, T]
        torch.cat([starts_pos, starts_neg], dim=1),       # Starts [B, 2*P*P]
        torch.cat([ends_pos, ends_neg], dim=1),           # Ends [B, 2*P*P]
        torch.cat([torch.zeros_like(mask_pos, dtype=torch.long), torch.ones_like(mask_neg, dtype=torch.long)], dim=1), # Strands (0=+, 1=-)
        torch.cat([mask_pos, mask_neg], dim=1)            # Mask [B, 2*P*P]
    )
