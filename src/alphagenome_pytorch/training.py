"""
Training utilities for AlphaGenome PyTorch.

Provides configuration, loss aggregation, optimizer/scheduler factories,
and metrics for training AlphaGenome models.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Any, Tuple
import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from . import losses


def _build_strand_channel_mask(strands: Sequence[str]) -> torch.Tensor:
    """Build a `[2, 1, C]` strand-channel mask from a per-track strand list.

    Mirrors upstream `GenomeTracksHead._get_strand_channel_mask` semantics:
      - axis 0 dim 0 (positive-strand bucket): `+` and `.` tracks contribute.
      - axis 0 dim 1 (negative-strand bucket): `-` and `.` tracks contribute.

    Args:
        strands: Sequence of one-character strand codes per track. Each must
            be one of `+`, `-`, `.`. Length defines C.

    Returns:
        Bool tensor of shape `[2, 1, C]`.
    """
    valid = {"+", "-", "."}
    invalid = sorted({s for s in strands if s not in valid})
    if invalid:
        raise ValueError(
            f"track strands must be one of {sorted(valid)}; got invalid values: {invalid}"
        )
    plus_compat = torch.tensor([s in ("+", ".") for s in strands], dtype=torch.bool)
    minus_compat = torch.tensor([s in ("-", ".") for s in strands], dtype=torch.bool)
    return torch.stack([plus_compat, minus_compat], dim=0).unsqueeze(1)  # [2, 1, C]


@dataclass
class AlphaGenomeTrainingConfig:
    """Training configuration matching paper parameters.
    
    Default values are from the AlphaGenome paper:
    - AdamW optimizer with β₁=0.9, β₂=0.999, ε=10⁻⁸
    - Weight decay: 0.4
    - Learning rate: 0.004 with linear warmup + cosine decay
    - 15,000 total steps (5,000 warmup + 10,000 decay)
    - Batch size: 64
    """
    learning_rate: float = 0.004
    weight_decay: float = 0.4
    warmup_steps: int = 5000
    total_steps: int = 15000
    batch_size: int = 64
    
    # AdamW hyperparameters
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Loss configuration
    multinomial_resolution: int = 128
    positional_weight: float = 5.0  # JAX production value (AG_MODEL.md line 321)


# Default per-head loss weights matching upstream JAX `HeadConfig.loss_weight`
# (google-deepmind/alphagenome_research, commit 7046b72). All heads weight 1.0
# except splice junctions, which upstream weights at 0.2 to balance the very
# different magnitude of the junction loss against the count-based heads.
DEFAULT_HEAD_WEIGHTS = {
    'atac': 1.0,
    'dnase': 1.0,
    'procap': 1.0,
    'cage': 1.0,
    'rna_seq': 1.0,
    'chip_tf': 1.0,
    'chip_histone': 1.0,
    'contact_maps': 1.0,
    'splice_sites': 1.0,
    'splice_site_usage': 1.0,
    'splice_junctions': 0.2,
}


class AlphaGenomeLoss(nn.Module):
    """Multi-head loss aggregation for AlphaGenome.

    Computes weighted loss across all output heads using appropriate
    loss functions for each head type.

    IMPORTANT: For correct training behavior, you MUST:
    1. Pass the model to this loss function: `AlphaGenomeLoss(model=model)`
    2. Call model with `return_scaled_predictions=True` during training
    3. Pass `organism_index` to the forward method

    Aggregation: the total loss is `sum(head_weight * head_loss) / num_heads`,
    where the divisor is the *count* of heads that produced a loss this step
    (not the sum of weights). A weight of 0.2 therefore down-weights that
    head's contribution by exactly 0.2× relative to a weight-1.0 head with
    the same residual, matching upstream JAX `HeadConfig.loss_weight` semantics.

    Args:
        model: AlphaGenome model instance. Required to access head modules for target scaling.
            Without this, targets will not be scaled to model space, resulting in incorrect gradients.
        heads: List of head names to compute loss for. If None, uses all heads.
        head_weights: Dict mapping head names to loss weights. Merged *on top
            of* `DEFAULT_HEAD_WEIGHTS`, so a partial dict only overrides the
            heads it names and leaves the rest at their upstream defaults
            (1.0 for count and splice-site heads, 0.2 for splice junctions).
            Pass `None` to use the defaults unchanged.
        multinomial_resolution: Resolution for multinomial loss computation. This is the
            segment size used to divide the sequence for multinomial loss. For JAX parity,
            use `seq_len` (full sequence as 1 segment) or `seq_len // 8` for 8 segments
            matching the model specification in their supplemental.
        positional_weight: Weight for positional component of multinomial loss.
            JAX production uses 5.0 (default), matching Borzoi.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        heads: Optional[List[str]] = None,
        head_weights: Optional[Dict[str, float]] = None,
        multinomial_resolution: int = 128,
        positional_weight: float = 5.0,  # JAX production value
        gene_loss_weights: Optional[Dict[str, float]] = None,
        gene_cross_track_weight: float = 5.0,
        track_strands: Optional[Dict[str, Sequence[str]]] = None,
    ):
        super().__init__()
        self.model = model
        self.heads = heads or list(DEFAULT_HEAD_WEIGHTS.keys())
        # Merge overrides onto a fresh copy of the defaults: a partial dict
        # only changes the heads it names (so splice_junctions keeps its 0.2
        # unless explicitly overridden), and we never alias/mutate the
        # module-level constant.
        self.head_weights = {**DEFAULT_HEAD_WEIGHTS, **(head_weights or {})}
        self.multinomial_resolution = multinomial_resolution
        self.positional_weight = positional_weight

        # Gene LFC loss config (B3.2 / Decima-style).
        # `gene_loss_weights[head]` is the OUTER multiplier on the head's gene
        # LFC term, not on the head's overall loss. Default empty means off.
        # `gene_cross_track_weight` is the INNER multiplier on the multinomial
        # (positional) component within the gene LFC term; paper value 5.0.
        self.gene_loss_weights: Dict[str, float] = dict(gene_loss_weights or {})
        self.gene_cross_track_weight = gene_cross_track_weight

        # Per-head strand-channel masks `[2, 1, C]` registered as buffers so
        # they follow `.to(device)` with the loss module. Stored under
        # mangled attribute names because `register_buffer` does not accept
        # nested keys; we look them up via `_get_strand_channel_mask(head)`.
        self._strand_mask_heads: List[str] = []
        if track_strands:
            for head, strands in track_strands.items():
                buffer_name = f"_strand_channel_mask__{head}"
                self.register_buffer(buffer_name, _build_strand_channel_mask(strands))
                self._strand_mask_heads.append(head)

    def _get_strand_channel_mask(self, head: str) -> Optional[torch.Tensor]:
        """Return the `[2, 1, C]` strand-channel mask for a head, or None."""
        return getattr(self, f"_strand_channel_mask__{head}", None)
    
    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, torch.Tensor],
        organism_index: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        gene_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-head losses and aggregate.

        Args:
            outputs: Model outputs dict. Each head maps to resolution dict or tensor.
            targets: Target values dict with same structure as outputs.
            organism_index: Organism indices (B,). Required for target scaling.
            masks: Optional boolean masks dict for each head.
            gene_mask: Optional `[B, S, 2, G]` gene-body mask for the gene
                LFC training loss. Only consumed by heads with a non-zero
                entry in `self.gene_loss_weights`.

        Returns:
            Dict with 'loss' (total), and per-head losses like 'atac_loss', etc.
        """
        masks = masks or {}
        head_losses = {}
        total_loss = torch.tensor(0.0, device=self._get_device(outputs))
        num_heads = 0

        for head in self.heads:
            if head not in outputs:
                continue

            head_output = outputs[head]
            head_target = targets.get(head)
            head_mask = masks.get(head)

            if head_target is None:
                continue

            # Compute loss based on head type
            head_loss = self._compute_head_loss(
                head, head_output, head_target, head_mask, organism_index,
                gene_mask=gene_mask,
            )

            head_losses[f'{head}_loss'] = head_loss
            total_loss = total_loss + self.head_weights.get(head, 1.0) * head_loss
            num_heads += 1

        # Average across heads
        if num_heads > 0:
            total_loss = total_loss / num_heads

        head_losses['loss'] = total_loss
        return head_losses
    
    def _compute_head_loss(
        self,
        head: str,
        output: Any,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
        organism_index: torch.Tensor,
        gene_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss for a single head.

        IMPORTANT: Assumes output is in model space (scaled predictions)
        and scales targets to match before loss computation.

        Uses multinomial loss for profile heads (ATAC, DNase, etc.)
        and MSE for aggregate/contact map heads. Optionally adds the gene
        LFC term (Decima-style cross-track loss) when this head has a
        non-zero entry in `self.gene_loss_weights`.
        """
        if head == 'splice_sites':
            logits = output['logits']
            classification_mask = torch.any(target > 0, dim=-1, keepdim=True)
            num_tracks = target.shape[-1]
            return losses.cross_entropy_loss_from_logits(
                y_pred_logits=logits,
                y_true=(1.0 - 1e-7) * target.float() + 1e-7 / num_tracks,
                mask=classification_mask,
                axis=-1,
            )

        if head == 'splice_site_usage':
            logits = output['logits']
            if mask is None:
                mask = torch.ones_like(target, dtype=torch.bool)
            return losses.binary_crossentropy_from_logits(
                y_pred=logits,
                y_true=torch.clamp(target.float(), 1e-7, 1.0 - 1e-7),
                mask=mask,
            )

        if head == 'splice_junctions':
            pred_pair = output['pred_counts']
            pairs_mask = output['splice_junction_mask']
            
            def _scale_junction_counts(counts):
                return torch.where(
                    counts > 10.0,
                    2.0 * torch.sqrt(counts * 10.0) - 10.0,
                    counts,
                )
                
            sum_acceptors_tgt = torch.sum(torch.where(pairs_mask, target.float(), 0.0), dim=-2)
            sum_acceptors_pred = torch.sum(torch.where(pairs_mask, pred_pair.float(), 0.0), dim=-2)
            accept_total_loss = losses.poisson_loss(
                y_true=_scale_junction_counts(sum_acceptors_tgt),
                y_pred=sum_acceptors_pred,
                mask=torch.any(pairs_mask, dim=-2)
            )
            
            sum_donors_tgt = torch.sum(torch.where(pairs_mask, target.float(), 0.0), dim=-3)
            sum_donors_pred = torch.sum(torch.where(pairs_mask, pred_pair.float(), 0.0), dim=-3)
            donor_total_loss = losses.poisson_loss(
                y_true=_scale_junction_counts(sum_donors_tgt),
                y_pred=sum_donors_pred,
                mask=torch.any(pairs_mask, dim=-3)
            )
            
            donor_ratios_loss = losses.cross_entropy_loss(
                y_true=target,
                y_pred=pred_pair,
                mask=pairs_mask,
                axis=-3,
            )
            acceptor_ratios_loss = losses.cross_entropy_loss(
                y_true=target,
                y_pred=pred_pair,
                mask=pairs_mask,
                axis=-2,
            )
            
            return donor_ratios_loss + acceptor_ratios_loss + 0.2 * (accept_total_loss + donor_total_loss)

        # Handle resolution dict outputs (e.g., {1: tensor, 128: tensor})
        resolution = None
        if isinstance(output, dict):
            # Use highest resolution available
            res_keys = [k for k in output.keys() if isinstance(k, int)]
            if res_keys:
                resolution = min(res_keys)  # 1bp > 128bp
                output = output[resolution]
                if isinstance(target, dict):
                    target = target[resolution]

        # Default mask if not provided
        # Loss functions expect mask to match data format
        if mask is None:
            batch_size = output.shape[0]
            # Try to infer format: if last dim is not 1 or 128 (approx), or if it's the track dim.
            # Usually num_tracks is known. But let's use a simpler heuristic or just support both.
            # If we assume NLC by default:
            is_channels_last = True
            if hasattr(self.model, 'heads') and head in self.model.heads:
                num_tracks = self.model.heads[head].num_tracks
                if output.shape[-2] == num_tracks and output.shape[-1] != num_tracks:
                    is_channels_last = False

            if is_channels_last:
                num_channels = output.shape[-1]
                mask = torch.ones((batch_size, 1, num_channels), dtype=torch.bool, device=output.device)
            else:
                num_channels = output.shape[-2]
                mask = torch.ones((batch_size, num_channels, 1), dtype=torch.bool, device=output.device)
        else:
            # If mask is provided, we assume it matches the output format.
            # We need to determine channels_last for scale/multinomial_loss.
            is_channels_last = (mask.shape[-2] == 1)

        # Contact maps use MSE (no scaling needed)
        if head == 'contact_maps':
            return losses.mse(y_pred=output, y_true=target, mask=mask)

        # For genome track heads: Scale targets to model space
        if self.model is not None and hasattr(self.model, 'heads') and head in self.model.heads and resolution is not None:
            head_module = self.model.heads[head]
            scaled_target = head_module.scale(target, organism_index, resolution, channels_last=is_channels_last)
        else:
            # Fallback: No scaling (for splice heads, or if model not provided)
            scaled_target = target

        # Profile heads use multinomial loss on scaled values
        result = losses.multinomial_loss(
            y_true=scaled_target,  # Now scaled!
            y_pred=output,         # Already scaled from forward pass
            mask=mask,
            multinomial_resolution=self.multinomial_resolution,
            positional_weight=self.positional_weight,
            channels_last=is_channels_last,
        )
        head_loss = result['loss']

        # Optional gene LFC (cross-track) loss. Mirrors upstream's
        # `_compute_cross_track_loss`; gated on (a) head has a non-zero
        # gene_loss_weight, (b) gene_mask is present this batch, and
        # (c) we are at 1bp resolution (matching upstream which only
        # threads gene_mask through resolution == 1).
        gene_w = self.gene_loss_weights.get(head, 0.0)
        if gene_w > 0 and gene_mask is not None and resolution == 1:
            # multinomial_loss internally normalizes to NLC; re-derive that
            # form here for the gene LFC einsum, which expects [B, S, C].
            if not is_channels_last:
                pred_nlc = output.transpose(-1, -2).contiguous()
                target_nlc = scaled_target.transpose(-1, -2).contiguous()
                track_mask_nlc = mask.transpose(-1, -2).contiguous()
            else:
                pred_nlc = output
                target_nlc = scaled_target
                track_mask_nlc = mask

            strand_channel_mask = self._get_strand_channel_mask(head)
            if strand_channel_mask is None:
                raise ValueError(
                    f"head '{head}' has gene_loss_weight={gene_w} but no "
                    f"track_strands were provided to AlphaGenomeLoss. "
                    f"Pass `track_strands={{'{head}': '<+/-/. per track>'}}` "
                    f"so the strand-channel mask can be built."
                )

            gene_loss, _ = self._compute_gene_lfc(
                predictions=pred_nlc,
                targets=target_nlc,
                targets_mask=track_mask_nlc,
                gene_mask=gene_mask,
                strand_channel_mask=strand_channel_mask,
            )
            head_loss = head_loss + gene_w * gene_loss

        return head_loss

    def _compute_gene_lfc(
        self,
        *,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: Optional[torch.Tensor],
        gene_mask: torch.Tensor,
        strand_channel_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Thin wrapper around `losses.gene_lfc_loss` using this loss's
        configured `gene_cross_track_weight`. Kept as a method for
        clarity at call sites and for backward-compat with existing tests.
        """
        return losses.gene_lfc_loss(
            predictions=predictions,
            targets=targets,
            targets_mask=targets_mask,
            gene_mask=gene_mask,
            strand_channel_mask=strand_channel_mask,
            gene_cross_track_weight=self.gene_cross_track_weight,
        )
    
    def _get_device(self, outputs: Dict) -> torch.device:
        """Get device from first tensor in outputs."""
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                return v.device
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, torch.Tensor):
                        return vv.device
        return torch.device('cpu')


def create_optimizer(
    model: nn.Module,
    config: AlphaGenomeTrainingConfig,
) -> torch.optim.AdamW:
    """Create AdamW optimizer with paper parameters.
    
    Args:
        model: Model to optimize.
        config: Training configuration.
        
    Returns:
        Configured AdamW optimizer.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: AlphaGenomeTrainingConfig,
) -> LambdaLR:
    """Create warmup + cosine decay scheduler.
    
    Linear warmup from 0 to learning_rate over warmup_steps,
    then cosine decay to 0 over remaining steps.
    
    Args:
        optimizer: Optimizer to schedule.
        config: Training configuration.
        
    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            # Linear warmup
            return step / max(1, config.warmup_steps)
        # Cosine decay
        progress = (step - config.warmup_steps) / max(
            1, config.total_steps - config.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)


__all__ = [
    'AlphaGenomeTrainingConfig',
    'AlphaGenomeLoss',
    'create_optimizer',
    'create_scheduler',
    'DEFAULT_HEAD_WEIGHTS',
    '_build_strand_channel_mask',
]
