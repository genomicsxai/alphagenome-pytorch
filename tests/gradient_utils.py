"""Gradient testing utilities for AlphaGenome backward pass verification."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn


@dataclass
class GradientCheckResult:
    """Results of gradient sanity checks for a parameter."""

    name: str
    has_grad: bool
    grad_norm: float
    has_nan: bool
    has_inf: bool

    @property
    def passed(self) -> bool:
        """Check if gradient passed all sanity checks."""
        return self.has_grad and not self.has_nan and not self.has_inf and self.grad_norm > 0


def check_all_gradients(model: nn.Module) -> Dict[str, GradientCheckResult]:
    """Check gradient status for all parameters in a model.

    Args:
        model: PyTorch model after backward() has been called.

    Returns:
        Dictionary mapping parameter names to their gradient check results.
    """
    results = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            results[name] = GradientCheckResult(
                name=name,
                has_grad=True,
                grad_norm=param.grad.norm().item(),
                has_nan=torch.isnan(param.grad).any().item(),
                has_inf=torch.isinf(param.grad).any().item(),
            )
        else:
            results[name] = GradientCheckResult(
                name=name,
                has_grad=False,
                grad_norm=0.0,
                has_nan=False,
                has_inf=False,
            )
    return results


def assert_all_params_have_gradients(
    model: nn.Module, exclude_patterns: Optional[List[str]] = None
):
    """Assert every parameter has a non-zero gradient.

    Args:
        model: PyTorch model after backward() has been called.
        exclude_patterns: List of patterns to exclude from checking.

    Raises:
        AssertionError: If any parameter is missing a gradient.
    """
    exclude_patterns = exclude_patterns or []

    missing = []
    for name, param in model.named_parameters():
        if any(pat in name for pat in exclude_patterns):
            continue
        if param.grad is None or param.grad.norm() == 0:
            missing.append(name)

    if missing:
        raise AssertionError(
            f"{len(missing)} parameters missing gradients:\n" + "\n".join(missing[:30])
        )


def assert_no_nan_inf_gradients(model: nn.Module):
    """Assert no NaN or Inf values in any gradient.

    Args:
        model: PyTorch model after backward() has been called.

    Raises:
        AssertionError: If any gradient contains NaN or Inf.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                raise AssertionError(f"NaN in gradient of {name}")
            if torch.isinf(param.grad).any():
                raise AssertionError(f"Inf in gradient of {name}")


def compute_combined_loss(outputs: Dict, include_splice: bool = False) -> torch.Tensor:
    """Compute loss from genome track heads for gradient computation.

    This function aggregates outputs from genome track heads and contact maps.
    By default, splice heads are excluded to match JAX parity testing behavior
    (splice heads have known forward pass differences that need separate fixing).

    Note: Only includes direct prediction keys (integer resolutions like 1, 128),
    excluding any 'scaled_predictions_*' keys to match JAX loss computation.

    Args:
        outputs: Dictionary of model outputs from AlphaGenome.forward()
        include_splice: Whether to include splice heads (default False for parity).

    Returns:
        Combined loss tensor.
    """
    loss = torch.tensor(0.0, device=_get_device(outputs))

    # Genome tracks heads
    # Use nanmean to handle NaN values from tracks without training data
    # Only include integer resolution keys (1, 128) to match JAX predictions_* filtering
    for head_name in [
        "atac",
        "dnase",
        "procap",
        "cage",
        "rna_seq",
        "chip_tf",
        "chip_histone",
    ]:
        if head_name in outputs:
            head_out = outputs[head_name]
            for res, pred in head_out.items():
                # Match JAX filtering: only include prediction keys (integer resolutions)
                # Excludes any 'scaled_predictions_*' style keys
                if isinstance(res, int):
                    loss = loss + torch.nanmean(pred)

    # Contact maps
    if "pair_activations" in outputs:
        loss = loss + outputs["pair_activations"].mean()

    # Splice heads - excluded by default to match JAX parity testing
    # JAX side comments these out due to known forward pass differences
    # See tests/integration/conftest.py lines 214-227
    if include_splice:
        if "splice_sites_classification" in outputs:
            loss = loss + outputs["splice_sites_classification"]["logits"].mean()
        if "splice_sites_usage" in outputs:
            loss = loss + outputs["splice_sites_usage"]["logits"].mean()
        # Skip junction - too memory intensive for gradient testing
        # if "splice_sites_junction" in outputs:
        #     loss = loss + outputs["splice_sites_junction"]["pred_counts"].mean()

    return loss


def compute_per_head_losses(outputs: Dict) -> Dict[str, torch.Tensor]:
    """Compute loss contribution from each head separately.

    This enables per-head loss parity testing to isolate where gradient
    drift originates between JAX and PyTorch implementations.

    Args:
        outputs: Dictionary of model outputs from AlphaGenome.forward()

    Returns:
        Dictionary mapping head name to its loss contribution tensor.
    """
    device = _get_device(outputs)
    losses = {}

    # Genome tracks heads
    for head_name in [
        "atac",
        "dnase",
        "procap",
        "cage",
        "rna_seq",
        "chip_tf",
        "chip_histone",
    ]:
        if head_name in outputs:
            head_loss = torch.tensor(0.0, device=device)
            head_out = outputs[head_name]
            for res, pred in head_out.items():
                # Only include integer resolution keys to match JAX
                if isinstance(res, int):
                    head_loss = head_loss + torch.nanmean(pred)
            losses[head_name] = head_loss

    # Contact maps
    if "pair_activations" in outputs:
        losses["contact_maps"] = outputs["pair_activations"].mean()

    # Splice heads (reported separately for debugging)
    if "splice_sites_classification" in outputs:
        losses["splice_sites_classification"] = (
            outputs["splice_sites_classification"]["logits"].mean()
        )
    if "splice_sites_usage" in outputs:
        losses["splice_sites_usage"] = outputs["splice_sites_usage"]["logits"].mean()
    if "splice_sites_junction" in outputs:
        losses["splice_sites_junction"] = (
            outputs["splice_sites_junction"]["pred_counts"].mean()
        )

    return losses


def compute_head_loss(outputs: Dict, head_name: str) -> torch.Tensor:
    """Compute loss from a single head for gradient isolation testing.

    Args:
        outputs: Dictionary of model outputs.
        head_name: Name of the head to compute loss from.

    Returns:
        Loss tensor from the specified head.
    """
    device = _get_device(outputs)

    if head_name in ["atac", "dnase", "procap", "cage", "rna_seq", "chip_tf", "chip_histone"]:
        head_out = outputs[head_name]
        loss = torch.tensor(0.0, device=device)
        for res, pred in head_out.items():
            loss = loss + torch.nanmean(pred)
        return loss

    elif head_name == "pair_activations":
        return outputs["pair_activations"].mean()

    elif head_name == "splice_sites_classification":
        return outputs["splice_sites_classification"]["logits"].mean()

    elif head_name == "splice_sites_usage":
        return outputs["splice_sites_usage"]["logits"].mean()

    elif head_name == "splice_sites_junction":
        return outputs["splice_sites_junction"]["pred_counts"].mean()

    else:
        raise ValueError(f"Unknown head name: {head_name}")


def _get_device(outputs: Dict) -> torch.device:
    """Extract device from outputs dictionary."""
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            return value.device
        elif isinstance(value, dict):
            for v in value.values():
                if isinstance(v, torch.Tensor):
                    return v.device
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
