"""Integration tests for training loss gradient parity.

This module tests gradient parity when using AlphaGenomeLoss (multinomial_loss).

CURRENT IMPLEMENTATION:
- Both JAX and PyTorch use experimental space (unscaled) predictions and targets
- This validates that multinomial_loss is implemented consistently between frameworks

FUTURE ENHANCEMENT:
- Test with model space (scaled) predictions and targets
- This would require accessing JAX's internal scaled predictions
- Would fully validate the scaling fix (loss computed in model space)

The scaling fix itself is validated by:
1. Unit tests (test_heads.py): scaling function reversibility
2. Unit tests (test_training.py): AlphaGenomeLoss with model reference
3. Real training runs: proper gradient flow in model space

NOTE: Training fixtures are defined in this file with module scope to:
1. Free memory after these tests complete
2. Keep training-specific fixtures close to the tests that use them
"""

import pytest
import numpy as np


# =============================================================================
# Training Fixtures (module-scoped for memory efficiency)
# =============================================================================


@pytest.fixture(scope="module")
def synthetic_targets(random_dna_sequence, run_jax_inference):
    """Generate synthetic targets for training loss testing.

    Creates targets by running inference and adding noise, ensuring we have
    realistic target distributions for testing loss computation.

    Returns dict with structure:
    {
        "human": {"atac": {1: array, 128: array}, "dnase": {...}, ...},
        "mouse": {...}
    }
    """
    import numpy as np

    targets = {}

    for org_name, org_idx in [("human", 0), ("mouse", 1)]:
        # Get predictions to use as base for targets
        predictions = run_jax_inference(random_dna_sequence, org_idx)

        org_targets = {}

        # Genome track heads
        for head_name in ['atac', 'dnase', 'procap', 'cage', 'rna_seq', 'chip_tf', 'chip_histone']:
            if head_name in predictions:
                head_out = predictions[head_name]
                head_targets = {}

                # Extract unscaled predictions for each resolution
                for key, val in head_out.items():
                    if key.startswith('predictions_'):
                        # Extract resolution (e.g., 'predictions_1' or 'predictions_128bp' -> 128)
                        res_str = key.split('_')[1].replace('bp', '')
                        res = int(res_str)
                        # Add small noise to predictions to create targets
                        noise = np.random.randn(*val.shape).astype(np.float32) * 0.1
                        target = np.maximum(val + noise, 0.01)  # Ensure positive
                        head_targets[res] = target

                if head_targets:
                    org_targets[head_name] = head_targets

        targets[org_name] = org_targets

    return targets


def _jax_targets_scaling(targets, track_means, resolution, apply_squashing=False):
    """Scale targets to model space (JAX implementation).

    Mirrors PyTorch's targets_scaling() from training.py.
    See AG_MODEL.md lines 300-306 for the original specification.

    Args:
        targets: Raw experimental target values
        track_means: Per-track mean values for normalization
        resolution: Resolution in bp (1 or 128) for proper scaling
        apply_squashing: If True, apply power law compression (RNA-seq only)

    Returns:
        Scaled targets in model space
    """
    import jax.numpy as jnp

    # Normalize by track means AND resolution (matches heads.py:71)
    targets = targets / (track_means * resolution + 1e-8)

    # Apply squashing for RNA-seq (power law compression)
    if apply_squashing:
        targets = jnp.power(targets, 0.75)

    # Soft clip: Where(targets > 10.0, 2 * Sqrt(x * 10.0) - 10.0, targets)
    targets = jnp.where(
        targets > 10.0,
        2.0 * jnp.sqrt(targets * 10.0) - 10.0,
        targets
    )

    return targets


@pytest.fixture(scope="module")
def run_jax_training_backward(jax_model):
    """Factory fixture to compute JAX gradients using training loss.

    Computes loss in model space (scaled predictions AND scaled targets) to match
    production training behavior from AG_MODEL.md.

    Key differences from simple parity testing:
    - Uses positional_weight=5.0 (production value)
    - Scales targets using targets_scaling() before loss computation
    - Both predictions and targets are in model space
    """
    import jax
    import jax.numpy as jnp
    import haiku as hk
    import jmp
    from alphagenome_research.model import model as jax_model_module
    from alphagenome_research.model import losses as jax_losses
    import numpy as np

    jmp_policy = jmp.get_policy('params=float32,compute=float32,output=float32')

    @hk.transform_with_state
    def _forward(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(jax_model_module.AlphaGenome, jmp_policy):
            return jax_model_module.AlphaGenome(jax_model._metadata)(dna_sequence, organism_index)

    def _loss_fn(params, state, dna_sequence, organism_index, targets_dict, track_means_dict):
        """Compute training loss matching production behavior.

        Uses scaled predictions AND scaled targets with positional_weight=5.0.
        This matches AG_MODEL.md lines 314-339.
        """
        (predictions, _), _ = _forward.apply(
            params, state, None, dna_sequence, organism_index
        )

        total_loss = jnp.array(0.0)
        num_losses = 0

        # Genome track heads - use multinomial_loss with production settings
        for head_name in ['atac', 'dnase', 'procap', 'cage', 'rna_seq', 'chip_tf', 'chip_histone']:
            if head_name not in predictions or head_name not in targets_dict:
                continue

            head_out = predictions[head_name]
            head_targets = targets_dict[head_name]
            apply_squashing = (head_name == 'rna_seq')

            # Use SCALED predictions (model space) for loss computation
            for key, pred_val in head_out.items():
                if key.startswith('scaled_predictions_'):
                    # Extract resolution (handle 'bp' suffix: scaled_predictions_128bp -> 128)
                    res_str = key.replace('scaled_predictions_', '').replace('bp', '')
                    res = int(res_str)

                    if res not in head_targets:
                        continue

                    # Get raw target and scale it to model space
                    target_raw = jnp.array(head_targets[res])

                    # Get track means for this head and resolution
                    track_means_key = f"{head_name}_{res}"
                    if track_means_key in track_means_dict:
                        track_means = jnp.array(track_means_dict[track_means_key])
                    else:
                        # Fallback: use ones (no scaling)
                        track_means = jnp.ones(target_raw.shape[-1])

                    # Scale targets to model space (matches PyTorch training.py)
                    target_scaled = _jax_targets_scaling(
                        target_raw, track_means, resolution=res, apply_squashing=apply_squashing
                    )

                    # Compute multinomial_resolution matching JAX production
                    # JAX uses 2^17 // resolution for full model (131072bp input)
                    # This creates 1 segment (full sequence multinomial loss)
                    # For tests with shorter inputs, use actual seq_len to match
                    seq_len = pred_val.shape[1]
                    multinomial_resolution = seq_len  # 1 segment (matches JAX production)

                    # Create mask (all valid)
                    mask = jnp.ones(
                        (*target_scaled.shape[:-2], 1, target_scaled.shape[-1]),
                        dtype=jnp.bool_
                    )

                    # Use JAX's multinomial_loss with production settings
                    loss_result = jax_losses.multinomial_loss(
                        y_true=target_scaled,
                        y_pred=pred_val,
                        mask=mask,
                        multinomial_resolution=multinomial_resolution,
                        positional_weight=5.0,  # Production value from AG_MODEL.md
                    )
                    total_loss = total_loss + loss_result['loss']
                    num_losses += 1

        # Ensure we computed at least some loss
        if num_losses == 0:
            sample_head = next(iter(predictions.keys()), None)
            sample_keys = list(predictions.get(sample_head, {}).keys()) if sample_head else []
            raise ValueError(
                f"No losses computed. Sample head '{sample_head}' has keys: {sample_keys}. "
                f"Run scripts/debug_jax_output_keys.py to diagnose."
            )

        return total_loss

    def _run(sequence: np.ndarray, organism_index: int, targets_dict: dict, track_means_dict: dict = None):
        """Run JAX training backward pass with production loss settings."""
        batch_size = sequence.shape[0]
        jax_input = jnp.array(sequence)
        jax_org = jnp.array([organism_index] * batch_size, dtype=jnp.int32)

        # Convert targets to JAX arrays
        jax_targets = {}
        for head_name, head_targets in targets_dict.items():
            jax_targets[head_name] = {
                res: jnp.array(target) for res, target in head_targets.items()
            }

        # Use empty dict if no track means provided
        if track_means_dict is None:
            track_means_dict = {}

        # Compute loss and gradients
        loss_value, grads = jax.value_and_grad(
            lambda p: _loss_fn(p, jax_model._state, jax_input, jax_org, jax_targets, track_means_dict)
        )(jax_model._params)

        # Flatten gradient tree
        flat_grads = {}
        def flatten(d, prefix=''):
            for k, v in d.items():
                key = f"{prefix}/{k}" if prefix else k
                if isinstance(v, dict):
                    flatten(v, key)
                else:
                    flat_grads[key] = np.array(v).astype(np.float32)

        flatten(grads)
        return float(loss_value), flat_grads

    return _run


@pytest.fixture(scope="module")
def run_pytorch_training_backward(pytorch_model):
    """Factory fixture to compute PyTorch gradients using manual per-resolution loss.

    Uses production training behavior from AG_MODEL.md:
    - positional_weight=5.0 (production value)
    - multinomial_resolution = 2^17 // resolution (matches JAX production)
    - Both predictions and targets are in model space
    """
    import torch
    from alphagenome_pytorch.losses import multinomial_loss

    def _run(sequence: np.ndarray, organism_index: int, targets_dict: dict):
        """Run PyTorch training backward pass with production loss settings."""
        batch_size = sequence.shape[0]

        pytorch_model.eval()

        device = next(pytorch_model.parameters()).device
        pt_input = torch.tensor(sequence, requires_grad=True, device=device)
        pt_org = torch.tensor([organism_index] * batch_size, dtype=torch.long, device=device)

        # Convert targets to PyTorch tensors (keep in experimental space)
        pt_targets = {}
        for head_name, head_targets in targets_dict.items():
            pt_targets[head_name] = {
                res: torch.tensor(target, device=device)
                for res, target in head_targets.items()
            }

        # Get predictions in MODEL SPACE (scaled predictions)
        pytorch_model.zero_grad()
        outputs = pytorch_model(
            pt_input,
            pt_org,
            return_scaled_predictions=True,  # Return model space
        )

        # Compute loss manually per-resolution to match JAX production
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_losses = 0

        for head_name in ['atac', 'dnase', 'procap', 'cage', 'rna_seq', 'chip_tf', 'chip_histone']:
            if head_name not in outputs or head_name not in pt_targets:
                continue

            head = pytorch_model.heads[head_name]
            head_outputs = outputs[head_name]
            head_targets = pt_targets[head_name]

            # Process each resolution separately with correct multinomial_resolution
            for res, pred in head_outputs.items():
                if res not in head_targets:
                    continue

                target_raw = head_targets[res]

                # Scale targets to model space using head's scale method
                target_scaled = head.scale(target_raw, pt_org, res)

                # Compute multinomial_resolution matching JAX production
                # JAX uses 2^17 // resolution for full model (131072bp input)
                # This creates 1 segment (full sequence multinomial loss)
                # For tests with shorter inputs, use actual seq_len to match
                seq_len = pred.shape[1]
                multinomial_resolution = seq_len  # 1 segment (matches JAX production)

                # Create mask (all valid)
                mask = torch.ones(
                    (*target_scaled.shape[:-2], 1, target_scaled.shape[-1]),
                    dtype=torch.bool,
                    device=device
                )

                # Compute loss using multinomial_loss directly
                loss_result = multinomial_loss(
                    y_true=target_scaled,
                    y_pred=pred,
                    mask=mask,
                    multinomial_resolution=multinomial_resolution,
                    positional_weight=5.0,  # Production value from AG_MODEL.md
                )
                total_loss = total_loss + loss_result['loss']
                num_losses += 1

        if num_losses == 0:
            raise ValueError("No losses computed")

        total_loss.backward()

        # Collect gradients
        grads = {}
        for name, param in pytorch_model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().cpu().numpy().astype(np.float32)

        return total_loss.item(), grads

    return _run


def _extract_track_means(pytorch_model, organism_index):
    """Extract track means from PyTorch model heads for JAX parity.

    Returns dict mapping '{head_name}_{resolution}' to track means array.
    """
    import torch

    track_means_dict = {}

    # Get heads from the model
    heads = pytorch_model.heads

    for head_name in ['atac', 'dnase', 'procap', 'cage', 'rna_seq', 'chip_tf', 'chip_histone']:
        if head_name not in heads:
            continue

        head = heads[head_name]
        if hasattr(head, 'track_means'):
            # track_means shape: (num_organisms, num_tracks)
            # Get for this organism
            means = head.track_means[organism_index].cpu().numpy()

            # Store for each resolution this head supports
            for res in head.resolutions:
                key = f"{head_name}_{res}"
                track_means_dict[key] = means

    return track_means_dict


@pytest.fixture(scope="module")
def cached_training_gradients(
    random_dna_sequence,
    synthetic_targets,
    run_jax_training_backward,
    run_pytorch_training_backward,
    pytorch_model,
):
    """Cache training gradients for both models.

    Uses production training behavior:
    - PyTorch: AlphaGenomeLoss with model=pytorch_model, positional_weight=5.0
    - JAX: multinomial_loss with target scaling, positional_weight=5.0

    Returns dict with structure:
    {
        "human": {
            "jax": (loss, {param_name: grad_array, ...}),
            "pytorch": (loss, {param_name: grad_array, ...})
        },
        "mouse": {...}
    }
    """
    import torch
    import gc

    cache = {}

    # Run PyTorch backward first for all organisms
    for org_name, org_idx in [("human", 0), ("mouse", 1)]:
        targets = synthetic_targets[org_name]
        cache[org_name] = {
            "pytorch": run_pytorch_training_backward(random_dna_sequence, org_idx, targets)
        }

    # Clear PyTorch GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # Run JAX backward for all organisms
    # Pass track means extracted from PyTorch model for target scaling parity
    for org_name, org_idx in [("human", 0), ("mouse", 1)]:
        targets = synthetic_targets[org_name]
        track_means_dict = _extract_track_means(pytorch_model, org_idx)
        cache[org_name]["jax"] = run_jax_training_backward(
            random_dna_sequence, org_idx, targets, track_means_dict
        )

    return cache


@pytest.fixture(scope="module")
def cached_training_losses(cached_training_gradients):
    """Extract just the loss values from cached training gradients.

    Returns dict with structure:
    {
        "human": {"jax": loss_value, "pytorch": loss_value},
        "mouse": {"jax": loss_value, "pytorch": loss_value}
    }
    """
    losses = {}
    for org_name in ["human", "mouse"]:
        jax_loss, _ = cached_training_gradients[org_name]["jax"]
        pt_loss, _ = cached_training_gradients[org_name]["pytorch"]
        losses[org_name] = {"jax": jax_loss, "pytorch": pt_loss}

    return losses


@pytest.fixture(scope="module")
def run_jax_training_per_head_loss(jax_model):
    """Factory fixture to compute per-head training losses in JAX.

    Uses production settings: positional_weight=5.0 and target scaling.
    """
    import jax
    import jax.numpy as jnp
    import haiku as hk
    import jmp
    from alphagenome_research.model import model as jax_model_module
    from alphagenome_research.model import losses as jax_losses

    jmp_policy = jmp.get_policy('params=float32,compute=float32,output=float32')

    @hk.transform_with_state
    def _forward(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(jax_model_module.AlphaGenome, jmp_policy):
            return jax_model_module.AlphaGenome(jax_model._metadata)(dna_sequence, organism_index)

    def _run(sequence: np.ndarray, organism_index: int, targets_dict: dict, track_means_dict: dict = None):
        """Compute per-head training losses for JAX model with production settings."""
        batch_size = sequence.shape[0]
        jax_input = jnp.array(sequence)
        jax_org = jnp.array([organism_index] * batch_size, dtype=jnp.int32)

        if track_means_dict is None:
            track_means_dict = {}

        (predictions, _), _ = _forward.apply(
            jax_model._params, jax_model._state, None, jax_input, jax_org
        )

        losses = {}

        for head_name in ['atac', 'dnase', 'procap', 'cage', 'rna_seq', 'chip_tf', 'chip_histone']:
            if head_name not in predictions or head_name not in targets_dict:
                continue

            head_out = predictions[head_name]
            head_targets = targets_dict[head_name]
            apply_squashing = (head_name == 'rna_seq')
            head_loss = jnp.array(0.0)

            for key, pred_val in head_out.items():
                if key.startswith('scaled_predictions_'):
                    # Extract resolution (handle 'bp' suffix: 'scaled_predictions_128bp' -> 128)
                    res_str = key.split('_')[2].replace('bp', '')
                    res = int(res_str)

                    if res not in head_targets:
                        continue

                    # Get raw target and scale it
                    target_raw = jnp.array(head_targets[res])

                    # Get track means for this head and resolution
                    track_means_key = f"{head_name}_{res}"
                    if track_means_key in track_means_dict:
                        track_means = jnp.array(track_means_dict[track_means_key])
                    else:
                        track_means = jnp.ones(target_raw.shape[-1])

                    # Scale targets to model space
                    target_scaled = _jax_targets_scaling(
                        target_raw, track_means, resolution=res, apply_squashing=apply_squashing
                    )

                    # Compute multinomial_resolution matching JAX production
                    # JAX uses 2^17 // resolution for full model (131072bp input)
                    # This creates 1 segment (full sequence multinomial loss)
                    # For tests with shorter inputs, use actual seq_len to match
                    seq_len = pred_val.shape[1]
                    multinomial_resolution = seq_len  # 1 segment (matches JAX production)

                    mask = jnp.ones((*target_scaled.shape[:-2], 1, target_scaled.shape[-1]), dtype=jnp.bool_)

                    loss_result = jax_losses.multinomial_loss(
                        y_true=target_scaled,
                        y_pred=pred_val,
                        mask=mask,
                        multinomial_resolution=multinomial_resolution,
                        positional_weight=5.0,  # Production value
                    )
                    head_loss = head_loss + loss_result['loss']

            losses[head_name] = float(head_loss)

        return losses

    return _run


@pytest.fixture(scope="module")
def run_pytorch_training_per_head_loss(pytorch_model):
    """Factory fixture to compute per-head training losses in PyTorch.

    Uses production settings with per-resolution multinomial_resolution = 2^17 // res.
    """
    import torch
    from alphagenome_pytorch.losses import multinomial_loss

    def _run(sequence: np.ndarray, organism_index: int, targets_dict: dict):
        """Compute per-head training losses for PyTorch model with production settings."""
        batch_size = sequence.shape[0]
        device = next(pytorch_model.parameters()).device

        pt_input = torch.tensor(sequence, device=device)
        pt_org = torch.tensor([organism_index] * batch_size, dtype=torch.long, device=device)

        # Convert targets (keep in experimental space)
        pt_targets = {}
        for head_name, head_targets in targets_dict.items():
            pt_targets[head_name] = {
                res: torch.tensor(target, device=device)
                for res, target in head_targets.items()
            }

        with torch.no_grad():
            outputs = pytorch_model(
                pt_input,
                pt_org,
                return_scaled_predictions=True,  # Model space (scaled predictions)
            )

        # Compute per-head losses with production settings (per-resolution)
        losses = {}
        for head_name in ['atac', 'dnase', 'procap', 'cage', 'rna_seq', 'chip_tf', 'chip_histone']:
            if head_name not in outputs or head_name not in pt_targets:
                continue

            head = pytorch_model.heads[head_name]
            head_outputs = outputs[head_name]
            head_targets = pt_targets[head_name]
            head_loss = 0.0

            # Process each resolution separately with correct multinomial_resolution
            for res, pred in head_outputs.items():
                if res not in head_targets:
                    continue

                target_raw = head_targets[res]

                # Scale targets to model space using head's scale method
                target_scaled = head.scale(target_raw, pt_org, res)

                # Compute multinomial_resolution matching JAX production
                # JAX uses 2^17 // resolution for full model (131072bp input)
                # This creates 1 segment (full sequence multinomial loss)
                # For tests with shorter inputs, use actual seq_len to match
                seq_len = pred.shape[1]
                multinomial_resolution = seq_len  # 1 segment (matches JAX production)

                # Create mask (all valid)
                mask = torch.ones(
                    (*target_scaled.shape[:-2], 1, target_scaled.shape[-1]),
                    dtype=torch.bool,
                    device=device
                )

                # Compute loss using multinomial_loss directly
                loss_result = multinomial_loss(
                    y_true=target_scaled,
                    y_pred=pred,
                    mask=mask,
                    multinomial_resolution=multinomial_resolution,
                    positional_weight=5.0,  # Production value
                )
                head_loss += loss_result['loss'].item()

            losses[head_name] = head_loss

        return losses

    return _run


@pytest.fixture(scope="module")
def cached_training_per_head_losses(
    random_dna_sequence,
    synthetic_targets,
    run_jax_training_per_head_loss,
    run_pytorch_training_per_head_loss,
    pytorch_model,
):
    """Cache per-head training losses for both models.

    Uses production settings:
    - PyTorch: model=pytorch_model (target scaling), positional_weight=5.0
    - JAX: target scaling, positional_weight=5.0

    Returns dict with structure:
    {
        "human": {
            "jax": {"atac": loss, "dnase": loss, ...},
            "pytorch": {"atac": loss, "dnase": loss, ...}
        },
        "mouse": {...}
    }
    """
    import torch
    import gc

    cache = {}

    # Run PyTorch first
    for org_name, org_idx in [("human", 0), ("mouse", 1)]:
        targets = synthetic_targets[org_name]
        cache[org_name] = {
            "pytorch": run_pytorch_training_per_head_loss(random_dna_sequence, org_idx, targets)
        }

    # Clear PyTorch GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # Run JAX with track means for target scaling parity
    for org_name, org_idx in [("human", 0), ("mouse", 1)]:
        targets = synthetic_targets[org_name]
        track_means_dict = _extract_track_means(pytorch_model, org_idx)
        cache[org_name]["jax"] = run_jax_training_per_head_loss(
            random_dna_sequence, org_idx, targets, track_means_dict
        )

    return cache


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.integration_jax
@pytest.mark.jax
class TestTrainingLossParity:
    """Test gradient parity using actual training loss computation."""

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_training_loss_value_parity(
        self, cached_training_losses, tolerances, organism
    ):
        """Test that training loss values match between JAX and PyTorch.

        This validates that loss computed with scaled predictions and
        scaled targets produces the same value in both frameworks.
        """
        jax_loss = cached_training_losses[organism]["jax"]
        pt_loss = cached_training_losses[organism]["pytorch"]

        rel_diff = abs(jax_loss - pt_loss) / (abs(jax_loss) + 1e-8)

        assert rel_diff < tolerances["rtol"], (
            f"Training loss mismatch ({organism}): "
            f"JAX={jax_loss:.6f}, PyTorch={pt_loss:.6f}, RelDiff={rel_diff:.4%}"
        )

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    @pytest.mark.parametrize(
        "head_name",
        ["atac", "dnase", "procap", "cage", "rna_seq", "chip_tf", "chip_histone"],
    )
    def test_per_head_training_loss_parity(
        self, cached_training_per_head_losses, tolerances, organism, head_name
    ):
        """Test per-head training loss values match between JAX and PyTorch.

        This helps isolate where any loss computation differences originate.
        """
        jax_losses = cached_training_per_head_losses[organism]["jax"]
        pt_losses = cached_training_per_head_losses[organism]["pytorch"]

        if head_name not in jax_losses or head_name not in pt_losses:
            pytest.skip(f"Head {head_name} not present in both frameworks")

        jax_loss = jax_losses[head_name]
        pt_loss = pt_losses[head_name]

        rel_diff = abs(jax_loss - pt_loss) / (abs(jax_loss) + 1e-8)

        assert rel_diff < tolerances["rtol"], (
            f"Per-head training loss mismatch for {head_name} ({organism}): "
            f"JAX={jax_loss:.6f}, PyTorch={pt_loss:.6f}, RelDiff={rel_diff:.4%}"
        )

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    @pytest.mark.parametrize(
        "component",
        [
            "encoder",
            "tower",
            "decoder",
            "embedder_1bp",
            "embedder_128bp",
            "heads.atac",
            "heads.dnase",
        ],
    )
    def test_training_gradient_parity_by_component(
        self,
        cached_training_gradients,
        param_name_mapping,
        tolerances,
        organism,
        component,
    ):
        """Test gradient parity when using actual training loss.

        This is the critical test that validates the scaling fix:
        - JAX computes loss in model space: loss(pred, scale(target))
        - PyTorch now does the same: loss(pred_scaled, scale(target))
        - Gradients should match much better than the nanmean tests
        """
        from tests.gradient_alignment import align_jax_gradient_to_pytorch
        from tests.integration_jax.comparison_utils import compare_gradients

        _, jax_grads = cached_training_gradients[organism]["jax"]
        _, pt_grads = cached_training_gradients[organism]["pytorch"]

        component_params = [n for n in pt_grads.keys() if n.startswith(component)]
        if not component_params:
            pytest.skip(f"No params for: {component}")

        failures = []
        for pt_name in component_params:
            jax_name = param_name_mapping.get(pt_name)
            if jax_name is None or jax_name not in jax_grads:
                continue

            pt_grad = pt_grads[pt_name]
            jax_grad = jax_grads[jax_name]

            try:
                jax_grad_aligned = align_jax_gradient_to_pytorch(
                    pt_name, jax_grad, pt_grad.shape
                )
            except ValueError as e:
                failures.append(f"{pt_name}: Alignment failed - {e}")
                continue

            # Use rtol_grad for gradients - backward pass accumulates more numerical
            # error than forward pass. Cosine similarity (>0.99) is the key metric.
            result = compare_gradients(
                pt_name,
                pt_grad,
                jax_grad_aligned,
                rtol=tolerances["rtol_grad"],
                cosine_threshold=0.99,
            )
            if not result.passed:
                failures.append(f"{pt_name}: {result.message}")

        assert len(failures) == 0, (
            f"Training gradient mismatch for {component} ({organism}):\n"
            + "\n".join(failures[:20])
        )

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_training_gradient_aggregation(
        self,
        cached_training_gradients,
        param_name_mapping,
        tolerances,
        organism,
    ):
        """Aggregate training gradient metrics by component.

        This provides a comprehensive view of gradient parity when using
        the actual training loss (scaled predictions and targets).
        """
        from tests.gradient_alignment import align_jax_gradient_to_pytorch
        from tests.integration_jax.comparison_utils import (
            compare_gradients,
            report_top_offenders,
        )

        _, jax_grads = cached_training_gradients[organism]["jax"]
        _, pt_grads = cached_training_gradients[organism]["pytorch"]

        all_results = []
        component_stats = {}

        for pt_name, pt_grad in pt_grads.items():
            jax_name = param_name_mapping.get(pt_name)
            if jax_name is None or jax_name not in jax_grads:
                continue

            jax_grad = jax_grads[jax_name]

            try:
                jax_aligned = align_jax_gradient_to_pytorch(
                    pt_name, jax_grad, pt_grad.shape
                )
            except ValueError:
                continue

            # Use rtol_grad for gradients - backward pass accumulates more numerical
            # error than forward pass. Cosine similarity (>0.99) is the key metric.
            result = compare_gradients(
                pt_name, pt_grad, jax_aligned, rtol=tolerances["rtol_grad"]
            )
            all_results.append(result)

            # Extract component name
            component = pt_name.split(".")[0]
            if component not in component_stats:
                component_stats[component] = {"total": 0, "passed": 0, "cosines": []}
            component_stats[component]["total"] += 1
            if result.passed:
                component_stats[component]["passed"] += 1
            component_stats[component]["cosines"].append(result.cosine_sim)

        # Report by component
        print(f"\n{organism.upper()} Training Gradient Parity by Component:")
        print("-" * 70)
        for comp, stats in sorted(component_stats.items()):
            pass_rate = (
                stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
            median_cos = np.median(stats["cosines"]) if stats["cosines"] else 0
            print(
                f"  {comp:25s}: {stats['passed']:3d}/{stats['total']:3d} passed "
                f"({pass_rate:5.1f}%), median cosine={median_cos:.4f}"
            )

        # Report top offenders
        print("\n" + report_top_offenders(all_results, k=10))

        # Calculate overall pass rate
        total_passed = sum(1 for r in all_results if r.passed)
        total_count = len(all_results)
        pass_rate = total_passed / total_count * 100 if total_count > 0 else 0

        print(
            f"\nOverall: {total_passed}/{total_count} parameters passed ({pass_rate:.1f}%)"
        )

        # Fail if pass rate is below threshold
        min_pass_rate = 95.0
        assert pass_rate >= min_pass_rate, (
            f"Only {pass_rate:.1f}% of parameters passed (expected >= {min_pass_rate}%)"
        )
