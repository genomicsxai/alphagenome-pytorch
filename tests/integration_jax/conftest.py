import os
# Prevent JAX from preallocating all GPU memory.
# These MUST be set before JAX is imported anywhere.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"  # Limit JAX to 40% of GPU

import pytest
import numpy as np

@pytest.fixture(scope="session")
def jax_model(jax_checkpoint_path):
    """Load JAX AlphaGenome model."""
    import jax
    import jax.numpy as jnp
    from alphagenome_research.model import dna_model
    from alphagenome_research.model import heads

    # Monkey patch GenomeTracksHead to sanitize track means
    # PyTorch replaces NaN track means with 0.0 to avoid NaN propagation.
    # JAX by default propagates NaN, which poisons gradients.
    # We patch JAX here to match PyTorch behavior for valid comparison.
    OriginalGenomeTracksHead = heads.GenomeTracksHead
    
    class PatchedGenomeTracksHead(OriginalGenomeTracksHead):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._track_means = jnp.nan_to_num(self._track_means)
            
    heads.GenomeTracksHead = PatchedGenomeTracksHead

    # Use GPU if available
    try:
        device = jax.devices("gpu")[0]
    except Exception:
        device = jax.devices("cpu")[0]

    model = dna_model.create(str(jax_checkpoint_path), device=device)
    return model


@pytest.fixture(scope="session")
def jax_output_types():
    """Return JAX OutputType enum mapping."""
    from alphagenome.models import dna_output

    return {
        "atac": dna_output.OutputType.ATAC,
        "dnase": dna_output.OutputType.DNASE,
        "procap": dna_output.OutputType.PROCAP,
        "cage": dna_output.OutputType.CAGE,
        "rna_seq": dna_output.OutputType.RNA_SEQ,
        "chip_tf": dna_output.OutputType.CHIP_TF,
        "chip_histone": dna_output.OutputType.CHIP_HISTONE,
        "contact_maps": dna_output.OutputType.CONTACT_MAPS,
        "splice_sites_classification": dna_output.OutputType.SPLICE_SITES,
        "splice_sites_usage": dna_output.OutputType.SPLICE_SITE_USAGE,
        "splice_sites_junction": dna_output.OutputType.SPLICE_JUNCTIONS,
    }


@pytest.fixture(scope="session")
def run_jax_inference(jax_model):
    """Factory fixture to run JAX inference."""
    import jax
    import jax.numpy as jnp
    from .fixture_utils import create_jax_forward_fn, jax_to_numpy

    # Create forward function with float32 precision (matches PyTorch)
    _forward, _apply_fn = create_jax_forward_fn(jax_model, use_float32=True)

    def _run(sequence: np.ndarray, organism_index: int):
        """Run JAX model inference using raw apply_fn for all resolutions.

        Args:
            sequence: One-hot encoded DNA (B, S, 4)
            organism_index: 0 for human, 1 for mouse

        Returns:
            Dict of predictions from JAX model
        """
        batch_size = sequence.shape[0]
        jax_input = jnp.array(sequence)
        jax_org = jnp.array([organism_index] * batch_size, dtype=jnp.int32)

        # Use our recreated apply_fn to get all resolutions (both 1bp and 128bp)
        outputs = _apply_fn(
            jax_model._params,
            jax_model._state,
            jax_input,
            jax_org,
        )

        # Convert to numpy float32
        return jax.tree.map(jax_to_numpy, outputs)

    return _run


@pytest.fixture(scope="session")
def run_pytorch_inference(pytorch_model):
    """Factory fixture to run PyTorch inference."""
    import torch
    from .fixture_utils import pytorch_to_numpy

    def _run(sequence: np.ndarray, organism_index: int):
        """Run PyTorch model inference.

        Args:
            sequence: One-hot encoded DNA (B, S, 4)
            organism_index: 0 for human, 1 for mouse

        Returns:
            Dict of predictions from PyTorch model
        """
        batch_size = sequence.shape[0]
        device = next(pytorch_model.parameters()).device
        pt_input = torch.tensor(sequence).to(device)
        pt_org = torch.tensor([organism_index] * batch_size, dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = pytorch_model(pt_input, pt_org)

        return pytorch_to_numpy(outputs)

    return _run


@pytest.fixture(scope="session")
def cached_jax_predictions(random_dna_sequence, run_jax_inference):
    """Cache JAX predictions only (no PyTorch).

    Use this fixture in JAX-only tests to avoid loading PyTorch model.

    Returns dict with structure:
    {
        "human": {...},
        "mouse": {...},
    }
    """
    from .fixture_utils import iterate_organisms

    cache = {}
    for org_name, org_idx in iterate_organisms():
        cache[org_name] = run_jax_inference(random_dna_sequence, org_idx)

    return cache


@pytest.fixture(scope="session")
def cached_pytorch_predictions(random_dna_sequence, run_pytorch_inference):
    """Cache PyTorch predictions only (no JAX).

    Use this fixture in PyTorch-only tests to avoid loading JAX model.
    This is the recommended fixture for tests that don't need JAX comparison.

    Returns dict with structure:
    {
        "human": {...},
        "mouse": {...},
    }
    """
    from .fixture_utils import iterate_organisms

    cache = {}
    for org_name, org_idx in iterate_organisms():
        cache[org_name] = run_pytorch_inference(random_dna_sequence, org_idx)

    return cache


@pytest.fixture(scope="session")
def cached_predictions(cached_jax_predictions, cached_pytorch_predictions):
    """Cache predictions for both models (JAX and PyTorch).

    This is a backward-compatibility fixture that combines cached_jax_predictions
    and cached_pytorch_predictions. Use framework-specific fixtures when possible
    to avoid loading both models unnecessarily.

    Returns dict with structure:
    {
        "human": {"jax": {...}, "pytorch": {...}},
        "mouse": {"jax": {...}, "pytorch": {...}},
    }
    """
    return {
        org: {
            "jax": cached_jax_predictions[org],
            "pytorch": cached_pytorch_predictions[org]
        }
        for org in ["human", "mouse"]
    }


@pytest.fixture(scope="session")
def cached_junction_with_shared_positions(
    random_dna_sequence, cached_jax_predictions, pytorch_model
):
    """Cache junction predictions using JAX's actual splice_site_positions.

    This fixture extracts the splice_site_positions that JAX actually used
    (from its junction output), then re-runs PyTorch with those same positions.
    This ensures both frameworks compute junction outputs at identical positions.

    This is necessary because JAX uses approx_max_k (approximate top-k) while
    PyTorch uses exact top-k, leading to different position selections even
    with identical classification probabilities.

    Returns dict with structure:
    {
        "human": {"jax": {...}, "pytorch": {...}},
        "mouse": {"jax": {...}, "pytorch": {...}},
    }
    """
    import torch
    from .fixture_utils import iterate_organisms, pytorch_to_numpy

    cache = {}
    device = next(pytorch_model.parameters()).device

    for org_name, org_idx in iterate_organisms():
        jax_out = cached_jax_predictions[org_name]

        # Use the actual positions JAX used (from its junction output)
        jax_positions = jax_out["splice_sites_junction"]["splice_site_positions"]
        positions_tensor = torch.tensor(jax_positions, dtype=torch.long, device=device)

        # Re-run PyTorch with JAX's actual positions
        batch_size = random_dna_sequence.shape[0]
        pt_input = torch.tensor(random_dna_sequence).to(device)
        pt_org = torch.tensor([org_idx] * batch_size, dtype=torch.long).to(device)

        with torch.no_grad():
            pt_out = pytorch_model(pt_input, pt_org, splice_site_positions=positions_tensor)

        cache[org_name] = {
            "jax": jax_out,
            "pytorch": pytorch_to_numpy(pt_out),
        }

    return cache


@pytest.fixture(scope="session")
def run_jax_backward(jax_model):
    """Factory fixture to compute JAX gradients."""
    import jax
    import jax.numpy as jnp
    from .fixture_utils import create_jax_loss_fn, flatten_jax_gradients

    # Create combined loss function with float32 precision
    _loss_fn = create_jax_loss_fn(jax_model, loss_type='combined', use_float32=True)

    def _run(sequence: np.ndarray, organism_index: int):
        """Run JAX backward pass and return gradients."""
        batch_size = sequence.shape[0]
        jax_input = jnp.array(sequence)
        jax_org = jnp.array([organism_index] * batch_size, dtype=jnp.int32)

        # Compute loss and gradients using JAX autodiff
        loss_value, grads = jax.value_and_grad(
            lambda p: _loss_fn(p, jax_model._state, jax_input, jax_org)
        )(jax_model._params)

        # Flatten gradient tree to dict with param names
        flat_grads = flatten_jax_gradients(grads)
        return float(loss_value), flat_grads

    return _run


@pytest.fixture(scope="session")
def run_pytorch_backward(pytorch_model):
    """Factory fixture to compute PyTorch gradients."""
    import torch
    from tests.gradient_utils import compute_combined_loss

    def _run(sequence: np.ndarray, organism_index: int):
        """Run PyTorch backward pass and return gradients.

        Args:
            sequence: DNA sequence input
            organism_index: Organism index (0=human, 1=mouse)
        """
        batch_size = sequence.shape[0]

        # Use eval mode to disable dropout but keep requires_grad=True
        pytorch_model.eval()

        device = next(pytorch_model.parameters()).device
        pt_input = torch.tensor(sequence, requires_grad=True, device=device)
        pt_org = torch.tensor([organism_index] * batch_size, dtype=torch.long, device=device)

        pytorch_model.zero_grad()
        # NOTE: Gradient tests use unscaled predictions (experimental space) to match
        # the JAX test which uses 'predictions_*' keys (also unscaled).
        # This is different from actual training which should use scaled predictions.
        outputs = pytorch_model(pt_input, pt_org)
        loss = compute_combined_loss(outputs, include_splice=True)
        loss.backward()

        grads = {}
        for name, param in pytorch_model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().cpu().numpy().astype(np.float32)

        return loss.item(), grads

    return _run


@pytest.fixture(scope="session")
def cached_jax_gradients(random_dna_sequence, run_jax_backward):
    """Cache JAX gradients only (no PyTorch).

    Returns dict with structure:
    {
        "human": (loss_value, gradients_dict),
        "mouse": (loss_value, gradients_dict),
    }
    """
    from .fixture_utils import iterate_organisms

    cache = {}
    for org_name, org_idx in iterate_organisms():
        cache[org_name] = run_jax_backward(random_dna_sequence, org_idx)

    return cache


@pytest.fixture(scope="session")
def cached_pytorch_gradients(random_dna_sequence, run_pytorch_backward):
    """Cache PyTorch gradients only (no JAX).

    Returns dict with structure:
    {
        "human": (loss_value, gradients_dict),
        "mouse": (loss_value, gradients_dict),
    }
    """
    from .fixture_utils import iterate_organisms

    cache = {}
    for org_name, org_idx in iterate_organisms():
        cache[org_name] = run_pytorch_backward(random_dna_sequence, org_idx)

    return cache


@pytest.fixture(scope="session")
def cached_gradients(cached_jax_gradients, cached_pytorch_gradients):
    """Cache gradients for both models (JAX and PyTorch).

    This is a backward-compatibility fixture. Use framework-specific fixtures
    when possible to avoid loading both models unnecessarily.

    Returns dict with structure:
    {
        "human": {"jax": (loss, grads), "pytorch": (loss, grads)},
        "mouse": {"jax": (loss, grads), "pytorch": (loss, grads)},
    }
    """
    return {
        org: {
            "jax": cached_jax_gradients[org],
            "pytorch": cached_pytorch_gradients[org]
        }
        for org in ["human", "mouse"]
    }


@pytest.fixture(scope="session")
def param_name_mapping(pytorch_model):
    """Mapping from PyTorch param names to JAX param names."""
    from alphagenome_pytorch.jax_compat.weight_mapping import map_pytorch_to_jax

    mapping = {}

    # Iterate over all PyTorch parameters and construct corresponding JAX key
    for pt_name, _ in pytorch_model.named_parameters():
        jax_key = map_pytorch_to_jax(pt_name)
        if jax_key:
            mapping[pt_name] = jax_key

    return mapping


@pytest.fixture(scope="session")
def run_jax_per_head_loss(jax_model):
    """Factory fixture to compute per-head losses in JAX."""
    import jax
    import jax.numpy as jnp
    import haiku as hk
    import jmp
    from alphagenome_research.model import model

    jmp_policy = jmp.get_policy('params=float32,compute=float32,output=float32')

    @hk.transform_with_state
    def _forward(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(model.AlphaGenome, jmp_policy):
            return model.AlphaGenome(jax_model._metadata)(dna_sequence, organism_index)

    def _run(sequence: np.ndarray, organism_index: int):
        """Compute per-head losses for JAX model."""
        batch_size = sequence.shape[0]
        jax_input = jnp.array(sequence)
        jax_org = jnp.array([organism_index] * batch_size, dtype=jnp.int32)

        (predictions, _), _ = _forward.apply(
            jax_model._params, jax_model._state, None, jax_input, jax_org
        )

        losses = {}

        # Genome tracks
        for head_name in ['atac', 'dnase', 'procap', 'cage', 'rna_seq', 'chip_tf', 'chip_histone']:
            if head_name in predictions:
                head_loss = jnp.array(0.0)
                head_out = predictions[head_name]
                for key, val in head_out.items():
                    if key.startswith('predictions_'):
                        head_loss = head_loss + jnp.nanmean(val)
                losses[head_name] = float(head_loss)

        # Contact maps
        if 'contact_maps' in predictions:
            cm_out = predictions['contact_maps']
            if isinstance(cm_out, dict) and 'predictions' in cm_out:
                losses['contact_maps'] = float(cm_out['predictions'].mean())

        return losses

    return _run


@pytest.fixture(scope="session")
def run_pytorch_per_head_loss(pytorch_model):
    """Factory fixture to compute per-head losses in PyTorch."""
    import torch
    from tests.gradient_utils import compute_per_head_losses

    def _run(sequence: np.ndarray, organism_index: int):
        """Compute per-head losses for PyTorch model."""
        batch_size = sequence.shape[0]
        device = next(pytorch_model.parameters()).device
        pt_input = torch.tensor(sequence).to(device)
        pt_org = torch.tensor([organism_index] * batch_size, dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = pytorch_model(pt_input, pt_org)

        losses = compute_per_head_losses(outputs)
        return {k: float(v) for k, v in losses.items()}

    return _run


@pytest.fixture(scope="session")
def cached_jax_per_head_losses(random_dna_sequence, run_jax_per_head_loss):
    """Cache JAX per-head losses only (no PyTorch).

    Returns dict with structure:
    {
        "human": {"atac": loss, "dnase": loss, ...},
        "mouse": {"atac": loss, "dnase": loss, ...},
    }
    """
    from .fixture_utils import iterate_organisms

    cache = {}
    for org_name, org_idx in iterate_organisms():
        cache[org_name] = run_jax_per_head_loss(random_dna_sequence, org_idx)

    return cache


@pytest.fixture(scope="session")
def cached_pytorch_per_head_losses(random_dna_sequence, run_pytorch_per_head_loss):
    """Cache PyTorch per-head losses only (no JAX).

    Returns dict with structure:
    {
        "human": {"atac": loss, "dnase": loss, ...},
        "mouse": {"atac": loss, "dnase": loss, ...},
    }
    """
    from .fixture_utils import iterate_organisms

    cache = {}
    for org_name, org_idx in iterate_organisms():
        cache[org_name] = run_pytorch_per_head_loss(random_dna_sequence, org_idx)

    return cache


@pytest.fixture(scope="session")
def cached_per_head_losses(cached_jax_per_head_losses, cached_pytorch_per_head_losses):
    """Cache per-head losses for both models (JAX and PyTorch).

    This is a backward-compatibility fixture. Use framework-specific fixtures
    when possible to avoid loading both models unnecessarily.

    Returns dict with structure:
    {
        "human": {"jax": {"atac": loss, ...}, "pytorch": {"atac": loss, ...}},
        "mouse": {"jax": {"atac": loss, ...}, "pytorch": {"atac": loss, ...}},
    }
    """
    return {
        org: {
            "jax": cached_jax_per_head_losses[org],
            "pytorch": cached_pytorch_per_head_losses[org]
        }
        for org in ["human", "mouse"]
    }


# =============================================================================
# Memory Management
# =============================================================================


def clear_gpu_memory():
    """Clear both JAX and PyTorch GPU memory caches."""
    import gc
    gc.collect()

    # Clear PyTorch cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    # Clear JAX cache
    try:
        import jax
        jax.clear_caches()
    except (ImportError, AttributeError):
        pass

    gc.collect()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clear GPU memory after each test to prevent OOM errors."""
    yield
    clear_gpu_memory()


