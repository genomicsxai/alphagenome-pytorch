"""Shared pytest fixtures for AlphaGenome testing."""

import pytest
import torch
import numpy as np
from pathlib import Path

# =============================================================================
# NUMERICAL PRECISION SETTINGS
# =============================================================================
# Disable TF32 to ensure reproducible gradient comparisons.
# TF32 (Tensor Float 32) on Ampere+ GPUs provides only 19-bit mantissa precision
# vs 23-bit for FP32, causing ~0.1-1% gradient drift between JAX and PyTorch.
# This is the most common cause of small gradient parity failures.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Enable deterministic mode for reproducibility (may impact performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--jax-checkpoint",
        action="store",
        default=None,
        help="Path to JAX checkpoint directory for comparison tests",
    )
    parser.addoption(
        "--torch-weights",
        action="store",
        default="model.pth",
        help="Path to PyTorch weights file",
    )
    parser.addoption(
        "--rtol",
        action="store",
        default=0.05,
        type=float,
        help="Relative tolerance for numerical comparison (5% default accounts for bfloat16 precision drift)",
    )
    parser.addoption(
        "--atol",
        action="store",
        default=1e-4,
        type=float,
        help="Absolute tolerance for numerical comparison",
    )
    parser.addoption(
        "--rtol-grad",
        action="store",
        default=0.08,
        type=float,
        help="Relative tolerance for gradient comparison (8% default - gradients accumulate more numerical error)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip JAX tests if JAX not available or checkpoint not provided.

    Marker meanings:
    - 'jax': Test requires JAX installation (component-level tests)
    - 'jax_comparison': Test requires JAX (component-level, in jax_comparison/)
    - 'integration': PyTorch-only full model tests (no JAX required)
    - 'integration_jax': Full model JAX comparison (requires --jax-checkpoint)
    - 'finetuning': Fine-tuning tests (no JAX required)
    """
    jax_checkpoint = config.getoption("--jax-checkpoint")

    import sys
    try:
        import jax
        # Verify JAX has the Array type (added in 0.4.1, required by scipy 1.16+)
        # scipy's array_api_compat checks sys.modules['jax'] and tries getattr(jax, 'Array')
        # which fails on old JAX versions and breaks scipy.stats functions
        _ = jax.Array
        jax_available = True
    except (ImportError, AttributeError):
        # If JAX is missing or too old, remove it from sys.modules
        # to prevent scipy's is_jax_array() from finding a broken JAX
        for mod_name in list(sys.modules.keys()):
            if mod_name == 'jax' or mod_name.startswith('jax.'):
                del sys.modules[mod_name]
        jax_available = False

    skip_jax = pytest.mark.skip(reason="JAX not installed")
    skip_no_checkpoint = pytest.mark.skip(reason="--jax-checkpoint not provided")

    for item in items:
        has_jax = "jax" in item.keywords
        has_jax_comparison = "jax_comparison" in item.keywords
        has_integration_jax = "integration_jax" in item.keywords
        has_finetuning = "finetuning" in item.keywords

        # Fine-tuning tests don't require JAX at all
        if has_finetuning:
            continue

        # Skip all JAX tests if JAX not available
        if (has_jax or has_jax_comparison or has_integration_jax) and not jax_available:
            item.add_marker(skip_jax)
        # Require checkpoint for integration_jax tests
        elif has_integration_jax and jax_checkpoint is None:
            item.add_marker(skip_no_checkpoint)



# Path Fixtures


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def mock_data_dir(project_root):
    """Ensure mock data exists and return path.

    If mock data files are missing, automatically regenerates them
    using the create_mock_data.py script.
    """
    mock_dir = project_root / "tests" / "fixtures" / "mock_data"

    # Check if all required files exist
    required_files = [
        "mock_genome.fa",
        "mock_positions.bed",
        "mock_rnaseq_track1.bw",
        "mock_atac_track1.bw",
    ]

    missing = [f for f in required_files if not (mock_dir / f).exists()]

    if missing:
        # Generate mock data
        import subprocess, sys
        script = project_root / "tests" / "create_mock_data.py"
        subprocess.run(
            [sys.executable, str(script), "--output_dir", str(mock_dir)],
            check=True
        )

    return mock_dir


@pytest.fixture(scope="session")
def jax_checkpoint_path(request):
    """Return JAX checkpoint path from CLI."""
    path = request.config.getoption("--jax-checkpoint")
    if path:
        return Path(path)
    pytest.skip("--jax-checkpoint not provided")


@pytest.fixture(scope="session")
def torch_weights_path(request, project_root):
    """Return PyTorch weights path."""
    path = request.config.getoption("--torch-weights")
    full_path = project_root / path if not Path(path).is_absolute() else Path(path)
    if not full_path.exists():
        pytest.skip(f"PyTorch weights not found: {full_path}")
    return full_path


# Tolerance Fixtures


@pytest.fixture(scope="session")
def tolerances(request):
    """Return tolerance settings."""
    return {
        "rtol": request.config.getoption("--rtol"),
        "atol": request.config.getoption("--atol"),
        "rtol_grad": request.config.getoption("--rtol-grad"),
    }


# Input Generation Fixtures


@pytest.fixture(scope="session")
def random_seed():
    """Fixed seed for reproducibility."""
    return 42


@pytest.fixture(scope="session")
def sequence_length():
    """Standard AlphaGenome input sequence length.

    Note: Full training length is 131072, but we use 16384 for tests
    to fit both JAX and PyTorch backward passes in GPU memory.
    This is sufficient for gradient parity testing.
    """
    return 16384


@pytest.fixture(scope="session")
def batch_size():
    """Default batch size for testing."""
    return 1


@pytest.fixture(scope="session")
def random_dna_sequence(random_seed, batch_size, sequence_length):
    """Generate random one-hot encoded DNA sequence."""
    np.random.seed(random_seed)
    seq_ints = np.random.randint(0, 4, size=(batch_size, sequence_length))
    one_hot = np.eye(4, dtype=np.float32)[seq_ints]
    return one_hot


# Model Loading Fixtures (PyTorch)


@pytest.fixture(scope="session")
def pytorch_model(torch_weights_path):
    """Load PyTorch AlphaGenome model (session-scoped for efficiency).

    Note: Track means are bundled with the model weights, so no separate
    track_means file is needed.
    """
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy

    model = AlphaGenome(
        num_organisms=2,
        dtype_policy=DtypePolicy.full_float32(),
    )

    state_dict = torch.load(torch_weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    # Use GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    return model


# Lightweight Model Fixture


@pytest.fixture(scope="session")
def small_model():
    """Create a lightweight AlphaGenome model for unit tests.

    This creates a minimal model instance (1 organism, float32) without
    loading any pretrained weights. Useful for testing model behavior,
    determinism, checkpointing, etc. without requiring weight files.

    Note: Session-scoped so the model is only created once per test session.
    """
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy

    model = AlphaGenome(
        num_organisms=1,
        dtype_policy=DtypePolicy.full_float32(),
    )
    model.eval()
    return model


# Head Configuration Fixtures


@pytest.fixture(scope="session")
def head_configs():
    """Return head configurations matching model.py."""
    return {
        "atac": {"num_tracks": 256, "resolutions": [1, 128], "apply_squashing": False},
        "dnase": {"num_tracks": 384, "resolutions": [1, 128], "apply_squashing": False},
        "procap": {"num_tracks": 128, "resolutions": [1, 128], "apply_squashing": False},
        "cage": {"num_tracks": 640, "resolutions": [1, 128], "apply_squashing": False},
        "rna_seq": {"num_tracks": 768, "resolutions": [1, 128], "apply_squashing": True},
        "chip_tf": {"num_tracks": 1664, "resolutions": [128], "apply_squashing": False},
        "chip_histone": {
            "num_tracks": 1152,
            "resolutions": [128],
            "apply_squashing": False,
        },
        "contact_maps": {
            "num_tracks": 28,
            "resolutions": ["pair"],
            "apply_squashing": False,
        },
    }


# Output Shape Fixtures


@pytest.fixture(scope="session")
def expected_output_shapes(batch_size, sequence_length, head_configs):
    """Calculate expected output shapes for each head."""
    shapes = {}
    for head_name, config in head_configs.items():
        shapes[head_name] = {}
        for res in config["resolutions"]:
            if res == 1:
                seq_len = sequence_length  # 131072
            elif res == 128:
                seq_len = sequence_length // 128  # 1024
            elif res == "pair":
                seq_len = sequence_length // 2048  # 64 x 64 (128bp * 16x pooling)
                shapes[head_name]["pair"] = (
                    batch_size,
                    seq_len,
                    seq_len,
                    config["num_tracks"],
                )
                continue
            shapes[head_name][res] = (batch_size, seq_len, config["num_tracks"])
    return shapes
