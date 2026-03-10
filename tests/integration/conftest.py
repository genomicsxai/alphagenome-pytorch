"""PyTorch integration test fixtures.

This module contains fixtures for PyTorch-only integration tests.
JAX comparison fixtures are in tests/integration_jax/conftest.py.

Most fixtures are inherited from the root tests/conftest.py, including:
- pytorch_model
- random_dna_sequence
- tolerances
- mock_data_dir
- torch_weights_path
"""

import pytest
from pathlib import Path

@pytest.fixture(scope="module")
def mock_data_dir():
    """Path to mock data directory for finetuning tests."""
    return Path(__file__).parent.parent / "fixtures" / "mock_data"
