"""Shared fixtures for JAX comparison tests."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Mark all tests in this directory as jax_comparison tests."""
    for item in items:
        # Add jax_comparison marker to all tests in this directory
        item.add_marker(pytest.mark.jax_comparison)


@pytest.fixture(scope="session")
def jax_available():
    """Check if JAX is available."""
    try:
        import jax
        return True
    except ImportError:
        pytest.skip("JAX not installed")


@pytest.fixture(scope="session")
def alphagenome_research_available():
    """Check if alphagenome_research is available."""
    try:
        from alphagenome_research.model import dna_model
        return True
    except ImportError:
        pytest.skip("alphagenome_research not installed")
