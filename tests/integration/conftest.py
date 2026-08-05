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
