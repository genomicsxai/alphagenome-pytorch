"""
JAX-PyTorch Equivalence Tests for AlphaGenome Losses.

These tests verify numerical equivalence between PyTorch and JAX implementations.
Run with: pytest tests/jax_comparison/test_losses_jax.py -v
"""

import pytest
import numpy as np

# Skip entire module if JAX is not available
pytest.importorskip("jax")
pytest.importorskip("jax.numpy")

import jax.numpy as jnp


@pytest.fixture
def jax_losses():
    """Import JAX losses module."""
    from alphagenome_research.model import losses as jax_losses
    return jax_losses


@pytest.fixture
def torch_losses():
    """Import PyTorch losses module."""
    import torch
    from alphagenome_pytorch import losses as torch_losses
    return torch_losses


@pytest.fixture
def torch():
    """Import torch module."""
    import torch
    return torch


@pytest.mark.jax
class TestPoissonLossEquivalence:
    """Verify poisson_loss matches JAX implementation."""
    
    def test_random_data(self, jax_losses, torch_losses, torch):
        """Test with random input data."""
        np.random.seed(42)
        y_true_np = np.abs(np.random.randn(3, 4)).astype(np.float32)
        y_pred_np = np.abs(np.random.randn(3, 4)).astype(np.float32)
        mask_np = np.ones((3, 4), dtype=bool)
        
        jax_result = jax_losses.poisson_loss(
            y_true=jnp.array(y_true_np),
            y_pred=jnp.array(y_pred_np),
            mask=jnp.array(mask_np),
        )
        
        torch_result = torch_losses.poisson_loss(
            y_true=torch.tensor(y_true_np),
            y_pred=torch.tensor(y_pred_np),
            mask=torch.tensor(mask_np),
        )
        
        np.testing.assert_almost_equal(
            float(jax_result), torch_result.item(), decimal=5
        )
    
    def test_with_partial_mask(self, jax_losses, torch_losses, torch):
        """Test with partial masking."""
        np.random.seed(123)
        y_true_np = np.abs(np.random.randn(2, 5)).astype(np.float32)
        y_pred_np = np.abs(np.random.randn(2, 5)).astype(np.float32)
        mask_np = np.array([[True, True, False, True, False],
                           [True, False, True, True, True]], dtype=bool)
        
        jax_result = jax_losses.poisson_loss(
            y_true=jnp.array(y_true_np),
            y_pred=jnp.array(y_pred_np),
            mask=jnp.array(mask_np),
        )
        
        torch_result = torch_losses.poisson_loss(
            y_true=torch.tensor(y_true_np),
            y_pred=torch.tensor(y_pred_np),
            mask=torch.tensor(mask_np),
        )
        
        np.testing.assert_almost_equal(
            float(jax_result), torch_result.item(), decimal=5
        )


@pytest.mark.jax
class TestMultinomialLossEquivalence:
    """Verify multinomial_loss matches JAX implementation."""
    
    def test_basic(self, jax_losses, torch_losses, torch):
        """Test with basic input."""
        y_true = np.array([[[10.0, 1.0, 3.0], [5.0, 2.0, 20.0]]]).astype(np.float32)
        y_pred = np.array([[[0.5, 2.5, 1.0], [2.5, 0.5, 1.0]]]).astype(np.float32)
        mask = np.array([[[True, True, True]]])
        
        jax_result = jax_losses.multinomial_loss(
            y_true=jnp.array(y_true),
            y_pred=jnp.array(y_pred),
            mask=jnp.array(mask),
            multinomial_resolution=1,
            positional_weight=1.0,
        )
        
        torch_result = torch_losses.multinomial_loss(
            y_true=torch.tensor(y_true),
            y_pred=torch.tensor(y_pred),
            mask=torch.tensor(mask),
            multinomial_resolution=1,
            positional_weight=1.0,
        )
        
        for key in ['loss', 'loss_total', 'loss_positional']:
            np.testing.assert_almost_equal(
                float(jax_result[key]),
                torch_result[key].item(),
                decimal=5,
                err_msg=f"Mismatch for {key}"
            )
    
    def test_with_resolution(self, jax_losses, torch_losses, torch):
        """Test with multinomial resolution > 1."""
        np.random.seed(42)
        y_true = np.abs(np.random.randn(1, 8, 3)).astype(np.float32)
        y_pred = np.abs(np.random.randn(1, 8, 3)).astype(np.float32) + 0.1
        mask = np.ones((1, 1, 3), dtype=bool)
        
        jax_result = jax_losses.multinomial_loss(
            y_true=jnp.array(y_true),
            y_pred=jnp.array(y_pred),
            mask=jnp.array(mask),
            multinomial_resolution=4,
            positional_weight=0.5,
        )
        
        torch_result = torch_losses.multinomial_loss(
            y_true=torch.tensor(y_true),
            y_pred=torch.tensor(y_pred),
            mask=torch.tensor(mask),
            multinomial_resolution=4,
            positional_weight=0.5,
        )
        
        for key in ['loss', 'loss_total', 'loss_positional']:
            np.testing.assert_almost_equal(
                float(jax_result[key]),
                torch_result[key].item(),
                decimal=5,
                err_msg=f"Mismatch for {key}"
            )


@pytest.mark.jax
class TestMSEEquivalence:
    """Verify mse matches JAX implementation."""
    
    def test_random_data(self, jax_losses, torch_losses, torch):
        """Test with random data."""
        np.random.seed(42)
        y_true_np = np.random.randn(3, 4).astype(np.float32)
        y_pred_np = np.random.randn(3, 4).astype(np.float32)
        mask_np = np.ones((3, 4), dtype=bool)
        
        jax_result = jax_losses.mse(
            y_true=jnp.array(y_true_np),
            y_pred=jnp.array(y_pred_np),
            mask=jnp.array(mask_np),
        )
        
        torch_result = torch_losses.mse(
            y_true=torch.tensor(y_true_np),
            y_pred=torch.tensor(y_pred_np),
            mask=torch.tensor(mask_np),
        )
        
        np.testing.assert_almost_equal(
            float(jax_result), torch_result.item(), decimal=5
        )


@pytest.mark.jax
class TestCrossEntropyFromLogitsEquivalence:
    """Verify cross_entropy_loss_from_logits matches JAX implementation."""
    
    def test_basic(self, jax_losses, torch_losses, torch):
        """Test with basic classification data."""
        np.random.seed(42)
        y_pred_logits = np.random.randn(2, 5, 4).astype(np.float32)
        # One-hot targets
        y_true = np.zeros((2, 5, 4), dtype=np.float32)
        indices = np.random.randint(0, 4, size=(2, 5))
        for b in range(2):
            for s in range(5):
                y_true[b, s, indices[b, s]] = 1.0
        mask = np.ones((2, 5, 4), dtype=bool)
        
        jax_result = jax_losses.cross_entropy_loss_from_logits(
            y_pred_logits=jnp.array(y_pred_logits),
            y_true=jnp.array(y_true),
            mask=jnp.array(mask),
            axis=-1,
        )
        
        torch_result = torch_losses.cross_entropy_loss_from_logits(
            y_pred_logits=torch.tensor(y_pred_logits),
            y_true=torch.tensor(y_true),
            mask=torch.tensor(mask),
            axis=-1,
        )
        
        np.testing.assert_almost_equal(
            float(jax_result), torch_result.item(), decimal=5
        )
    
    def test_with_partial_mask(self, jax_losses, torch_losses, torch):
        """Test with partial masking."""
        np.random.seed(123)
        y_pred_logits = np.random.randn(2, 3, 4).astype(np.float32)
        y_true = np.zeros((2, 3, 4), dtype=np.float32)
        y_true[:, :, 0] = 1.0  # All same class
        mask = np.array([[[True, True, True, True],
                          [False, False, False, False],
                          [True, True, True, True]],
                         [[True, True, True, True],
                          [True, True, True, True],
                          [False, False, False, False]]], dtype=bool)
        
        jax_result = jax_losses.cross_entropy_loss_from_logits(
            y_pred_logits=jnp.array(y_pred_logits),
            y_true=jnp.array(y_true),
            mask=jnp.array(mask),
            axis=-1,
        )
        
        torch_result = torch_losses.cross_entropy_loss_from_logits(
            y_pred_logits=torch.tensor(y_pred_logits),
            y_true=torch.tensor(y_true),
            mask=torch.tensor(mask),
            axis=-1,
        )
        
        np.testing.assert_almost_equal(
            float(jax_result), torch_result.item(), decimal=5
        )


@pytest.mark.jax
class TestBinaryCrossEntropyFromLogitsEquivalence:
    """Verify binary_crossentropy_from_logits matches JAX implementation."""
    
    def test_basic(self, jax_losses, torch_losses, torch):
        """Test with random binary targets."""
        np.random.seed(42)
        y_pred = np.random.randn(3, 5).astype(np.float32)
        y_true = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
        mask = np.ones((3, 5), dtype=bool)
        
        jax_result = jax_losses.binary_crossentropy_from_logits(
            y_pred=jnp.array(y_pred),
            y_true=jnp.array(y_true),
            mask=jnp.array(mask),
        )
        
        torch_result = torch_losses.binary_crossentropy_from_logits(
            y_pred=torch.tensor(y_pred),
            y_true=torch.tensor(y_true),
            mask=torch.tensor(mask),
        )
        
        np.testing.assert_almost_equal(
            float(jax_result), torch_result.item(), decimal=5
        )
    
    def test_extreme_logits(self, jax_losses, torch_losses, torch):
        """Test numerical stability with extreme logits."""
        y_pred = np.array([[10.0, -10.0, 5.0, -5.0]], dtype=np.float32)
        y_true = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        mask = np.ones((1, 4), dtype=bool)
        
        jax_result = jax_losses.binary_crossentropy_from_logits(
            y_pred=jnp.array(y_pred),
            y_true=jnp.array(y_true),
            mask=jnp.array(mask),
        )
        
        torch_result = torch_losses.binary_crossentropy_from_logits(
            y_pred=torch.tensor(y_pred),
            y_true=torch.tensor(y_true),
            mask=torch.tensor(mask),
        )
        
        np.testing.assert_almost_equal(
            float(jax_result), torch_result.item(), decimal=5
        )


@pytest.mark.jax
class TestCrossEntropyLossEquivalence:
    """Verify cross_entropy_loss matches JAX implementation."""
    
    def test_basic(self, jax_losses, torch_losses, torch):
        """Test with count data."""
        np.random.seed(42)
        y_true = np.abs(np.random.randn(2, 4, 3)).astype(np.float32) + 0.1
        y_pred = np.abs(np.random.randn(2, 4, 3)).astype(np.float32) + 0.1
        mask = np.ones((2, 4, 3), dtype=bool)
        
        jax_result = jax_losses.cross_entropy_loss(
            y_true=jnp.array(y_true),
            y_pred=jnp.array(y_pred),
            mask=jnp.array(mask),
            axis=-1,
        )
        
        torch_result = torch_losses.cross_entropy_loss(
            y_true=torch.tensor(y_true),
            y_pred=torch.tensor(y_pred),
            mask=torch.tensor(mask),
            axis=-1,
        )
        
        np.testing.assert_almost_equal(
            float(jax_result), torch_result.item(), decimal=5
        )
    
    def test_with_mask(self, jax_losses, torch_losses, torch):
        """Test with partial masking."""
        np.random.seed(123)
        y_true = np.abs(np.random.randn(2, 3, 4)).astype(np.float32) + 0.1
        y_pred = np.abs(np.random.randn(2, 3, 4)).astype(np.float32) + 0.1
        mask = np.array([[[True, True, False, True],
                          [True, True, True, True],
                          [False, True, True, True]],
                         [[True, True, True, False],
                          [True, False, True, True],
                          [True, True, True, True]]], dtype=bool)
        
        jax_result = jax_losses.cross_entropy_loss(
            y_true=jnp.array(y_true),
            y_pred=jnp.array(y_pred),
            mask=jnp.array(mask),
            axis=-1,
        )
        
        torch_result = torch_losses.cross_entropy_loss(
            y_true=torch.tensor(y_true),
            y_pred=torch.tensor(y_pred),
            mask=torch.tensor(mask),
            axis=-1,
        )
        
        np.testing.assert_almost_equal(
            float(jax_result), torch_result.item(), decimal=5
        )
