"""
JAX-PyTorch gradient equivalence tests.

Verifies that PyTorch backward pass produces gradients matching JAX.
Uses jax.grad() to compute reference gradients.
"""

import pytest
import numpy as np
import torch
import sys
import os


# Module-level tolerance constants for JAX-PyTorch comparison
RTOL_F32 = 1e-6
ATOL_F32 = 1e-8
RTOL_F64 = 2e-7
ATOL_F64 = 2e-8


# Fixtures for JAX/PyTorch comparison

@pytest.fixture
def jax_losses():
    """Load JAX loss functions."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from alphagenome_research.model import losses as jax_losses_module
    return jax_losses_module


@pytest.fixture
def pytorch_losses():
    """Load PyTorch loss functions."""
    from alphagenome_pytorch import losses as pt_losses_module
    return pt_losses_module


# =============================================================================
# Gradient Tests for Individual Loss Functions
# =============================================================================

@pytest.mark.unit
@pytest.mark.jax
class TestPoissonLossGradient:
    """Test gradients for poisson_loss."""
    
    def test_gradient_wrt_predictions(self, jax_losses, pytorch_losses):
        """Compare ∂poisson_loss/∂y_pred between JAX and PyTorch."""
        import jax
        import jax.numpy as jnp
        
        np.random.seed(42)
        y_true = np.abs(np.random.randn(2, 8, 3).astype(np.float32)) + 0.1
        y_pred = np.abs(np.random.randn(2, 8, 3).astype(np.float32)) + 0.1
        mask = np.ones((2, 1, 3), dtype=bool)
        
        # JAX gradient
        def jax_loss_fn(y_pred_jax):
            return jax_losses.poisson_loss(
                y_true=jnp.array(y_true),
                y_pred=y_pred_jax,
                mask=jnp.array(mask),
            )
        
        jax_grad = jax.grad(jax_loss_fn)(jnp.array(y_pred))
        jax_grad_np = np.array(jax_grad)
        
        # PyTorch gradient
        y_pred_pt = torch.tensor(y_pred, requires_grad=True)
        loss_pt = pytorch_losses.poisson_loss(
            y_true=torch.tensor(y_true),
            y_pred=y_pred_pt,
            mask=torch.tensor(mask),
        )
        loss_pt.backward()
        pt_grad_np = y_pred_pt.grad.numpy()
        
        np.testing.assert_allclose(jax_grad_np, pt_grad_np, rtol=RTOL_F32, atol=ATOL_F32)

    def test_gradient_wrt_targets(self, jax_losses, pytorch_losses):
        """Compare ∂poisson_loss/∂y_true between JAX and PyTorch."""
        import jax
        import jax.numpy as jnp

        np.random.seed(43)
        y_true = np.abs(np.random.randn(2, 8, 3).astype(np.float32)) + 0.1
        y_pred = np.abs(np.random.randn(2, 8, 3).astype(np.float32)) + 0.1
        mask = np.ones((2, 1, 3), dtype=bool)

        def jax_loss_fn(y_true_jax):
            return jax_losses.poisson_loss(
                y_true=y_true_jax,
                y_pred=jnp.array(y_pred),
                mask=jnp.array(mask),
            )

        jax_grad = jax.grad(jax_loss_fn)(jnp.array(y_true))
        jax_grad_np = np.array(jax_grad)

        y_true_pt = torch.tensor(y_true, requires_grad=True)
        loss_pt = pytorch_losses.poisson_loss(
            y_true=y_true_pt,
            y_pred=torch.tensor(y_pred),
            mask=torch.tensor(mask),
        )
        loss_pt.backward()
        pt_grad_np = y_true_pt.grad.numpy()

        np.testing.assert_allclose(jax_grad_np, pt_grad_np, rtol=RTOL_F32, atol=ATOL_F32)

    def test_gradient_with_partial_mask(self, jax_losses, pytorch_losses):
        """Compare gradients when some tracks are masked out."""
        import jax
        import jax.numpy as jnp

        np.random.seed(44)
        y_true = np.abs(np.random.randn(2, 8, 3).astype(np.float32)) + 0.1
        y_pred = np.abs(np.random.randn(2, 8, 3).astype(np.float32)) + 0.1
        # Mask out the second track entirely
        mask = np.ones((2, 1, 3), dtype=bool)
        mask[:, :, 1] = False

        def jax_loss_fn(y_pred_jax):
            return jax_losses.poisson_loss(
                y_true=jnp.array(y_true),
                y_pred=y_pred_jax,
                mask=jnp.array(mask),
            )

        jax_grad = jax.grad(jax_loss_fn)(jnp.array(y_pred))
        jax_grad_np = np.array(jax_grad)

        y_pred_pt = torch.tensor(y_pred, requires_grad=True)
        loss_pt = pytorch_losses.poisson_loss(
            y_true=torch.tensor(y_true),
            y_pred=y_pred_pt,
            mask=torch.tensor(mask),
        )
        loss_pt.backward()
        pt_grad_np = y_pred_pt.grad.numpy()

        np.testing.assert_allclose(jax_grad_np, pt_grad_np, rtol=RTOL_F32, atol=ATOL_F32)
        # Masked track should have zero gradient
        np.testing.assert_allclose(pt_grad_np[:, :, 1], 0.0, atol=1e-7)


@pytest.mark.unit
@pytest.mark.jax
class TestMSEGradient:
    """Test gradients for mse loss."""
    
    def test_gradient_wrt_predictions(self, jax_losses, pytorch_losses):
        """Compare ∂mse/∂y_pred between JAX and PyTorch."""
        import jax
        import jax.numpy as jnp
        
        np.random.seed(42)
        y_true = np.random.randn(2, 8, 3).astype(np.float32)
        y_pred = np.random.randn(2, 8, 3).astype(np.float32)
        mask = np.ones((2, 8, 3), dtype=bool)
        
        # JAX gradient
        def jax_loss_fn(y_pred_jax):
            return jax_losses.mse(
                y_true=jnp.array(y_true),
                y_pred=y_pred_jax,
                mask=jnp.array(mask),
            )
        
        jax_grad = jax.grad(jax_loss_fn)(jnp.array(y_pred))
        jax_grad_np = np.array(jax_grad)
        
        # PyTorch gradient
        y_pred_pt = torch.tensor(y_pred, requires_grad=True)
        loss_pt = pytorch_losses.mse(
            y_true=torch.tensor(y_true),
            y_pred=y_pred_pt,
            mask=torch.tensor(mask),
        )
        loss_pt.backward()
        pt_grad_np = y_pred_pt.grad.numpy()
        
        np.testing.assert_allclose(jax_grad_np, pt_grad_np, rtol=RTOL_F32, atol=ATOL_F32)

    def test_gradient_with_partial_mask(self, jax_losses, pytorch_losses):
        """Compare MSE gradients when some tracks are masked out."""
        import jax
        import jax.numpy as jnp

        np.random.seed(44)
        y_true = np.random.randn(2, 8, 3).astype(np.float32)
        y_pred = np.random.randn(2, 8, 3).astype(np.float32)
        mask = np.ones((2, 8, 3), dtype=bool)
        mask[:, :, 0] = False  # Mask out first track

        def jax_loss_fn(y_pred_jax):
            return jax_losses.mse(
                y_true=jnp.array(y_true),
                y_pred=y_pred_jax,
                mask=jnp.array(mask),
            )

        jax_grad = jax.grad(jax_loss_fn)(jnp.array(y_pred))
        jax_grad_np = np.array(jax_grad)

        y_pred_pt = torch.tensor(y_pred, requires_grad=True)
        loss_pt = pytorch_losses.mse(
            y_true=torch.tensor(y_true),
            y_pred=y_pred_pt,
            mask=torch.tensor(mask),
        )
        loss_pt.backward()
        pt_grad_np = y_pred_pt.grad.numpy()

        np.testing.assert_allclose(jax_grad_np, pt_grad_np, rtol=RTOL_F32, atol=ATOL_F32)
        # Masked track should have zero gradient
        np.testing.assert_allclose(pt_grad_np[:, :, 0], 0.0, atol=1e-7)


@pytest.mark.unit
@pytest.mark.jax
class TestMultinomialLossGradient:
    """Test gradients for multinomial_loss."""
    
    def test_gradient_wrt_predictions(self, jax_losses, pytorch_losses):
        """Compare ∂multinomial_loss/∂y_pred between JAX and PyTorch."""
        import jax
        import jax.numpy as jnp
        
        np.random.seed(42)
        y_true = np.abs(np.random.randn(2, 8, 3).astype(np.float32)) + 0.1
        y_pred = np.abs(np.random.randn(2, 8, 3).astype(np.float32)) + 0.1
        mask = np.ones((2, 1, 3), dtype=bool)
        
        # JAX gradient
        def jax_loss_fn(y_pred_jax):
            result = jax_losses.multinomial_loss(
                y_true=jnp.array(y_true),
                y_pred=y_pred_jax,
                mask=jnp.array(mask),
                multinomial_resolution=4,
                positional_weight=1.0,
            )
            return result['loss']
        
        jax_grad = jax.grad(jax_loss_fn)(jnp.array(y_pred))
        jax_grad_np = np.array(jax_grad)
        
        # PyTorch gradient
        y_pred_pt = torch.tensor(y_pred, requires_grad=True)
        loss_dict = pytorch_losses.multinomial_loss(
            y_true=torch.tensor(y_true),
            y_pred=y_pred_pt,
            mask=torch.tensor(mask),
            multinomial_resolution=4,
            positional_weight=1.0,
        )
        loss_dict['loss'].backward()
        pt_grad_np = y_pred_pt.grad.numpy()
        
        np.testing.assert_allclose(jax_grad_np, pt_grad_np, rtol=RTOL_F32, atol=ATOL_F32)


@pytest.mark.unit
@pytest.mark.jax
class TestCrossEntropyFromLogitsGradient:
    """Test gradients for cross_entropy_loss_from_logits."""
    
    def test_gradient_wrt_logits(self, jax_losses, pytorch_losses):
        """Compare ∂cross_entropy/∂logits between JAX and PyTorch."""
        import jax
        import jax.numpy as jnp
        
        np.random.seed(42)
        logits = np.random.randn(2, 8, 4).astype(np.float32)
        # Normalize to probabilities for y_true
        y_true = np.exp(np.random.randn(2, 8, 4).astype(np.float32))
        y_true = y_true / y_true.sum(axis=-1, keepdims=True)
        mask = np.ones((2, 8, 4), dtype=bool)
        
        # JAX gradient
        def jax_loss_fn(logits_jax):
            return jax_losses.cross_entropy_loss_from_logits(
                y_pred_logits=logits_jax,
                y_true=jnp.array(y_true),
                mask=jnp.array(mask),
                axis=-1,
            )
        
        jax_grad = jax.grad(jax_loss_fn)(jnp.array(logits))
        jax_grad_np = np.array(jax_grad)
        
        # PyTorch gradient
        logits_pt = torch.tensor(logits, requires_grad=True)
        loss_pt = pytorch_losses.cross_entropy_loss_from_logits(
            y_pred_logits=logits_pt,
            y_true=torch.tensor(y_true),
            mask=torch.tensor(mask),
            axis=-1,
        )
        loss_pt.backward()
        pt_grad_np = logits_pt.grad.numpy()
        
        np.testing.assert_allclose(jax_grad_np, pt_grad_np, rtol=RTOL_F32, atol=ATOL_F32)


@pytest.mark.unit
@pytest.mark.jax
class TestBinaryCrossEntropyFromLogitsGradient:
    """Test gradients for binary_crossentropy_from_logits."""
    
    def test_gradient_wrt_logits(self, jax_losses, pytorch_losses):
        """Compare ∂bce/∂logits between JAX and PyTorch."""
        import jax
        import jax.numpy as jnp
        
        np.random.seed(42)
        logits = np.random.randn(2, 8, 3).astype(np.float32)
        y_true = (np.random.rand(2, 8, 3) > 0.5).astype(np.float32)
        mask = np.ones((2, 8, 3), dtype=bool)
        
        # JAX gradient
        def jax_loss_fn(logits_jax):
            return jax_losses.binary_crossentropy_from_logits(
                y_pred=logits_jax,
                y_true=jnp.array(y_true),
                mask=jnp.array(mask),
            )
        
        jax_grad = jax.grad(jax_loss_fn)(jnp.array(logits))
        jax_grad_np = np.array(jax_grad)
        
        # PyTorch gradient
        logits_pt = torch.tensor(logits, requires_grad=True)
        loss_pt = pytorch_losses.binary_crossentropy_from_logits(
            y_pred=logits_pt,
            y_true=torch.tensor(y_true),
            mask=torch.tensor(mask),
        )
        loss_pt.backward()
        pt_grad_np = logits_pt.grad.numpy()
        
        np.testing.assert_allclose(jax_grad_np, pt_grad_np, rtol=RTOL_F32, atol=ATOL_F32)


@pytest.mark.unit
@pytest.mark.jax
class TestCrossEntropyLossGradient:
    """Test gradients for cross_entropy_loss."""
    
    def test_gradient_wrt_predictions(self, jax_losses, pytorch_losses):
        """Compare ∂cross_entropy/∂y_pred between JAX and PyTorch."""
        import jax
        import jax.numpy as jnp
        
        np.random.seed(42)
        y_true = np.abs(np.random.randn(2, 8, 4).astype(np.float32)) + 0.1
        y_pred = np.abs(np.random.randn(2, 8, 4).astype(np.float32)) + 0.1
        mask = np.ones((2, 8, 4), dtype=bool)
        
        # JAX gradient
        def jax_loss_fn(y_pred_jax):
            return jax_losses.cross_entropy_loss(
                y_true=jnp.array(y_true),
                y_pred=y_pred_jax,
                mask=jnp.array(mask),
                axis=-1,
            )
        
        jax_grad = jax.grad(jax_loss_fn)(jnp.array(y_pred))
        jax_grad_np = np.array(jax_grad)
        
        # PyTorch gradient
        y_pred_pt = torch.tensor(y_pred, requires_grad=True)
        loss_pt = pytorch_losses.cross_entropy_loss(
            y_true=torch.tensor(y_true),
            y_pred=y_pred_pt,
            mask=torch.tensor(mask),
            axis=-1,
        )
        loss_pt.backward()
        pt_grad_np = y_pred_pt.grad.numpy()
        
        np.testing.assert_allclose(jax_grad_np, pt_grad_np, rtol=RTOL_F32, atol=ATOL_F32)


# =============================================================================
# Parametrized Batch/Dtype Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.jax
class TestGradientBatchAndDtype:
    """Test gradient equivalence across batch sizes and dtypes."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_poisson_gradient_batch_dtype(self, jax_losses, pytorch_losses, batch_size, dtype):
        """Compare Poisson gradients across batch sizes and dtypes."""
        import jax
        import jax.numpy as jnp

        np.random.seed(42)
        y_true = np.abs(np.random.randn(batch_size, 8, 3).astype(dtype)) + 0.1
        y_pred = np.abs(np.random.randn(batch_size, 8, 3).astype(dtype)) + 0.1
        mask = np.ones((batch_size, 1, 3), dtype=bool)

        def jax_loss_fn(y_pred_jax):
            return jax_losses.poisson_loss(
                y_true=jnp.array(y_true),
                y_pred=y_pred_jax,
                mask=jnp.array(mask),
            )

        jax_grad = jax.grad(jax_loss_fn)(jnp.array(y_pred))
        jax_grad_np = np.array(jax_grad)

        torch_dtype = torch.float32 if dtype == np.float32 else torch.float64
        y_pred_pt = torch.tensor(y_pred, requires_grad=True, dtype=torch_dtype)
        loss_pt = pytorch_losses.poisson_loss(
            y_true=torch.tensor(y_true, dtype=torch_dtype),
            y_pred=y_pred_pt,
            mask=torch.tensor(mask),
        )
        loss_pt.backward()
        pt_grad_np = y_pred_pt.grad.numpy()

        rtol = RTOL_F32 if dtype == np.float32 else RTOL_F64
        atol = ATOL_F32 if dtype == np.float32 else ATOL_F64
        np.testing.assert_allclose(jax_grad_np, pt_grad_np, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_mse_gradient_batch_dtype(self, jax_losses, pytorch_losses, batch_size, dtype):
        """Compare MSE gradients across batch sizes and dtypes."""
        import jax
        import jax.numpy as jnp

        np.random.seed(42)
        y_true = np.random.randn(batch_size, 8, 3).astype(dtype)
        y_pred = np.random.randn(batch_size, 8, 3).astype(dtype)
        mask = np.ones((batch_size, 8, 3), dtype=bool)

        def jax_loss_fn(y_pred_jax):
            return jax_losses.mse(
                y_true=jnp.array(y_true),
                y_pred=y_pred_jax,
                mask=jnp.array(mask),
            )

        jax_grad = jax.grad(jax_loss_fn)(jnp.array(y_pred))
        jax_grad_np = np.array(jax_grad)

        torch_dtype = torch.float32 if dtype == np.float32 else torch.float64
        y_pred_pt = torch.tensor(y_pred, requires_grad=True, dtype=torch_dtype)
        loss_pt = pytorch_losses.mse(
            y_true=torch.tensor(y_true, dtype=torch_dtype),
            y_pred=y_pred_pt,
            mask=torch.tensor(mask),
        )
        loss_pt.backward()
        pt_grad_np = y_pred_pt.grad.numpy()

        rtol = RTOL_F32 if dtype == np.float32 else RTOL_F64
        atol = ATOL_F32 if dtype == np.float32 else ATOL_F64
        np.testing.assert_allclose(jax_grad_np, pt_grad_np, rtol=rtol, atol=atol)
