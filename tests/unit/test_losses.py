"""
Unit tests for AlphaGenome PyTorch losses.

PyTorch-only tests (no JAX dependency).
For JAX equivalence tests, see tests/unit/test_losses_jax.py
"""

import pytest
import torch
import numpy as np

from alphagenome_pytorch import losses


@pytest.mark.unit
class TestSafeMaskedMean:
    """Tests for _safe_masked_mean helper."""
    
    def test_no_mask(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = losses._safe_masked_mean(x)
        assert torch.isclose(result, torch.tensor(2.5))
    
    def test_with_mask(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, True, False, False])
        result = losses._safe_masked_mean(x, mask)
        assert torch.isclose(result, torch.tensor(1.5))
    
    def test_all_masked(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([False, False, False])
        result = losses._safe_masked_mean(x, mask)
        assert torch.isclose(result, torch.tensor(0.0))


@pytest.mark.unit
class TestPoissonLoss:
    """Tests for poisson_loss."""
    
    def test_perfect_prediction(self):
        """Loss should be near zero when pred == true."""
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.ones(3, dtype=torch.bool)
        
        loss = losses.poisson_loss(y_true=y_true, y_pred=y_pred, mask=mask)
        assert loss.item() < 1e-5
    
    def test_positive_loss(self):
        """Loss should be positive for wrong predictions."""
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([3.0, 1.0, 2.0])
        mask = torch.ones(3, dtype=torch.bool)
        
        loss = losses.poisson_loss(y_true=y_true, y_pred=y_pred, mask=mask)
        assert loss.item() > 0


@pytest.mark.unit
class TestMultinomialLoss:
    """Tests for multinomial_loss - ported from JAX tests."""
    
    def test_multinomial_loss_masking(self):
        """Tests that masking correctly zeros out predictions and targets."""
        y_true = torch.tensor([[[10.0, 1.0, 3.0], [5.0, 2.0, 20.0]]])
        y_pred = torch.tensor([[[0.5, 2.5, 1.0], [2.5, 0.5, 1.0]]])
        
        loss_full = losses.multinomial_loss(
            y_true=y_true,
            y_pred=y_pred,
            mask=torch.tensor([[[True, True, True]]]),
            multinomial_resolution=1,
            positional_weight=1.0,
        )['loss']
        
        loss_masked = losses.multinomial_loss(
            y_true=y_true,
            y_pred=y_pred,
            mask=torch.tensor([[[True, True, False]]]),
            multinomial_resolution=1,
            positional_weight=1.0,
        )['loss']
        
        y_true_zero = torch.tensor([[[10.0, 1.0], [5.0, 2.0]]])
        y_pred_zero = torch.tensor([[[0.5, 2.5], [2.5, 0.5]]])
        
        loss_truncated = losses.multinomial_loss(
            y_true=y_true_zero,
            y_pred=y_pred_zero,
            mask=torch.tensor([[[True, True]]]),
            multinomial_resolution=1,
            positional_weight=1.0,
        )['loss']
        
        np.testing.assert_almost_equal(
            loss_masked.item(), loss_truncated.item(), decimal=5
        )
        assert loss_masked.item() < loss_full.item()
    
    def test_multinomial_loss_resolution_aggregation(self):
        """Tests the resolution aggregation logic."""
        y_true = torch.ones((1, 4, 1))
        y_pred = torch.ones((1, 4, 1))
        mask = torch.ones((1, 1, 1), dtype=torch.bool)
        
        out_res1 = losses.multinomial_loss(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            multinomial_resolution=1,
            positional_weight=1.0,
        )
        
        out_res4 = losses.multinomial_loss(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            multinomial_resolution=4,
            positional_weight=1.0,
        )
        
        assert torch.isfinite(out_res1['loss']).all()
        assert torch.isfinite(out_res4['loss']).all()
        np.testing.assert_almost_equal(out_res1['max_sum_preds'].item(), 1.0)
        np.testing.assert_almost_equal(out_res4['max_sum_preds'].item(), 4.0)

    def test_count_weight_scales_count_component(self):
        """Tests that count_weight scales the count loss while other outputs remain unchanged."""
        y_true = torch.tensor([[[10.0, 5.0], [5.0, 10.0]]])
        y_pred = torch.tensor([[[8.0, 6.0], [4.0, 12.0]]])
        mask = torch.ones((1, 1, 2), dtype=torch.bool)

        out_cw1 = losses.multinomial_loss(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            multinomial_resolution=2,
            positional_weight=1.0,
            count_weight=1.0,
        )

        out_cw2 = losses.multinomial_loss(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            multinomial_resolution=2,
            positional_weight=1.0,
            count_weight=2.0,
        )

        # The individual components should be identical
        np.testing.assert_almost_equal(
            out_cw1['loss_total'].item(), out_cw2['loss_total'].item(), decimal=6
        )
        np.testing.assert_almost_equal(
            out_cw1['loss_positional'].item(), out_cw2['loss_positional'].item(), decimal=6
        )
        np.testing.assert_almost_equal(
            out_cw1['max_sum_preds'].item(), out_cw2['max_sum_preds'].item(), decimal=6
        )
        np.testing.assert_almost_equal(
            out_cw1['max_preds'].item(), out_cw2['max_preds'].item(), decimal=6
        )
        np.testing.assert_almost_equal(
            out_cw1['max_targets'].item(), out_cw2['max_targets'].item(), decimal=6
        )

        # Combined loss should differ by exactly loss_total (the scaled component)
        loss_diff = out_cw2['loss'].item() - out_cw1['loss'].item()
        np.testing.assert_almost_equal(
            loss_diff, out_cw1['loss_total'].item(), decimal=6
        )

    def test_multinomial_loss_ncl_parity(self):
        """Tests parity between NCL and NLC formats."""
        # NLC: (B, S, C) = (1, 4, 2)
        y_true_nlc = torch.tensor([[[10.0, 1.0], [5.0, 2.0], [8.0, 3.0], [2.0, 10.0]]])
        y_pred_nlc = torch.tensor([[[9.0, 1.5], [6.0, 1.8], [7.5, 3.2], [2.5, 9.5]]])
        mask_nlc = torch.tensor([[[True, True]]])  # (1, 1, 2)

        out_nlc = losses.multinomial_loss(
            y_true=y_true_nlc,
            y_pred=y_pred_nlc,
            mask=mask_nlc,
            multinomial_resolution=2,
            positional_weight=1.0,
            channels_last=True,
        )

        # NCL: (B, C, S) = (1, 2, 4)
        y_true_ncl = y_true_nlc.transpose(1, 2)
        y_pred_ncl = y_pred_nlc.transpose(1, 2)
        mask_ncl = mask_nlc.transpose(1, 2)  # (1, 2, 1)

        out_ncl = losses.multinomial_loss(
            y_true=y_true_ncl,
            y_pred=y_pred_ncl,
            mask=mask_ncl,
            multinomial_resolution=2,
            positional_weight=1.0,
            channels_last=False,
        )

        assert torch.isclose(out_nlc['loss'], out_ncl['loss'])
        assert torch.isclose(out_nlc['loss_total'], out_ncl['loss_total'])
        assert torch.isclose(out_nlc['loss_positional'], out_ncl['loss_positional'])


@pytest.mark.unit
class TestMSE:
    """Tests for mse loss."""
    
    def test_perfect_prediction(self):
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.ones(3, dtype=torch.bool)
        
        loss = losses.mse(y_pred=y_pred, y_true=y_true, mask=mask)
        assert loss.item() == 0.0
    
    def test_simple_error(self):
        y_true = torch.tensor([0.0, 0.0])
        y_pred = torch.tensor([1.0, 2.0])
        mask = torch.ones(2, dtype=torch.bool)
        
        loss = losses.mse(y_pred=y_pred, y_true=y_true, mask=mask)
        # (1^2 + 2^2) / 2 = 2.5
        assert torch.isclose(loss, torch.tensor(2.5))


@pytest.mark.unit
class TestCrossEntropyLossFromLogits:
    """Tests for cross_entropy_loss_from_logits."""
    
    def test_basic(self):
        # Simple 3-class classification
        y_pred_logits = torch.tensor([[1.0, 0.0, 0.0]])
        y_true = torch.tensor([[1.0, 0.0, 0.0]])  # One-hot
        mask = torch.ones((1, 3), dtype=torch.bool)
        
        loss = losses.cross_entropy_loss_from_logits(
            y_pred_logits=y_pred_logits,
            y_true=y_true,
            mask=mask,
            axis=-1,
        )
        assert torch.isfinite(loss)
        assert loss.item() >= 0
    
    def test_perfect_prediction_low_loss(self):
        """Strong logit for correct class should give low loss."""
        y_pred_logits = torch.tensor([[10.0, -10.0, -10.0]])
        y_true = torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.ones((1, 3), dtype=torch.bool)
        
        loss = losses.cross_entropy_loss_from_logits(
            y_pred_logits=y_pred_logits,
            y_true=y_true,
            mask=mask,
            axis=-1,
        )
        assert loss.item() < 0.1


@pytest.mark.unit
class TestBinaryCrossEntropyFromLogits:
    """Tests for binary_crossentropy_from_logits."""
    
    def test_perfect_prediction(self):
        # Large positive logit for true=1, large negative for true=0
        y_pred = torch.tensor([10.0, -10.0])
        y_true = torch.tensor([1.0, 0.0])
        mask = torch.ones(2, dtype=torch.bool)
        
        loss = losses.binary_crossentropy_from_logits(
            y_pred=y_pred, y_true=y_true, mask=mask
        )
        assert loss.item() < 1e-3
    
    def test_numerical_stability(self):
        """Test with extreme logits for numerical stability."""
        y_pred = torch.tensor([100.0, -100.0, 50.0])
        y_true = torch.tensor([1.0, 0.0, 1.0])
        mask = torch.ones(3, dtype=torch.bool)
        
        loss = losses.binary_crossentropy_from_logits(
            y_pred=y_pred, y_true=y_true, mask=mask
        )
        assert torch.isfinite(loss)


@pytest.mark.unit
class TestCrossEntropyLoss:
    """Tests for cross_entropy_loss on counts."""
    
    def test_basic(self):
        """Basic test with count data."""
        y_true = torch.tensor([[1.0, 2.0, 3.0]])
        y_pred = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.ones((1, 3), dtype=torch.bool)
        
        loss = losses.cross_entropy_loss(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            axis=-1,
        )
        assert torch.isfinite(loss)
        assert loss.item() >= 0
    
    def test_non_negative(self):
        """Loss should be non-negative."""
        torch.manual_seed(42)
        y_true = torch.abs(torch.randn(2, 5)) + 0.1
        y_pred = torch.abs(torch.randn(2, 5)) + 0.1
        mask = torch.ones((2, 5), dtype=torch.bool)
        
        loss = losses.cross_entropy_loss(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            axis=-1,
        )
        assert loss.item() >= 0
    
    def test_with_mask(self):
        """Test masking functionality."""
        y_true = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        y_pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask_full = torch.ones((1, 4), dtype=torch.bool)
        mask_partial = torch.tensor([[True, True, False, False]])
        
        loss_full = losses.cross_entropy_loss(
            y_true=y_true, y_pred=y_pred, mask=mask_full, axis=-1
        )
        loss_partial = losses.cross_entropy_loss(
            y_true=y_true, y_pred=y_pred, mask=mask_partial, axis=-1
        )
        
        # Both should be valid
        assert torch.isfinite(loss_full)
        assert torch.isfinite(loss_partial)


@pytest.mark.unit
class TestExtremeValues:
    """Tests for numerical stability with extreme inputs."""

    def test_poisson_loss_tiny_predictions(self):
        """Poisson loss with very small predictions should not produce NaN/Inf."""
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1e-7, 1e-7, 1e-7])
        mask = torch.ones(3, dtype=torch.bool)

        loss = losses.poisson_loss(y_true=y_true, y_pred=y_pred, mask=mask)
        assert torch.isfinite(loss), f"Loss is {loss.item()}"

    def test_poisson_loss_large_predictions(self):
        """Poisson loss with very large predictions should not produce NaN/Inf."""
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1e7, 1e7, 1e7])
        mask = torch.ones(3, dtype=torch.bool)

        loss = losses.poisson_loss(y_true=y_true, y_pred=y_pred, mask=mask)
        assert torch.isfinite(loss), f"Loss is {loss.item()}"

    def test_mse_extreme_difference(self):
        """MSE with large differences should still be finite."""
        y_true = torch.tensor([0.0, 0.0])
        y_pred = torch.tensor([1e6, -1e6])
        mask = torch.ones(2, dtype=torch.bool)

        loss = losses.mse(y_pred=y_pred, y_true=y_true, mask=mask)
        assert torch.isfinite(loss)


@pytest.mark.unit
class TestGradientThroughLoss:
    """Tests that backward() produces finite gradients for all loss functions."""

    def test_poisson_loss_gradient(self):
        """Poisson loss should produce finite gradients."""
        y_true = torch.abs(torch.randn(2, 8, 3)) + 0.1
        y_pred = torch.abs(torch.randn(2, 8, 3)) + 0.1
        y_pred.requires_grad_(True)
        mask = torch.ones(2, 1, 3, dtype=torch.bool)

        loss = losses.poisson_loss(y_true=y_true, y_pred=y_pred, mask=mask)
        loss.backward()

        assert y_pred.grad is not None
        assert torch.all(torch.isfinite(y_pred.grad))

    def test_mse_gradient(self):
        """MSE should produce finite gradients."""
        y_true = torch.randn(2, 8, 3)
        y_pred = torch.randn(2, 8, 3, requires_grad=True)
        mask = torch.ones(2, 8, 3, dtype=torch.bool)

        loss = losses.mse(y_pred=y_pred, y_true=y_true, mask=mask)
        loss.backward()

        assert y_pred.grad is not None
        assert torch.all(torch.isfinite(y_pred.grad))

    def test_multinomial_loss_gradient(self):
        """Multinomial loss should produce finite gradients."""
        y_true = torch.abs(torch.randn(2, 8, 3)) + 0.1
        y_pred = torch.abs(torch.randn(2, 8, 3)) + 0.1
        y_pred.requires_grad_(True)
        mask = torch.ones(2, 1, 3, dtype=torch.bool)

        result = losses.multinomial_loss(
            y_true=y_true, y_pred=y_pred, mask=mask,
            multinomial_resolution=4, positional_weight=1.0,
        )
        result['loss'].backward()

        assert y_pred.grad is not None
        assert torch.all(torch.isfinite(y_pred.grad))

    def test_cross_entropy_from_logits_gradient(self):
        """Cross entropy from logits should produce finite gradients."""
        logits = torch.randn(2, 8, 4, requires_grad=True)
        y_true = torch.softmax(torch.randn(2, 8, 4), dim=-1)
        mask = torch.ones(2, 8, 4, dtype=torch.bool)

        loss = losses.cross_entropy_loss_from_logits(
            y_pred_logits=logits, y_true=y_true, mask=mask, axis=-1,
        )
        loss.backward()

        assert logits.grad is not None
        assert torch.all(torch.isfinite(logits.grad))

    def test_bce_from_logits_gradient(self):
        """Binary cross entropy from logits should produce finite gradients."""
        logits = torch.randn(2, 8, 3, requires_grad=True)
        y_true = (torch.rand(2, 8, 3) > 0.5).float()
        mask = torch.ones(2, 8, 3, dtype=torch.bool)

        loss = losses.binary_crossentropy_from_logits(
            y_pred=logits, y_true=y_true, mask=mask,
        )
        loss.backward()

        assert logits.grad is not None
        assert torch.all(torch.isfinite(logits.grad))


@pytest.mark.unit
class TestGoldenValues:
    """Golden value tests: verify exact outputs for known inputs.

    These catch silent numerical changes that property-based tests miss.
    If a test here fails, the loss function's numerical behavior has changed.
    """

    def test_poisson_loss_golden(self):
        """Poisson loss golden value."""
        y_pred = torch.tensor([[[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]])
        y_true = torch.tensor([[[0.8, 2.2, 2.9], [0.6, 1.4, 2.6]]])
        mask = torch.ones(1, 1, 3, dtype=torch.bool)

        loss = losses.poisson_loss(y_true=y_true, y_pred=y_pred, mask=mask)
        assert torch.isclose(loss, torch.tensor(0.0079382462), atol=1e-7)

    def test_mse_golden(self):
        """MSE loss golden value."""
        y_pred = torch.tensor([[[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]])
        y_true = torch.tensor([[[0.8, 2.2, 2.9], [0.6, 1.4, 2.6]]])
        mask = torch.ones(1, 1, 3, dtype=torch.bool)

        loss = losses.mse(y_true=y_true, y_pred=y_pred, mask=mask)
        assert torch.isclose(loss, torch.tensor(0.0199999977), atol=1e-7)

    def test_cross_entropy_from_logits_golden(self):
        """Cross-entropy from logits golden value."""
        logits = torch.tensor([[[2.0, 0.5, -1.0, 0.1]]])
        targets = torch.tensor([[[0.9, 0.05, 0.03, 0.02]]])
        mask = torch.ones(1, 1, 1, dtype=torch.bool)

        loss = losses.cross_entropy_loss_from_logits(
            y_pred_logits=logits, y_true=targets, mask=mask, axis=-1,
        )
        assert torch.isclose(loss, torch.tensor(0.5554059148), atol=1e-7)

    def test_binary_crossentropy_from_logits_golden(self):
        """Binary cross-entropy from logits golden value."""
        y_pred = torch.tensor([[[2.0, -1.0, 0.5]]])
        y_true = torch.tensor([[[1.0, 0.0, 1.0]]])
        mask = torch.ones(1, 1, 3, dtype=torch.bool)

        loss = losses.binary_crossentropy_from_logits(
            y_pred=y_pred, y_true=y_true, mask=mask,
        )
        assert torch.isclose(loss, torch.tensor(0.3047555685), atol=1e-7)

    def test_multinomial_loss_golden(self):
        """Multinomial loss golden value."""
        y_pred = torch.tensor([[[1.0, 2.0, 0.5, 3.0], [0.5, 1.5, 0.8, 2.0]]])
        y_true = torch.tensor([[[10.0, 30.0, 5.0, 20.0], [8.0, 15.0, 3.0, 12.0]]])
        mask = torch.ones(1, 1, 4, dtype=torch.bool)

        result = losses.multinomial_loss(
            y_true=y_true, y_pred=y_pred, mask=mask,
            multinomial_resolution=2, positional_weight=1.0,
        )
        assert torch.isclose(result['loss'], torch.tensor(26.4599399567), atol=1e-4)
        assert torch.isclose(result['loss_total'], torch.tensor(17.7364959717), atol=1e-4)
        assert torch.isclose(result['loss_positional'], torch.tensor(8.7234439850), atol=1e-4)
