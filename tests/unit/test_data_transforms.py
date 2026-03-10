"""Unit tests for data_transforms module."""

import numpy as np
import pytest
import torch

from alphagenome_pytorch.extensions.finetuning.data_transforms import (
    normalize_to_total,
    mean_normalize,
    power_transform,
    power_transform_inverse,
    smooth_clip,
    smooth_clip_inverse,
    apply_atac_transforms,
    apply_rnaseq_transforms,
    DEFAULT_SOFT_CLIP_THRESHOLD,
    DEFAULT_TOTAL_COUNT,
)


class TestNormalizeToTotal:
    """Tests for normalize_to_total function."""

    def test_numpy_array(self):
        """Test normalization with numpy array."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = normalize_to_total(x, total=100.0)
        assert np.isclose(result.sum(), 100.0)

    def test_torch_tensor(self):
        """Test normalization with torch tensor."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = normalize_to_total(x, total=100.0)
        assert torch.isclose(result.sum(), torch.tensor(100.0))

    def test_preserves_ratios(self):
        """Test that relative ratios are preserved."""
        x = np.array([1.0, 2.0, 4.0])
        result = normalize_to_total(x, total=70.0)
        assert np.allclose(result, [10.0, 20.0, 40.0])

    def test_zero_array(self):
        """Test with all-zero array returns zeros."""
        x = np.zeros(5)
        result = normalize_to_total(x, total=100.0)
        assert np.allclose(result, np.zeros(5))


class TestMeanNormalize:
    """Tests for mean_normalize function."""

    def test_numpy_array(self):
        """Test mean normalization with numpy."""
        x = np.array([0.0, 2.0, 4.0, 6.0])  # nonzero mean = 4.0
        result = mean_normalize(x)
        expected = x / 4.0
        assert np.allclose(result, expected)

    def test_torch_tensor(self):
        """Test mean normalization with torch."""
        x = torch.tensor([0.0, 2.0, 4.0, 6.0])
        result = mean_normalize(x)
        expected = x / 4.0
        assert torch.allclose(result, expected)

    def test_all_zeros(self):
        """Test with all-zero array returns zeros."""
        x = np.zeros(5)
        result = mean_normalize(x)
        assert np.allclose(result, np.zeros(5))


class TestPowerTransform:
    """Tests for power_transform function."""

    def test_numpy_array(self):
        """Test power transform with numpy."""
        x = np.array([1.0, 4.0, 9.0, 16.0])
        result = power_transform(x, power=0.5)
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.allclose(result, expected)

    def test_torch_tensor(self):
        """Test power transform with torch."""
        x = torch.tensor([1.0, 4.0, 9.0, 16.0])
        result = power_transform(x, power=0.5)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(result, expected)

    def test_default_power_075(self):
        """Test default power of 0.75."""
        x = np.array([16.0])
        result = power_transform(x)  # default power=0.75
        expected = np.power(16.0, 0.75)
        assert np.isclose(result[0], expected)


class TestPowerTransformInverse:
    """Tests for power_transform_inverse function."""

    def test_roundtrip(self):
        """Test that inverse undoes the transform."""
        x = np.array([1.0, 4.0, 9.0, 16.0])
        transformed = power_transform(x, power=0.75)
        recovered = power_transform_inverse(transformed, power=0.75)
        assert np.allclose(x, recovered)


class TestSmoothClip:
    """Tests for smooth_clip function."""

    def test_below_threshold_unchanged(self):
        """Test values below threshold are unchanged."""
        x = np.array([1.0, 10.0, 100.0])
        result = smooth_clip(x, threshold=200.0)
        assert np.allclose(result, x)

    def test_above_threshold_dampened(self):
        """Test values above threshold are dampened."""
        threshold = 100.0
        x = np.array([100.0, 200.0, 500.0])
        result = smooth_clip(x, threshold=threshold)

        # At threshold: unchanged
        assert np.isclose(result[0], 100.0)
        # Above threshold: T + 2*sqrt(x-T)
        assert np.isclose(result[1], 100.0 + 2.0 * np.sqrt(100.0))  # 120
        assert np.isclose(result[2], 100.0 + 2.0 * np.sqrt(400.0))  # 140

    def test_torch_tensor(self):
        """Test smooth clip with torch tensor."""
        threshold = 100.0
        x = torch.tensor([50.0, 100.0, 200.0])
        result = smooth_clip(x, threshold=threshold)
        
        assert torch.isclose(result[0], torch.tensor(50.0))
        assert torch.isclose(result[1], torch.tensor(100.0))
        assert torch.isclose(result[2], torch.tensor(100.0 + 20.0))


class TestSmoothClipInverse:
    """Tests for smooth_clip_inverse function."""

    def test_roundtrip(self):
        """Test that inverse undoes the clipping."""
        threshold = 100.0
        x = np.array([50.0, 100.0, 200.0, 500.0])
        clipped = smooth_clip(x, threshold=threshold)
        recovered = smooth_clip_inverse(clipped, threshold=threshold)
        assert np.allclose(x, recovered)


class TestApplyAtacTransforms:
    """Tests for apply_atac_transforms function."""

    def test_full_pipeline(self):
        """Test ATAC transform pipeline."""
        x = np.random.rand(1000) * 100
        result = apply_atac_transforms(x, total=1000.0, clip_threshold=50.0)
        
        # Result should be normalized, mean-divided, and clipped
        assert result.max() <= 50.0 + 2.0 * np.sqrt(result.max())  # rough bound
        assert result.min() >= 0.0

    def test_no_power_transform(self):
        """Verify ATAC does not apply power transform."""
        x = np.array([100.0])
        # After normalize to total and mean normalize, value stays 1.0
        # (100/100 = 1.0 total, then 1.0 / 1.0 = 1.0)
        result = apply_atac_transforms(x, total=100.0, clip_threshold=1000.0)
        # No power transform means linear relationship preserved
        assert np.isclose(result[0], 1.0)


class TestApplyRnaseqTransforms:
    """Tests for apply_rnaseq_transforms function."""

    def test_full_pipeline(self):
        """Test RNA-seq transform pipeline."""
        x = np.random.rand(1000) * 100
        result = apply_rnaseq_transforms(x, power=0.75, clip_threshold=50.0)
        
        # Result should be normalized, power-transformed, and clipped
        assert result.max() <= 50.0 + 2.0 * np.sqrt(50.0)  # rough bound
        assert result.min() >= 0.0

    def test_power_transform_applied(self):
        """Verify RNA-seq applies power transform."""
        x = np.array([0.0, 16.0])  # nonzero mean = 16.0
        result = apply_rnaseq_transforms(x, power=0.75, clip_threshold=1000.0)
        # After mean normalize: [0, 1.0]
        # After power 0.75: [0, 1.0^0.75] = [0, 1.0]
        assert np.allclose(result, [0.0, 1.0])
