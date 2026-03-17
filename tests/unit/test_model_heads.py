"""Unit tests for model heads parameter validation."""

import pytest
import torch

from alphagenome_pytorch import AlphaGenome


@pytest.mark.unit
class TestHeadsParameter:
    """Tests for the heads parameter in AlphaGenome.forward()."""

    @pytest.fixture
    def model(self):
        """Create a minimal model for testing."""
        return AlphaGenome()

    def test_unknown_head_raises_error(self, model):
        """Specifying unknown head names should raise ValueError."""
        dna = torch.randn(1, 131072, 4)
        organism = torch.tensor([0])

        with pytest.raises(ValueError, match="Unknown head names"):
            model(dna, organism, heads=("nonexistent_head",))

    def test_unknown_head_error_message_includes_available(self, model):
        """Error message should include list of available heads."""
        dna = torch.randn(1, 131072, 4)
        organism = torch.tensor([0])

        with pytest.raises(ValueError, match="Available heads:"):
            model(dna, organism, heads=("nonexistent",))

    def test_valid_head_succeeds(self, model):
        """Valid head names should work without error."""
        dna = torch.randn(1, 131072, 4)
        organism = torch.tensor([0])

        # Should not raise
        result = model(dna, organism, heads=("atac",), resolutions=(128,))
        assert "atac" in result
        # Other heads should NOT be in the result
        assert "dnase" not in result
        assert "cage" not in result

    def test_multiple_valid_heads(self, model):
        """Multiple valid heads should work."""
        dna = torch.randn(1, 131072, 4)
        organism = torch.tensor([0])

        result = model(dna, organism, heads=("atac", "dnase"), resolutions=(128,))
        assert "atac" in result
        assert "dnase" in result
        assert "cage" not in result

    def test_heads_none_returns_all(self, model):
        """heads=None should compute all heads."""
        dna = torch.randn(1, 131072, 4)
        organism = torch.tensor([0])

        result = model(dna, organism, heads=None, resolutions=(128,))
        # Should have all the main heads
        assert "atac" in result
        assert "dnase" in result
        assert "cage" in result
        assert "rna_seq" in result
        assert "chip_tf" in result
        assert "chip_histone" in result

    def test_pair_activations_head(self, model):
        """pair_activations should be a valid head name."""
        dna = torch.randn(1, 131072, 4)
        organism = torch.tensor([0])

        result = model(dna, organism, heads=("pair_activations",), resolutions=(128,))
        assert "pair_activations" in result
        # Other heads should NOT be in the result
        assert "atac" not in result
