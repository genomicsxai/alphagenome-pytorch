"""Unit tests for model heads parameter validation."""

import types

import pytest
import torch


@pytest.mark.unit
class TestHeadsParameter:
    """Tests for the heads parameter in AlphaGenome.forward()."""

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

    def test_contact_maps_head(self, model):
        """contact_maps should be a valid head name."""
        dna = torch.randn(1, 131072, 4)
        organism = torch.tensor([0])

        result = model(dna, organism, heads=("contact_maps",), resolutions=(128,))
        assert "contact_maps" in result
        # Other heads should NOT be in the result
        assert "atac" not in result

    def test_splice_junction_only_does_not_return_classification(self, model):
        """Junction-only requests may compute classification internally, but should not return it."""

        class _DummySpliceHead(torch.nn.Module):
            def __init__(self, name, seq_len):
                super().__init__()
                self.name = name
                self.seq_len = seq_len

            def forward(self, *args, **kwargs):
                if self.name == "classification":
                    return {"probs": torch.zeros(1, self.seq_len, 5)}
                return {"pred_counts": torch.zeros(1, 4, 512)}

        seq_len = 1024
        dna = torch.randn(1, seq_len, 4)
        organism = torch.tensor([0])

        model._compute_embeddings_ncl = types.MethodType(
            lambda self, dna_sequence, organism_index, resolutions: (
                torch.zeros(1, 1536, seq_len),
                torch.zeros(1, 3072, seq_len // 128),
                torch.zeros(1, 1, 1, 128),
                True,
            ),
            model,
        )
        model.splice_sites_classification_head = _DummySpliceHead("classification", seq_len)
        model.splice_sites_usage_head = None
        model.splice_sites_junction_head = _DummySpliceHead("junction", seq_len)
        model.contact_maps_head = None
        model.heads = torch.nn.ModuleDict()

        result = model(dna, organism, heads=("splice_junctions",))

        assert "splice_junctions" in result
        assert "splice_sites" not in result

    def test_default_splice_junction_mask_matches_reference_metadata(self):
        """The reference metadata exposes 367 splice-junction tissues for human and mouse."""
        from alphagenome_pytorch import AlphaGenome

        model = AlphaGenome(num_organisms=2)
        tissue_counts = model.splice_sites_junction_head.tissue_mask.sum(dim=1)
        assert tissue_counts.tolist() == [367, 367]
