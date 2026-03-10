"""
Unit tests for fine-tuning extensions (heads only).

Tests the create_finetuning_head factory function.
For dataset tests, see tests/unit/test_finetuning_datasets.py.
For integration tests with full model, see tests/integration/test_finetuning_heads_integration.py.
"""

import pytest
import torch

from alphagenome_pytorch.extensions.finetuning.heads import (
    create_finetuning_head,
    ASSAY_TYPES,
)

@pytest.mark.unit
class TestCreateFinetuningHeadATAC:
    """Tests for create_finetuning_head with ATAC assay type."""

    def test_forward_shape(self):
        """Test forward pass produces correct shapes."""
        head = create_finetuning_head('atac', n_tracks=5)
        # Input NCL format: (B, C, S)
        embeddings_dict = {
            1: torch.randn(2, 1536, 131072),
            128: torch.randn(2, 3072, 1024),
        }
        organism_index = torch.tensor([0, 0])

        outputs = head(embeddings_dict, organism_index)

        # Output NLC format: (B, S, T)
        assert 1 in outputs
        assert 128 in outputs
        assert outputs[1].shape == (2, 131072, 5)
        assert outputs[128].shape == (2, 1024, 5)

    def test_output_non_negative(self):
        """Test output is non-negative (softplus activation)."""
        head = create_finetuning_head('atac', n_tracks=3, resolutions=(128,))
        # Input NCL format: (B, C, S)
        embeddings_dict = {128: torch.randn(2, 3072, 1024)}
        organism_index = torch.tensor([0, 0])

        outputs = head(embeddings_dict, organism_index)

        assert (outputs[128] >= 0).all()

    def test_single_resolution(self):
        """Test single 128bp resolution."""
        head = create_finetuning_head('atac', n_tracks=5, resolutions=(128,))
        # Input NCL format: (B, C, S)
        embeddings_dict = {128: torch.randn(2, 3072, 1024)}
        organism_index = torch.tensor([0, 0])

        outputs = head(embeddings_dict, organism_index)
        assert 128 in outputs
        assert 1 not in outputs
        assert outputs[128].shape == (2, 1024, 5)

    def test_no_squashing(self):
        """Test ATAC head does not apply squashing."""
        head = create_finetuning_head('atac', n_tracks=3)
        assert not head.apply_squashing


@pytest.mark.unit
class TestCreateFinetuningHeadRNASeq:
    """Tests for create_finetuning_head with RNA-seq assay type."""

    def test_forward_dual_resolution(self):
        """Test forward pass with both 1bp and 128bp."""
        head = create_finetuning_head('rna_seq', n_tracks=5, resolutions=(1, 128))
        # Input NCL format: (B, C, S)
        embeddings_dict = {
            1: torch.randn(2, 1536, 100),
            128: torch.randn(2, 3072, 100),
        }
        organism_index = torch.tensor([0, 0])

        outputs = head(embeddings_dict, organism_index)

        # Output NLC format: (B, S, T)
        assert 1 in outputs
        assert 128 in outputs
        assert outputs[1].shape == (2, 100, 5)
        assert outputs[128].shape == (2, 100, 5)

    def test_forward_single_resolution(self):
        """Test forward pass with only 128bp."""
        head = create_finetuning_head('rna_seq', n_tracks=3, resolutions=(128,))
        # Input NCL format: (B, C, S)
        embeddings_dict = {128: torch.randn(2, 3072, 100)}
        organism_index = torch.tensor([0, 0])

        outputs = head(embeddings_dict, organism_index)

        assert 128 in outputs
        assert 1 not in outputs
        assert outputs[128].shape == (2, 100, 3)

    def test_output_non_negative(self):
        """Test output is non-negative (softplus activation)."""
        head = create_finetuning_head('rna_seq', n_tracks=3, resolutions=(128,))
        # Input NCL format: (B, C, S)
        embeddings_dict = {128: torch.randn(2, 3072, 100)}
        organism_index = torch.tensor([0, 0])

        outputs = head(embeddings_dict, organism_index)

        assert (outputs[128] >= 0).all()

@pytest.mark.unit
class TestAssayTypes:
    """Tests for ASSAY_TYPES configuration dict."""

    def test_all_assay_types_present(self):
        """Test all expected assay types are defined."""
        expected = {'rna_seq', 'atac', 'dnase', 'procap', 'cage', 'chip_tf', 'chip_histone'}
        assert set(ASSAY_TYPES.keys()) == expected

    def test_only_rnaseq_has_squashing(self):
        """Test only RNA-seq has apply_squashing=True."""
        for assay_type, config in ASSAY_TYPES.items():
            if assay_type == 'rna_seq':
                assert config['apply_squashing'] is True
            else:
                assert config['apply_squashing'] is False

    def test_chip_types_128bp_only(self):
        """Test ChIP types default to 128bp resolution only."""
        assert ASSAY_TYPES['chip_tf']['default_resolutions'] == (128,)
        assert ASSAY_TYPES['chip_histone']['default_resolutions'] == (128,)

    def test_other_types_dual_resolution(self):
        """Test non-ChIP types default to dual resolution."""
        dual_res_types = ['rna_seq', 'atac', 'dnase', 'procap', 'cage']
        for assay_type in dual_res_types:
            assert ASSAY_TYPES[assay_type]['default_resolutions'] == (1, 128)


@pytest.mark.unit
class TestCreateFinetuningHeadAllModalities:
    """Tests for create_finetuning_head with all supported modalities."""

    @pytest.mark.parametrize("assay_type", ['dnase', 'procap', 'cage'])
    def test_dual_resolution_modalities(self, assay_type):
        """Test modalities that support both 1bp and 128bp."""
        head = create_finetuning_head(assay_type, n_tracks=5)
        # Input NCL format: (B, C, S)
        embeddings_dict = {
            1: torch.randn(2, 1536, 100),
            128: torch.randn(2, 3072, 100),
        }
        organism_index = torch.tensor([0, 0])

        outputs = head(embeddings_dict, organism_index)

        # Output NLC format: (B, S, T)
        assert 1 in outputs
        assert 128 in outputs
        assert outputs[1].shape == (2, 100, 5)
        assert outputs[128].shape == (2, 100, 5)
        assert not head.apply_squashing

    @pytest.mark.parametrize("assay_type", ['chip_tf', 'chip_histone'])
    def test_chip_modalities_default_128bp(self, assay_type):
        """Test ChIP modalities default to 128bp only."""
        head = create_finetuning_head(assay_type, n_tracks=10)
        # Input NCL format: (B, C, S)
        embeddings_dict = {128: torch.randn(2, 3072, 100)}
        organism_index = torch.tensor([0, 0])

        outputs = head(embeddings_dict, organism_index)

        assert 128 in outputs
        assert 1 not in outputs
        assert outputs[128].shape == (2, 100, 10)
        assert not head.apply_squashing

    def test_chip_modality_override_resolution(self):
        """Test ChIP modality can override default resolution."""
        head = create_finetuning_head('chip_tf', n_tracks=10, resolutions=(1, 128))
        # Input NCL format: (B, C, S)
        embeddings_dict = {
            1: torch.randn(2, 1536, 100),
            128: torch.randn(2, 3072, 100),
        }
        organism_index = torch.tensor([0, 0])

        outputs = head(embeddings_dict, organism_index)

        assert 1 in outputs
        assert 128 in outputs

    def test_invalid_assay_type_raises(self):
        """Test invalid assay type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid assay type"):
            create_finetuning_head('invalid_type', n_tracks=5)

    def test_invalid_resolution_raises(self):
        """Test invalid resolution raises ValueError."""
        with pytest.raises(ValueError, match="Invalid resolution"):
            create_finetuning_head('atac', n_tracks=5, resolutions=(64,))

    @pytest.mark.parametrize("assay_type", list(ASSAY_TYPES.keys()))
    def test_all_modalities_output_non_negative(self, assay_type):
        """Test all modalities produce non-negative outputs."""
        head = create_finetuning_head(assay_type, n_tracks=3, resolutions=(128,))
        # Input NCL format: (B, C, S)
        embeddings_dict = {128: torch.randn(2, 3072, 100)}
        organism_index = torch.tensor([0, 0])

        outputs = head(embeddings_dict, organism_index)

        assert (outputs[128] >= 0).all()


@pytest.mark.unit
class TestFinetuningHeadGradients:
    """Tests for gradient flow through finetuning heads."""

    def test_gradient_flows_through_head(self):
        """Gradients should flow through the head to embeddings."""
        head = create_finetuning_head('atac', n_tracks=3, resolutions=(128,))
        # Input NCL format: (B, C, S)
        embeddings = torch.randn(2, 3072, 100, requires_grad=True)
        embeddings_dict = {128: embeddings}
        organism_index = torch.tensor([0, 0])

        outputs = head(embeddings_dict, organism_index)
        loss = outputs[128].sum()
        loss.backward()

        assert embeddings.grad is not None
        assert torch.all(torch.isfinite(embeddings.grad))

    @pytest.mark.parametrize("n_tracks", [1, 5, 50])
    def test_different_track_counts(self, n_tracks):
        """Head should produce correct output for various track counts."""
        head = create_finetuning_head('atac', n_tracks=n_tracks, resolutions=(128,))
        # Input NCL format: (B, C, S)
        embeddings_dict = {128: torch.randn(1, 3072, 100)}
        organism_index = torch.tensor([0])

        outputs = head(embeddings_dict, organism_index)
        assert outputs[128].shape == (1, 100, n_tracks)
